# agent/llm_brain.py
import anthropic
import json
import re
import time
from typing import Optional
from agent.object_registry import ObjectRegistry
from agent.lessons import read_lessons_section
from agent.world_model import read_world_model, render_for_system_prompt
from recording import Recorder, LLMSessionLog


MODELS = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-7",
}

DEFAULT_MODEL = "haiku"


def resolve_model(short_or_full: str) -> str:
    return MODELS.get(short_or_full, short_or_full)


def prompt_model_choice() -> str:
    print("\nChoose Claude model for this session:")
    print("  1) Haiku  4.5  - fastest, cheapest, default for routine tasks")
    print("  2) Sonnet 4.6  - balanced")
    print("  3) Opus   4.7  - most capable, slowest, for novel/complex tasks")
    try:
        choice = input("\nSelection [1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""
    if choice in ("", "1"): return "haiku"
    if choice == "2": return "sonnet"
    if choice == "3": return "opus"
    print(f"Unrecognized '{choice}' - using haiku")
    return "haiku"


SYSTEM_PROMPT_TEMPLATE = """\
You are the control brain for a UFACTORY xArm6 mounted on a 700mm linear rail \
in a benchmark pick-and-place environment.

## Your 7 degrees of freedom
- Rail: 0-700mm linear axis along X. Move rail FIRST to get the arm near the target.
- Joints 1-6: rotational axes of the xArm6.

## Command vocabulary (output as JSON array)
- home            params: {{}}  — drive rail to 350mm and all six joints to 0 (the canonical home pose). Use this for "go home", "reset", "return to start", or to put the arm in a known safe pose between sub-tasks.
- wave_goodbye    params: {{"n_waves": 3}}  — move to mid-rail and wave the arm side-to-side N times. Use this for "wave goodbye", "say goodbye", "wave at me", "say hi", or similar greeting/farewell gestures. Default 3 waves; pass n_waves to change.
- set_rail        params: position_mm, speed_mm_s
- move_to         params: x, y, z, roll, pitch, yaw, speed_mm_s (mm, deg)
- set_joints      params: angles_deg (6 floats)
- gripper_open    params: {{}}
- gripper_close   params: {{}}
- place_tube_in_rack  params: {{"rack_name": "left_tube_rack" | "right_tube_rack"}}  — while holding a tube, this finds the first open slot in the named rack, flies the arm there, and seats the tube. Use this when the task says "put the tube in the other rack" / "place tube in an open slot" / "move tube to rack X".
- push_object     params: {{"target_name": <body name>, "to_x_mm": <float>, "to_y_mm": <float>}}  — slide/carry an object across the bench to a target xy, or past the bench edge to push it off (it falls to the floor under gravity). Bench extents: x ∈ [-750, +750] mm, y ∈ [-450, +450] mm. To push something OFF the bench, pass a target xy past those bounds (e.g. y=550 for past the front edge).
  Valid `target_name` values (any of these can be pushed):
    - Cubes: `red_cube`, `green_cube`, `blue_cube`
    - Bins:  `red_bin`, `green_bin`, `blue_bin`
    - Tube racks: `left_tube_rack`, `right_tube_rack` (each contains 3 tubes that come along automatically)
    - Falcon tubes: `tube_L1`, `tube_L2`, `tube_L3`, `tube_R1`, `tube_R2`, `tube_R3`
  When the task references multiple objects ("all objects", "everything on the bench", "clear the table", "all the things"), iterate ALL of the bodies above and emit one push_object per body. **Do not skip racks** — pushing a rack carries its 3 tubes with it, so a single push_object on a rack removes 4 things from the bench at once. The full "clear the table" sequence is: 3 cubes + 3 bins + 2 racks = 8 push_object calls (the 6 tubes inside racks are handled by the rack pushes).
  **IMPORTANT**: whenever the user says "push" / "slide" / "shove" / "knock off" / "drop off the edge" / similar, ALWAYS use push_object. Do NOT use move_to + gripper_close + gripper_open for these tasks — that's the pick-and-place pattern, which produces the wrong motion for push tasks.
- get_pose        params: {{}}
- search_workspace  params: object_name
- wait            params: seconds
- done            params: message

## Motion planning rules
1. ALWAYS call set_rail FIRST. Use optimal_rail_mm from the registry as your target.
2. For pick-and-place: rail to object -> move_to above -> lower -> gripper_close ->
   move_to lift height -> rail to destination -> move_to above bin -> gripper_open
   (release ABOVE the bin opening; the cube drops into the bin).
3. Keep speed_mm_s <= 100. Use 50-80 for grasps, 100 for transit.
4. Object coordinates from the registry are in millimeters in world frame.
   The arm base translates along the rail; pass world coordinates to move_to.
5. **Heights:** bench top is at z=750mm. Cube tops sit at z=780mm. Bin opening
   tops are at z=810mm. Grasp cubes at z=795mm (just above cube top). Lift to
   z=870mm for safe transit. Release at z=830mm above the bin (never dive
   into the bin -- the gripper is too wide and will clip the walls). The
   cube drops the remaining distance into the bin under gravity.

   **Falcon tube heights:** tubes stand in racks at z=818mm (body center).
   Cap top is at z=881mm. Tube rack walls extend up to z=810mm. To pick up
   a tube, lift it OUT of the rack before traversing -- approach from above
   to z=920mm (well clear of rack walls), descend to z=850mm to grasp the
   cap, then lift to z=1000mm for transit. To place a tube back in a rack
   slot, lower to z=900mm and release; the tube drops the last ~80mm into
   the slot under gravity.
6. If a task is ambiguous, output done() with a message asking for clarification.

## Output format
JSON array ONLY - no prose, no markdown fences. Example for "put red cube in red bin":
[
  {{"action": "set_rail",      "params": {{"position_mm": 150, "speed_mm_s": 100}}}},
  {{"action": "move_to",       "params": {{"x": -200, "y": 150, "z": 830, "roll": 180, "pitch": 0, "yaw": 0, "speed_mm_s": 80}}}},
  {{"action": "move_to",       "params": {{"x": -200, "y": 150, "z": 795, "roll": 180, "pitch": 0, "yaw": 0, "speed_mm_s": 50}}}},
  {{"action": "gripper_close", "params": {{}}}},
  {{"action": "move_to",       "params": {{"x": -200, "y": 150, "z": 870, "roll": 180, "pitch": 0, "yaw": 0, "speed_mm_s": 80}}}},
  {{"action": "move_to",       "params": {{"x": -200, "y": 350, "z": 870, "roll": 180, "pitch": 0, "yaw": 0, "speed_mm_s": 80}}}},
  {{"action": "move_to",       "params": {{"x": -200, "y": 350, "z": 830, "roll": 180, "pitch": 0, "yaw": 0, "speed_mm_s": 50}}}},
  {{"action": "gripper_open",  "params": {{}}}},
  {{"action": "done",          "params": {{"message": "Red cube placed in red bin"}}}}
]

## Active speed cap
{speed_cap_section}

## Current scene registry
{registry_context}

## What we know about this scene and arm (from prior sessions)
Accumulated cross-task invariants. High-confidence entries have held
across 3+ sessions. Provisional entries are from a single session and
are NOT yet validated -- you may rely on the high/medium ones; the
provisional ones are hypotheses to test rather than facts to respect.
{world_model_section}

## Lessons from past runs
These are auto-appended after each task. Most recent first. Read them and
avoid repeating mistakes (e.g. a height that previously caused a collision).
{lessons_section}
"""


class LLMBrain:

    def __init__(self, arm, registry: ObjectRegistry,
                 recorder: Optional[Recorder] = None,
                 model: str = DEFAULT_MODEL):
        self.arm = arm
        self.registry = registry
        self.recorder = recorder
        self.client = anthropic.Anthropic()
        self.model_full = resolve_model(model)
        self.model_short = model if model in MODELS else "custom"
        self.history = []
        # Speed cap applied to move_to / set_rail / set_joints. None means
        # "no clamp" (the crazy_fast tier). Default tier is "medium" with an
        # 80 mm/s cap -- set explicitly via set_speed_cap() by the episode
        # loop after it infers the tier from the task prompt.
        self.speed_tier = "medium"
        self.speed_cap_mm_s: Optional[float] = 80.0
        print(f"[LLMBrain] Using model: {self.model_short} ({self.model_full})")

    def set_speed_cap(self, tier: str, cap_mm_s: Optional[float]) -> None:
        """Set the dispatch-time speed clamp. `tier` is the human-readable
        name for logs ('medium', 'fast', etc.); `cap_mm_s` is the numeric
        ceiling (None = no clamp, only used for 'crazy_fast')."""
        self.speed_tier = tier
        self.speed_cap_mm_s = cap_mm_s

    def prepare_for_task(self, task: str,
                         override_tier: Optional[str] = None) -> None:
        """Per-task setup hook: infer the speed tier from the task prompt
        and apply the cap. Idempotent for the same task string (skips
        re-inference on a repeated call -- useful when an outer loop
        re-invokes execute_task with the same prompt augmented in
        different ways).

        Non-fatal: any failure leaves the existing cap in place (defaults
        to 'medium'/80 mm/s on a fresh brain).

        `override_tier`, when set, SKIPS the Haiku inference entirely and
        uses the named tier as the session ceiling. Wired to the
        `--speed-tier` CLI flag in the entry-point scripts -- gives the
        user a deterministic safety override that wins over any prompt
        interpretation. Invalid names log a warning and fall back to the
        usual inference path.

        Call this from any entry point that drives the brain (run_task,
        auto_play, run_task_augmented, EpisodeRetry); skipping it means
        the brain keeps the default medium cap regardless of what the
        prompt says.
        """
        # Cache key is the raw task string so a per-episode call from
        # EpisodeRetry (which augments the task with constraints/successes
        # blocks) won't trigger re-inference if the underlying task is
        # the same. The augmented form should be passed to execute_task,
        # but prepare_for_task should be called with the ORIGINAL task.
        if getattr(self, "_speed_cap_task", None) == task:
            return

        # "auto" is an explicit opt-in to the Haiku inference path. It
        # exists so callers can self-document their intent (--speed-tier
        # auto reads better in a saved command than omitting the flag).
        # Treat it as no override.
        if override_tier == "auto":
            override_tier = None

        # CLI override path: deterministic, no Haiku call.
        if override_tier is not None:
            from agent.dynamic_grader import SPEED_TIERS
            if override_tier in SPEED_TIERS:
                cap = SPEED_TIERS[override_tier]
                self.set_speed_cap(override_tier, cap)
                self._speed_cap_task = task
                cap_str = "UNCAPPED" if cap is None else f"{cap:.0f} mm/s"
                print(f"[LLMBrain] Speed cap (CLI override): "
                      f"{override_tier} ({cap_str})")
                return
            else:
                print(f"[LLMBrain] Invalid --speed-tier '{override_tier}'; "
                      f"falling back to Haiku inference.")

        try:
            from agent.dynamic_grader import (infer_speed_cap, SPEED_TIERS,
                                              DEFAULT_SPEED_TIER)
            tier = infer_speed_cap(task)
        except Exception as e:
            print(f"[LLMBrain] Speed inference unavailable (non-fatal): "
                  f"{type(e).__name__}: {e}; keeping current cap.")
            self._speed_cap_task = task
            return

        cap = SPEED_TIERS.get(tier, SPEED_TIERS[DEFAULT_SPEED_TIER])
        self.set_speed_cap(tier, cap)
        self._speed_cap_task = task
        cap_str = "UNCAPPED" if cap is None else f"{cap:.0f} mm/s"
        print(f"[LLMBrain] Speed cap for this task: {tier} ({cap_str})")

    def _render_speed_cap_section(self) -> str:
        """Render the active-cap section for the system prompt so the LLM
        produces compliant speeds up-front rather than waiting for the
        dispatch clamp to chop them down."""
        # The per-command tier vocabulary applies regardless of session
        # tier, so it's the same in every case.
        per_command_block = (
            "\n\n"
            "**Per-command speed_tier (optional):** any move_to / set_rail / "
            "set_joints command may include `\"speed_tier\": \"<tier>\"` in "
            "its params to use a different cap for that one command. Valid "
            "tier names: `crazy_fast` (no cap), `fast` (120 mm/s), `medium` "
            "(80 mm/s), `slow` (40 mm/s), `very_slow` (15 mm/s). The "
            "per-command tier is clamped to the SESSION tier above (you "
            "can DOWNGRADE per command but you cannot exceed the session "
            "ceiling). Use this when the task mentions different speeds "
            "for different phases, e.g. \"pick up quickly then carefully "
            "insert\":\n"
            "```\n"
            "{\"action\": \"move_to\", \"params\": {\"x\":..., \"y\":..., \"z\":..., \"speed_mm_s\": 100, \"speed_tier\": \"fast\"}}    "
            "// fast pickup phase\n"
            "...\n"
            "{\"action\": \"move_to\", \"params\": {\"x\":..., \"y\":..., \"z\":..., \"speed_mm_s\": 30, \"speed_tier\": \"slow\"}}     "
            "// careful insert phase\n"
            "```\n"
            "If you omit `speed_tier` the session cap applies."
        )

        if self.speed_cap_mm_s is None:
            return ("Tier: **crazy_fast** -- NO session-level speed cap. "
                    "You may use any speed_mm_s value, but stay realistic "
                    "(the arm's mechanical limits still apply)."
                    + per_command_block)
        cap = self.speed_cap_mm_s
        # Suggest grasp ~ half the cap, transit at the cap. These are
        # ratios that worked in the prior 100-mm/s regime (80 grasp / 100
        # transit -> grasp ~ 0.8 * cap, transit = cap).
        grasp = max(int(round(cap * 0.5)), 5)
        transit = int(round(cap))
        return (
            f"Tier: **{self.speed_tier}** -- session-level ceiling of "
            f"{cap:.0f} mm/s. speed_mm_s on every motion command will be "
            f"CLAMPED to <= {cap:.0f} mm/s (or deg/s for joints). "
            f"Suggested values within the cap: ~{grasp} mm/s for grasps "
            f"and approaches, ~{transit} mm/s for transit. Do not request "
            f"speeds above {cap:.0f}; they will be silently clamped and "
            f"the move will actually go slower than you specified."
            + per_command_block
        )

    def _effective_speed_mm_s(self, per_cmd_tier: Optional[str] = None,
                              default: float = 100.0) -> float:
        """Return the smallest of (default, per-command cap, session cap),
        treating None caps as 'no constraint' on that side. Used by
        push_object dispatch -- the macro takes a single speed_mm_s value
        and applies it to its internal moves, so we need to resolve the
        effective cap up-front rather than per-move."""
        candidates = [default]
        if per_cmd_tier:
            from agent.dynamic_grader import SPEED_TIERS
            if per_cmd_tier in SPEED_TIERS and SPEED_TIERS[per_cmd_tier] is not None:
                candidates.append(SPEED_TIERS[per_cmd_tier])
        if self.speed_cap_mm_s is not None:
            candidates.append(self.speed_cap_mm_s)
        return float(min(candidates))

    def _clamp_speed(self, requested: float, units: str = "mm/s",
                     per_cmd_tier: Optional[str] = None) -> float:
        """Clamp a single speed value against the effective cap.

        Effective cap = min(per_command_cap, session_cap), honouring None
        (None means uncapped for whichever side carries it):
          - both None -> uncapped, passthrough
          - one None -> the other wins
          - both set -> the smaller value wins (per-command can downgrade
            below the session ceiling but cannot exceed it)

        When `per_cmd_tier` is supplied but not a recognised tier name we
        ignore it and fall back to the session cap. Logs every clamp,
        attributing it to whichever cap actually bit.
        """
        # Look up per-command cap from tier name. We import lazily to avoid
        # forcing the dynamic_grader module at every import of llm_brain.
        per_cap = None
        per_tier_label = None
        if per_cmd_tier:
            from agent.dynamic_grader import SPEED_TIERS
            if per_cmd_tier in SPEED_TIERS:
                per_cap = SPEED_TIERS[per_cmd_tier]
                per_tier_label = per_cmd_tier
            else:
                print(f"[Agent] Unknown per-command speed_tier "
                      f"'{per_cmd_tier}'; ignoring.")

        session_cap = self.speed_cap_mm_s
        # Compute effective cap.
        if per_cap is None and session_cap is None:
            effective = None
            cap_source = None
        elif per_cap is None:
            effective = session_cap
            cap_source = ("session", self.speed_tier)
        elif session_cap is None:
            effective = per_cap
            cap_source = ("per-command", per_tier_label)
        else:
            if per_cap <= session_cap:
                effective = per_cap
                cap_source = ("per-command", per_tier_label)
            else:
                effective = session_cap
                cap_source = ("session ceiling", self.speed_tier)
                print(f"[Agent] Per-command tier '{per_tier_label}' "
                      f"({per_cap:.0f}) exceeds session ceiling "
                      f"'{self.speed_tier}' ({session_cap:.0f}); using "
                      f"session ceiling.")

        if effective is None:
            return requested
        if requested > effective:
            src_kind, src_label = cap_source
            print(f"[Agent] Clamping speed {requested:.0f} -> "
                  f"{effective:.0f} {units} "
                  f"({src_kind} cap: {src_label})")
            return effective
        return requested

    def execute_task(self, task_prompt: str, dry_run: bool = False) -> dict:
        llm_log = None
        if self.recorder is not None and self.recorder.is_recording:
            llm_log = LLMSessionLog(self.recorder, self.model_full, task_prompt)
            llm_log.log_prompt()

        lessons = read_lessons_section(current_task=task_prompt)
        wm = read_world_model()
        wm_section = render_for_system_prompt(wm, include_provisional=True)
        if wm.scene_changed and wm_section:
            wm_section = ("**Scene XML has changed since these observations "
                          "were recorded -- treat all entries as untested.**"
                          "\n\n" + wm_section)
        system = SYSTEM_PROMPT_TEMPLATE.format(
            registry_context=self.registry.to_llm_context(),
            speed_cap_section=self._render_speed_cap_section(),
            world_model_section=wm_section if wm_section
                                else "(no cross-task world model yet)",
            lessons_section=lessons if lessons else "(no prior lessons yet)",
        )
        self.history.append({"role": "user", "content": task_prompt})

        t_start = time.time()
        try:
            response = self.client.messages.create(
                model=self.model_full, max_tokens=2048,
                system=system, messages=self.history,
            )
        except anthropic.APIError as e:
            if llm_log:
                llm_log.log_response(f"API ERROR: {e}", time.time() - t_start)
                llm_log.close()
            print(f"[LLMBrain] Claude API error: {e}")
            return {"commands": [], "results": [], "raw": str(e), "error": True}
        latency = time.time() - t_start
        raw = response.content[0].text
        in_tok  = getattr(response.usage, "input_tokens", 0)
        out_tok = getattr(response.usage, "output_tokens", 0)
        if llm_log:
            llm_log.log_response(raw, latency, in_tok, out_tok)
        print(f"[LLMBrain] Response in {latency:.1f}s "
              f"({in_tok}->{out_tok} tokens)")
        self.history.append({"role": "assistant", "content": raw})

        try:
            commands = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"(\[.*\])", raw, re.DOTALL)
            if m:
                commands = json.loads(m.group(1))
            else:
                if llm_log:
                    llm_log.log_parse_error(f"No JSON array:\n{raw}")
                    llm_log.close()
                raise ValueError(f"LLM returned non-JSON:\n{raw}")
        if llm_log:
            llm_log.log_parsed(commands)

        results = []
        if not dry_run:
            results = self._run(commands, llm_log)

        if llm_log:
            llm_log.close()
        return {"commands": commands, "results": results, "raw": raw,
                "latency_s": latency,
                "input_tokens": in_tok, "output_tokens": out_tok}

    def _run(self, commands, llm_log=None):
        results = []
        for cmd in commands:
            action = cmd["action"]
            params = cmd.get("params", {})
            result = self._dispatch(action, params)
            results.append({"action": action, "result": result})
            if llm_log:
                llm_log.log_dispatch(action, params, result)
            if self.recorder and self.recorder.is_recording:
                self.recorder.log_command("llm_dispatch",
                    {"action": action, "params": params, "result": result})
            print(f"[Agent] {action}({params}) -> {result}")
            if result != 0 and action not in ("done", "wait", "get_pose"):
                print("[Agent] Command failed - halting sequence")
                break
        return results

    def _move_to(self, p):
        speed = p.get("speed_mm_s", p.get("speed", 100.0))
        speed = self._clamp_speed(speed, per_cmd_tier=p.get("speed_tier"))
        return self.arm.set_position(
            x=p["x"], y=p["y"], z=p["z"],
            roll=p.get("roll", 0.0), pitch=p.get("pitch", 0.0),
            yaw=p.get("yaw", 0.0),
            speed=speed, wait=p.get("wait", True),
        )

    def _set_rail(self, p):
        speed = self._clamp_speed(p.get("speed_mm_s", 50.0),
                                  per_cmd_tier=p.get("speed_tier"))
        return self.arm.set_rail_position(
            position_mm=p["position_mm"],
            speed_mm_s=speed,
            wait=p.get("wait", True),
        )

    def _set_joints(self, p):
        # set_joints is angular (deg/s), not linear (mm/s). The cap is
        # nominally a linear speed limit, so we apply it directly to deg/s
        # only as a conservative ceiling -- a joint moving at >80 deg/s is
        # already too fast for cautious operation. Tune separately if joint
        # speed becomes its own concern.
        requested = p.get("speed_deg_s", p.get("speed", 30.0))
        speed = self._clamp_speed(requested, units="deg/s",
                                  per_cmd_tier=p.get("speed_tier"))
        return self.arm.set_servo_angle(
            angle=p["angles_deg"],
            speed=speed,
            wait=p.get("wait", True),
        )

    def _get_pose(self, p):
        ee = self.arm.get_position()
        rail = self.arm.get_rail_position()
        print(f"[Agent] Pose: ee={ee}  rail={rail}")
        return 0

    def _search(self, name):
        obj = self.registry.find(name)
        if obj is None:
            print(f"[Agent] '{name}' not in registry"); return 1
        print(f"[Agent] Found '{name}' at {obj.position_xyz_m}  "
              f"rail: {obj.optimal_rail_mm}mm")
        return 0

    def _dispatch(self, action, params):
        d = {
            "home":             lambda p: self.arm.go_home(wait=p.get("wait", True)),
            "wave_goodbye":     lambda p: self.arm.wave_goodbye(
                                    n_waves=int(p.get("n_waves", 3))),
            "set_rail":         self._set_rail,
            "move_to":          self._move_to,
            "set_joints":       self._set_joints,
            "gripper_open":     lambda p: self.arm.open_lite6_gripper(),
            "gripper_close":    lambda p: self.arm.close_lite6_gripper(),
            "place_tube_in_rack": lambda p: self.arm.place_tube_in_rack(
                                      rack_name=p["rack_name"]),
            "push_object":      lambda p: self.arm.push_object(
                                      target_name=p["target_name"],
                                      to_x_mm=float(p["to_x_mm"]),
                                      to_y_mm=float(p["to_y_mm"]),
                                      speed_mm_s=self._effective_speed_mm_s(
                                          p.get("speed_tier"))),
            "get_pose":         self._get_pose,
            "wait":             lambda p: time.sleep(p.get("seconds", 1)) or 0,
            "done":             lambda p: print(f"[Done] {p.get('message','')}") or 0,
            "search_workspace": lambda p: self._search(p["object_name"]),
        }
        h = d.get(action)
        if h is None:
            print(f"[Agent] Unknown action: {action}"); return -1
        return h(params)
