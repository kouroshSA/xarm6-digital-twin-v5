# agent/llm_brain.py
import anthropic
import json
import re
import time
from typing import Optional
from agent.object_registry import ObjectRegistry
from agent.lessons import read_lessons_section
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

## Current scene registry
{registry_context}

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
        print(f"[LLMBrain] Using model: {self.model_short} ({self.model_full})")

    def execute_task(self, task_prompt: str, dry_run: bool = False) -> dict:
        llm_log = None
        if self.recorder is not None and self.recorder.is_recording:
            llm_log = LLMSessionLog(self.recorder, self.model_full, task_prompt)
            llm_log.log_prompt()

        lessons = read_lessons_section(current_task=task_prompt)
        system = SYSTEM_PROMPT_TEMPLATE.format(
            registry_context=self.registry.to_llm_context(),
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
        return self.arm.set_position(
            x=p["x"], y=p["y"], z=p["z"],
            roll=p.get("roll", 0.0), pitch=p.get("pitch", 0.0),
            yaw=p.get("yaw", 0.0),
            speed=speed, wait=p.get("wait", True),
        )

    def _set_rail(self, p):
        return self.arm.set_rail_position(
            position_mm=p["position_mm"],
            speed_mm_s=p.get("speed_mm_s", 50.0),
            wait=p.get("wait", True),
        )

    def _set_joints(self, p):
        return self.arm.set_servo_angle(
            angle=p["angles_deg"],
            speed=p.get("speed_deg_s", p.get("speed", 30.0)),
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
                                      to_y_mm=float(p["to_y_mm"])),
            "get_pose":         self._get_pose,
            "wait":             lambda p: time.sleep(p.get("seconds", 1)) or 0,
            "done":             lambda p: print(f"[Done] {p.get('message','')}") or 0,
            "search_workspace": lambda p: self._search(p["object_name"]),
        }
        h = d.get(action)
        if h is None:
            print(f"[Agent] Unknown action: {action}"); return -1
        return h(params)
