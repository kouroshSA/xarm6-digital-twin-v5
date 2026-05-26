# agent/prompt_variants.py
import anthropic
import json
import os
import re
import subprocess
import tempfile
from typing import List


VARIANT_GEN_SYSTEM_PROMPT = """\
You are generating paraphrased variants of a robot pick-and-place command.
The robot is an xArm6 on a linear rail in a benchmark scene with three RGB
cubes (red, green, blue) and three matching bins.

Given an original command, produce N paraphrased variants that:
1. Preserve the EXACT task intent - same source cube, same destination bin,
   same action verb category. "Put red cube in red bin" can become "place
   the red block in the red container" but NOT "put green cube in red bin".
2. Vary surface form - synonyms, word order, formality, contractions.
3. Cover a range of phrasings: formal ("Could you please place the red cube
   into the red container?") to casual ("red in red plz").
4. Are distinct - no two identical, none merely punctuation-different.

Output ONLY a JSON array of strings. No prose, no markdown fences.
"""


def generate_variants(original_prompt: str, n_variants: int,
                      model: str = "claude-haiku-4-5-20251001") -> List[str]:
    client = anthropic.Anthropic()
    user_msg = (
        f"Original command: \"{original_prompt}\"\n\n"
        f"Generate exactly {n_variants} paraphrased variants. "
        f"Output as a JSON array of strings."
    )
    response = client.messages.create(
        model=model, max_tokens=1024,
        system=VARIANT_GEN_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text
    try:
        variants = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"(\[.*\])", raw, re.DOTALL)
        if not m:
            raise ValueError(f"Variant generator returned non-JSON:\n{raw}")
        variants = json.loads(m.group(1))
    if not isinstance(variants, list):
        raise ValueError(f"Expected JSON array, got: {type(variants)}")
    variants = [str(v).strip() for v in variants if isinstance(v, str) and v.strip()]
    if len(variants) < n_variants:
        raise ValueError(
            f"Asked for {n_variants}, got {len(variants)}:\n{variants}"
        )
    return variants[:n_variants]


EPISODE_GEN_SYSTEM_PROMPT = """\
You are generating a varied script of robotic task prompts for a benchmark
scene. The robot is a UFACTORY xArm6 on a 700mm linear rail. The scene has:
  - Three RGB cubes (red, green, blue) on a bench
  - Three matching bins (red, green, blue) behind the cubes
  - A gripper that can pick up cubes
The robot understands these high-level primitives:
  - "go home" / "reset"                       -> home pose
  - "wave goodbye N times"                    -> N side-to-side waves
  - "put the <color> cube in the <color> bin" -> pick-and-place
  - "put all cubes in the <color> bin"        -> sort to one bin
  - "sort the cubes by color"                 -> 3 pick-and-places
  - "move the arm to <coords>"                -> direct Cartesian move

Generate N DIVERSE natural-language task prompts that exercise these primitives.
Rules:
  1. Mix simple (one verb) and compound (two or three steps) tasks.
  2. Vary phrasing: formal, casual, terse.
  3. No two prompts should be near-duplicates.
  4. Include at least one gesture (wave/home) and at least one pick-and-place
     among any N >= 3.
  5. Keep each prompt under 25 words.

Output ONLY a JSON array of strings. No prose, no markdown fences.
"""


def generate_episodes(n_episodes: int,
                      model: str = "claude-haiku-4-5-20251001") -> List[str]:
    """Ask Claude for N diverse task prompts. Returns list of strings."""
    client = anthropic.Anthropic()
    user_msg = (
        f"Generate exactly {n_episodes} diverse robotic task prompts. "
        f"Output as a JSON array of strings."
    )
    response = client.messages.create(
        model=model, max_tokens=2048,
        system=EPISODE_GEN_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text
    try:
        tasks = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"(\[.*\])", raw, re.DOTALL)
        if not m:
            raise ValueError(f"Episode generator returned non-JSON:\n{raw}")
        tasks = json.loads(m.group(1))
    if not isinstance(tasks, list):
        raise ValueError(f"Expected JSON array, got: {type(tasks)}")
    tasks = [str(t).strip() for t in tasks if isinstance(t, str) and t.strip()]
    if len(tasks) < n_episodes:
        raise ValueError(
            f"Asked for {n_episodes}, got {len(tasks)}:\n{tasks}"
        )
    return tasks[:n_episodes]


def preview_and_confirm(original: str, variants: List[str]) -> List[str]:
    print("\n" + "=" * 70)
    print("  VARIANT PROMPTS")
    print("=" * 70)
    print(f"\n  Original (cycle 1):\n    \"{original}\"")
    print(f"\n  Variants (cycles 2..{len(variants)+1}):")
    for i, v in enumerate(variants, start=2):
        print(f"    {i}: \"{v}\"")
    print("\n" + "-" * 70)

    while True:
        try:
            ans = input("Accept? [Y/n/e=edit in $EDITOR]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return []
        if ans in ("", "y"):
            return variants
        if ans == "n":
            return []
        if ans == "e":
            edited = _edit_in_editor(original, variants)
            if edited:
                print("\nEdited list:")
                print(f"  Original: \"{original}\"")
                for i, v in enumerate(edited, start=2):
                    print(f"  {i}: \"{v}\"")
                variants = edited
                continue
            else:
                print("Edit cancelled - keeping previous list.")
                continue
        print("Please answer y, n, or e.")


def _edit_in_editor(original: str, variants: List[str]) -> List[str]:
    editor = os.environ.get("EDITOR", "nano")
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("# One variant per line. Lines starting with # are comments.\n")
        f.write(f"# Original (DO NOT EDIT - reference only):\n# {original}\n#\n")
        f.write("# Variants - edit, add, remove freely. Save and close to confirm.\n\n")
        for v in variants:
            f.write(v + "\n")
        tmp = f.name
    try:
        subprocess.run([editor, tmp], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Editor failed: {e}")
        os.unlink(tmp); return []
    with open(tmp) as f:
        lines = [ln.strip() for ln in f
                 if ln.strip() and not ln.strip().startswith("#")]
    os.unlink(tmp)
    return lines
