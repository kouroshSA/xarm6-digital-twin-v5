# agent/world_model.py
"""
Phase 3 of the learning architecture: cross-task knowledge file.

`world_model.md` accumulates *invariants* -- facts that hold across tasks,
not specific to one. Each new Opus session-review (Phase 2) emits a set of
`cross_task_observations` alongside its per-task writeup; those are
merged into the world model here, then injected into the LLMBrain system
prompt at construction so every new task starts with the system's
accumulated knowledge of how the scene and arm behave.

File format (human-editable):

  # World Model

  scene_version: <sha256[:12] of envs/lab_scene.xml>
  last_updated: <ISO timestamp>

  ## Geometric and kinematic invariants
  - <observation text on a single line>
    Corroborated by: <task> (<date>), <task> (<date>), ...
    Confidence: <provisional|medium|high>

  ## Object-class regularities
  - ...

  ## Primitive-behavior knowledge
  - ...

  ## Grader and stringency regularities
  - ...

Confidence is derived on read from len(corroborations):
  1 = provisional, 2 = medium, 3+ = high. It's written into the file too
  for human readability, but it is the corroboration count that's
  authoritative -- callers should rely on `.confidence` not on what the
  file claims.

If the current `envs/lab_scene.xml` hash differs from the file's
`scene_version`, the reader flags `scene_changed=True`. The system-prompt
injector includes a banner about this so the model knows entries may be
stale; the Opus reviewer also sees the banner so it knows to re-validate
provisional entries against the current scene.
"""
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Anchor to the project root (.../xarm_lab_twin) so reads/writes resolve to
# the same files regardless of the CWD the entry point was launched from.
# Without this, a run started outside xarm_lab_twin/ silently splits its
# cross-run memory across directories and misflags scene_changed.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent   # .../xarm_lab_twin
WORLD_MODEL_FILE = _PROJECT_ROOT / "world_model.md"
SCENE_XML_PATH   = _PROJECT_ROOT / "envs" / "lab_scene.xml"

# Section name in the file -> internal category key.
SECTIONS: Tuple[Tuple[str, str], ...] = (
    ("Geometric and kinematic invariants", "geometric"),
    ("Object-class regularities",          "object_class"),
    ("Primitive-behavior knowledge",       "primitive"),
    ("Grader and stringency regularities", "grader"),
)
SECTION_TITLE = {key: title for title, key in SECTIONS}
SECTION_KEYS = tuple(key for _title, key in SECTIONS)


def confidence_from_count(n: int) -> str:
    if n >= 3:
        return "high"
    if n == 2:
        return "medium"
    return "provisional"


@dataclass
class WorldEntry:
    """One observation with its corroboration history."""
    text: str
    corroborations: List[Dict[str, str]] = field(default_factory=list)
    # Each corroboration: {"task": "<short task label>", "date": "YYYY-MM-DD"}

    @property
    def confidence(self) -> str:
        return confidence_from_count(len(self.corroborations))

    def add_corroboration(self, task: str, date: str) -> bool:
        """Record a new corroborating session. Dedup by (task, date) so
        a single session can't inflate its own confidence by emitting the
        same observation twice."""
        for c in self.corroborations:
            if c.get("task") == task and c.get("date") == date:
                return False
        self.corroborations.append({"task": task, "date": date})
        return True


@dataclass
class WorldModel:
    scene_version: str = ""              # last-recorded scene hash
    last_updated:  str = ""              # ISO timestamp
    entries: Dict[str, List[WorldEntry]] = field(default_factory=dict)
    scene_changed: bool = False          # current XML hash != scene_version

    def section(self, key: str) -> List[WorldEntry]:
        return self.entries.setdefault(key, [])


# ---------------------------------------------------------------------------
# Scene hash
# ---------------------------------------------------------------------------

def compute_scene_hash(scene_path: Path = None) -> str:
    """sha256 of envs/lab_scene.xml (first 12 hex chars). Empty if missing."""
    if scene_path is None:
        scene_path = SCENE_XML_PATH
    if not scene_path.exists():
        return ""
    return hashlib.sha256(scene_path.read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Parse / serialise
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$")
_ENTRY_HEAD_RE = re.compile(r"^-\s+(.+?)\s*$")
_INDENT_LINE_RE = re.compile(r"^\s{2,}(.+?)\s*$")
_HEADER_KV_RE = re.compile(r"^([a-z_]+):\s*(.+?)\s*$")
_CORROB_ITEM_RE = re.compile(r"^(?P<task>.+?)\s*\((?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})\)\s*$")


def _parse_corroborations(value: str) -> List[Dict[str, str]]:
    """Parse 'task1 (YYYY-MM-DD), task2 (YYYY-MM-DD)' into structured list.

    Robust to commas inside task labels: we split on `), ` so the date
    parenthesis is the boundary, then re-attach the trailing `)`.
    """
    out: List[Dict[str, str]] = []
    if not value.strip():
        return out
    chunks = [c.strip() for c in value.split("), ")]
    for i, chunk in enumerate(chunks):
        if i < len(chunks) - 1 and not chunk.endswith(")"):
            chunk = chunk + ")"
        m = _CORROB_ITEM_RE.match(chunk)
        if m:
            out.append({"task": m.group("task").strip(),
                        "date": m.group("date")})
    return out


def read_world_model(wm_file: Path = None,
                     scene_path: Path = None) -> WorldModel:
    """Parse world_model.md into a WorldModel. Returns an empty (but
    valid) model if the file doesn't exist. Sets `scene_changed=True` if
    the current scene XML's hash differs from the file's recorded one.
    """
    if wm_file is None:
        wm_file = WORLD_MODEL_FILE
    current_hash = compute_scene_hash(scene_path)

    if not wm_file.exists():
        return WorldModel(scene_version=current_hash,
                          last_updated="",
                          entries={k: [] for k in SECTION_KEYS},
                          scene_changed=False)

    wm = WorldModel(entries={k: [] for k in SECTION_KEYS})
    lines = wm_file.read_text().splitlines()
    section_key: Optional[str] = None
    current_entry: Optional[WorldEntry] = None
    pending_field: Optional[str] = None  # which `<field>:` we're filling

    for raw in lines:
        # Header key/value lines (scene_version, last_updated) only valid
        # before we've hit a section header.
        m = _HEADER_KV_RE.match(raw)
        if m and section_key is None:
            k, v = m.group(1), m.group(2)
            if k == "scene_version":
                wm.scene_version = v
            elif k == "last_updated":
                wm.last_updated = v
            continue

        sm = _SECTION_RE.match(raw)
        if sm:
            title = sm.group(1).strip()
            section_key = None
            for t, key in SECTIONS:
                if t == title:
                    section_key = key
                    break
            current_entry = None
            pending_field = None
            continue

        if section_key is None:
            continue

        em = _ENTRY_HEAD_RE.match(raw)
        if em:
            current_entry = WorldEntry(text=em.group(1))
            wm.section(section_key).append(current_entry)
            pending_field = None
            continue

        im = _INDENT_LINE_RE.match(raw)
        if im and current_entry is not None:
            content = im.group(1)
            if content.startswith("Corroborated by:"):
                value = content[len("Corroborated by:"):].strip()
                current_entry.corroborations = _parse_corroborations(value)
                pending_field = "corroborations"
            elif content.startswith("Confidence:"):
                # Derived on read -- ignore the stored value.
                pending_field = None
            else:
                # Continuation of the observation text (rare; supported
                # for human edits that wrap a long observation).
                current_entry.text = current_entry.text + " " + content
                pending_field = None
            continue

        # Anything else (blank line, etc.) resets the entry continuation.
        pending_field = None

    wm.scene_changed = bool(wm.scene_version and current_hash
                            and wm.scene_version != current_hash)
    if not wm.scene_version:
        wm.scene_version = current_hash
    return wm


def _render_entry(entry: WorldEntry) -> str:
    lines = [f"- {entry.text}"]
    if entry.corroborations:
        items = ", ".join(f"{c['task']} ({c['date']})"
                          for c in entry.corroborations)
        lines.append(f"  Corroborated by: {items}")
    lines.append(f"  Confidence: {entry.confidence}")
    return "\n".join(lines)


def write_world_model(wm: WorldModel, wm_file: Path = None,
                      scene_path: Path = None) -> Path:
    """Write the WorldModel back to disk; refresh scene_version and
    last_updated so the file reflects the version of the scene the
    entries were written against."""
    if wm_file is None:
        wm_file = WORLD_MODEL_FILE
    current_hash = compute_scene_hash(scene_path)
    wm.scene_version = current_hash or wm.scene_version
    wm.last_updated  = datetime.now().isoformat(timespec="seconds")

    parts = [
        "# World Model",
        "",
        f"scene_version: {wm.scene_version}",
        f"last_updated: {wm.last_updated}",
        "",
        "Cross-task invariants learned from past sessions. Each entry's",
        "confidence is derived from its corroboration count (1=provisional,",
        "2=medium, 3+=high). If the scene XML changes, all entries are",
        "flagged as untested until re-corroborated in the new scene.",
        "",
    ]
    for title, key in SECTIONS:
        parts.append(f"## {title}")
        parts.append("")
        entries = wm.entries.get(key, [])
        if not entries:
            parts.append("_(no entries yet)_")
        else:
            for entry in entries:
                parts.append(_render_entry(entry))
        parts.append("")

    wm_file.write_text("\n".join(parts).rstrip() + "\n")
    return wm_file


# ---------------------------------------------------------------------------
# Merge logic: integrating cross-task observations from a review.
# ---------------------------------------------------------------------------

def update_from_review(wm: WorldModel,
                       cross_task_observations: List[Dict],
                       task_label: str,
                       date: Optional[str] = None) -> List[Dict]:
    """Merge a review's cross-task observations into the WorldModel.

    Each observation is a dict from the Opus reviewer; see
    review_session.py for the exact schema. The shape we use here:
      {
        "text": str,                         # the observation
        "category": str,                     # one of SECTION_KEYS
        "merge_with_index": Optional[int],   # index in section's entries
                                             # to merge into, or null=new
      }

    Opus is responsible for deciding whether an observation matches an
    existing entry; we just execute the decision. Returns a log of what
    happened so the caller can print it for human audit.
    """
    if date is None:
        date = datetime.now().date().isoformat()
    log: List[Dict] = []

    for obs in cross_task_observations or []:
        text = (obs.get("text") or "").strip()
        category = obs.get("category", "")
        if not text or category not in SECTION_KEYS:
            log.append({"action": "skipped",
                        "reason": "missing text or invalid category",
                        "obs": obs})
            continue

        section = wm.section(category)
        merge_idx = obs.get("merge_with_index")

        if (isinstance(merge_idx, int)
                and 0 <= merge_idx < len(section)):
            target = section[merge_idx]
            added = target.add_corroboration(task_label, date)
            log.append({
                "action": "merged" if added else "merged_duplicate",
                "category": category,
                "into_index": merge_idx,
                "into_text": target.text,
                "new_text": text,
                "new_confidence": target.confidence,
            })
            continue

        # New entry.
        new = WorldEntry(text=text)
        new.add_corroboration(task_label, date)
        section.append(new)
        log.append({
            "action": "new",
            "category": category,
            "index": len(section) - 1,
            "text": text,
            "confidence": new.confidence,
        })

    return log


def persist_constraints(constraints: List[str], task_label: str,
                        category: str = "geometric",
                        wm_file: Path = None, scene_path: Path = None) -> Optional[Path]:
    """Append free-text learned constraints as provisional entries.

    Deterministic fallback for when the Opus review didn't run. Entries land
    with a single corroboration (confidence='provisional') so the prompt
    injector labels them untested and trims them first under budget pressure.
    Returns the path written, or None if there was nothing to persist.
    """
    constraints = [c.strip() for c in (constraints or []) if c and c.strip()]
    if not constraints:
        return None
    wm = read_world_model(wm_file, scene_path)
    obs = [{"text": c, "category": category, "merge_with_index": None}
           for c in constraints]
    update_from_review(wm, obs, task_label=task_label)
    return write_world_model(wm, wm_file, scene_path)


# ---------------------------------------------------------------------------
# Rendering for system-prompt injection.
# ---------------------------------------------------------------------------

# Soft cap on injected chars (~ token budget). The injector trims by
# dropping provisional entries first, then medium, prioritising
# geometric+primitive sections when space is tight.
_MAX_INJECTED_CHARS = 8_000  # ~2k tokens at 4 chars/token


def render_for_system_prompt(wm: WorldModel,
                             include_provisional: bool = True,
                             max_chars: int = _MAX_INJECTED_CHARS) -> str:
    """Build the markdown block to splice into LLMBrain's system prompt.

    High-confidence entries are stated plainly. Medium are framed as
    "observed across N sessions". Provisional are included only when
    `include_provisional=True` and explicitly labelled as untested.
    Scene-changed banner is prepended when set.
    """
    if not any(wm.entries.get(k) for k in SECTION_KEYS):
        return ""

    parts: List[str] = []
    if wm.scene_changed:
        parts.append(
            "**Scene XML has changed since these observations were "
            "recorded -- treat all entries below as UNTESTED in the "
            "current scene until corroborated again.**"
        )
        parts.append("")

    # Section priority for trimming.
    section_priority = ("geometric", "primitive", "object_class", "grader")

    def render_entry(entry: WorldEntry) -> str:
        if entry.confidence == "high":
            return f"- {entry.text}"
        if entry.confidence == "medium":
            return (f"- {entry.text}  _(observed across "
                    f"{len(entry.corroborations)} sessions)_")
        # provisional
        return f"- {entry.text}  _(provisional -- 1 session, untested)_"

    rendered_sections: List[Tuple[str, str, List[str]]] = []
    for key in section_priority:
        section = wm.entries.get(key, [])
        if not section:
            continue
        lines: List[str] = []
        for entry in section:
            if entry.confidence == "provisional" and not include_provisional:
                continue
            lines.append(render_entry(entry))
        if lines:
            rendered_sections.append((key, SECTION_TITLE[key], lines))

    if not rendered_sections:
        return ""

    def assemble(sections_to_use) -> str:
        body: List[str] = parts.copy()
        for _key, title, lines in sections_to_use:
            body.append(f"### {title}")
            body.extend(lines)
            body.append("")
        return "\n".join(body).rstrip()

    out = assemble(rendered_sections)
    if len(out) <= max_chars:
        return out

    # Trim: drop provisional entries first, then medium, then by section
    # priority (drop grader first, geometric last).
    def trim_provisional(sections):
        new = []
        for key, title, lines in sections:
            kept = [ln for ln in lines if "_(provisional" not in ln]
            if kept:
                new.append((key, title, kept))
        return new

    out = assemble(trim_provisional(rendered_sections))
    if len(out) <= max_chars:
        return out

    # Still too big -- drop the lowest-priority sections.
    sections = trim_provisional(rendered_sections)
    while len(sections) > 1 and len(assemble(sections)) > max_chars:
        sections = sections[:-1]
    return assemble(sections)
