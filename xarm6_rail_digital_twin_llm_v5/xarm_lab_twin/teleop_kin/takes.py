# teleop_kin/takes.py
"""Take lifecycle shared by the twin and real-arm teach runners.

One Recorder per take, created by a factory, started and stopped by a key;
Delete soft-deletes (recordings/trash/, recoverable with
`recording.restore_session`). Same lifecycle as run_spacemouse.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from recording import soft_delete_session


class TakeControl:
    def __init__(self, recorder=None, factory: Optional[Callable] = None,
                 task_label: str = "", log: Callable[[str], None] = print,
                 on_start: Optional[Callable] = None):
        self.rec = recorder
        self.factory = factory
        self.task_label = task_label
        self.log = log
        #: Called with the recorder right after a take starts, so a runner can
        #: log context (robot model, mode) as the take's first commands.
        self.on_start = on_start
        self.last_saved: Optional[str] = None

    @property
    def recording(self) -> bool:
        return bool(self.rec is not None and self.rec.is_recording)

    def log_command(self, name: str, payload: dict) -> None:
        if self.recording:
            self.rec.log_command(name, payload)

    def toggle(self) -> None:
        if self.rec is None:
            self.log("recording disabled (--no-record); ignored.")
            return
        if not self.rec.is_recording:
            self.rec.start()
            if self.on_start is not None:
                self.on_start(self.rec)
            self.log("recording STARTED  (Insert to stop, Delete to discard)")
        else:
            self.stop(kept=True)

    def stop(self, kept: bool = True) -> Optional[str]:
        """Stop a take in progress (if any). Returns the saved path."""
        if not self.recording:
            return None
        path = self.rec.stop(kept=kept, task_label=self.task_label)
        if kept:
            self.last_saved = str(path) if path else None
            self.log(f"recording STOPPED -> {path}")
        if self.factory is not None:
            self.rec = self.factory()
        return str(path) if path else None

    def discard(self) -> None:
        """Stop-and-trash the take in progress, or, if none is in progress,
        trash the take saved last."""
        if self.recording:
            path = self.stop(kept=False)
            if path is not None:
                soft_delete_session(Path(path))
            self.log("take DISCARDED (moved to recordings/trash/)")
        elif self.last_saved:
            soft_delete_session(Path(self.last_saved))
            self.log(f"last take DISCARDED: {self.last_saved}")
            self.last_saved = None
        else:
            self.log("nothing to discard")
