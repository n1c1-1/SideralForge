"""Runs the pipeline in a detached worker process and follows it through the journal (events.jsonl).

The worker survives the closing of the interface: reopening the project re-attaches to the running process.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from ..core.journal import read_events
from ..core.project import Project
from ..core.runner import _run_alive, request_stop


class RunController(QObject):
    event = Signal(dict)          # every journal event
    runningChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Project | None = None
        self._pos = 0
        self._running = False
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.poll)
        self.stage_times: dict[str, float] = {}
        self._started = 0.0

    def attach(self, project: Project):
        self.project = project
        ev = project.logs / "events.jsonl"
        self._pos = ev.stat().st_size if ev.exists() else 0
        self._running = _run_alive(project)
        self.runningChanged.emit(self._running)
        self.timer.start()

    def history(self, n: int = 400) -> list[dict]:
        """Last events of the journal (to fill the log view when a project is opened)."""
        if not self.project:
            return []
        evs, _ = read_events(self.project.logs / "events.jsonl", 0)
        return evs[-n:]

    def start(self, until: str | None = None, force: list[str] | None = None) -> bool:
        if not self.project or self.is_running():
            return False
        args = ["-m", "astro_gpu.cli", "run", str(self.project.root)]
        if until:
            args += ["--until", until]
        if force:
            args += ["--force", *force]
        log = self.project.logs / "worker.log"
        self.project.logs.mkdir(parents=True, exist_ok=True)
        ok = QProcess.startDetached(sys.executable, args, str(self.project.root))
        if isinstance(ok, tuple):
            ok = ok[0]
        if ok:
            log.write_text(f"started {time.ctime()}\n")
            # the lock file appears when the worker has initialised
            for _ in range(40):
                if (self.project.logs / "run.lock").exists():
                    break
                time.sleep(0.05)
            self._started = time.time()
            self._set_running(True)
        return bool(ok)

    def stop(self):
        if self.project:
            request_stop(self.project)

    def is_running(self) -> bool:
        return bool(self.project) and _run_alive(self.project)

    def _set_running(self, v: bool):
        if v != self._running:
            self._running = v
            self.runningChanged.emit(v)

    def poll(self):
        if not self.project:
            return
        evs, self._pos = read_events(self.project.logs / "events.jsonl", self._pos)
        for e in evs:
            self.event.emit(e)
        # grace period right after a start: the worker may not have written its lock file yet
        alive = _run_alive(self.project) or (time.time() - self._started < 8)
        self._set_running(bool(alive))


def stage_history_seconds(stage: str) -> float | None:
    p = Path.home() / ".astro_gpu" / "timings.json"
    try:
        d = json.loads(p.read_text())
        v = d.get(stage)
        return float(sorted(v)[len(v) // 2]) if v else None
    except Exception:  # noqa: BLE001
        return None
