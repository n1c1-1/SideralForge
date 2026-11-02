"""ApiWorker: the thread that does what touches the disk or takes time, so that the GUI never blocks.

Workspace scans, the file parts of the keys, describe, outputs, arrays, thumbnails, imports: each request carries a
channel; a new request on a channel replaces the one still waiting there, and only the result of the newest request
of a channel is delivered. Results come back in the GUI thread as signals.

    worker = ApiWorker(parent)
    worker.done.connect(on_done)                 # (channel, request number, result)
    worker.failed.connect(on_failed)             # (channel, request number, error text)
    n = worker.submit("home", scan_home, rows, recent)

Also here, as plain functions run by the worker: ``machine_info()`` (the GPUs, cores and memory) and
``disk_usage(folders)``; and ``norm_path(path)``, a path as Windows compares it (no disk access).
"""
from __future__ import annotations

import itertools
import os
import shutil
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QObject, Qt, QThread, Signal, Slot


class _Thread(QThread):
    result = Signal(str, int, bool, object)          # channel, request number, ok, value (or error text)

    def __init__(self, owner: ApiWorker):
        super().__init__()
        self.owner = owner

    def run(self):
        while True:
            job = self.owner._next()
            if job is None:
                return
            channel, n, fn, args, kwargs = job
            try:
                value, ok = fn(*args, **kwargs), True
            except Exception as e:  # noqa: BLE001 - every failure goes back to the GUI as text
                value, ok = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=6)}", False
            self.result.emit(channel, n, ok, value)


class ApiWorker(QObject):
    done = Signal(str, int, object)                  # channel, request number, result
    failed = Signal(str, int, str)                   # channel, request number, error text

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cond = threading.Condition()
        self._waiting: dict[str, tuple] = {}         # channel → (n, fn, args, kwargs), the newest not started
        self._order: list[str] = []                  # channels in the order of their requests
        self._latest: dict[str, int] = {}            # channel → number of its newest request
        self._inflight = 0                           # requests not yet delivered or dropped
        self._seq = itertools.count(1)
        self._stopping = False
        self._thread = _Thread(self)
        self._thread.result.connect(self._deliver, Qt.QueuedConnection)
        self._thread.start()
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.stop)

    # ------------------------------------------------------------ GUI thread
    def submit(self, channel: str, fn, *args, **kwargs) -> int:
        """Run ``fn(*args, **kwargs)`` in the worker; returns the request number. A request still waiting on the
        same channel is dropped."""
        n = next(self._seq)
        with self._cond:
            self._latest[channel] = n
            if channel in self._waiting:
                self._inflight -= 1
            else:
                self._order.append(channel)
            self._waiting[channel] = (n, fn, args, kwargs)
            self._inflight += 1
            self._cond.notify()
        return n

    def cancel(self, channel: str):
        """Drop the waiting request of a channel, and the result of the one running."""
        with self._cond:
            self._latest[channel] = 0
            if self._waiting.pop(channel, None) is not None:
                self._order.remove(channel)
                self._inflight -= 1

    def busy(self) -> bool:
        return self._inflight > 0

    def wait(self, timeout: float = 30.0) -> bool:
        """Process events until every request is delivered (tests and scripted walkthroughs). False on timeout."""
        end = time.monotonic() + timeout
        app = QCoreApplication.instance()
        while self.busy() and time.monotonic() < end:
            app.processEvents()
            time.sleep(0.005)
        app.processEvents()
        return not self.busy()

    def stop(self, timeout_ms: int = 5000):
        """End the thread after the request it is running."""
        with self._cond:
            self._stopping = True
            self._waiting.clear()
            self._order.clear()
            self._cond.notify()
        self._thread.wait(timeout_ms)

    @Slot(str, int, bool, object)
    def _deliver(self, channel, n, ok, value):
        self._inflight -= 1
        if self._latest.get(channel) != n:
            return                                   # a newer request on this channel, or cancelled
        if ok:
            self.done.emit(channel, n, value)
        else:
            self.failed.emit(channel, n, value)

    # ------------------------------------------------------------ worker thread
    def _next(self):
        with self._cond:
            while not self._stopping and not self._order:
                self._cond.wait()
            if self._stopping:
                return None
            channel = self._order.pop(0)
            n, fn, args, kwargs = self._waiting.pop(channel)
            return channel, n, fn, args, kwargs


# ---------------------------------------------------------------------------------------------- jobs

@dataclass
class Gpu:
    device: str                                      # "cuda:0"
    name: str                                        # the name the driver gives
    memory_gb: float


@dataclass
class Machine:
    gpus: list[Gpu] = field(default_factory=list)
    cores: int = 0
    memory_gb: float = 0.0

    def gpu(self, device: str) -> Gpu | None:
        return next((g for g in self.gpus if g.device == device), None)


def _short(name: str) -> str:
    return " ".join(w for w in name.split() if w != "GeForce")


def machine_info() -> Machine:
    """The GPUs (in CUDA's order, the device names the engine uses), the CPU cores and the memory."""
    gpus: list[Gpu] = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                pr = torch.cuda.get_device_properties(i)
                gpus.append(Gpu(f"cuda:{i}", _short(pr.name), pr.total_memory / 2 ** 30))
    except Exception:  # noqa: BLE001 - no torch or no driver: try nvidia-smi
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
            for i, line in enumerate(x for x in out.splitlines() if x.strip()):
                name, mib = line.rsplit(",", 1)
                gpus.append(Gpu(f"cuda:{i}", _short(name.strip()), float(mib) / 1024))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    mem = 0.0
    try:
        import psutil
        mem = psutil.virtual_memory().total / 2 ** 30
    except Exception:  # noqa: BLE001
        pass
    return Machine(gpus, os.cpu_count() or 0, mem)


def norm_path(x) -> str:
    """A path as Windows compares it: case and separators made alike (no disk access)."""
    return os.path.normcase(os.path.normpath(str(x)))


def disk_usage(folders: dict[str, str]) -> dict[str, tuple[float, float] | None]:
    """{name: (used GB, total GB) of the drive holding the folder}, None when the drive is not there."""
    out: dict[str, tuple[float, float] | None] = {}
    for name, folder in folders.items():
        p = Path(folder) if folder else None
        while p is not None and not p.exists() and p.parent != p:
            p = p.parent
        try:
            u = shutil.disk_usage(p) if p is not None and p.exists() else None
        except OSError:
            u = None
        out[name] = (u.used / 1e9, u.total / 1e9) if u else None
    return out
