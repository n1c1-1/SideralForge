"""gros_projets: the GUI process's memory over repeated close / reopen of the same project (does each reopening keep
memory?). Args: project folder, number of reopenings."""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Explorer  # noqa: E402
from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import psutil  # noqa: E402

root = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ex = Explorer("gros_projets")
ex.open()
ex.settle(10)


def quiet(sec=1.0):
    last = t0 = time.perf_counter()
    while time.perf_counter() - last < sec and time.perf_counter() - t0 < 120:
        a = time.perf_counter()
        QApplication.processEvents()
        if time.perf_counter() - a > 0.015 or ex.w.worker.busy():
            last = time.perf_counter()
        time.sleep(0.005)


rss = []
for i in range(n):
    ex.w.open_project(root)
    quiet()
    ex.w.close_project()
    quiet(0.5)
    # the deletions the application's event loop makes (deleteLater): processEvents alone never makes them, so the
    # closed project's widgets were counted as kept (round 4 fixes)
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
    gc.collect()
    rss.append(round(psutil.Process(os.getpid()).memory_info().rss / 2 ** 20, 1))
    print("RSS", i + 1, rss[-1], flush=True)
print("RSS_ALL", rss, "growth/reopen", round((rss[-1] - rss[1]) / max(1, n - 2), 1), flush=True)
ex.close()
