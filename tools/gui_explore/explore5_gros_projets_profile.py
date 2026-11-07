"""gros_projets: where the GUI thread spends its time when a large project opens and when a block is placed
(cProfile of the GUI thread, top functions by cumulative time). Arg: the project folder."""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Explorer  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

root = Path(sys.argv[1])
ex = Explorer("gros_projets")
ex.open()
ex.settle(10)


def drain(quiet=1.0):
    last = time.perf_counter()
    t0 = last
    while time.perf_counter() - last < quiet and time.perf_counter() - t0 < 120:
        a = time.perf_counter()
        QApplication.processEvents()
        if time.perf_counter() - a > 0.015 or ex.w.worker.busy():
            last = time.perf_counter()
        time.sleep(0.002)


def prof(tag, fn):
    pr = cProfile.Profile()
    pr.enable()
    fn()
    drain()
    pr.disable()
    s = io.StringIO()
    st = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    st.print_stats(45)
    print("=====", tag)
    print("\n".join(x for x in s.getvalue().splitlines() if "astro_gpu" in x or "ncalls" in x)[:9000], flush=True)
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(15)
    print("----- tottime", tag)
    print(s.getvalue()[-4500:], flush=True)


prof("open", lambda: ex.w.open_project(root))
tgt = next(b for b in ex.blocks() if b.node.key == "debruitage")
ex.view.centerOn(tgt)
ex.settle(3)
QTest.mouseClick(ex.view.viewport(), Qt.LeftButton, Qt.NoModifier, ex.point_of(tgt))
drain(0.5)
prof("palette double-click", lambda: QTest.mouseDClick(ex.card("gradient"), Qt.LeftButton, Qt.NoModifier,
                                                       QPoint(20, 12)))
ex.close()
