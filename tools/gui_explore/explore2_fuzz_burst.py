"""Round 2 fuzz: Ctrl+Z / Ctrl+Y pressed fast, as a user does, without waiting for the window between two presses.
After random edits (a round-2 sequence's first gestures), all of them undone in a burst, then redone in a burst:
the project must be the start one, then the end one.

    python explore2_fuzz_burst.py <first seed> <n>
"""
import os
import sys
from pathlib import Path

A, N = int(sys.argv[1]), int(sys.argv[2])
sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore2_fuzz as G  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

fz = G.Fuzz2("fr")
fz.panels = False
ex = fz.ex
EDITS = {"dbl_card", "drag_card", "menu", "cross", "del", "d", "to_palette", "pull_port", "wire"}
for seed in range(A, A + N):
    seq = G.make_sequence(seed)
    acts = [a for a in seq["actions"] if a[0] in EDITS][:10]
    res = fz.run({"seed": seed, "start": seq["start"], "actions": acts}, f"b{seed}", watchdog=60)
    s = ex.w.s
    if s is None:
        continue
    fz.ensure_graph([])
    end = fz.state()
    n = len(s._undo)
    fz.errs.clear()
    fz.activate()
    for _ in range(n):                                  # fast: no wait between presses
        QTest.keyClick(ex.w, Qt.Key_Z, Qt.ControlModifier)
    fz.wait_ready()
    ex.settle(6)
    start = fz.state()
    mid = len(s._undo)
    d0 = fz.diff(fz._first if hasattr(fz, "_first") else start, start)
    for _ in range(n):
        QTest.keyClick(ex.w, Qt.Key_Y, Qt.ControlModifier)
    fz.wait_ready()
    ex.settle(6)
    back = fz.state()
    d1 = fz.diff(end, back)
    errs = [e.strip().splitlines()[-1] for e in fz.errs]
    fz.errs.clear()
    print("SEED", seed, seq["start"], "edits", len(acts), "undo entries", n, "after undo burst", mid, "after redo burst", len(s._undo), "| end≠back:", d1,
          "|", fz.explain(end, back, d1[0])[:400] if d1 else "", "| errs", errs[:3], flush=True)
os._exit(0)
