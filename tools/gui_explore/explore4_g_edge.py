"""Round 4 (agent G): a palette card held near the right edge of the graph at 1366x768: the view scrolls (17)."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_post_common import CHAINS, Explorer, build_chain, start  # noqa: E402
from PySide6.QtCore import QMimeData, QPoint, Qt  # noqa: E402
from PySide6.QtGui import QDragMoveEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

start(600)
ex = Explorer("g4", size=(1366, 768))
ex.open(ex.project("session"))
ex.w.activateWindow()
build_chain(ex, CHAINS["stars"])
v = ex.view
hb = v.horizontalScrollBar()
print("range", hb.minimum(), hb.maximum(), "value", hb.value(), "policy", v.horizontalScrollBarPolicy())
vp = v.viewport()
ex.drag_palette_hover("gradient", QPoint(vp.width() // 2, vp.height() // 2))
print("after enter: range", hb.minimum(), hb.maximum(), "value", hb.value())
edge = QPoint(vp.width() - 8, vp.height() // 2)
mime = QMimeData()
mime.setText("block:gradient")
for _ in range(3):
    QApplication.sendEvent(vp, QDragMoveEvent(edge, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    ex.settle(2)
t0 = time.monotonic()
while time.monotonic() - t0 < 1.5:
    ex.settle(1)
print("held 1.5 s at the edge: value", hb.value(), "timer", v._edge_timer.isActive(), "pos", v._edge_pos)
ex.drag_leave()
# the control round's gesture: a new hover at the edge, then settle(20)
h0 = hb.value()
t0 = time.monotonic()
ex.drag_palette_hover("gradient", QPoint(vp.width() - 5, vp.height() // 2))
ex.settle(20)
print("control round gesture: %.2f s held, value %d -> %d" % (time.monotonic() - t0, h0, hb.value()))
ex.drag_leave()
ex.close()
