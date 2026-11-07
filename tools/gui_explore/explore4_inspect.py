"""Round 4 (agent I): the control round's points on Inspect, the run window, the panels and the words, replayed with
real gestures in the offscreen window. Prints what each shows; screenshots in GUI_EXPLORE_OUT.

    python explore4_inspect.py fr
"""
from __future__ import annotations

import faulthandler
import sys

faulthandler.dump_traceback_later(400, exit=True)
from harness import Explorer, stray_qt_errors  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton  # noqa: E402

LANG = sys.argv[1] if len(sys.argv) > 1 else "fr"
stray_qt_errors()
ex = Explorer("inspect4", lang=LANG, size=(1366, 768))


def band():
    st = ex.w.tune.strip
    return [b.text() for b in st.blocks.findChildren(QPushButton) if b.isVisible()]


def lead_visible():
    st = ex.w.tune.strip
    return [b.text() for b in st.findChildren(QPushButton) if b.isVisible() and b.parent() is st]


def head():
    t = ex.w.tune
    return [w.text() for w in t.findChildren(QLabel) if w.isVisible() and w.text() and
            (w.parent() is not None and t.pages.currentIndex() == 0)][:6]


def chain(keys):
    for k in keys:
        ex.double_click_card(k)
        ex.settle(3)


# 1, 5: a long chain at 1366, Stars/background opened: "◂ Graph" and the list stay; no Output; the graph's names
ex.open(ex.project("session", f"I1 {LANG}"))
ex.click_block(ex.block("sources") or ex.blocks()[0])
chain(("calibration", "registration", "selection", "stacking", "astrometry", "gradient", "color", "denoise", "stars"))
ex.double_click_block(ex.block("etoiles"))
ex.settle(8)
st = ex.w.tune.strip
back = next(b for b in st.findChildren(QPushButton) if "◂" in b.text())
x = back.mapTo(ex.w, back.rect().topLeft()).x()
print("1. band:", band(), "| ◂ at x =", x, "visible", back.isVisible(), "| scroll",
      ex.w.tune.strip_scroll.horizontalScrollBar().value(), flush=True)
ex.shot(f"i1_band_stars_{LANG}")
ex.w.set_mode(0)
ex.settle(4)

# 6, 2, 3: Gradient 2 switched off: Inspect, its band button, Run to here, a pause on it
ex.click_block(ex.block("fond"))
ex.double_click_card("gradient")
ex.settle(4)
ex.click_block(ex.block("fond~2"))
ex.settle(3)
ex.right_click_block(ex.block("fond~2"), "Pause")
ex.settle(3)
ex.click_block(ex.block("fond~2"))
ex.key(Qt.Key_D)
ex.settle(4)
g2 = ex.block("fond~2")
print("3. tooltip of Gradient 2 off:", g2.toolTip().replace("\n", " | "), flush=True)
ex.click_block(g2)
ex.key(Qt.Key_F5, mod=Qt.ShiftModifier)
ex.settle(6)
d = ex.w.run.dialog if hasattr(ex.w.run, "dialog") else None
dl = [x for x in ex.dialogs()]
for x in dl:
    print("2. run window:", x.windowTitle(), "|", [t for t in ex.texts(x) if t][:14], flush=True)
    ex.shot(f"i2_runhere_off_{LANG}", x)
    x.reject()
ex.settle(3)
ex.click_block(ex.block("debruitage"))
ex.key(Qt.Key_F5)
ex.settle(6)
for x in ex.dialogs():
    print("3. F5 run window:", [t for t in ex.texts(x) if t][:16], flush=True)
    ex.shot(f"i3_f5_pause_off_{LANG}", x)
    x.reject()
ex.settle(3)
ex.double_click_block(ex.block("fond~2"))
ex.settle(8)
print("6. Inspect Gradient 2 off:", [t for t in ex.texts(ex.w.tune) if "Gradient" in t or "désactiv" in t
                                     or "off" in t][:6], flush=True)
b2 = next((b for b in ex.w.tune.strip.blocks.findChildren(QPushButton) if b.text() == "Gradient 2"), None)
print("6. band button:", b2.text() if b2 else None, "|", b2.toolTip() if b2 else "", "|",
      b2.styleSheet()[:60] if b2 else "", flush=True)
ex.shot(f"i6_inspect_g2_off_{LANG}")
ex.w.set_mode(0)
ex.settle(4)

# 9: Gradient 2 on again: its Method cards, chosen like the first one's
ex.click_block(ex.block("fond~2"))
ex.key(Qt.Key_D)
ex.settle(4)
ex.click_block(ex.block("fond~2"))
ex.settle(4)
pn = ex.page.panel
cards = [w for w in pn.findChildren(QLabel) if w.isVisible() and w.text() in ("RBF", "Polynôme", "Polynomial",
                                                                               "Automatique", "Automatic")]
print("9. Gradient 2 panel method texts:", [c.text() for c in cards],
      "| greyed 'Fixé':", any("options du workflow" in t or "options of the workflow" in t for t in ex.texts(pn)),
      flush=True)
from astro_gpu.client.panel import MethodCard  # noqa: E402
mc = [c for c in pn.findChildren(MethodCard) if c.isVisible()]
print("9. method cards:", len(mc), flush=True)
if len(mc) > 1:
    target = next(c for c in mc if not c.on) if hasattr(mc[0], "on") else mc[1]
    ex.click(target)
    ex.settle(5)
    from astro_gpu.client.model import method_of  # noqa: E402
    print("9. after a click: Gradient 2 method", method_of(ex.s.path, "fond~2"), "| Gradient 1",
          method_of(ex.s.path, "fond"), "| status:", ex.w.status.text(), flush=True)
ex.shot(f"i9_g2_method_{LANG}")

# 10, 12: Stacking's drizzle scale, the Astrometry block before anything computed
ex.click_block(ex.block("empilement"))
ex.settle(4)
eds = [(w.placeholderText(), w.text()) for w in ex.page.panel.findChildren(QLineEdit) if w.isVisible()]
print("10. stacking panel line edits:", eds, flush=True)
print("10. scale texts:", [t for t in ex.texts(ex.page.panel) if "drizzle" in t.lower()][:4], flush=True)
ex.shot(f"i10_stacking_{LANG}")
print("12. astrometry block:", ex.block("astrometrie").summary, flush=True)

# 7: the last block after a deletion — needs computed results: covered by tests (test_inspect)

# 4, 5: two sessions: the band from each Session, its Calibration; Inspect of a session's Calibration
ex.open(ex.project("two_sessions", f"I4 {LANG}"))
ex.click_block(ex.block("session", sid=1) or ex.blocks()[0])
chain(("calibration", "registration", "stacking"))
cal = ex.block("calibration", sid=1)
print("4. calibration@1 block:", cal.toolTip().replace("\n", " | ") if cal else None, flush=True)
if cal is not None:
    ex.double_click_block(cal)
    ex.settle(8)
    print("4/5. band:", band(), flush=True)
    print("4. Inspect texts:", [t for t in ex.texts(ex.w.tune) if t in ("À jour", "Up to date", "Pas encore calculé",
                                                                         "Not computed yet")], flush=True)
    ex.shot(f"i4_cal1_{LANG}")
    ex.w.set_mode(0)

# 5, 13: Image project: band "Image › Gradient › Post"; status line at opening
ex.open(ex.project("image", f"I13 {LANG}"))
ex.settle(6)
print("13. status line:", ex.w.status.text(), flush=True)
ex.click_block(ex.block("image"))
ex.double_click_card("gradient")
ex.settle(4)
ex.double_click_block(ex.block("fond"))
ex.settle(8)
print("5. image band:", band(), flush=True)
ex.shot(f"i5_image_band_{LANG}")
ex.w.set_mode(0)
print("anomalies:", ex.notes(), flush=True)
ex.close()
