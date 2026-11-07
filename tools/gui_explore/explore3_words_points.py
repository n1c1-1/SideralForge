"""Words and layout, round 3: the round-2 new points replayed with real gestures (image project at opening, the
palette reasons, "off", the Inspect band at 1366, two sessions' band, "Sortie" blocks, same thing two names).
Writes <out>/points_<lang>_<w>.json and screenshots to <out>/points/.

    python explore3_words_points.py fr 1366 768
"""
from __future__ import annotations

import faulthandler
import json
import re
import sys
import time
import traceback

faulthandler.dump_traceback_later(500, exit=True)
from harness import Explorer, stray_qt_errors  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QEnterEvent, QTextDocument  # noqa: E402
from PySide6.QtWidgets import QAbstractButton, QApplication, QLabel, QPushButton, QWidget  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import graph as graphmod  # noqa: E402
from astro_gpu.client import i18n  # noqa: E402

LANG = sys.argv[1] if len(sys.argv) > 1 else "fr"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 1366
H = int(sys.argv[3]) if len(sys.argv) > 3 else 768
T = i18n._
stray_qt_errors()
ex = Explorer("words_layout", lang=LANG, size=(W, H))
SH = ex.out / "points"
SH.mkdir(exist_ok=True)
R: dict = {}
SAID: list[str] = []
_st = graphmod.GraphView.show_toast
graphmod.GraphView.show_toast = lambda self, t: (SAID.append(str(t)), _st(self, t))[1]
_fl = appmod.Main.flash
appmod.Main.flash = lambda self, t: (SAID.append("[barre] " + str(t)), _fl(self, t))[1]


def plain(t):
    if t and (re.search(r"<[a-zA-Z/!]", t) or "&nbsp;" in t):
        d = QTextDocument()
        d.setHtml(t)
        t = d.toPlainText()
    return (t or "").strip()


def shot(name, w=None):
    f = SH / f"{LANG}_{W}_{name}.png"
    ex.settle(2)
    (w or ex.w).grab().save(str(f))
    return str(f)


def texts(root=None):
    root = root or ex.w
    out = []
    for x in [root] + root.findChildren(QWidget):
        if not x.isVisible():
            continue
        if isinstance(x, QLabel) and x.text():
            out.append(plain(x.text()))
        elif isinstance(x, QAbstractButton) and x.text():
            out.append(x.text())
        if x.toolTip():
            out.append("tip: " + plain(x.toolTip()))
    return out


def scene_texts():
    out = []
    for it in ex.view.scene().items():
        for a in ("title", "summary", "text"):
            v = getattr(it, a, None)
            if isinstance(v, str) and v.strip():
                out.append(v)
        if it.toolTip():
            out.append("tip: " + plain(it.toolTip()))
    return out


def reasons():
    page = ex.page
    out = {}
    for b, card in list(page.cards.items()):
        if not card.reason:
            continue
        QApplication.sendEvent(card, QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(card.mapToGlobal(QPoint(5, 5)))))
        ex.settle(2)
        out[card.title] = page.hint.text()
        QApplication.sendEvent(card, QEnterEvent(QPointF(-5, -5), QPointF(-5, -5), QPointF(-5, -5)))
        page.hide_hint()
    return out


def place(*keys):
    for k in keys:
        ex.double_click_card(k)
        ex.settle(4)


def blk(key, nth=0):
    its = sorted([x for x in ex.blocks() if x.node.key.split("~")[0] == key], key=lambda x: x.pos().x())
    return its[nth] if len(its) > nth else None


def rich(two=False):
    ex.open(ex.project("two_sessions" if two else "session", f"P{LANG}{W}{time.strftime('%H%M%S')}{int(two)}"))
    ex.settle(8)
    ex.click_block(ex.block("session") or ex.block("sources") or ex.blocks()[0])
    place("calibration", "registration", "selection", "stacking", "astrometry", "gradient", "color", "denoise",
          "stars", "gradient", "denoise")


def band_state():
    """The Inspect band: its buttons, the open one, and how much of it is in view."""
    bs = [b for b in ex.w.tune.findChildren(QPushButton) if b.isVisible()]
    out = []
    for b in bs:
        vr = b.visibleRegion().boundingRect()
        out.append([b.text(), b.isChecked(), b.width(), vr.width()])
    return out


def section(name, fn):
    print("==", name, flush=True)
    SAID.clear()
    try:
        R[name] = fn()
    except Exception as e:  # noqa: BLE001
        R[name] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2000:]}
        print("ERROR", name, e, flush=True)
    (ex.out / f"points_{LANG}_{W}.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")


# N1, N5: an image project at opening, nothing clicked
def s_image_open():
    ex.open(ex.project("image"))
    ex.settle(8)
    out = {"status": ex.w.status.text(), "reasons": reasons(), "panel": texts(ex.page.panel)[:30],
           "head": [t for t in texts(ex.page) if "Blocs" in t or "Blocks" in t or "workflow" in t.lower()][:10]}
    out["shot"] = shot("image_open")
    ex.click_block(ex.block("image"))
    out["reasons_after_click"] = reasons()
    out["status_after_click"] = ex.w.status.text()
    return out


# N2: Stars reason in a session project, "Sortie" blocks after To Post on Stacking
def s_sortie():
    ex.open(ex.project("session"))
    ex.settle(8)
    ex.click_block(ex.block("session") or ex.block("sources") or ex.blocks()[0])
    place("calibration")
    out = {"reasons_cal": reasons()}
    out["shot_reason"] = shot("reason_after_calibration")
    place("registration", "selection", "stacking", "astrometry", "gradient", "color")
    st = blk("empilement") or blk("stacking")
    out["menu_stack"] = [a.get("text") for a in ex.right_click_block(st, T("To Post"))]
    ex.settle(6)
    out["toasts"] = list(SAID)
    out["titles"] = [x.title for x in sorted(ex.blocks(), key=lambda x: x.pos().x())]
    out["scene"] = [t for t in scene_texts() if "Sortie" in t or "Output" in t]
    out["shot"] = shot("after_to_post_stacking")
    return out


# N0, N4, N6, N7: a full chain, Gradient 2 switched off, the Log, Inspect of Stars at this size, Ctrl+Z
def s_rich():
    rich()
    out = {}
    g2 = blk("fond", 1)
    out["menu_g2"] = [a.get("text") for a in ex.right_click_block(g2, T("Switch off"))]
    ex.settle(4)
    out["toasts_off"] = list(SAID)
    out["scene_off"] = [t for t in scene_texts() if re.search(r"\boff\b|Désactiv|Switched|switched", t)]
    out["shot_off"] = shot("gradient2_off")
    SAID.clear()
    ex.undo()
    ex.settle(4)
    out["toasts_undo"] = list(SAID)
    out["status_undo"] = ex.w.status.text()
    SAID.clear()
    ex.redo()
    ex.settle(4)
    out["toasts_redo"] = list(SAID)
    out["status_redo"] = ex.w.status.text()
    # undo after a refused delete
    d = blk("debruitage")
    if d is not None:
        ex.click_block(d)
        SAID.clear()
        ex.key(Qt.Key_Delete)
        out["toasts_refused"] = list(SAID)
        SAID.clear()
        ex.undo()
        out["toasts_undo2"] = list(SAID)
        out["status_undo2"] = ex.w.status.text()
    out["all_texts"] = sorted(set(t for t in texts() + scene_texts()
                                  if re.search(r"Jusqu|Run to|[Ff]lux|Flow|Retir|Remove|Pause|off\b|Images|Frames", t)))
    ex.w.b_log.setChecked(True)
    ex.settle(3)
    t = ex.w.logp.table
    out["log"] = [" | ".join(t.item(r, c).text() for c in range(t.columnCount()) if t.item(r, c))
                  for r in range(t.rowCount())]
    out["shot_log"] = shot("log")
    ex.w.b_log.setChecked(False)
    # Inspect of Stars/background: is its band button in view?
    s = blk("etoiles")
    ex.double_click_block(s)
    ex.settle(8)
    out["band_stars"] = band_state()
    out["inspect_texts"] = texts(ex.w.tune)[:80]
    out["shot_band"] = shot("inspect_stars_band")
    out["win"] = [ex.w.width(), ex.w.height()]
    ex.w.set_mode(0)
    ex.settle(3)
    return out


# N10: two sessions, Inspect of Gradient: is Calibration in the band?
def s_two():
    rich(two=True)
    g = blk("fond", 0)
    ex.double_click_block(g)
    ex.settle(8)
    out = {"band": band_state(), "shot": shot("two_sessions_band")}
    ex.w.set_mode(0)
    return out


ONLY = sys.argv[4].split(",") if len(sys.argv) > 4 else ["image_open", "sortie", "rich", "two"]
OLD = ex.out / f"points_{LANG}_{W}.json"
if OLD.exists():
    R.update(json.loads(OLD.read_text(encoding="utf-8")))
for name, fn in (("image_open", s_image_open), ("sortie", s_sortie), ("rich", s_rich), ("two", s_two)):
    if name in ONLY:
        section(name, fn)
ex.close()
print("DONE", flush=True)
