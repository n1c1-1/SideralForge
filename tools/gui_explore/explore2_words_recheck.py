"""Words and layout, round 2: the round-1 anomalies replayed with real gestures. Writes the measures to
<out>/recheck_raw_<lang>_<w>.json and screenshots to <out>/recheck/ (the verdicts, recheck.jsonl, are written after a
look at them).

    python explore2_words_recheck.py fr 1366 768 [section,...]
"""
from __future__ import annotations

import faulthandler
import json
import re
import sys
import time
import traceback

faulthandler.dump_traceback_later(900, exit=True)
from harness import Explorer, stray_qt_errors  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QEnterEvent, QTextDocument  # noqa: E402
from PySide6.QtWidgets import (QAbstractButton, QApplication, QComboBox, QLabel, QPushButton,  # noqa: E402
                               QWidget)

from astro_gpu.client import graph as graphmod  # noqa: E402
from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.client.widgets import ElidedLabel  # noqa: E402

LANG = sys.argv[1] if len(sys.argv) > 1 else "fr"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 1366
H = int(sys.argv[3]) if len(sys.argv) > 3 else 768
SECTIONS = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] != "all" else ["all"]
T = i18n._
stray_qt_errors()
ex = Explorer("words_layout", lang=LANG, size=(W, H))
SH = ex.out / "recheck"
SH.mkdir(exist_ok=True)
RAW = ex.out / f"recheck_raw_{LANG}_{W}.json"
R: dict = json.loads(RAW.read_text(encoding="utf-8")) if RAW.exists() else {}
TOASTS: list[str] = []
_st = graphmod.GraphView.show_toast
graphmod.GraphView.show_toast = lambda self, t: (TOASTS.append(str(t)), _st(self, t))[1]


def on(name):
    return "all" in SECTIONS or name in SECTIONS


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
    """(text, kind) of the visible widgets: labels, buttons (raw text, & kept), combos, tooltips."""
    root = root or ex.w
    out = []
    for x in [root] + root.findChildren(QWidget):
        if not x.isVisible():
            continue
        if isinstance(x, ElidedLabel):
            out.append((x.full_text(), "label"))
        elif isinstance(x, QLabel) and x.text():
            out.append((plain(x.text()), "label"))
        elif isinstance(x, QAbstractButton) and x.text():
            out.append((x.text(), "button"))
        elif isinstance(x, QComboBox):
            out += [(x.itemText(i), "combo") for i in range(x.count())]
        if x.toolTip():
            out.append((plain(x.toolTip()), "tip"))
    return out


def scene_texts():
    out = []
    for it in ex.view.scene().items():
        for a in ("title", "summary", "text"):
            v = getattr(it, a, None)
            if callable(v):
                try:
                    v = v()
                except TypeError:
                    v = None
            if isinstance(v, str) and v.strip():
                out.append((v, f"graph {type(it).__name__}.{a}"))
        if it.toolTip():
            out.append((plain(it.toolTip()), f"graph {type(it).__name__} tip"))
    return out


def log_lines():
    ex.w.b_log.setChecked(True)
    ex.settle(3)
    t = ex.w.logp.table
    return [" | ".join(t.item(r, c).text() for c in range(t.columnCount()) if t.item(r, c)) for r in range(t.rowCount())]


def sel(key, nth=0):
    its = sorted([x for x in ex.blocks() if x.node.key.split("~")[0] == key], key=lambda x: x.pos().x())
    if len(its) > nth:
        ex.click_block(its[nth])
        its = sorted([x for x in ex.blocks() if x.node.key.split("~")[0] == key], key=lambda x: x.pos().x())
        return its[nth] if len(its) > nth else None     # a click may redraw the graph: the item drawn now
    return None


def place(*keys):
    for k in keys:
        ex.double_click_card(k)
        ex.settle(4)


def rich(two_sessions=False):
    """A session project, the chain built from the palette with Gradient and Denoise placed twice."""
    root = ex.project("two_sessions" if two_sessions else "session", f"Chaine {LANG} {time.strftime('%H%M%S')}")
    ex.open(root)
    ex.settle(8)
    it = ex.block("session") or ex.block("sources") or (ex.blocks()[0] if ex.blocks() else None)
    if it is not None:
        ex.click_block(it)
    if two_sessions:
        z = ex.ghosts()
    place("calibration", "registration", "selection", "stacking", "astrometry", "gradient", "color", "denoise",
          "stars", "gradient", "denoise")
    return root


def section(name, fn):
    if not on(name):
        return
    print("==", name, flush=True)
    TOASTS.clear()
    try:
        R[name] = fn()
    except Exception as e:  # noqa: BLE001
        R[name] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2000:]}
        print("ERROR", name, e, flush=True)
    RAW.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")


def widest(root, n=6):
    rows = []
    for x in root.findChildren(QWidget):
        if x.isVisible():
            m = max(x.minimumSizeHint().width(), x.minimumWidth())
            rows.append((m, type(x).__name__, x.objectName(), (x.text()[:40] if hasattr(x, "text") and
                                                                isinstance(getattr(x, "text")(), str) else "")
                         if not isinstance(x, QComboBox) else ""))
    rows.sort(key=lambda r: -r[0])
    return rows[:n]


# 1. Inspect widens the window at 1366
def s_width():
    out = {}
    for label, maker in (("legacy", lambda: ex.open(ex.project("legacy"))), ("rich", rich)):
        maker()
        ex.settle(6)
        for key in ("etoiles", "fond", "debruitage"):
            ex.w.set_mode(0)
            ex.w.resize(W, H)
            ex.settle(4)
            its = sorted([x for x in ex.blocks() if x.node.key.startswith(key)], key=lambda x: -x.pos().x())
            if not its:
                continue
            ex.double_click_block(its[0])
            ex.settle(8)
            r = {"inspect_w": ex.w.width(), "min": ex.w.minimumSizeHint().width(), "mode": ex.w.mode}
            r["shot"] = shot(f"width_{label}_{key}")
            r["band"] = [b.text() for b in ex.w.tune.findChildren(QPushButton) if b.isVisible()][:40]
            r["widest"] = widest(ex.w.tune)
            ex.w.set_mode(0)
            ex.settle(4)
            ex.w.resize(W, H)
            ex.settle(4)
            r["after_back_w"] = ex.w.width()
            out[f"{label}/{key}"] = r
    return out


# 2, 10, 13. The Log after switch off, copies, branch, send, template; the numbered copies in Inspect / run window
def s_log():
    out = {}
    rich()
    g2 = sel("fond", 1)
    out["menu_g2"] = [a.get("text") for a in ex.right_click_block(g2, T("Switch off"))]
    ex.settle(4)
    out["toasts_off"] = list(TOASTS)
    g2 = sorted([x for x in ex.blocks() if x.node.key.startswith("fond")], key=lambda x: x.pos().x())
    out["graph_titles"] = [x.title for x in sorted(ex.blocks(), key=lambda x: x.pos().x())]
    ex.double_click_block(g2[-1])
    ex.settle(8)
    out["inspect_g2_texts"] = [t for t in texts(ex.w.tune)][:120]
    out["inspect_g2_shot"] = shot("inspect_gradient2_off")
    ex.w.set_mode(0)
    ex.settle(3)
    g1 = sorted([x for x in ex.blocks() if x.node.key.startswith("fond")], key=lambda x: x.pos().x())[0]
    ex.double_click_block(g1)
    ex.settle(8)
    out["inspect_g1_texts"] = [t for t in texts(ex.w.tune)][:120]
    out["inspect_g1_shot"] = shot("inspect_gradient1")
    ex.w.set_mode(0)
    ex.settle(3)
    # a branch: Denoise under the Stacking
    ex.drag_palette_hover("denoise", ex.view.viewport().rect().center())
    br = [g for g in ex.ghosts() if getattr(g, "branch_key", None)]
    pt = ex.ghost_center(br[0]) if br else None
    ex.drag_leave()
    if pt is not None:
        ex.drag_palette("denoise", pt)
        ex.settle(6)
    out["toasts_branch"] = list(TOASTS)
    st = sel("empilement")
    if st is not None:
        out["menu_stack"] = [a.get("text") for a in ex.right_click_block(st, T("To Post"))]
        ex.settle(4)
    a = sel("astrometrie")
    if a is not None:
        ex.right_click_block(a, T("Delete"))
        ex.settle(4)
    out["toasts"] = list(TOASTS)
    out["log"] = log_lines()
    out["log_shot"] = shot("log")
    # the run window
    d = ex.w.run.launch()
    ex.settle(8)
    if d is not None:
        out["run_texts"] = texts(d)
        out["run_shot"] = shot("run_window", d)
        d.hide()
    out["device_combo"] = [ex.w.device_cb.itemText(i) for i in range(ex.w.device_cb.count())]
    ex.w.b_log.setChecked(False)
    return out


# 3, 9, 15, 16: every text of a full graph, its menus, its panels: names
def s_names():
    out = {"texts": [], "menus": {}}
    rich()
    out["texts"] += texts() + scene_texts()
    for it in list(ex.blocks()):
        k = it.node.key + (f"@{it.node.sid}" if it.node.sid is not None else "")
        try:
            ex.click_block(it)
            out["texts"] += [(t, f"panel {k} {w}") for t, w in texts(ex.page.panel)]
            its = [x for x in ex.blocks() if x.node is it.node] or [it]
            out["menus"][k] = ex.right_click_block(its[0])
        except RuntimeError:
            continue
    # palette cards and their reasons
    for b, card in list(ex.page.cards.items()):
        out["texts"].append((card.title, "palette card"))
        if card.toolTip():
            out["texts"].append((card.toolTip(), "palette tip"))
        if card.reason:
            out["texts"].append((str(card.reason), "palette reason"))
    out["shot"] = shot("names_graph")
    return out


# 4, 14. The template window
def s_gallery():
    out = {}

    def inside(d):
        try:
            out["size"] = [d.width(), d.height(), d.minimumSizeHint().width(), d.minimumSizeHint().height()]
            out["tabs"] = [d.tabs.tabText(i) for i in range(d.tabs.count())]
            for i in range(d.tabs.count()):
                d.tabs.setCurrentIndex(i)
                ex.settle(2)
                for c in list(d.cards):
                    d.pick(c.t)
                    ex.settle(1)
                    key = c.t.get("key") or c.t.get("name")
                    tx = texts(d)
                    out[f"{d.tab_keys[i]}/{key}"] = tx
                    if key in ("sho", "dualband", "full", "Full"):
                        out[f"shot_{key}"] = shot(f"gallery_{key}", d)
        finally:
            d.reject()
    ex.open()
    ex.dialog_handler = inside
    ex.w.create_project(str(ex.ws / f"Galerie_{LANG}_{W}"), "Galerie", "M 42")
    ex.settle(12)
    ex.dialog_handler = None
    return out


# 5. Places on each other
def s_places():
    out = {}
    from astro_gpu.client.graph import BH, BlockItem, GhostItem
    for kind in ("two_sessions", "session_empty_wf", "session+image", "jpg_mistake"):
        ex.open(ex.project(kind))
        ex.settle(8)
        gs = [g for g in ex.ghosts()]
        z = ex.new_flow_zone()
        if z is not None and z not in gs:
            gs.append(z)
        over = []
        for i, a in enumerate(gs):
            ra = a.sceneBoundingRect()
            for b in gs[i + 1:]:
                ir = ra.intersected(b.sceneBoundingRect())
                if ir.width() > 3 and ir.height() > 3:
                    over.append([getattr(a, "text", ""), getattr(b, "text", ""), round(ir.width()), round(ir.height())])
            for bl in ex.blocks():
                ir = ra.intersected(bl.sceneBoundingRect())
                if ir.width() > 3 and ir.height() > 3:
                    over.append([getattr(a, "text", ""), "block " + bl.title, round(ir.width()), round(ir.height())])
        out[kind] = {"over": over, "ghosts": [getattr(g, "text", "") for g in gs], "shot": shot(f"places_{kind}")}
        # while dragging an image block
        ex.drag_palette_hover("denoise", ex.view.viewport().rect().center())
        gs = list(ex.ghosts()) + ([ex.new_flow_zone()] if ex.new_flow_zone() is not None else [])
        over = []
        for i, a in enumerate(gs):
            for b in gs[i + 1:]:
                ir = a.sceneBoundingRect().intersected(b.sceneBoundingRect())
                if ir.width() > 3 and ir.height() > 3:
                    over.append([getattr(a, "text", ""), getattr(b, "text", ""), round(ir.width()), round(ir.height())])
        out[kind]["drag_over"] = over
        out[kind]["drag_shot"] = shot(f"places_{kind}_drag")
        ex.drag_leave()
    return out


# 6. Run to here's tooltip in Inspect, two sessions, Denoise twice
def s_runhere():
    out = {}
    root = ex.project("two_sessions", f"RH {LANG}")
    ex.open(root)
    ex.settle(8)
    s1 = ex.block("session", sid=1) or ex.blocks()[0]
    ex.click_block(s1)
    place("calibration", "registration", "selection", "stacking", "astrometry", "gradient", "color", "denoise",
          "stars", "denoise")
    out["graph"] = [x.title for x in sorted(ex.blocks(), key=lambda x: (x.pos().x(), x.pos().y()))]
    for key in ("etoiles", "selection", "debruitage"):
        its = sorted([x for x in ex.blocks() if x.node.key.startswith(key)], key=lambda x: -x.pos().x())
        if not its:
            continue
        ex.double_click_block(its[0])
        ex.settle(8)
        rb = [b for b in ex.w.tune.findChildren(QPushButton) if b.isVisible() and b.text().startswith(T("Run to here"))]
        out[key] = {"buttons": [b.text() for b in rb], "tips": [b.toolTip() for b in rb], "shot": shot(f"runhere_{key}")}
        ex.w.set_mode(0)
        ex.settle(3)
    return out


# 7. Graph scale at this size
def s_scale():
    rich()
    ex.settle(6)
    k = ex.view.transform().m11()
    hs = ex.view.horizontalScrollBar()
    from PySide6.QtGui import QFont, QFontMetricsF
    f2 = QFontMetricsF(QFont("Segoe UI", 8.5))
    cut = [[it.title, it.summary] for it in ex.blocks() if it.summary and f2.horizontalAdvance(it.summary) > it.w - 15]
    out = {"scale": k, "hscroll_max": hs.maximum() if hs.isVisible() else 0, "cut_summaries": cut,
           "shot": shot("scale_rich")}
    ex.w.b_log.setChecked(True)
    ex.settle(4)
    out["with_log_scale"] = ex.view.transform().m11()
    out["with_log_view_h"] = ex.view.height()
    out["shot_log"] = shot("scale_rich_log")
    ex.w.b_log.setChecked(False)
    return out


# 8. & in English buttons and menus
def s_amp():
    ex.open(ex.project("legacy"))
    ex.settle(6)
    out = {"buttons": []}
    for key in ("etoiles", "fond"):
        it = ex.block(key)
        if it is None:
            continue
        ex.click_block(it)
        out["buttons"] += [t for t, w in texts(ex.page.panel) if w == "button"]
        out.setdefault("menus", {})[key] = ex.right_click_block(ex.block(key))
        ex.double_click_block(ex.block(key))
        ex.settle(8)
        out["buttons"] += [t for t, w in texts(ex.w.tune) if w == "button"]
        out[f"shot_{key}"] = shot(f"amp_inspect_{key}")
        ex.w.set_mode(0)
        ex.settle(3)
    out["amp"] = sorted({t for t in out["buttons"] if "&" in t})
    return out


# 11. Band title cut, palette reasons out of the page with the Log open
def s_band():
    from astro_gpu.client.graph import BandItem
    from PySide6.QtGui import QFont, QFontMetricsF
    ex.open(ex.project("image"))
    ex.settle(6)
    ex.click_block(ex.block("image"))
    place("gradient")
    fb = QFont("Segoe UI", 9)
    fb.setBold(True)
    fm = QFontMetricsF(fb)
    bands = [[it.text, round(fm.horizontalAdvance(it.text.upper())), round(it.rect.width() - 20)]
             for it in ex.view.scene().items() if isinstance(it, BandItem)]
    out = {"bands": bands, "shot": shot("band")}
    ex.w.b_log.setChecked(True)
    ex.settle(4)
    page = ex.page
    hints = []
    for b, card in list(page.cards.items()):
        if not card.reason:
            continue
        QApplication.sendEvent(card, QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(card.mapToGlobal(QPoint(5, 5)))))
        ex.settle(2)
        h = page.hint
        g = h.geometry()
        hints.append([card.title, h.isVisible(), g.y(), g.bottom(), page.height(), h.text()[:120]])
        if b == "composition":
            out["hint_shot"] = shot("band_hint_log")
        QApplication.sendEvent(card, QEnterEvent(QPointF(-5, -5), QPointF(-5, -5), QPointF(-5, -5)))
        page.hide_hint()
    out["hints"] = hints
    ex.w.b_log.setChecked(False)
    return out


# 12. New project: the unknown-target message
def s_newproj():
    out = {}
    ex.open()

    def inside(d):
        try:
            d.target.setText("Truc inconnu 42")
            d._check_target()
            ex.settle(3)
            labs = [x for x in d.findChildren(QLabel) if x.isVisible() and x.text()]
            out["labels"] = [[plain(x.text())[:120], x.height(), x.heightForWidth(x.width()) if x.wordWrap() else
                              x.sizeHint().height(), x.width(), x.sizeHint().width()] for x in labs]
            out["size"] = [d.width(), d.height()]
            out["shot"] = shot("newproj_unknown", d)
        finally:
            d.reject()
    ex.dialog_handler = inside
    ex.w.home.new_project()
    ex.settle(6)
    ex.dialog_handler = None
    return out


# 18. Misleading messages: the status line, the reasons in the menus of Denoise and Output
def s_status():
    out = {}
    for kind in ("two_sessions", "session_empty_wf", "legacy"):
        ex.open(ex.project(kind))
        ex.settle(6)
        out[kind] = {"texts": [t for t, w in texts() if w in ("label", "tip")][:80],
                     "run_tip": ex.w.bt_run.toolTip(), "run_text": ex.w.bt_run.text()}
        if kind == "legacy":
            for key in ("debruitage", "post", "astrometrie", "empilement"):
                it = ex.block(key)
                if it is not None:
                    out[kind][f"menu_{key}"] = ex.right_click_block(it)
        out[kind]["shot"] = shot(f"status_{kind}")
    return out


# 15. External names: the Stacking panel, About, Settings (spec mode)
def s_external():
    out = {}
    ex.open(ex.project("mono"))
    ex.settle(6)
    it = ex.block("empilement")
    if it is not None:
        ex.click_block(it)
        out["stack_panel"] = texts(ex.page.panel)
    it = ex.block("astrometrie")
    if it is None:
        sel("empilement")
        place("astrometry")
        it = ex.block("astrometrie")
    if it is not None:
        ex.click_block(it)
        out["astrometry_panel"] = texts(ex.page.panel)

    def grab(name):
        def h(d):
            try:
                ex.settle(3)
                if hasattr(d, "nav"):
                    for i in range(d.nav.count()):
                        d.nav.setCurrentRow(i)
                        ex.settle(2)
                        out[f"{name}/{i}"] = texts(d)
                else:
                    out[name] = texts(d)
            finally:
                d.reject()
        return h
    mb = ex.w.menuBar()

    def trig(path):
        acts = mb.actions()
        a = None
        for t in path:
            a = next((x for x in acts if x.text().replace("&", "") == t.replace("&", "")), None)
            if a is None:
                raise LookupError(f"{t!r} not in {[x.text() for x in acts]}")
            acts = a.menu().actions() if a.menu() is not None else []
        a.trigger()
    for name, path in (("about", [T("Help"), T("About")]), ("settings", [T("Edit"), T("Settings…")])):
        ex.dialog_handler = grab(name)
        try:
            trig(path)
        except LookupError as e:
            out[name + "_err"] = str(e)
        ex.settle(6)
    ex.w.set_spec(True)
    ex.settle(3)
    ex.dialog_handler = grab("settings_spec")
    try:
        trig([T("Edit"), T("Settings…")])
    except LookupError as e:
        out["spec_err"] = str(e)
    ex.settle(6)
    ex.w.set_spec(False)
    ex.dialog_handler = None
    return out


section("width", s_width)
section("log", s_log)
section("names", s_names)
section("gallery", s_gallery)
section("places", s_places)
section("runhere", s_runhere)
section("scale", s_scale)
section("amp", s_amp)
section("band", s_band)
section("newproj", s_newproj)
section("status", s_status)
section("external", s_external)
RAW.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
ex.close()
print("DONE", RAW, flush=True)
