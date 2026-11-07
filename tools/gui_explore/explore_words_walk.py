"""Words and layout, the walk: every screen of the window in one language at one size, its texts and its layout
problems written to <out>/walk_<lang>_<w>x<h>.json (the analysis, explore_words_report.py, compares the languages and
writes the anomalies).

    python explore_words_walk.py fr 1366 768 [part ...]

A screen: {"texts": [[text, where]], "layout": [problem], "missing": [English texts asked without a French entry],
"shot": png}. Nothing is computed (the harness blocks the runs); the projects are the harness's throw-away ones.
"""
from __future__ import annotations

import faulthandler
import json
import re
import sys
import time
import traceback
from pathlib import Path

faulthandler.dump_traceback_later(int(dict(enumerate(sys.argv)).get(5, 1500)), exit=True)

from harness import TMP, Explorer, stray_qt_errors  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QAction, QEnterEvent, QFont, QFontMetricsF, QTextDocument  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QAbstractScrollArea,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QGraphicsSimpleTextItem,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QStyle,
    QStyleOptionButton,
    QTabBar,
    QTableWidget,
    QTextEdit,
    QToolButton,
    QWidget,
)

from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.client.widgets import ElidedLabel  # noqa: E402

LANG = sys.argv[1] if len(sys.argv) > 1 else "fr"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 1920
H = int(sys.argv[3]) if len(sys.argv) > 3 else 1080
PARTS = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] != "all" else ["all"]
TAG = f"{LANG}_{W}x{H}"

stray_qt_errors()
ex = Explorer("words_layout", lang=LANG, size=(W, H))
SHOTS = ex.out / "shots" / TAG
SHOTS.mkdir(parents=True, exist_ok=True)
RESULT = ex.out / f"walk_{TAG}.json"
DATA: dict[str, dict] = {}
if RESULT.exists() and PARTS != ["all"]:
    DATA = json.loads(RESULT.read_text(encoding="utf-8"))
TOASTS: list[str] = []
_seen_missing: set[str] = set()


def part(name: str) -> bool:
    return "all" in PARTS or name in PARTS


# ------------------------------------------------------------------ what the app says on its own (toasts, flashes)
from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import graph as graphmod  # noqa: E402

_show_toast = graphmod.GraphView.show_toast


def _rec_toast(self, text):
    TOASTS.append(str(text))
    return _show_toast(self, text)


graphmod.GraphView.show_toast = _rec_toast
_flash = appmod.Main.flash


def _rec_flash(self, text):
    TOASTS.append("[status] " + str(text))
    return _flash(self, text)


appmod.Main.flash = _rec_flash


# ------------------------------------------------------------------ texts
def plain(t: str) -> str:
    if not t:
        return ""
    if re.search(r"<[a-zA-Z/!]", t) or "&nbsp;" in t or "&lt;" in t:
        d = QTextDocument()
        d.setHtml(t)
        t = d.toPlainText()
    return re.sub(r"&(?=[^&\s])", "", t).replace("&&", "&").strip()


def desc(x) -> str:
    n = x.objectName()
    return type(x).__name__ + (f"#{n}" if n else "")


def widget_texts(root: QWidget) -> list[list[str]]:
    out = []

    def add(t, where):
        t = plain(t) if isinstance(t, str) else ""
        if t and not re.fullmatch(r"[\s\W\d]*", t):
            out.append([t, where])
    ws = [root] + root.findChildren(QWidget)
    if root.isWindow():
        add(root.windowTitle(), "window title")
    for x in ws:
        if not x.isVisible():
            continue
        d = desc(x)
        if isinstance(x, ElidedLabel):
            add(x.full_text(), d)
        elif isinstance(x, QLabel):
            add(x.text(), d)
        elif isinstance(x, QAbstractButton):
            add(x.text(), d)
        elif isinstance(x, QLineEdit):
            add(x.placeholderText(), d + " placeholder")
            if x.isReadOnly():
                add(x.text(), d)
        elif isinstance(x, QComboBox):
            for i in range(x.count()):
                add(x.itemText(i), d + " item")
                tip = x.itemData(i, Qt.ToolTipRole)
                if isinstance(tip, str):
                    add(tip, d + " item tip")
        elif isinstance(x, QTabBar):
            for i in range(x.count()):
                add(x.tabText(i), d + " tab")
                add(x.tabToolTip(i), d + " tab tip")
        elif isinstance(x, QTableWidget):
            for c in range(x.columnCount()):
                it = x.horizontalHeaderItem(c)
                if it is not None:
                    add(it.text(), d + " header")
            for r in range(min(x.rowCount(), 60)):
                for c in range(x.columnCount()):
                    it = x.item(r, c)
                    if it is not None:
                        add(it.text(), d + " cell")
                        add(it.toolTip(), d + " cell tip")
        elif isinstance(x, QListWidget):
            for i in range(x.count()):
                add(x.item(i).text(), d + " item")
                add(x.item(i).toolTip(), d + " item tip")
        elif isinstance(x, QGroupBox):
            add(x.title(), d)
        elif isinstance(x, (QPlainTextEdit, QTextEdit)):
            add(x.placeholderText(), d + " placeholder")
            if x.isReadOnly():
                add(x.toPlainText()[:4000], d)
        add(x.toolTip(), d + " tooltip")
        if isinstance(x, QWidget) and x.isWindow() and x is not root:
            add(x.windowTitle(), "window title")
    return out


def scene_texts() -> list[list[str]]:
    out = []
    page = ex.w.graph if ex.w is not None else None
    if page is None:
        return out
    for it in page.graph.scene().items():
        kind = type(it).__name__
        for attr in ("title", "summary", "text"):
            v = getattr(it, attr, None)
            if callable(v):
                try:
                    v = v()
                except TypeError:
                    v = None
            if isinstance(v, str) and v.strip():
                out.append([v, f"graph {kind}.{attr}"])
        if it.toolTip():
            out.append([it.toolTip(), f"graph {kind} tooltip"])
    return out


def menu_texts(items, where) -> list[list[str]]:
    out = []
    for a in items:
        if a.get("text") and a["text"] != "---":
            out.append([plain(a["text"].split("\t")[0]), where + (" (greyed)" if not a.get("enabled", True) else "")])
        if a.get("tip"):
            out.append([a["tip"], where + " tip"])
        if a.get("sub"):
            out += menu_texts(a["sub"], where + " > " + a["text"])
    return out


# ------------------------------------------------------------------ layout
def _text_width(fm, t):
    return max((fm.horizontalAdvance(line) for line in t.split("\n")), default=0)


def layout_issues(root: QWidget) -> list[str]:
    out = []
    top = root.window()
    ws = [root] + root.findChildren(QWidget)
    for x in ws:
        if not x.isVisible() or x.width() <= 0:
            continue
        d = desc(x)
        txt = ""
        try:
            if isinstance(x, ElidedLabel):
                txt = x.full_text()
            elif isinstance(x, QLabel):
                txt = x.text()
                pm = x.pixmap()
                if (pm is not None and not pm.isNull()) or not plain(txt):
                    txt = ""
                elif x.wordWrap():
                    cr = x.contentsRect()
                    need = x.heightForWidth(x.width())
                    if need > x.height() + 3:
                        out.append(f"cut (bottom): {d} «{plain(txt)[:90]}» needs {need} px high, has {x.height()}")
                else:
                    need = x.sizeHint().width()
                    if need > x.width() + 2 and not x.toolTip():
                        out.append(f"truncated: {d} «{plain(txt)[:90]}» needs {need} px, has {x.width()}")
            elif isinstance(x, QPushButton) or isinstance(x, (QCheckBox, QRadioButton)):
                txt = x.text()
                if txt:
                    opt = QStyleOptionButton()
                    x.initStyleOption(opt)
                    se = QStyle.SE_PushButtonContents if isinstance(x, QPushButton) else (
                        QStyle.SE_CheckBoxContents if isinstance(x, QCheckBox) else QStyle.SE_RadioButtonContents)
                    have = x.style().subElementRect(se, opt, x).width()
                    need = _text_width(x.fontMetrics(), plain(txt)) + (x.iconSize().width() + 4 if not x.icon().isNull() else 0)
                    if need > have + 4:
                        out.append(f"truncated: {d} «{plain(txt)[:90]}» text {need} px, room {have} px")
            elif isinstance(x, QComboBox) and x.count():
                txt = x.currentText()
                need = x.fontMetrics().horizontalAdvance(txt) + 28
                if need > x.width() + 2:
                    out.append(f"truncated: {d} «{txt[:90]}» needs ≈{need} px, has {x.width()}")
            elif isinstance(x, QTabBar) and x.count():
                tot = sum(x.tabSizeHint(i).width() for i in range(x.count()))
                if tot > x.width() + 2:
                    out.append(f"tabs do not fit (scroll arrows or cut): {d} needs {tot} px, has {x.width()}")
                txt = x.tabText(0)
        except RuntimeError:
            continue
        if not txt:
            continue
        # clipped by an ancestor (not a scrolling viewport) or outside the window
        r = QRect(QPoint(0, 0), x.size())
        p = x
        while p is not None and not p.isWindow():
            par = p.parentWidget()
            if par is None:
                break
            r.translate(p.pos())
            if isinstance(par.parentWidget(), QAbstractScrollArea) and par is par.parentWidget().viewport():
                break
            if not QRect(QPoint(0, 0), par.size()).adjusted(-2, -2, 2, 2).contains(r):
                lost = r.intersected(QRect(QPoint(0, 0), par.size()))
                out.append(f"clipped by {desc(par)} ({par.width()}×{par.height()}): {d} «{plain(txt)[:70]}» at "
                           f"{r.x()},{r.y()} {r.width()}×{r.height()}" + (" — invisible" if lost.isEmpty() else ""))
                break
            p = par
    # text widgets on top of each other (siblings)
    by_parent: dict[int, list] = {}
    for x in ws:
        if x.isVisible() and not x.isWindow() and isinstance(x, (QLabel, QAbstractButton, QComboBox, QLineEdit)) and \
                (not isinstance(x, QLabel) or plain(x.text())):
            by_parent.setdefault(id(x.parentWidget()), []).append(x)
    for sibs in by_parent.values():
        for i, a in enumerate(sibs):
            for b in sibs[i + 1:]:
                ir = a.geometry().intersected(b.geometry())
                if ir.width() > 3 and ir.height() > 3:
                    ta = plain(a.text()) if hasattr(a, "text") and isinstance(a.text(), str) else desc(a)
                    tb = plain(b.text()) if hasattr(b, "text") and isinstance(b.text(), str) else desc(b)
                    out.append(f"overlap: {desc(a)} «{ta[:40]}» and {desc(b)} «{tb[:40]}» ({ir.width()}×{ir.height()})")
    return out


def graph_issues() -> list[str]:
    """The graph's drawn items: titles and summaries cut by their block, blocks on blocks, ghost or band texts wider
    than their box, labels on blocks, the floating buttons on blocks, the scale."""
    from astro_gpu.client.graph import BH, BandItem, BlockItem, GhostItem
    out = []
    page = ex.w.graph
    if page is None:
        return out
    view = page.graph
    items = view.scene().items()
    blocks = [it for it in items if isinstance(it, BlockItem)]
    f = QFont("Segoe UI", 10)
    f.setBold(True)
    fm = QFontMetricsF(f)
    f2 = QFont("Segoe UI", 8.5)
    fm2 = QFontMetricsF(f2)
    for it in blocks:
        title = ("⊘ " if it.node.grey else "") + it.title
        tw = it.w - 18
        if fm.horizontalAdvance(title) > tw:
            out.append(f"block title cut: «{title}» ({fm.horizontalAdvance(title):.0f} px in {tw:.0f})")
        if it.summary and fm2.horizontalAdvance(it.summary) > it.w - 15:
            out.append(f"block summary cut: «{it.title}»: «{it.summary}» ({fm2.horizontalAdvance(it.summary):.0f} px in "
                       f"{it.w - 15:.0f})")
    rects = [(it, QRectF(it.pos(), it.pos() + QPointF(it.w, BH))) for it in blocks]
    for i, (a, ra) in enumerate(rects):
        for b, rb in rects[i + 1:]:
            ir = ra.intersected(rb)
            if ir.width() > 2 and ir.height() > 2:
                out.append(f"blocks overlap: «{a.title}» and «{b.title}» ({ir.width():.0f}×{ir.height():.0f})")
    fg = QFontMetricsF(QFont("Segoe UI", 9))
    fb = QFont("Segoe UI", 9)
    fb.setBold(True)
    fbm = QFontMetricsF(fb)
    for it in items:
        if isinstance(it, GhostItem) and it.text and fg.horizontalAdvance(it.text) > it.rect.width() - 4:
            out.append(f"place text wider than its box: «{it.text}» ({fg.horizontalAdvance(it.text):.0f} px in "
                       f"{it.rect.width():.0f})")
        if isinstance(it, GhostItem):
            r = it.rect
            for b, rb in rects:
                ir = r.intersected(rb)
                if ir.width() > 4 and ir.height() > 4 and not getattr(it, "branch_key", None):
                    out.append(f"place «{it.text}» on block «{b.title}»")
        if isinstance(it, GhostItem):
            for o in items:
                if o is not it and isinstance(o, GhostItem) and id(o) > id(it):
                    ir = it.rect.intersected(o.rect)
                    if ir.width() > 4 and ir.height() > 4:
                        out.append(f"places on each other: «{it.text}» and «{o.text}» ({ir.width():.0f}×{ir.height():.0f})")
        if isinstance(it, BandItem) and fbm.horizontalAdvance(it.text.upper()) > it.rect.width() - 20:
            out.append(f"band title wider than its band: «{it.text.upper()}» ({fbm.horizontalAdvance(it.text.upper()):.0f}"
                       f" px in {it.rect.width() - 20:.0f})")
        if isinstance(it, QGraphicsSimpleTextItem):
            r = it.sceneBoundingRect()
            for b, rb in rects:
                if r.intersects(rb):
                    out.append(f"wire label «{it.text()}» over block «{b.title}»")
    # the floating "Show all" button and the toast over blocks
    for w_, name in ((view.b_all, "Show all button"),):
        if w_.isVisible():
            g = QRectF(w_.geometry())
            for b, rb in rects:
                vr = QRectF(view.mapFromScene(rb).boundingRect())
                if vr.intersects(g):
                    out.append(f"{name} «{w_.text()}» over block «{b.title}»")
    k = view.transform().m11()
    if k < 0.75:
        out.append(f"graph drawn at {k:.2f} of its size (block titles ≈ {10 * k:.1f} pt)")
    hs = view.horizontalScrollBar()
    if hs.isVisible() and hs.maximum() > 0:
        out.append(f"graph scrolls horizontally ({hs.maximum()} px hidden) at scale {k:.2f}")
    return out


# ------------------------------------------------------------------ screens
def new_missing() -> list[str]:
    m = sorted(i18n.missing() - _seen_missing)
    _seen_missing.update(m)
    return m


def screen(name: str, root: QWidget | None = None, extra: list | None = None, graph: bool = False,
           shot_widget: QWidget | None = None, layout_root: QWidget | None = None):
    ex.settle(3)
    root = root or ex.w
    rec = {"texts": [], "layout": [], "missing": [], "shot": "", "toasts": []}
    try:
        rec["texts"] = widget_texts(root) + (extra or [])
        if graph:
            rec["texts"] += scene_texts()
        rec["layout"] = layout_issues(layout_root or root) + (graph_issues() if graph else [])
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    rec["missing"] = new_missing()
    rec["toasts"] = list(TOASTS)
    TOASTS.clear()
    rec["popups"] = [list(p) for p in ex.popups]
    ex.popups.clear()
    f = SHOTS / (re.sub(r"[^\w.-]+", "_", name)[:90] + ".png")
    try:
        (shot_widget or root.window()).grab().save(str(f))
        rec["shot"] = str(f)
    except RuntimeError:
        pass
    DATA[name] = rec
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {len(rec['texts'])} texts, {len(rec['layout'])} layout, "
          f"{len(rec['missing'])} missing" + (" ERROR" if "error" in rec else ""), flush=True)
    return rec


def save():
    RESULT.write_text(json.dumps(DATA, ensure_ascii=False, indent=1), encoding="utf-8")


def step(title, fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:  # noqa: BLE001
        from harness import RunsBlocked
        if isinstance(e, RunsBlocked):
            return None
        DATA.setdefault("_errors", {"texts": [], "layout": [], "missing": [], "errors": []})["errors"].append(
            f"{title}: {type(e).__name__}: {e}\n{traceback.format_exc()[-1800:]}")
        print("ERROR", title, type(e).__name__, e, flush=True)
        return None


def node_id(it) -> str:
    n = it.node
    return f"{n.key}" + (f"@{n.sid}" if n.sid is not None else "") + f"[{','.join(str(q.pid) for q in n.paths)}]"


def find_block(nid: str):
    for it in ex.blocks():
        if node_id(it) == nid:
            return it
    return None


def walk_graph(label: str, inspect: bool = True, hints: bool = True, menus: bool = True):
    """The graph page of the open project: the page, each block's panel and menu, the palette's greyed reasons, each
    block's Inspect."""
    ex.w.set_mode(0)
    ex.settle(4)
    screen(f"graph/{label}", graph=True)
    if hints:
        hint_walk(label)
    ids = [node_id(it) for it in ex.blocks()]
    for nid in ids:
        it = find_block(nid)
        if it is None:
            continue
        step(f"click {nid}", ex.click_block, it)
        screen(f"panel/{label}/{nid}", root=ex.page.panel, layout_root=ex.page.panel, shot_widget=ex.w)
        if menus:
            it = find_block(nid)
            if it is not None:
                items = step(f"menu {nid}", ex.right_click_block, it) or []
                DATA[f"menu/{label}/{nid}"] = {"texts": menu_texts(items, f"menu of {nid}"), "layout": [],
                                               "missing": new_missing(), "shot": "", "items": items}
    if inspect:
        for nid in ids:
            it = find_block(nid)
            if it is None or it.node.key in ("final",):
                continue
            step(f"inspect {nid}", ex.double_click_block, it)
            ex.settle(6)
            if ex.w.mode != 0:
                screen(f"inspect/{label}/{nid}")
            step("back to graph", ex.w.set_mode, 0)
            ex.settle(3)
    save()


def hint_walk(label: str):
    """Each greyed palette card hovered: the message that says why, where it shows."""
    page = ex.page
    texts, lay = [], []
    for b, card in list(page.cards.items()):
        if not card.reason:
            texts.append([card.title, "palette card"])
            if card.toolTip():
                texts.append([card.toolTip(), "palette card tooltip"])
            continue
        QApplication.sendEvent(card, QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(card.mapToGlobal(QPoint(5, 5)))))
        ex.settle(2)
        h = page.hint
        texts.append([card.title, "palette card (greyed)"])
        texts.append([h.text(), f"palette hint of {b}"])
        if h.isVisible():
            g = h.geometry()
            if g.bottom() > page.height() or g.right() > page.width() or g.top() < 0:
                lay.append(f"palette hint of «{card.title}» goes out of the page: {g.x()},{g.y()} {g.width()}×{g.height()}"
                           f" in {page.width()}×{page.height()}")
            if h.heightForWidth(h.width()) > h.height() + 3:
                lay.append(f"palette hint of «{card.title}» cut at the bottom")
            # the card itself visible (the palette scrolls in a short window)
            vp = page.palette_bar.viewport()
            cr = QRect(card.mapTo(vp, QPoint(0, 0)), card.size())
            if not vp.rect().contains(cr):
                lay.append(f"palette card «{card.title}» not fully visible without scrolling ({cr.y()}..{cr.bottom()} in "
                           f"{vp.height()})")
        else:
            lay.append(f"palette hint of «{card.title}» not shown on hover")
        if b in ("composition", "fusion"):
            screen(f"hint/{label}/{b}", graph=False, extra=[[h.text(), "hint"]])
        QApplication.sendEvent(card, QEnterEvent(QPointF(-5, -5), QPointF(-5, -5), QPointF(-5, -5)))
        page.hide_hint()
    sb = page.palette_bar.verticalScrollBar()
    if sb.maximum() > 0 and not sb.isVisible():
        lay.append("the palette is taller than its window but shows no scroll bar")
    DATA[f"palette/{label}"] = {"texts": texts, "layout": lay, "missing": new_missing(), "shot": "",
                                "scroll": [sb.maximum(), sb.isVisible()]}


def keep_open(d):
    """dialog_handler: the dialog stays open (the walk reads it, then closes it)."""
    return None


def dialog_screen(name: str, opener, inside=None, close=True):
    """A modal dialog: opened by ``opener``; ``inside(d)`` runs while it is open (default: one screen); closed."""
    got = {}

    def h(d):
        try:
            got["d"] = d
            ex.settle(3)
            if inside is not None:
                inside(d)
            else:
                screen(name, root=d, shot_widget=d)
                size_check(name, d)
        except Exception as e:  # noqa: BLE001
            DATA.setdefault("_errors", {"texts": [], "layout": [], "missing": [], "errors": []})["errors"].append(
                f"{name}: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
        finally:
            if close:
                try:
                    d.reject()
                except RuntimeError:
                    pass
    ex.dialog_handler = h
    step(f"open {name}", opener)
    ex.settle(6)
    ex.dialog_handler = None
    return got.get("d")


def size_check(name, d):
    """A dialog larger than the screen's usable room (the screen less the taskbar and the title bar)."""
    room_w, room_h = W, H - 40 - 31
    s = d.size()
    mh = d.minimumSizeHint()
    lay = DATA[name]["layout"] if name in DATA else []
    if s.width() > room_w or s.height() > room_h:
        lay.append(f"dialog {s.width()}×{s.height()} larger than the room of a {W}×{H} screen ({room_w}×{room_h})")
    if mh.width() > room_w or mh.height() > room_h:
        lay.append(f"dialog's minimum {mh.width()}×{mh.height()} larger than the room of a {W}×{H} screen")
    if name in DATA:
        DATA[name]["layout"] = lay
        DATA[name]["size"] = [s.width(), s.height(), mh.width(), mh.height()]


def menubar_screen(name):
    mb = ex.w.menuBar()
    texts, lay = [], []

    def walk(m: QMenu, where):
        for a in m.actions():
            if a.isSeparator():
                continue
            t = plain(a.text())
            texts.append([t, where + (" (greyed)" if not a.isEnabled() else "")])
            if a.toolTip() and a.toolTip() != a.text().replace("&", ""):
                texts.append([a.toolTip(), where + " tip"])
            if a.menu() is not None:
                walk(a.menu(), where + " > " + t)
    for a in mb.actions():
        texts.append([plain(a.text()), "menu bar"])
        if a.menu() is not None:
            walk(a.menu(), plain(a.text()))
    DATA[name] = {"texts": texts, "layout": lay, "missing": new_missing(), "shot": ""}


def trigger(menu_path: list[str]):
    """A menu bar action by its texts (translated)."""
    mb = ex.w.menuBar()
    acts = mb.actions()
    a = None
    for t in menu_path:
        a = next((x for x in acts if plain(x.text()) == t), None)
        if a is None:
            raise LookupError(f"no menu item {t!r} in {[plain(x.text()) for x in acts]}")
        acts = a.menu().actions() if a.menu() is not None else []
    a.trigger()


T = i18n._


# ================================================================== the walk
t0 = time.time()
w = ex.open()
ex.settle(10)
sz = w.size()
DATA["_window"] = {"texts": [], "layout": [] if (sz.width(), sz.height()) == (W, H) else
                   [f"the window asked {W}×{H} is {sz.width()}×{sz.height()}"], "missing": [], "shot": "",
                   "min": [w.minimumSizeHint().width(), w.minimumSizeHint().height()]}

if part("home"):
    # Home, empty bench folder, then with projects (the cards)
    screen("home/empty")
    menubar_screen("menubar/home")
    for k in ("session", "image", "mono"):
        ex.project(k, f"Projet {k} au nom un peu long pour la carte")
    step("rescan", ex.w.home.rescan)
    ex.settle(15)
    screen("home/cards")
    # New project dialog: empty, then a name that exists, a target found and one not found
    dialog_screen("dialog/new_project", ex.w.home.new_project)

    def np_inside(d):
        d.name.setText("Projet session au nom un peu long pour la carte".replace("session", "session"))
        d.name.setText("Projet session au nom un peu long pour la carte")
        d.target.setText("M 31")
        d._check_target()
        ex.settle(2)
        screen("dialog/new_project_filled", root=d, shot_widget=d)
        d.target.setText("Truc inconnu 42")
        d._check_target()
        ex.settle(2)
        screen("dialog/new_project_unknown", root=d, shot_widget=d)
    dialog_screen("dialog/new_project_filled", ex.w.home.new_project, np_inside)
    save()

if part("gallery"):
    # the start: a new project opens the template window; every tab, every card
    def gal_inside(d):
        for i in range(d.tabs.count()):
            d.tabs.setCurrentIndex(i)
            ex.settle(2)
            tab = d.tab_keys[i]
            for j, c in enumerate(list(d.cards)):
                d.pick(c.t)
                ex.settle(1)
                screen(f"gallery/{tab}/{j}_{c.t.get('key') or c.t.get('name')}", root=d, shot_widget=d)
            if not d.cards:
                screen(f"gallery/{tab}/empty", root=d, shot_widget=d)
        size_check(next(k for k in reversed(list(DATA)) if k.startswith("gallery/")), d)
        DATA["gallery/_size"] = {"texts": [], "layout": [], "missing": [], "shot": "",
                                 "size": [d.width(), d.height(), d.minimumSizeHint().width(),
                                          d.minimumSizeHint().height()]}
        size_check("gallery/_size", d)
    ex.dialog_handler = lambda d: (gal_inside(d), d.reject()) if d.__class__.__name__ == "TemplateGallery" else d.reject()
    step("create project", ex.w.create_project, str(ex.ws / "Galerie"), "Galerie", "M 42")
    ex.settle(15)
    ex.dialog_handler = None
    screen("graph/new_blank_after_gallery_cancel", graph=True)
    save()

KINDS = ["blank", "session", "session_empty_wf", "image", "session+image", "two_sessions", "two_flows", "mono",
         "legacy", "jpg_mistake"]
if part("kinds"):
    for k in KINDS:
        root = ex.project(k)
        step(f"open {k}", ex.open, root)
        ex.settle(10)
        walk_graph(k, inspect=k in ("session", "image", "mono", "legacy", "jpg_mistake"))
        if k == "two_flows":
            # the gallery on a project with workflows: the Copy tab
            def gal2(d):
                for i in range(d.tabs.count()):
                    d.tabs.setCurrentIndex(i)
                    ex.settle(2)
                    screen(f"gallery_copy/{d.tab_keys[i]}", root=d, shot_widget=d)
            dialog_screen("gallery_copy", ex.page.new_workflow, gal2)
    save()

if part("rich"):
    # a full chain built from the palette on a session project, with repeated blocks, a split, a combine, a branch,
    # a second flow, switches, pauses, sends to Post
    root = ex.project("session", "Chaine complete")
    ex.open(root)
    ex.settle(8)
    it = ex.block("session") or (ex.blocks()[0] if ex.blocks() else None)
    if it is not None:
        ex.click_block(it)
    for b in ("calibration", "registration", "selection", "stacking", "astrometry", "gradient", "color", "denoise",
              "stars", "gradient", "denoise"):
        step(f"place {b}", ex.double_click_card, b)
        ex.settle(4)
        DATA.setdefault("_placed", {"texts": [], "layout": [], "missing": [], "shot": "", "log": []})["log"].append(
            [b, [node_id(x) for x in ex.blocks()], list(TOASTS)])
    screen("graph/rich_chain", graph=True)
    step("combine", ex.double_click_card, "composition")
    ex.settle(4)
    screen("graph/rich_chain_combine", graph=True)
    # a branch: a Gradient dropped under the Stacking... wherever the graph offers it
    c = ex.drag_palette_hover("denoise", ex.view.viewport().rect().center())
    branch = [g for g in ex.ghosts() if getattr(g, "branch_key", None)]
    pt = ex.ghost_center(branch[len(branch) // 2]) if branch else None
    screen("graph/rich_drag_denoise_places", graph=True)
    ex.drag_leave()
    if pt is not None:
        step("branch", ex.drag_palette, "denoise", pt)
        ex.settle(6)
    screen("graph/rich_branch", graph=True)
    # a second flow: a Session on the bottom zone
    z = ex.new_flow_zone()
    if z is not None:
        step("second flow", ex.drag_palette, "session", ex.ghost_center(z))
        ex.settle(6)
    screen("graph/rich_two_flows", graph=True)
    # actions from the menus: switch off, pause, send to Post, a refused delete, rename, template, duplicate
    for nid_key, pick in (("fond", "Switch off" if LANG == "en" else T("Switch off")),
                          ("debruitage", T("Review pause")),
                          ("empilement", T("To Post")),
                          ("astrometrie", T("Delete")),
                          ("calibration", T("Delete"))):
        it = next((x for x in ex.blocks() if x.node.key == nid_key or x.node.key.startswith(nid_key)), None)
        if it is None:
            continue
        items = step(f"menu {nid_key}", ex.right_click_block, it, pick) or []
        ex.settle(6)
        screen(f"action/{nid_key}_{re.sub(r'\\W+', '_', pick)[:20]}", graph=True,
               extra=menu_texts(items, f"menu of {nid_key}"))
    it = next((x for x in ex.blocks() if x.node.key == "post"), None)
    if it is not None:
        dialog_screen("dialog/rename_workflow", lambda: ex.right_click_block(it, T("Rename")))
        it = next((x for x in ex.blocks() if x.node.key == "post"), None)
        dialog_screen("dialog/save_as_template", lambda: ex.right_click_block(it, T("Save as template…")))
        it = next((x for x in ex.blocks() if x.node.key == "post"), None)
        step("duplicate", ex.right_click_block, it, T("Duplicate"))
        ex.settle(6)
        screen("action/duplicate", graph=True)
    s0 = next((x for x in ex.blocks() if x.node.key == "session"), None)
    if s0 is not None:
        dialog_screen("dialog/rename_session", lambda: ex.right_click_block(s0, T("Rename session")))
    walk_graph("rich_final", inspect=True, hints=True)
    # the Log
    ex.w.b_log.setChecked(True)
    ex.settle(4)
    screen("log/rich", root=ex.w.logp, shot_widget=ex.w)
    # Ctrl+Z a few times: the toasts and the status line
    for i in range(3):
        ex.undo()
    screen("action/undo3", graph=True)
    for i in range(3):
        ex.redo()
    screen("action/redo3", graph=True)
    # the plan window, Run to here
    d = step("plan", ex.w.run.launch)
    ex.settle(8)
    if d is not None:
        screen("run/plan_all", root=d, shot_widget=d)
        size_check("run/plan_all", d)
        d.hide()
    sel = next((x for x in ex.blocks() if x.node.key == "fond" or x.node.key.startswith("fond")), None)
    if sel is not None:
        ex.click_block(sel)
        d = step("run to here", ex.w.run.run_to_here)
        ex.settle(8)
        if d is not None:
            screen("run/plan_to_here", root=d, shot_widget=d)
            d.hide()
    # Run button: start (blocked: a start that failed)
    step("run button", ex.click, ex.w.bt_run)
    ex.settle(8)
    for dd in ex.dialogs():
        if dd.__class__.__name__ == "RunDialog":
            screen("run/after_run_button", root=dd, shot_widget=dd)
            dd.hide()
    screen("status/after_run_button")
    # Post
    step("post", ex.w.set_mode, 2)
    ex.settle(15)
    screen("post/rich")
    menubar_screen("menubar/project")
    save()

if part("fusion"):
    root = ex.project("mono", "Mono fusion")
    ex.open(root)
    ex.settle(8)
    step("fusion", ex.double_click_card, "fusion")
    ex.settle(6)
    walk_graph("mono_fusion", inspect=True)
    # an image project, its split, its sends
    root = ex.project("image", "Image etoiles")
    ex.open(root)
    ex.settle(8)
    it = ex.block("image")
    if it is not None:
        ex.click_block(it)
    for b in ("gradient", "stars"):
        step(f"place {b}", ex.double_click_card, b)
        ex.settle(4)
    it = next((x for x in ex.blocks() if x.node.key == "etoiles"), None)
    if it is not None:
        items = ex.right_click_block(it, T("Starless to Post"))
        ex.settle(5)
        screen("action/starless_to_post", graph=True, extra=menu_texts(items, "menu of split"))
        it = next((x for x in ex.blocks() if x.node.key == "etoiles"), None)
        items = ex.right_click_block(it, T("+ Recombine"))
        ex.settle(5)
        screen("action/new_combine", graph=True, extra=menu_texts(items, "menu of split"))
    walk_graph("image_split", inspect=True)
    ex.w.b_log.setChecked(True)
    ex.settle(3)
    screen("log/image_split", root=ex.w.logp, shot_widget=ex.w)
    save()

if part("settings"):
    def settings_inside(d):
        for i in range(d.nav.count()):
            d.nav.setCurrentRow(i)
            ex.settle(2)
            screen(f"settings/{i}_{d.keys[i]}", root=d, shot_widget=d)
            size_check(f"settings/{i}_{d.keys[i]}", d)
    dialog_screen("settings", lambda: trigger([T("Edit"), T("Settings…")]), settings_inside)
    dialog_screen("dialog/about", lambda: trigger([T("Help"), T("About")]))
    # Spec mode on: the ◇ items
    step("spec on", ex.w.set_spec, True)
    ex.settle(4)
    menubar_screen("menubar/spec")
    if ex.w.graph is not None:
        screen("graph/spec_mode", graph=True)
    dialog_screen("settings_spec", lambda: trigger([T("Edit"), T("Settings…")]), lambda d: [
        (d.nav.setCurrentRow(i), ex.settle(2), screen(f"settings_spec/{i}_{d.keys[i]}", root=d, shot_widget=d))
        for i in range(d.nav.count())])
    step("spec off", ex.w.set_spec, False)
    save()

if part("switch") and LANG == "en":
    # the language changed at runtime: what stays in English once French is chosen
    root = ex.project("session", "Bascule")
    ex.open(root)
    ex.settle(6)
    step("lang fr", ex.w.change_language, "fr")
    ex.settle(10)
    screen("switch/graph_after_fr", graph=True)
    menubar_screen("switch/menubar_after_fr")
    save()

DATA["_meta"] = {"texts": [], "layout": [], "missing": sorted(i18n.missing()), "shot": "", "lang": LANG, "size": [W, H],
                 "seconds": round(time.time() - t0), "tmp": str(TMP)}
save()
ex.close()
print("DONE", TAG, round(time.time() - t0), "s", flush=True)
