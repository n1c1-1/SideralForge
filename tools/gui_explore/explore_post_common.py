"""Shared helpers of the post_wires exploration (wires to Post, layers, Split stars, Combine, the Post page)."""
from __future__ import annotations

import faulthandler
import os
import re
import sys
import traceback

from harness import Explorer, stray_qt_errors  # noqa: F401  (sets the environment first)
from PySide6.QtCore import QPoint, QPointF, Qt, QEvent
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtWidgets import QApplication, QGraphicsPathItem, QGraphicsSimpleTextItem, QLabel, QAbstractButton, QWidget

from astro_gpu.client.graph import BH, WireItem, BlockItem, GhostItem  # noqa: E402

INTERNAL = re.compile(r"render_simple|star_removal|\bdebruitage\b|\betoiles\b|\bastrometrie\b|"
                      r"[a-z]~\d|\bstarless\b|\bcombined\b|\bStacking\b|\bpid\b|\blid\b|Traceback|Error:|\bNone\b|"
                      r"\{[a-z_]+\}|Combine stars|Split stars|stars alone|Starless|\blayers?\b|\bOutput\b|"
                      r"\bRaw stack\b|\boutput\b|inspect:|file:")
ENGLISH_HINT = re.compile(r"\b(the|layer|layers|to Post|Not computed|Open Post|stars alone|Starless image|Remove|"
                          r"Send|Take|output|workflow's|Combine stars|Split stars|block)\b")


SLOT_ERRORS: list[str] = []


def _hook(t, v, tb):
    SLOT_ERRORS.append("".join(traceback.format_exception(t, v, tb)))
    sys.__excepthook__(t, v, tb)


def start(timeout: int = 900):
    faulthandler.dump_traceback_later(timeout, exit=True)
    stray_qt_errors()
    sys.excepthook = _hook


class P:
    """The exploration helpers over an Explorer."""

    def __init__(self, ex: Explorer):
        self.ex = ex
        self.seen_titles: set[str] = set()

    # ------------------------------------------------------------ reading the graph
    @property
    def s(self):
        return self.ex.s

    def wires(self) -> list[WireItem]:
        return [it for it in self.ex.view.scene().items() if isinstance(it, WireItem)]

    def plain_paths(self) -> list[QGraphicsPathItem]:
        return [it for it in self.ex.view.scene().items() if isinstance(it, QGraphicsPathItem)
                and not isinstance(it, WireItem)]

    def post_arrows(self) -> list:
        """The dashed gold arrows of images sent to Post (``_post_wire``)."""
        return [it for it in self.plain_paths() if it.pen().color().name() == "#c9a227" and it.toolTip()]

    def final(self):
        return self.ex.view.final_item

    def final_summary(self) -> str:
        f = self.final()
        return f.summary if f is not None else "<no Post block>"

    def graph_layers(self):
        g = self.ex.view.graph
        return list(g.post_layers) if g is not None else []

    def saved_layers(self):
        return list(((getattr(self.s.p, "post", None) or {}).get("layers")) or [])

    def blocks_desc(self) -> list[str]:
        out = []
        for it in sorted(self.ex.blocks(), key=lambda b: (b.node.col, b.node.y)):
            n = it.node
            out.append(f"{n.key}[{','.join(str(q.pid) for q in n.paths)}] '{it.title}' / '{it.summary}'"
                       + (" port" if it.ports else ""))
        return out

    def dump(self, tag: str):
        print(f"--- {tag}")
        print("  paths:", [(q.pid, q.name, q.end, [n['block'] for n in (self.s.p.workflow(q.pid).blocks or [])])
                           for q in self.s.paths])
        print("  blocks:", self.blocks_desc())
        print("  wires(removable):", [w.toolTip().split("\n")[0] for w in self.wires()])
        print("  arrows:", [a.toolTip() for a in self.post_arrows()])
        print("  final:", self.final_summary(), "graph_layers:", self.graph_layers())
        print("  saved:", [(x.get("kind"), x.get("pid"), x.get("part"), x.get("name")) for x in self.saved_layers()],
              "edited:", (getattr(self.s.p, "post", None) or {}).get("edited"))
        print("  undo:", self.s.can_undo(), "redo:", self.s.can_redo(), "popups:", self.ex.popups[-3:],
              "modals:", self.ex.modals[-3:])

    def block(self, key, pid=None, sid=None):
        return self.ex.block(key, pid, sid)

    def blocks_of(self, key):
        return [it for it in self.ex.blocks() if it.node.key == key]

    # ------------------------------------------------------------ gestures on wires
    def wire_point(self, w, t: float = 0.5) -> QPoint:
        return self.ex.view.mapFromScene(w.mapToScene(w.path().pointAtPercent(t)))

    def item_at(self, pt: QPoint):
        return self.ex.view.itemAt(pt)

    def free_point_on(self, w) -> QPoint | None:
        """A point of the wire where the wire itself is the top item (not under a block)."""
        for t in (0.5, 0.4, 0.6, 0.3, 0.7, 0.25, 0.75, 0.2, 0.8, 0.15, 0.85):
            pt = self.wire_point(w, t)
            if self.item_at(pt) is w:
                return pt
        return None

    def click_wire(self, w) -> bool:
        pt = self.free_point_on(w)
        if pt is None:
            return False
        vp = self.ex.view.viewport()
        self.ex.hover(vp, pt)
        self.ex.click(vp, pt)
        return True

    def right_click_at(self, pt: QPoint, pick: str | None = None):
        vp = self.ex.view.viewport()
        self.ex.menu_pick = pick
        n = len(self.ex.menus)
        QApplication.sendEvent(vp, QContextMenuEvent(QContextMenuEvent.Mouse, pt, vp.mapToGlobal(pt)))
        self.ex.settle()
        return self.ex.menus[-1] if len(self.ex.menus) > n else []

    def right_click_wire(self, w, pick=None):
        pt = self.free_point_on(w)
        if pt is None:
            return None
        return self.right_click_at(pt, pick)

    def activate(self):
        """Offscreen, the window loses its active state after some gestures: the shortcuts need it (harness artefact)."""
        if QApplication.activeWindow() is not self.ex.w:
            self.ex.w.activateWindow()
            self.ex.settle(2)

    def key(self, k, widget=None, mod=Qt.NoModifier):
        self.activate()
        self.ex.key(k, widget, mod)

    def del_key(self):
        self.key(Qt.Key_Delete)

    def undo(self):
        self.activate()
        self.ex.undo()

    def redo(self):
        self.activate()
        self.ex.redo()

    # ------------------------------------------------------------ ports
    def pull_port(self, item, what=None, to: str = "post"):
        """Pull a block's port (``what``: None, "starless", "stars") onto Post or onto the Composition ghost."""
        ex = self.ex
        vp = ex.view.viewport()
        y = next((py for w, py in item.ports if w == what), BH / 2)
        a = ex.view.mapFromScene(item.mapToScene(QPointF(item.w - 1, y)))
        if to == "post":
            b = ex.view.mapFromScene(ex.view.final_item.sceneBoundingRect().center())
        else:
            b = None
        ex._mouse(vp, QEvent.MouseButtonPress, a)
        ex.settle(1)
        if to != "post":
            g = getattr(ex.view, "_comp_ghost", None)
            if g is None:
                ex._mouse(vp, QEvent.MouseButtonRelease, a, Qt.LeftButton, Qt.NoButton)
                ex.settle()
                return "no ghost"
            b = ex.view.mapFromScene(g.sceneBoundingRect().center())
        for t in (0.3, 0.6, 1.0):
            ex._mouse(vp, QEvent.MouseMove, QPoint(int(a.x() + (b.x() - a.x()) * t), int(a.y() + (b.y() - a.y()) * t)),
                      Qt.NoButton, Qt.LeftButton)
        ex._mouse(vp, QEvent.MouseButtonRelease, b, Qt.LeftButton, Qt.NoButton)
        ex.settle(8)
        return "ok"

    # ------------------------------------------------------------ texts
    def check_texts(self, where: str, texts: list[str], steps=None):
        """Internal words in what is shown, noted once per distinct word."""
        for t in texts:
            for m in INTERNAL.finditer(t):
                key = f"internal:{m.group(0)}:{t[:40]}"
                if key in self.seen_titles:
                    continue
                self.seen_titles.add(key)
                print(f"TEXTHIT [{where}] {m.group(0)!r} in {t[:200]!r}", flush=True)

    def menu_texts(self, items) -> list[str]:
        out = []
        for d in items or []:
            if d.get("text") and d["text"] != "---":
                out.append(d["text"])
                if d.get("tip"):
                    out.append(d["tip"])
            out += self.menu_texts(d.get("sub"))
        return out

    def panel_texts(self) -> list[str]:
        pn = self.ex.page.panel
        out = []
        for x in [pn] + pn.findChildren(QWidget):
            if not x.isVisibleTo(pn):
                continue
            if isinstance(x, (QLabel, QAbstractButton)) and x.text():
                out.append(x.text())
            if x.toolTip():
                out.append(x.toolTip())
        return out

    def panel_buttons(self) -> list[tuple[str, bool, str]]:
        pn = self.ex.page.panel
        return [(b.text(), b.isEnabled(), b.toolTip()) for b in pn.findChildren(QAbstractButton) if b.isVisibleTo(pn)]

    def toast(self) -> str:
        t = self.ex.view.toast
        return t.text() if t.isVisible() else ""

    def once(self, key: str) -> bool:
        if key in self.seen_titles:
            return False
        self.seen_titles.add(key)
        return True

    def note_once(self, key, severity, title, detail="", steps=None, shot=True):
        """A candidate anomaly: printed always; written (with its screenshot) only when EXPLORE_NOTE=1 (after it was
        reproduced on a dry run)."""
        if not self.once("note:" + key):
            return
        print(f"CANDIDATE [{severity}] {key}: {title}\n    {detail[:600]}", flush=True)
        if os.environ.get("EXPLORE_NOTE") == "1":
            self.ex.note(severity, title, detail, shot=shot, steps=steps)

    def take_slot_errors(self) -> list[str]:
        out, SLOT_ERRORS[:] = list(SLOT_ERRORS), []
        return out

    def check_slot_errors(self, where: str, steps=None):
        """Exceptions raised inside Qt slots (printed by PySide, caught by our excepthook): one candidate each."""
        for tb in self.take_slot_errors():
            last = tb.strip().splitlines()[-1]
            self.note_once("slot:" + last[:80], "bloquant", f"Exception Python pendant : {where}",
                           f"{last}\n{tb[-2000:]}", steps=steps)

    def guard(self, title, fn, *a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa: BLE001
            if self.once("exc:" + title + type(e).__name__):
                self.ex.note("bloquant", f"Exception : {title}", f"{type(e).__name__}: {e}\n"
                             f"{traceback.format_exc()[-2500:]}")
            print("EXC", title, e, flush=True)
            return None


def build_chain(ex: Explorer, blocks: list[str]):
    """Blocks placed one by one by double-click on the palette (the current workflow)."""
    for b in blocks:
        ex.double_click_card(b)
        ex.settle(6)


def session_project(ex: Explorer, name=None):
    root = ex.project("session", name)
    ex.open(root)
    ex.settle(10)
    ex.w.activateWindow()                              # the shortcuts (Del, D, Ctrl+Z) need the active window
    ex.settle()
    return root


CHAINS = {
    "stacking": ["calibration", "registration", "stacking"],
    "astrometry": ["calibration", "registration", "stacking", "astrometry"],
    "gradient": ["calibration", "registration", "stacking", "astrometry", "gradient"],
    "denoise": ["calibration", "registration", "stacking", "astrometry", "gradient", "denoise"],
    "color": ["calibration", "registration", "stacking", "astrometry", "gradient", "denoise", "color"],
    "stars": ["calibration", "registration", "stacking", "astrometry", "gradient", "denoise", "color", "stars"],
}
FR = {"stacking": "Empilement", "astrometry": "Astrométrie", "gradient": "Gradient", "denoise": "Débruitage",
      "color": "Couleur", "stars": "Split étoiles et fond"}
VIEWKEY = {"stacking": "empilement", "astrometry": "astrometrie", "gradient": "fond", "denoise": "debruitage",
           "color": "couleur", "stars": "etoiles"}


def steps_chain(end: str) -> list[str]:
    return ["Nouveau projet avec une session de 15 brutes", "Double-clic dans la palette : " +
            ", ".join(FR.get(b, b.capitalize()) for b in CHAINS[end])]
