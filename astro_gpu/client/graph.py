"""The graph of the project: Graph mode's centre, and the page that holds it (``GraphPage``).

Columns follow the processing order; a block sits in its column and the wires draw themselves. A block shared by
several workflows is drawn once, and the branch splits where they start to differ. Optional blocks are added by
dragging a card of the toolbar onto a workflow: while one is dragged, the places where it can go appear. The output of
Stacking has a port that is pulled onto Post.

The Stars block has two ports, its two images: the starless image (top) and the stars alone (bottom). A port pulled
onto Post makes that image a layer of Post; pulled onto the Composition section, a new Stars and background block,
which recombines the two (its settings are those of its layer in Post, also in the panel here). Their wires carry the
image's name; a Stars and background block, or an image sent as it is, is taken out of Post by its menu or its ×.

``GraphPage`` is what the window mounts: the toolbar of optional blocks, the ``GraphView`` and the block ``Panel``, over
the ``ProjectModel``. It makes the edits (add or remove a block, new workflow, rename, duplicate, delete, an output sent
to Post) and tells the window, by signals, what it must do (open Inspect or Post, import frames).

The window (app.py ``graph_page``) builds it again with the project screen and connects its signals: ``openBlock``
and ``openPost`` (Inspect or Post), ``addSession`` (a new session), ``variantMade`` (Inspect opens
the variant compared with its original), ``changed`` (the Run button and the status line).

Issue #43: the sessions calibrated each on its own sit as rows of a Session block and its Calibration block before the
chain; their wires join into one bar before Registration (``_join``). A block the user cut is drawn hatched in grey
with its wire across it; a block greyed by a cut has a dashed amber outline, ⊘ and its reason: the image passes
through both. A block is cut or restored by its menu, its panel's switch or the D key. A workflow from an image starts
with an Image block on its own lane.

Issue #44: zoom and navigation are the Post image's: the wheel at the pointer, + and −, F / Ctrl+0 (Show all), 1 / Ctrl+1
(the readable size, 100 %), the presets' menu, a pan by the middle button, Space + drag or a left drag on the empty area,
a minimap in the corner when part of the graph is out of view. The view, once moved by the user, stays through rebuilds
and with the project (the settings, ``GraphView.GRAPH_VIEW_KEY``); below the readable size the blocks show their titles
only. The right panel can fold to icons or hide (``layout.Dock``), the palette fold or hide.

The scene's items: ``BlockItem`` (a block), ``GhostItem`` (where a dragged tool can go), ``BandItem`` (a phase band);
``ToolCard`` is a card of the toolbar.
"""
from __future__ import annotations

import textwrap
import time
from pathlib import Path

from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QDrag,
    QFont,
    QFontMetrics,
    QFontMetricsF,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyleOptionGraphicsItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core import workflow as wfl
from . import prefs
from .blocks import (
    BREAK_COL,
    BY_KEY,
    FRAME_KEYS,
    IMAGE_KEYS,
    PAUSE_COL,
    REVIEWED_COL,
    STAR_OUTPUTS,
    STATE_WORD,
    STEPS,
    TOOL_COLUMN,
    TOOLS,
    WARN_MARK,
    Graph,
    Node,
    block_title,
    kbase,
    node_title,
)
from .dialogs import ask_name
from .gallery import TemplateGallery, TemplatePage
from .i18n import _, hm, number, plural
from .icons import BLOCK_ROLES, IconLabel, draw, icon, icon_only, set_icon
from .imageview import ZOOM_PRESETS, ZOOM_STEP
from .layout import ARROW_H, ARROW_W, Dock, fold_button, pinch_factor, wheel_gesture
from .model import NotInEngine, cuttable, node_key, view_key  # noqa: F401  (cuttable: older callers)
from .panel import Panel
from .widgets import ACCENT, BAD, NEW, OK, WARN, Spinner, clear, label

BW, BH = 120, 54            # block size (a column is wider when its title needs it: TITLE_PAD)
GAP = 12                    # between two columns: where the branches split
BEND = 22                   # a wire to Post: the length of its bends onto its lane and into Post
LANE = 80                   # lane pitch
SROW = BH + 14              # the pitch of the sessions' rows (issue #43)
CUT_COL, GREY_COL = "#6b7280", "#c9a227"
BROKEN_COL = "#e05252"                          # a block that cannot run at all (GUI exploration 1.1)
TOP = 74                    # room for the phase labels
MARGIN = 8                  # around the graph, left and right
MIN_SCALE = 0.75            # the smallest the default view goes to show the whole chain; below, the text cannot be read
#                             (9.2: 0.6 made titles of 6 pt; GUI exploration 2: 0.65 still unreadable at 1366 x 768,
#                             titles of 7.5 pt here): the view keeps this scale, shows the left end (Frames) and scrolls
#                             right, Post pinned at the right edge ("Post ›"), a drag near the edge scrolls. Show all
#                             goes lower
MIN_ZOOM, MAX_ZOOM = 0.1, 8.0  # issue #44: the zoom the user can choose (100 % is the blocks at their designed size)
MAP_W, MAP_H = 190.0, 130.0  # the minimap, as the Inspect viewer's (issue #44)
EDGE = 40                   # a drag this near the view's edge scrolls it
EDGE_WAIT = 0.35            # s held near the edge before it scrolls (a drag that only crosses the edge moves nothing)
TITLE_PAD = 22              # a block's title, its state stripe and its padding
STAR_MAX = 260               # the star blocks' columns: as wide as their long names (issue #36)
OUT_MAX = BW + 30           # the widest Output column: a longer workflow name loses its middle (whole in the tooltip)
STATE_COL = {"done": OK, "pending": "#5d6470", "stale": WARN, "running": ACCENT, "failed": BAD,
             "interrupted": WARN, "skipped": "#5d6470"}
# the blocks whose output can go straight to the post: only where it gives an image no workflow already gives
# (before the background step, an output still carries the sky gradient; astrometry does not change the image)
EXITS = {"empilement"}
OPTIONAL_BLOCKS = ("debruitage", "etoiles", "deconv")
#: the ports of a block: (what it sends, its height on the block's right edge); None is a workflow's image (an exit)
PORTS = {"empilement": ((None, BH / 2),), "etoiles": (("starless", BH * 0.32), ("stars", BH * 0.68))}
#: the colours of the Stars block's two images: their ports and their wires
IMAGE_COL = {"starless": "#7aa2d6", "stars": "#e0c45c"}
NO_PORT = "-"


def elide(text, font, width):
    return QFontMetrics(font).elidedText(text, Qt.ElideRight, int(width))


def review_tip(bp) -> str:
    """What the review point of a block says, for its tooltip: bp is Graph.breakpoint's (state, have, total)."""
    state, have, total = bp
    text = {"pending": _("Review point: the run stops here for your review."),
            "paused": _("Paused: check this block (Inspect), then Continue from the Run window."),
            "reviewed": _("Reviewed: the run goes through while these results stay as they are.")}[state]
    if have < total:
        text += " " + plural(have, "Pause set in {n} workflow of {total}.", "Pause set in {n} workflows of {total}.",
                             total=total)
    return text


POST_WHAT = "Where the workflows' images become layers: stretched, balanced, blended and exported."


def block_what(node: Node) -> str:
    """What a block does, for its tooltip: a paragraph of its own, wrapped (a plain tooltip never wraps by itself)."""
    what = POST_WHAT if node.key == "final" else BY_KEY.get("session" if node.key == "sources" else kbase(node.key),
                                                            {}).get("what")
    return textwrap.fill(_(what), 70) + "\n" if what else ""


class BlockItem(QGraphicsItem):
    def __init__(self, view, node: Node, title, summary, state, selected, dim, removable, w: float = BW, bp=None):
        super().__init__()
        self.view, self.node, self.w = view, node, w
        # a warning at the head of the summary (issue #58) is its mark, drawn as an icon
        self.warned = summary.startswith(WARN_MARK)
        summary = summary[len(WARN_MARK):] if self.warned else summary
        self.title, self.summary, self.state = title, summary, state
        self.selected, self.dim, self.removable = selected, dim, removable
        self.bp = bp                           # None, or Graph.breakpoint: (state, workflows that pause here, of how many)
        self.hover = False
        self.target = False                    # the Post block, while an output is pulled over it
        self._wire = False
        self.exit = view.has_port(node) if hasattr(view, "has_port") else node.key in EXITS
        self.ports = PORTS.get(node.key, ()) if node.key not in EXITS else ()
        if self.exit and not self.ports:                  # its image, pulled onto Post: a layer (issue #47)
            self.ports = ((None, BH / 2),)
        if node.key == "fusion" and node.paths and ((node.paths[0].source or {}).get("fusion") or {}).get("stars"):
            self.ports = PORTS["etoiles"]                 # a starless fusion: its two images, as the Split block's
        self.port = bool(self.ports)
        self.setAcceptHoverEvents(True)
        # a block switched off is not computed: never "Up to date" (round 2), its line below says what it is
        tip = f"{title}\n{block_what(node)}{summary}" + ("\n" + _(STATE_WORD[state]) if node.key not in
                                                          ("sources", "final", "session") and not node.cut else "")
        # a reason is a sentence of its own: its full stop is not doubled by the one after {why}
        if node.broken:
            tip += "\n" + _("Cannot run: {why}. Run is refused until it is fixed.").format(why=node.broken.rstrip("."))
        if node.cut:
            tip += "\n" + _("Switched off: the image passes through. D or the menu switches it on again.")
        elif node.grey and node.key == "composition":     # a Combine layers to come (GUI exploration 4, Q10)
            tip += "\n" + _("Comes with the lights: once they are added, Post combines the Stars block's two images")
        elif node.grey:
            tip += "\n" + _("Cannot run: {why}. The image passes through.").format(why=node.grey.rstrip("."))
        if self.exit:
            tip += "\n" + _("Output ●: drag it onto Post to make a layer")
        elif node.key == "etoiles":
            tip += "\n" + _("● top: the starless image; ● bottom: the stars alone. Drag one onto Post to make it a "
                            "layer, or onto Composition for a Combine layers block")
        if bp is not None and node.cut:                  # kept, but a block switched off is not run: no stop now
            tip += "\n" + _("Review point kept: it waits for the block to be switched on again (D); runs go "
                            "through meanwhile.")
        elif bp is not None:
            tip += "\n" + review_tip(bp)
        self.setToolTip(tip.strip())

    def boundingRect(self):                                           # noqa: N802 - Qt's name
        left = 12 if self.bp is not None else 2         # the review dot sits on the corner
        return QRectF(-left, -left, self.w + left + (10 if self.port else 2), BH + left + 2)

    def _x_rect(self):
        return QRectF(self.w - 20, 4, 16, 16)

    def _on_port(self, pos):
        """The port under ``pos`` (what it sends: None for an exit), or NO_PORT."""
        for what, y in self.ports:
            if (pos - QPointF(self.w, y)).manhattanLength() < (10 if len(self.ports) > 1 else 14):
                return what
        return NO_PORT

    def paint_review(self, p: QPainter):
        """The review point, on the block's top-left corner like a debugger's dot: red while the run will stop here,
        yellow with the pause bars while it is stopped here for the user, a green tick once the user checked these
        results. A ring instead of a disc when only some of the workflows going through the block have it."""
        state, have, total = self.bp
        c = QPointF(2, 2)
        col = QColor({"pending": BREAK_COL, "paused": PAUSE_COL, "reviewed": REVIEWED_COL}[state])
        r = 9.5 if state == "paused" else 8.0
        disc = have == total and state != "reviewed"
        dark = QColor("#15171c")
        p.save()
        p.setPen(QPen(dark, 2.4))                                      # a cut-out ring so it reads on any block
        p.setBrush(col if disc else dark)
        p.drawEllipse(c, r, r)
        if not disc:
            p.setPen(QPen(col, 2.0))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(c, r - 1.4, r - 1.4)
        if state == "paused":                                          # the pause bars
            p.setPen(Qt.NoPen)
            p.setBrush(dark if disc else col)
            p.drawRoundedRect(QRectF(c.x() - 4.4, c.y() - 4.6, 3.2, 9.2), 1.0, 1.0)
            p.drawRoundedRect(QRectF(c.x() + 1.2, c.y() - 4.6, 3.2, 9.2), 1.0, 1.0)
        elif state == "reviewed":                                      # the tick
            p.setPen(QPen(col, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPolyline([QPointF(c.x() - 3.4, c.y() + 0.2), QPointF(c.x() - 1.0, c.y() + 2.8),
                            QPointF(c.x() + 3.6, c.y() - 2.6)])
        p.restore()

    def paint(self, p: QPainter, opt, widget=None):
        p.setRenderHint(QPainter.Antialiasing)
        p.setOpacity(0.55 if self.dim or self.node.cut else 1.0)
        w = self.w
        r = QRectF(0, 0, w, BH)
        ghost = self.node.key == "sources" and self.state == "pending"
        fill = "#1d2129" if ghost else ("#2b3446" if self.selected or self.target else
                                        ("#272c36" if self.hover else "#22262e"))
        paused = self.bp is not None and self.bp[0] == "paused"
        pen = QPen(QColor(ACCENT if self.selected or self.target else (PAUSE_COL if paused else "#3a4150")),
                   2 if self.selected or self.target or paused else 1, Qt.DashLine if ghost else Qt.SolidLine)
        p.setPen(pen)
        p.setBrush(QColor(fill))
        p.drawRoundedRect(r, 7, 7)
        if self.node.cut:                              # cut by the user: hatched, the wire drawn across it
            p.save()
            p.setClipPath(_rounded(r))
            p.setPen(QPen(QColor("#3a4150"), 1.0))
            for x in range(-int(BH), int(w) + 1, 9):
                p.drawLine(QPointF(x, BH), QPointF(x + BH, 0))
            p.restore()
            p.setPen(QPen(QColor("#9aa3b2"), 1.8))
            p.drawLine(QPointF(0, BH / 2), QPointF(w, BH / 2))
        if self.node.grey:                             # greyed by a cut upstream: a dashed amber outline
            p.setPen(QPen(QColor(GREY_COL), 1.4, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r.adjusted(1, 1, -1, -1), 7, 7)
        if self.node.broken:                           # it cannot run at all (1.1): circled in red, Run refused
            p.setPen(QPen(QColor(BROKEN_COL), 2.2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r.adjusted(-2, -2, 2, 2), 8, 8)
        if self.node.key not in ("sources", "final", "session") or self.state != "pending":
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(CUT_COL if self.node.cut else (GREY_COL if self.node.grey else STATE_COL[self.state])))
            path = QPainterPath()
            path.addRoundedRect(QRectF(0, 0, 8, BH), 4, 4)
            clip = QPainterPath()
            clip.addRect(QRectF(0, 0, 4, BH))
            p.drawPath(path.intersected(clip))
        f = QFont("Segoe UI", 10)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor("#e8ebf0"))
        mark = ("failed", "error") if self.node.broken else ("off", "dim") if self.node.cut else \
            ("not_computable", "warning") if self.node.grey else None
        x0 = 11 + (20 if mark else 0)                     # the block's state, before its title
        tw = w - 7 - x0 - (18 if self.removable and self.hover else 0)
        # a workflow's name loses its middle (two variants differ at their end; the whole name is in the tooltip), the
        # engine's words their end
        mode = Qt.ElideMiddle if self.node.key == "post" else Qt.ElideRight
        readable = QStyleOptionGraphicsItem.levelOfDetailFromTransform(p.worldTransform()) >= MIN_SCALE - 1e-6
        p.drawText(QRectF(x0, 6, tw, 20) if readable else QRectF(x0, 0, tw, BH), Qt.AlignLeft | Qt.AlignVCenter,
                   QFontMetrics(f).elidedText(self.title, mode, int(tw)))
        if mark:
            draw(p, QRectF(11, 8 if readable else (BH - 16) / 2, 16, 16), mark[0], mark[1])
        # issue #44: below the readable size the subtitles go, so that a zoomed-out graph is still a map of titles
        if readable:
            f2 = QFont("Segoe UI", 8.5)
            p.setFont(f2)
            p.setPen(QColor("#8b93a1"))
            sx = 11 + (20 if self.warned else 0)
            p.drawText(QRectF(sx, 28, w - 4 - sx, 18), Qt.AlignLeft | Qt.AlignVCenter, elide(self.summary, f2, w - 4 - sx))
            if self.warned:
                draw(p, QRectF(11, 29, 16, 16), "warning", "warning")
        if self.removable and self.hover:
            draw(p, self._x_rect(), "remove", "error")
        if self.node.key == "composition":                 # its two inputs, in the colours of the Stars block's outputs
            for what, y in PORTS["etoiles"]:
                p.setPen(QPen(QColor(IMAGE_COL[what]), 1.5))
                p.setBrush(QColor(IMAGE_COL[what]))
                p.drawEllipse(QPointF(0, y), 4.5, 4.5)
        for what, y in self.ports:
            big = self.hover or self.selected or self._wire
            col = IMAGE_COL.get(what, ACCENT)
            p.setPen(QPen(QColor(col if big or what else "#4a5263"), 1.5))
            p.setBrush(QColor(col if big or what else "#15171c"))
            r = 6.5 if big else 4.5
            p.drawEllipse(QPointF(w, y), r, r)
        if self.bp is not None:
            self.paint_review(p)

    def hoverEnterEvent(self, e):                                     # noqa: N802
        self.hover = True
        self.update()

    def hoverMoveEvent(self, e):                                      # noqa: N802
        self.setCursor(Qt.OpenHandCursor if self._on_port(e.pos()) != NO_PORT else Qt.ArrowCursor)

    def hoverLeaveEvent(self, e):                                     # noqa: N802
        self.hover = False
        self.update()

    def mousePressEvent(self, e):                                     # noqa: N802
        if e.button() == Qt.LeftButton:
            port = self._on_port(e.pos())
            if port != NO_PORT:
                self._wire = True
                self.view.start_wire(self, port)
            elif self.removable and self._x_rect().contains(e.pos()):
                self.view.remove_block(self.node, None)
            else:
                self._press = e.screenPos()
                self.view.blockClicked.emit(self.node)
        e.accept()

    def mouseMoveEvent(self, e):                                      # noqa: N802
        if self._wire:
            self.view.move_wire(e.scenePos())
            return
        press = getattr(self, "_press", None)
        if press is not None and self.removable and \
                (e.buttons() & Qt.LeftButton) and \
                (e.screenPos() - press).manhattanLength() >= QApplication.startDragDistance():
            self._press = None                            # issue #54: dragged out to the palette, it is removed
            self.view.drag_out(self.node)

    def mouseReleaseEvent(self, e):                                   # noqa: N802
        if self._wire:
            self._wire = False
            self.view.end_wire(e.scenePos())
        e.accept()

    def mouseDoubleClickEvent(self, e):                               # noqa: N802
        if self.node.key == "composition":
            self.view.open_layer(self.node)
        else:
            self.view.blockOpened.emit(self.node)
        e.accept()

    def contextMenuEvent(self, e):                                    # noqa: N802
        self.view.blockClicked.emit(self.node)          # the block of the menu is the one selected (8.6, as editors)
        self.view.block_menu(self.node, e.screenPos())
        e.accept()


class WireItem(QGraphicsPathItem):
    """A wire the user can take away (issue #52): an image sent to Post as a layer. Hovered it lights up, a click
    selects it, Del or its menu's Remove takes the layer away. The wires of a chain are not selectable (no item)."""

    def __init__(self, view, path, pen, on_remove, tip: str, pid: int | None = None):
        super().__init__(path)
        self.view, self.base_pen, self.on_remove, self.pid = view, QPen(pen), on_remove, pid
        self.setPen(pen)
        self.setAcceptHoverEvents(True)
        self.setToolTip(tip + "\n" + _("Click to select it, Del to remove it"))
        self.selected = False
        self.part: tuple | None = None                 # a Stars/background block's image sent as it is: (pid, part)

    def shape(self):                                                  # noqa: D102 - a wide band to hit
        st = QPainterPathStroker()
        st.setWidth(16)
        return st.createStroke(self.path())

    def _look(self, hot: bool):
        pen = QPen(self.base_pen)
        if hot or self.selected:
            pen.setColor(QColor(ACCENT))
            pen.setWidthF(self.base_pen.widthF() + 1.2)
        self.setPen(pen)

    def hoverEnterEvent(self, e):                                     # noqa: N802
        self._look(True)

    def hoverLeaveEvent(self, e):                                     # noqa: N802
        self._look(False)

    def mousePressEvent(self, e):                                     # noqa: N802
        self.view.select_wire(self)
        e.accept()

    def contextMenuEvent(self, e):                                    # noqa: N802
        self.view.select_wire(self)
        m = QMenu(self.view)
        if self.pid is not None:                      # a workflow's image: the Output hidden in the wire, its menu
            for key, text in (("rename", "Rename"), ("duplicate", "Duplicate"), ("template", "Save template…")):
                m.addAction(_(text)).triggered.connect(
                    lambda _checked=False, k=key: self.view.workflowMenu.emit(k, self.pid))
            m.addSeparator()
        m.addAction(_("Remove") + "\t" + _("Del")).triggered.connect(lambda _checked=False: self.on_remove())   # not the flag
        if self.pid is not None:
            m.addSeparator()
            m.addAction(_("Delete workflow")).triggered.connect(
                lambda _checked=False: self.view.workflowMenu.emit("delete", self.pid))
        m.exec(e.screenPos())


class ChainWireItem(QGraphicsPathItem):
    """A wire that is the chain itself (issue #52): nothing to select or take away, and it says so, hovered or clicked
    (a block taken away takes its wires with it)."""

    def __init__(self, view, path, pen, tip: str = ""):
        super().__init__(path)
        self.view = view
        self.setPen(pen)
        self.setToolTip((tip + "\n" if tip else "") + _("Part of the chain: it follows the blocks, it cannot be removed"))

    def shape(self):                                                  # noqa: D102 - a wide band to hit, as WireItem's
        st = QPainterPathStroker()
        st.setWidth(16)
        return st.createStroke(self.path())

    def mousePressEvent(self, e):                                     # noqa: N802
        self.view.show_toast(_("Chain wire: not removable"))
        e.accept()


def _line(x1, y1, x2, y2) -> QPainterPath:
    path = QPainterPath(QPointF(x1, y1))
    path.lineTo(QPointF(x2, y2))
    return path


def _rounded(r: QRectF) -> QPainterPath:
    path = QPainterPath()
    path.addRoundedRect(r, 7, 7)
    return path


class GhostItem(QGraphicsItem):
    """A dashed place: where a dragged tool can go, or the next workflow; ``role``: an icon before its text."""

    def __init__(self, rect: QRectF, text, on_click=None, pid=None, role: str | None = None):
        super().__init__()
        self.rect, self.text, self.on_click, self.pid, self.role = rect, text, on_click, pid, role
        self.hot = False
        self.far = False                                 # another workflow's place: shown when the cursor is on its row
        self.setAcceptHoverEvents(True)

    def boundingRect(self):                                           # noqa: N802
        return self.rect.adjusted(-2, -2, 2, 2)

    def paint(self, p, opt, widget=None):
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(ACCENT if self.hot else "#4a5263"), 1.6 if self.hot else 1.2, Qt.DashLine))
        p.setBrush(QColor("#1d2633") if self.hot else Qt.NoBrush)
        p.drawRoundedRect(self.rect, 7, 7)
        f = QFont("Segoe UI", 9)
        p.setFont(f)
        p.setPen(QColor(ACCENT if self.hot or self.on_click else "#8b93a1"))
        if self.role:                                    # the icon and the text centred together
            tw = QFontMetricsF(f).horizontalAdvance(self.text)
            x = self.rect.center().x() - (tw + 20) / 2
            draw(p, QRectF(x, self.rect.center().y() - 8, 16, 16), self.role, "selected" if self.hot else "dim")
            p.drawText(QRectF(x + 20, self.rect.y(), tw + 2, self.rect.height()), Qt.AlignVCenter | Qt.AlignLeft, self.text)
        else:
            p.drawText(self.rect, Qt.AlignCenter, self.text)

    def hoverEnterEvent(self, e):                                     # noqa: N802
        if self.on_click:
            self.hot = True
            self.update()

    def hoverLeaveEvent(self, e):                                     # noqa: N802
        if self.on_click:
            self.hot = False
            self.update()

    def mousePressEvent(self, e):                                     # noqa: N802
        if self.on_click:
            self.on_click()
        e.accept()


class LinkItem(QGraphicsItem):
    """A link on the blank page: words alone, underlined under the mouse (dashed is for the places a block can go)."""

    def __init__(self, center: QPointF, text: str, on_click):
        super().__init__()
        self.text, self.on_click = text, on_click
        self.font = QFont("Segoe UI", 10)
        w = QFontMetricsF(self.font).horizontalAdvance(text) + 8
        self.rect = QRectF(center.x() - w / 2, center.y() - 12, w, 24)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setZValue(7)

    def boundingRect(self):                                           # noqa: N802
        return self.rect

    def paint(self, p, opt, widget=None):
        f = QFont(self.font)
        f.setUnderline(self.isUnderMouse())
        p.setFont(f)
        p.setPen(QColor(ACCENT))
        p.drawText(self.rect, Qt.AlignCenter, self.text)

    def hoverEnterEvent(self, e):                                     # noqa: N802
        self.update()

    def hoverLeaveEvent(self, e):                                     # noqa: N802
        self.update()

    def mousePressEvent(self, e):                                     # noqa: N802
        self.on_click()
        e.accept()


class ProjectItem(QGraphicsItem):
    """The Project block: above the first object, out of the flow; what it says of the target by a dot (green known,
    orange unknown to the catalogues here, grey empty). A click shows its panel."""

    DOTS = {"known": OK, "unknown": WARN, "empty": "#6b7382"}

    def __init__(self, view, rect: QRectF, title: str, summary: str, dot: str, selected: bool, tip: str):
        super().__init__()
        self.view, self.rect, self.title, self.summary, self.dot, self.selected = view, rect, title, summary, dot, selected
        self.setToolTip(tip)
        self.setZValue(6)

    def boundingRect(self):                                           # noqa: N802
        return self.rect.adjusted(-2, -2, 2, 2)

    def paint(self, p, opt, widget=None):
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(ACCENT if self.selected else "#3a4150"), 2 if self.selected else 1))
        p.setBrush(QColor("#2b3446" if self.selected else "#22262e"))
        p.drawRoundedRect(self.rect, 7, 7)
        c = QPointF(self.rect.right() - 14, self.rect.top() + 14)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.DOTS[self.dot]))
        p.drawEllipse(c, 5, 5)
        f = QFont("Segoe UI", 10)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor("#e8ebf0"))
        w = self.rect.width() - 34
        p.drawText(QRectF(self.rect.x() + 11, self.rect.y() + 6, w, 20), Qt.AlignLeft | Qt.AlignVCenter,
                   QFontMetrics(f).elidedText(self.title, Qt.ElideRight, int(w)))
        f2 = QFont("Segoe UI", 8.5)
        p.setFont(f2)
        p.setPen(QColor("#8b93a1"))
        p.drawText(QRectF(self.rect.x() + 11, self.rect.y() + 28, self.rect.width() - 15, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   elide(self.summary, f2, self.rect.width() - 15))

    def mousePressEvent(self, e):                                     # noqa: N802
        self.view.projectClicked.emit()
        e.accept()


class AlignItem(QGraphicsItem):
    """The Align images block: one block across the lanes of its images, a wire in and a wire out on each lane."""

    def __init__(self, view, nodes: list[Node], rect: QRectF, title: str, summary: str, state: str, selected: bool):
        super().__init__()
        self.view, self.nodes, self.rect, self.title, self.summary, self.state, self.selected = \
            view, nodes, rect, title, summary, state, selected
        self.hover = False
        self.setAcceptHoverEvents(True)
        self.setZValue(5)
        self.setToolTip(f"{title}\n{block_what(nodes[0])}")

    def boundingRect(self):                                           # noqa: N802
        return self.rect.adjusted(-2, -2, 2, 2)

    def _x_rect(self):
        return QRectF(self.rect.right() - 20, self.rect.top() + 4, 16, 16)

    def paint(self, p, opt, widget=None):
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(ACCENT if self.selected else "#3a4150"), 2 if self.selected else 1))
        p.setBrush(QColor("#2b3446" if self.selected else "#22262e"))
        p.drawRoundedRect(self.rect, 7, 7)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(STATE_COL.get(self.state, "#5d6470")))
        p.drawRoundedRect(QRectF(self.rect.x(), self.rect.y() + 3, 4, self.rect.height() - 6), 2, 2)
        c = self.rect.center()
        f = QFont("Segoe UI", 10)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor("#e8ebf0"))
        p.drawText(QRectF(self.rect.x() + 10, c.y() - 22, self.rect.width() - 14, 22), Qt.AlignCenter | Qt.TextWordWrap, self.title)
        f2 = QFont("Segoe UI", 8.5)
        p.setFont(f2)
        p.setPen(QColor("#8b93a1"))
        p.drawText(QRectF(self.rect.x() + 6, c.y() + 2, self.rect.width() - 10, 18), Qt.AlignCenter,
                   elide(self.summary, f2, self.rect.width() - 10))
        if self.hover:
            draw(p, self._x_rect(), "remove", "error")

    def hoverEnterEvent(self, e):                                     # noqa: N802
        self.hover = True
        self.update()

    def hoverLeaveEvent(self, e):                                     # noqa: N802
        self.hover = False
        self.update()

    def mousePressEvent(self, e):                                     # noqa: N802
        if e.button() == Qt.LeftButton and self._x_rect().contains(e.pos()):
            self.view.remove_block(self.nodes[0], None)
        elif e.button() == Qt.LeftButton:
            self.view.blockClicked.emit(self.nodes[0])
        e.accept()

    def mouseDoubleClickEvent(self, e):                               # noqa: N802
        self.view.blockOpened.emit(self.nodes[0])
        e.accept()

    def contextMenuEvent(self, e):                                    # noqa: N802
        self.view.blockClicked.emit(self.nodes[0])
        m = QMenu(self.view)
        m.addAction(_("Delete") + "\t" + _("Del")).triggered.connect(lambda: self.view.remove_block(self.nodes[0], None))
        m.exec(e.screenPos())
        m.deleteLater()
        e.accept()


class BandItem(QGraphicsItem):
    def __init__(self, rect, text):
        super().__init__()
        self.rect, self.text = rect, text
        self.setZValue(-10)

    def boundingRect(self):                                           # noqa: N802
        return self.rect

    def paint(self, p, opt, widget=None):
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#191c22"))
        p.drawRoundedRect(self.rect, 10, 10)
        f = QFont("Segoe UI", 9)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor("#6b7280"))
        # a band one column wide ("LINEAR PROCESSING" over one image block): its title on two lines, never cut (GUI
        # exploration 2); a single word still too long loses its end, with "…"
        r = self.rect.adjusted(12, 8, -8, 0)
        fm = QFontMetricsF(f)
        words = [w if fm.horizontalAdvance(w) <= r.width() else fm.elidedText(w, Qt.ElideRight, r.width())
                 for w in self.text.upper().split()]
        p.drawText(r, Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, " ".join(words))


class GraphView(QGraphicsView):
    blockClicked = Signal(object)          # Node
    blockOpened = Signal(object)           # Node: double-click, to Inspect
    newWorkflow = Signal()
    toolDropped = Signal(str, int)         # tool key, workflow pid
    blockDropped = Signal(str, int, int)   # issue #47: a palette block (engine name) dropped on a workflow at an index
    sourceDropped = Signal(str)            # issue #47: a Session or an Image dropped on the graph
    postLinkRemoved = Signal(int)          # issue #52: a workflow's link to Post taken away (its wire)
    branchDropped = Signal(str, int, str)  # issue #47: a block dropped under a block: a new branch from there
    removeTool = Signal(object, object)    # Node, list of pids or None (all the workflows of the block)
    workflowMenu = Signal(str, int)        # action, pid
    sendToPost = Signal(object)            # Node whose output was pulled to the Post block
    pauseToggled = Signal(object)          # Node: "Review pause" was chosen (set it, or take it off)
    reviewMarked = Signal(object)          # Node: "Mark reviewed" was chosen
    partToPost = Signal(object, str)       # a Stars node's image (starless | stars) pulled onto Post
    partOutOfPost = Signal(object, str)    # a Stars node's image sent as it is: taken out of Post (#29)
    newComposition = Signal(object)        # a Stars node's port pulled onto Composition: a new Stars and background block
    layerRemoved = Signal(int)             # a Stars and background block, or an image sent as it is: its Post layer
    openLayer = Signal(int)                # Open in Post (a Post layer's id)
    cutToggled = Signal(object, object)    # Node, list of pids (None: every workflow of the block): cut or restore it
    newSession = Signal()                  # "+ Session" (issue #43)
    templatesAsked = Signal()              # the blank page's link
    projectClicked = Signal()              # the Project block
    sessionMenu = Signal(str, int)         # action (rename, remove), session id
    postOpened = Signal()                  # "Open Post" of the Post block's menu
    deselected = Signal()                  # 2nd round, decision 5: a click on empty space, Escape, or a wire chosen
    filesDropped = Signal(int, list)       # files or folders dragged from the Explorer onto a Session: session id, paths
    wireSelected = Signal(object)          # a wire to Post chosen (WireItem): its layer in the panel (control round)

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.s = project
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(QBrush(QColor("#15171c")))
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setAcceptDrops(True)
        self.drag_tool = None
        self.drag_block = None                                         # issue #47: the palette block being dragged
        self.palette_flow = lambda: None                 # the workflow the palette places in (the page's); None: all
        self.project_selected = lambda: False            # the Project block is the one selected (the page's)
        self.ghosts: list[GhostItem] = []
        self.graph: Graph | None = None
        self._sel_item = None
        self.final_item = None
        self._fitting = False
        self._next_x: dict[str, float] = {}
        self.toast = QLabel(self)
        self.toast.setStyleSheet("background: #262a33; border: 1px solid #3a4150; border-radius: 6px; "
                                 "padding: 6px 10px; color: #e8ebf0;")
        self.toast.hide()
        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self.toast.hide)
        self.wait = QLabel(self)                                       # while the worker reads the project
        self.wait.setStyleSheet("color: #8b93a1; background: transparent;")
        self.wait.setAlignment(Qt.AlignCenter)
        self.wait.hide()
        self.veil = QFrame(self)                                       # no graph before the project is read (#86)
        self.veil.setStyleSheet("background: #15171c;")
        self.veil.hide()
        self.spinner = Spinner(self)
        self.spinner.hide()
        self.b_all = QPushButton(_("Show all"), self)
        self.b_all.setCheckable(True)
        self.b_all.setObjectName("toggle")
        self.b_all.setToolTip(_("The whole graph in the window, smaller; off: readable size, scrolling"))
        # kept with the project, as the selection (GUI exploration 4, Q9)
        self.b_all.setChecked(bool(getattr(getattr(project, "p", None), "show_all", False)))
        self.b_all.toggled.connect(self._show_all)
        # issue #44: zoom and position, as the Post image's: "auto" is today's view (b_all, or the readable fit);
        # once the user zooms or moves it, it is his: kept through rebuilds and with the project
        self._zoomed = False
        self._k, self._c = 1.0, QPointF()
        self._rebuilding = False
        self._view_timer = QTimer(self)               # the settings written at most twice a second, as a pan goes
        self._view_timer.setSingleShot(True)
        self._view_timer.setInterval(500)
        self._view_timer.timeout.connect(self._write_view)
        saved = self._saved_view()
        if saved is not None:
            self._zoomed = True
            self._k, self._c = saved
        self._pan = None                               # the pan under way: (start, h value, v value, moved, empty click)
        self._space = False                            # Space is held: a left drag pans
        self._sel_seen = None                          # the selection last shown (a new one is scrolled into view)
        self.map = GraphMap(self)
        self.map.hide()
        self.b_zoom = QPushButton("100 %", self)
        self.b_zoom.setToolTip(_("Zoom (wheel, + and −)"))
        self.b_zoom.setMinimumWidth(62)
        zm = QMenu(self.b_zoom)
        zm.addAction(_("Show all"), lambda: self.b_all.setChecked(True))
        for z in ZOOM_PRESETS:
            zm.addAction(f"{number(z * 100, 1 if z * 100 % 1 else 0)} %", lambda zz=z: self.zoom_to(zz))
        self.b_zoom.setMenu(zm)
        # GUI exploration 2: a long chain at a readable scale goes past the view's right edge; Post stays at hand: a
        # button at the top right corner while it is out of view, a click scrolls to it
        self.b_post = QPushButton("", self)
        self.b_post.clicked.connect(self.show_post)
        self.b_post.hide()
        for bar in (self.horizontalScrollBar(), self.verticalScrollBar()):
            bar.valueChanged.connect(lambda _v: self._scrolled())
        # a drag (a palette block, a placed block, a port's wire) held near the view's edge scrolls the view that way
        self._edge_pos: QPoint | None = None
        self._edge_since = 0.0
        self._wiring = False
        self._edge_timer = QTimer(self)
        self._edge_timer.setInterval(30)
        self._edge_timer.timeout.connect(self._edge_tick)

    # ------------------------------------------------------------ layout
    @staticmethod
    def col_of(n: Node) -> str:
        """A block's column: the frame blocks' and the outputs' own; an image block's, its rank among its workflow's
        image blocks ("i0", "i1"...: they are placed in any order, more than once, issue #47)."""
        if n.key in ("session", "image"):
            return "sources"
        if kbase(n.key) in IMAGE_KEYS:
            return f"i{max(n.slot, 0)}"
        return n.key

    def columns(self, g: Graph):
        """The columns the graph uses, left to right: a block the workflows do not have has no column (issue #47)."""
        used = {self.col_of(n) for n in g.nodes if n.paths}
        if self.drag_block == "fusion":             # issue #53: its place is in the Composition section
            used.add("fusion")
        if "etoiles" in used or "fusion" in used:   # the Composition section: the stars recombined, the fusions
            used.add("composition")
        slots = [int(c[1:]) for c in used if c[:1] == "i" and c[1:].isdigit()]
        out = ["sources"] + [c for c in FRAME_KEYS if c in used]
        out += [f"i{k}" for k in range(max(slots) + 1 if slots else 0)]
        out += [c for c in ("etoiles", "synth", "post", "fusion", "composition") if c in used]
        return out + ["final"]

    @staticmethod
    def column_width(key: str, names=()) -> float:
        """A column is as wide as a block, or wider when its title needs it in the language shown ("Post-traitement").
        The outputs carry the workflows' names: their column widens to the longest name, up to OUT_MAX; a longer name
        loses its middle ("Classic stack…adapted") and is whole in the tooltip."""
        f = QFont("Segoe UI", 10)
        f.setBold(True)
        fm = QFontMetricsF(f)
        if key == "post":
            need = max((fm.horizontalAdvance(n) for n in names), default=0.0) + TITLE_PAD
            return max(BW, min(OUT_MAX, need))
        # the star blocks' names are long (issue #36): their columns widen to them, as the outputs' do
        cap = STAR_MAX if key in ("etoiles", "composition") else BW + 40
        return max(BW, min(cap, fm.horizontalAdvance(block_title(None, key)) + TITLE_PAD))

    @staticmethod
    def lane_order(g: Graph) -> list[int]:
        """The workflows' lanes in the order of the tree (depth first): the workflows of every block sit on lanes next
        to each other, so that a block is drawn between its own lanes only, and a workflow that leaves the trunk before
        a block never runs behind it. A workflow that ends on a block the others go on from (a branch sent to Post
        from a block in the middle) takes the lane right after the lanes of that block's own subtree: never one after
        another channel's (control round: two Combine channels channels' blocks on top of each other), nor after the trunk's
        other branches (a workflow ending on Color drawn below a branch leaving at Gradient overlapped it)."""
        out: list[int] = []
        todo: list = [g.root]
        while todo:
            n = todo.pop()
            if isinstance(n, list):                       # a block's subtree walked: the workflows ending on it
                out += [pid for pid in n if pid not in out]
                continue
            if not n.children:
                out += [p.pid for p in n.paths if p.pid not in out]
                continue
            ending = [p.pid for p in n.paths if g.leaves.get(p.pid) is n] if n is not g.root else []
            if ending:
                todo.append(ending)                       # popped once every child below it is walked
            todo += reversed(n.children)
        return out + [p.pid for p in g.s.paths if p.pid not in out]

    @staticmethod
    def session_rows(g: Graph, order: list[int], frames: list[int], ghost: bool = False
                     ) -> tuple[dict[int, float], dict[int, float]]:
        """The lanes (workflow -> y) and the sessions' rows (session -> y), from 0 (issue #43): each session level with
        the chains it feeds, with the other sessions of those chains (a filter's second night with its filter's
        chain, not under the last chain: GUI exploration 4); rows never closer than SROW, the lanes below moved down
        when the rows need the room. A session no workflow reads: after the others. A night takes a row more per
        Calibration placed in parallel on it, and one for the place of a new one (``ghost``, while one is dragged)."""
        rel = {pid: i * LANE for i, pid in enumerate(order)}
        fr = set(frames)
        idx = {pid: i for i, pid in enumerate(order) if pid in fr}
        groups: dict[tuple, list[int]] = {}
        for sn, _cn in g.sessions:
            groups.setdefault(tuple(sorted(idx[p.pid] for p in sn.paths if p.pid in idx)), []).append(sn.sid)
        rows: dict[int, float] = {}
        prev = None
        high = {sn.sid: 1 + len(getattr(g, "calib_forks", {}).get(sn.sid, [])) + int(ghost) for sn, _cn in g.sessions}
        for key in sorted(groups, key=lambda k: (not k, k)):
            sids = groups[key]
            span = sum(high[sid] for sid in sids)
            if key:
                centre = (rel[order[key[0]]] + rel[order[key[-1]]]) / 2
            else:
                centre = prev + SROW + (span - 1) * SROW / 2 if prev is not None else 0.0
            first = centre - (span - 1) * SROW / 2
            if prev is not None and first < prev + SROW:
                d = prev + SROW - first
                first += d
                for pid in order[key[0]:] if key else []:  # its chains and those below: down, level with it
                    rel[pid] += d
            at = first
            for sid in sids:
                rows[sid] = at
                at += high[sid] * SROW
            prev = at - SROW
        return rel, rows

    def rebuild(self):
        sc = self.scene()
        # issue #24: a change (or a stage event of a run) builds the graph again; where the user had scrolled stays,
        # when the graph keeps its size
        was = (sc.sceneRect(), self.horizontalScrollBar().value(), self.verticalScrollBar().value())
        self._rebuilding = True
        sc.clear()
        self.ghosts = []
        self.sel_wire = None                              # issue #52: a wire selected before is gone with the scene
        self._sel_item = None
        self.final_item = None
        self.b_all.setText(_("Show all"))
        self.b_all.setToolTip(_("The whole graph in the window, smaller; off: readable size, scrolling"))
        self.b_all.adjustSize()
        self.b_zoom.setToolTip(_("Zoom (wheel, + and −)"))
        s = self.s
        g = Graph(s)
        self.graph = g
        cols = self.columns(g)
        # issue #47: while a palette block is dragged, a column opens before each block it can go in front of (and
        # after the last one): its places, shown as the block itself, the blocks after them moved aside
        targets = self._drop_targets(g, cols) if self.drag_block in wfl.IO else []
        if targets:
            gaps = {c for _pid, _idx, c in targets}
            cols = [x for c in cols for x in ((("g:" + c), c) if c in gaps else (c,))]
        cw = {}
        for c in cols:
            if c.startswith("g:"):                          # an opened column: a block's width
                cw[c] = BW
            elif c[:1] == "i" and c[1:].isdigit():          # an image rank: as wide as its widest block's name
                cw[c] = max((self.column_width(n.key) for n in g.nodes if n.paths and self.col_of(n) == c),
                            default=BW)
            else:
                cw[c] = self.column_width(c, [p.name for p in s.paths])
        cx, x = {}, 0.0
        for c in cols:
            cx[c] = x
            x += cw[c] + GAP
        self._next_x = {c: cx[c] + cw[c] + GAP for c in cols}       # where the column after it starts
        order = self.lane_order(g)
        frames = [pid for pid in order if getattr(s.view(pid), "source", None) is None]
        order = frames + [pid for pid in order if pid not in frames]      # the image workflows' lanes last
        n_lanes = max(1, len(s.paths))
        # issue #43: the sessions' rows, each level with the chains it feeds (session_rows)
        rel, rows = self.session_rows(g, order, frames, self.drag_block == "calibration" and bool(g.sessions))
        top = TOP + max(0.0, -min(rows.values(), default=0.0))
        lanes = {pid: top + rel[pid] for pid in order}
        mid = (lanes[order[0]] + lanes[order[-1]]) / 2 if order else top + LANE / 2
        fmid = (lanes[frames[0]] + lanes[frames[-1]]) / 2 if frames else top
        for sn, cn in g.sessions:
            sn.y = cn.y = top + rows[sn.sid]
            for i, c in enumerate(g.calib_forks.get(sn.sid, [])):          # under the night's own Calibration
                c.y = sn.y + (i + 1) * SROW
        s_last = max((n.y for pair in g.sessions for n in pair + tuple(g.calib_forks.get(pair[0].sid, []))),
                     default=fmid)                                          # the lowest session's row
        others = [pid for pid in order if pid not in frames]
        if others and (s.sources_loaded or s.p.sessions):   # the image lanes: below the sessions and "+ Session"
            last = s_last
            low = last + BH / 2 + 8 + BH * 0.6 + 10 + BH / 2
            shift = max(0.0, low - lanes[others[0]])
            for pid in others:
                lanes[pid] += shift
        for n in g.nodes:
            if n.key == "session" or n.sid is not None:
                pass
            elif n.key in ("sources", "final") or not n.paths:
                n.y = fmid if n.key == "sources" else mid
            else:
                ys = [lanes[p.pid] for p in n.paths]
                n.y = (min(ys) + max(ys)) / 2
            col = self.col_of(n)
            n.col = cx.get(col, 0.0)
            n.w = cw.get(col, BW)
        seen: dict[int, int] = {}
        for n in g.compositions:                   # two blocks of one workflow: one a little below the other
            k = seen.get(n.paths[0].pid, 0)
            seen[n.paths[0].pid] = k + 1
            n.y += k * (BH + 6)
        # 2nd round: two blocks of one column (a fusion's Combine layers and a workflow's) never drawn over each other
        for c in ("composition", "fusion"):
            low = None
            for n in sorted((n for n in g.nodes if n.paths and self.col_of(n) == c), key=lambda n: n.y):
                if low is not None and n.y < low:
                    n.y = low
                low = n.y + BH + 6
        self._lanes, self._cx, self._cw = lanes, cx, cw
        bottom = max(top + n_lanes * LANE, max(lanes.values(), default=top) + LANE,
                     s_last + SROW if g.sessions else fmid + SROW / 2) + LANE
        # phase bands
        bands = [("Pre-processing", [c for c in cols if c in FRAME_KEYS]),
                 ("Linear processing", [c for c in cols if (c[:1] == "i" and c[1:].isdigit()) or c in ("etoiles", "synth")]),
                 ("Composition", [c for c in cols if c in ("fusion", "composition")])]
        for title, span in bands:
            if not span:
                continue
            pad = GAP / 2 - 1                                         # two bands never touch
            x0, x1 = cx[span[0]] - pad, cx[span[-1]] + cw[span[-1]] + pad
            sc.addItem(BandItem(QRectF(x0, 0, x1 - x0, bottom), _(title)))
        cur = s.path
        sel = g.node_of(cur, s.sel_step) if s.sel_step not in ("sources", "final") else (
            g.root if s.sel_step == "sources" else g.final)
        # wires along the chains; the sessions' wires join into one bar before their first block (issue #43)
        firsts = [n for n in g.root.children if n.key != "image"]
        for n in g.nodes:
            if n.key in ("sources", "composition", "session") or n.sid is not None or n is g.final:
                continue
            if n.key in ("image", "fusion") or (g.sessions and n in firsts):
                continue
            par = n.parent if n.parent is not None else g.root
            self._wire(par, n, cur is not None and cur in n.paths)
        for sn, cn in g.sessions:
            if cn.paths:                                  # a session's Calibration, when its workflows have one
                self._wire(sn, cn, True)
            for c in g.calib_forks.get(sn.sid, []):       # and those placed in parallel: a fork from the Session
                self._wire(sn, c, cur is not None and cur in c.paths)
        for t in (firsts if g.sessions else []):   # each first block from the sessions its workflows stack
            src = []
            for sn, cn in g.sessions:
                if set(map(id, sn.paths)) & set(map(id, t.paths)):      # without Calibration: from the Session
                    src += [c for c in [cn] + g.calib_forks.get(sn.sid, [])
                            if set(map(id, c.paths)) & set(map(id, t.paths))] or [sn]
            if len(src) > 1:
                self._join(src, [t], cur)
            elif src:
                self._wire(src[0], t, cur is not None and cur in t.paths)
        # issue #43: the fusions, from the outputs of the workflows they combine
        for n in g.nodes:
            for a in g.fusion_inputs.get(id(n), []):
                self._wire(a, n, cur is not None and (cur in n.paths or cur in a.paths))
        # issue #43: "+ Session" under the sessions (or under the one Session block), where a session goes
        self.plus_session_ghost = None
        self.plus_session_pid = None                      # the flow it joins: the one it is drawn with
        if s.sources_loaded:
            last = max((sn for sn, _cn in g.sessions), key=lambda n: n.y) if g.sessions else g.root   # the lowest
            gh = GhostItem(QRectF(cx["sources"], last.y + BH / 2 + 8, cw["sources"], BH * 0.6), _("Session"),
                           self.newSession.emit, role="add")
            cur_flow = s.path if s.path is not None and s.path.source is None else None
            if g.sessions and last.sid is not None:       # it joins the flow of the session it is drawn under
                cur_flow = next((q for q in s.paths if q.source is None and q.sessions is not None
                                 and last.sid in q.sessions), cur_flow)
            self.plus_session_pid = cur_flow.pid if cur_flow is not None else None
            gh.setToolTip(_("Add a session to “{name}”: a night of lights, with its own calibration frames").format(
                name=cur_flow.name) if cur_flow is not None and len([q for q in s.paths if q.source is None]) > 1
                else _("Add a session: a night of lights, with its own calibration frames"))   # which flow (expl. 2)
            gh.hot = self.drag_block == "session"         # a Session dragged: one of its two places (8.2)
            gh.setZValue(7)                               # never under the new-flow zone (5.3)
            sc.addItem(gh)
            self.plus_session_ghost = gh
        for p in s.paths:                          # a workflow's Output image, when it is a layer of Post
            if p.pid in g.to_post:
                self._wire(g.leaves[p.pid], g.final, cur is not None and p is cur,
                           on_remove=lambda q=p: self.postLinkRemoved.emit(q.pid),
                           tip=_("“{name}”: its image to Post as a layer").format(name=p.name), pid=p.pid,
                           via_y=lanes.get(p.pid))
            elif p.pid in g.awaits_post and p.pid in g.leaves:   # the same wire, dashed until the image exists
                grey = next((n for n in g.nodes if n.key == "composition" and n.grey and p in n.paths), None)
                self._wire(grey or g.leaves[p.pid], g.final, False, via_y=None if grey else lanes.get(p.pid),
                           waiting=True, tip=_("“{name}”: its image comes to Post with the lights").format(name=p.name))
        # the Stars block's images: into its Stars and background blocks, or straight to Post
        for n in g.compositions:
            strong = cur is not None and cur in n.paths
            for i, (what, word) in enumerate(STAR_OUTPUTS):
                self._image_wire(n.parent, PORTS["etoiles"][i][1], n, BH * (0.32 + 0.36 * i), what, _(word), strong)
            self._wire(n, g.final, strong)
        for src, what in g.direct:
            i = [w for w, _y in PORTS["etoiles"]].index(what)
            self._image_wire(src, PORTS["etoiles"][i][1], g.final, BH * (0.4 + 0.2 * i), what,
                             _(dict(STAR_OUTPUTS)[what]), cur is not None and cur in src.paths, True)
        # the images sent to Post (from Inspect, or a block's output): an arrow from their block to Post
        for n in self._post_sources(g):
            self._post_wire(n, g.final)
        # side inputs of the selected block
        if sel is not None and sel.key not in ("sources", "final") and not sel.grey and not sel.cut:
            for src, lab in g.side_inputs(sel):
                self._side_wire(src, sel, lab)
        # blocks
        self._add_align_blocks(g, sel)
        for n in g.nodes:
            if n.key == "align_images":
                continue                                  # drawn together, across their lanes
            if n.key not in ("sources", "final", "session") and not n.paths:
                continue                                  # a Session is drawn even when no workflow reads it (1.8)
            if n.key == "sources" and (g.sessions or (s.paths and not g.root.paths and not s.p.sessions)):
                continue                                  # the sessions' blocks stand for it; or only image workflows
            if n.key in ("sources", "final") and not s.paths and not s.p.sessions:
                continue                                  # a blank page is empty (issue #47)
            title = node_title(s, n)
            removable = n.key == "composition" or (n.key in ("image", "fusion", "calibration") and bool(n.paths)) or \
                n.key == "session" or (n.key == "sources" and len(s.p.sessions) == 1) or \
                (n.sid is None and bool(n.paths) and kbase(n.key) not in ("sources", "final", "session", "post"))
            dim = cur is not None and n.key not in ("sources", "final", "session") and n.sid is None \
                and cur not in n.paths
            it = BlockItem(self, n, title, g.summary(n), g.state(n), n is sel or (sel is not None and n.key ==
                           "composition" and sel.key == "composition" and n.lid == sel.lid), dim, removable, n.w,
                           g.breakpoint(n))
            it.setPos(n.col, n.y - BH / 2)
            it.setZValue(5)
            sc.addItem(it)
            if n is g.final:
                self.final_item = it
            if n is sel:
                self._sel_item = it
        # the next workflow
        y = max(TOP + len(s.paths) * LANE, max(lanes.values(), default=TOP) + LANE) - BH / 2 if s.paths \
            else mid - BH / 2
        inner = [c for c in cols if c not in ("sources", "final", "fusion", "composition")] or []
        if inner:
            x0, x1 = cx[inner[0]], cx[inner[-1]] + cw[inner[-1]]
        else:                                          # a blank page (issue #47): the palette's blocks start it
            x0 = cx["sources"]
            x1 = max(cx["final"] + cw["final"], x0 + 4 * BW)
            y = mid + BH
        # the templates only when the project is made: here, blocks dropped one by one
        if s.paths or s.p.sessions:                      # under every block and "+ Session", never over them (8.1)
            low = [n.y + BH / 2 for n in g.nodes if getattr(n, "y", None) is not None and (n.paths or n.key == "session")]
            ps = self.plus_session_ghost
            if ps is not None:
                low.append(ps.rect.bottom())
            if self.drag_block == "calibration" and g.sessions:   # the row open for a new one: never under this zone
                low.append(s_last + SROW + BH / 2)
            y = max([y] + [v + 14 for v in low])
        text = _("Workflow: Session/Image") if s.paths or s.p.sessions else _("Start: Session/Image")
        need = QFontMetricsF(QFont("Segoe UI", 9)).horizontalAdvance(text) + 60
        x1 = max(x1, x0 + need)                          # its text whole, however short the chain
        gh = GhostItem(QRectF(x0, y, x1 - x0, BH), text, None, role="add")
        gh.new_flow = True                               # issue #47: a parallel flow starts here
        gh.hot = self.drag_block in ("session", "image")
        sc.addItem(gh)
        self.new_flow_ghost = gh
        if not s.paths and not s.p.sessions and s.ready:
            sc.addItem(LinkItem(QPointF((x0 + x1) / 2, y + BH + 22), _("Start from a template"), self.templatesAsked.emit))
        # issue #47: while an image or a frame block is dragged, a new branch can start under each block it can follow
        if self.drag_block in wfl.IO and (wfl.IO[self.drag_block] == (wfl.IMAGE, wfl.IMAGE)
                                          or self.drag_block in wfl.FRAME_BLOCKS):
            title = block_title(None, view_key(self.drag_block))
            seen: set[int] = set()
            for p in self._flows():
                for key in s.branch_keys(p):
                    if not s.can_branch(p, key, self.drag_block):
                        continue
                    n, row = self._branch_anchor(g, p, key)
                    if n is None or id(n) in seen:
                        continue
                    seen.add(id(n))
                    r = QRectF(n.col + 6, n.y + BH / 2 + 4, n.w - 12, LANE - BH - 8) if not row else \
                        QRectF(n.col + 4, n.y + SROW - BH / 2 + 4, n.w - 8, BH - 8)   # in the gap, not on the
                    #                                                                    block under it (8.1)
                    bg = GhostItem(r, _("new branch"), None, p.pid, role="branch")
                    bg.branch_key = key
                    bg.setToolTip(_("A new branch from here: what comes before is shared (computed once), then "
                                    "{block}; it goes to Post as a layer of its own").format(block=title))
                    bg.setZValue(8)
                    sc.addItem(bg)
                    self.ghosts.append(bg)
        if self.drag_block == "composition" and "composition" in cx:     # issue #53: after each split block
            title = block_title(None, "composition")
            flows = self._flows()
            for n in g.nodes:
                if n.key == "etoiles" and n.paths:
                    r = QRectF(cx["composition"] + 4, n.y + BH / 2 + 6, cw["composition"] - 8, BH - 8)
                    gh = GhostItem(r, title, None, next(q for q in flows if q in n.paths).pid)
                    gh.node = n
                    gh.setZValue(8)
                    sc.addItem(gh)
                    self.ghosts.append(gh)
        if self.drag_block == "fusion":                  # issue #53: in the Composition column, under its Combine channels blocks
            below = [n.y for n in g.nodes if n.paths and self.col_of(n) == "fusion"]
            yc = max(below) + BH + 6 if below else mid
            r = QRectF(cx["fusion"] + 4, yc - BH / 2 + 4, cw["fusion"] - 8, BH - 8)
            gh = GhostItem(r, block_title(None, "fusion"), None)
            gh.setZValue(8)
            sc.addItem(gh)
            self.ghosts.append(gh)
        # issue #47: where the dragged palette block can go: the block itself, dashed, in each opened column
        for pid, idx, c in targets:
            p = s.view(pid) if pid != NEW_FRAMES else None
            title = block_title(None, view_key(self.drag_block))
            r = QRectF(cx["g:" + c] + 4, lanes.get(pid, fmid) - BH / 2 + 4, cw["g:" + c] - 8, BH - 8)
            gh = GhostItem(r, title, None, pid)
            gh.idx = idx
            gh.setToolTip(_('{block} here, in "{name}"').format(block=title, name=p.name) if p is not None else
                          _("{block} here, after the sessions").format(block=title))
            gh.setZValue(8)
            sc.addItem(gh)
            self.ghosts.append(gh)
        self._add_project(g, sel, cx, cw, gh_top=self.new_flow_ghost.rect.top())
        r = sc.itemsBoundingRect().adjusted(-MARGIN, -10, MARGIN, 24)
        sc.setSceneRect(r)
        own = self.palette_flow()
        for gh in self.ghosts:                            # the graph's size is set by all of them: nothing moves
            gh.far = own is not None and gh.pid not in (None, NEW_FRAMES, own.pid)
            gh.setVisible(not gh.far)
        self.show_wait()
        self.map.rows = [(it.sceneBoundingRect(), QColor(STATE_COL.get(it.state, "#5d6470")))
                         for it in sc.items() if isinstance(it, BlockItem)]
        self._rebuilding = False
        self.fit()
        old, h0, v0 = was
        if not self._zoomed and not old.isEmpty() and abs(old.width() - r.width()) < 1 and \
                abs(old.height() - r.height()) < 1:
            self.horizontalScrollBar().setValue(h0)
            self.verticalScrollBar().setValue(v0)
        # issue #44: a block selected from the panel, a run or Inspect, off-screen, is scrolled into view (zoom kept);
        # the block selected when the project opens does not move the default view
        seen = (cur.pid if cur is not None else None, s.sel_step, getattr(s, "sel_layer", None))
        if self._sel_seen is not None and seen != self._sel_seen:
            self.reveal(self._sel_item)
        self._sel_seen = seen

    def _add_align_blocks(self, g: Graph, sel):
        """Each group of aligned images: one block across their lanes (each lane's node keeps its wires)."""
        s = self.s
        groups: dict[int, list[Node]] = {}
        for n in g.nodes:
            if n.key == "align_images":
                groups.setdefault(n.paths[0].source["align"], []).append(n)
        for ref, nodes in groups.items():
            nodes.sort(key=lambda n: n.y)
            r = QRectF(nodes[0].col, nodes[0].y - BH / 2, nodes[0].w, nodes[-1].y - nodes[0].y + BH)
            name = next((n.paths[0].name for n in nodes if n.paths[0].pid == ref), "")
            nodes.sort(key=lambda n: n.paths[0].pid == ref)   # a click shows an image that moves, not the reference
            state = g.state(next(n for n in nodes if n.paths[0].pid != ref))
            self.scene().addItem(AlignItem(self, nodes, r, _("Align images"), _("on {name}").format(name=name), state,
                                           sel is not None and any(n is sel for n in nodes)))
            if any(n is sel for n in nodes):
                self._sel_item = None

    def _add_project(self, g: Graph, sel, cx: dict, cw: dict, gh_top: float):
        """The Project block, above the first object (the blank page: above its start zone); its wire to Astrometry only
        while one of the two is selected."""
        s = self.s
        first = [n.y for n in g.nodes if n.key in ("sources", "session", "image") and getattr(n, "y", None) is not None
                 and (n.paths or n.key == "session" or s.p.sessions)]
        x, w = cx.get("sources", 0.0), cw.get("sources", BW)
        y = (min(first) - BH / 2 if first and (s.paths or s.p.sessions) else gh_top) - BH - 14
        st = s.target_state()
        mine = self.project_selected()
        it = ProjectItem(self, QRectF(x, y, w, BH), st.name or _("Project"), s.name, st.dot, mine,
                         _("The project: target, comments, capture data") + "\n" + st.note)
        self.scene().addItem(it)
        self.project_item = it
        for n in g.nodes:
            if kbase(n.key) == "astrometrie" and n.paths and (mine or n is sel):
                self._project_wire(it, n)

    def _project_wire(self, it: ProjectItem, n: Node):
        x1, y1 = it.rect.right(), it.rect.center().y()
        x2, y2 = n.col + n.w / 2, n.y - BH / 2
        path = QPainterPath(QPointF(x1, y1))
        path.cubicTo(QPointF(x2, y1), QPointF(x2, y1), QPointF(x2, y2))
        wire = self.scene().addPath(path, QPen(QColor(ACCENT), 1.5, Qt.DashLine))
        wire.setZValue(2)
        wire.setToolTip(_("Astrometry starts from the target of the project"))

    def _branch_anchor(self, g: Graph, p, key: str) -> tuple[Node | None, bool]:
        """The block a new branch's place is drawn under, and whether a whole row is open there: a Calibration beside
        a night's own (its lowest), any other under the block it follows; ``None`` for what is not drawn there."""
        if key == "sources":
            return g.node_of(p, "calibration"), False
        if key.startswith("session@"):
            sid = int(key.partition("@")[2])
            cn = next((c for sn, c in g.sessions if sn.sid == sid), None)
            forks = g.calib_forks.get(sid, [])
            return (forks[-1] if forks else cn), True
        if key == "calibration" and g.sessions:           # the frames of every night: under the last Calibration
            sn, cn = g.sessions[-1]
            return (g.calib_forks.get(sn.sid) or [cn])[-1], False
        n = g.node_of(p, key)
        return (None if n is not None and n.sid is not None else n), False

    def _flows(self) -> list:
        """The workflows a dragged palette block can go in: all of them, the one the palette says ("for …") first: its
        places are the ones always shown (a large project would light a hundred at once; GUI exploration 4, Q11)."""
        p = self.palette_flow()
        return [p] + [q for q in self.s.paths if q is not p] if p is not None and p in self.s.paths else list(self.s.paths)

    def _drop_targets(self, g: Graph, cols: list[str]) -> list[tuple[int, int, str]]:
        """(workflow, index in its graph, the column it opens before) of each place the dragged block can go: before
        the block it would go in front of; after the last block, before the column that follows it."""
        s, b = self.s, self.drag_block
        out = []
        for p in self._flows():
            graph = s.graph(p)
            shown = [(j, g.node_of(p, node_key(n))) for j, n in enumerate(graph)]
            shown = [(j, nd) for j, nd in shown if nd is not None and nd.sid is None]
            for idx in s.place_indices(p, b):
                after = next((nd for j, nd in shown if j >= idx), None)
                before = next((nd for j, nd in reversed(shown) if j < idx), None)
                if after is not None:
                    c = self.col_of(after)
                elif before is not None:
                    i = cols.index(self.col_of(before)) if self.col_of(before) in cols else len(cols) - 2
                    c = cols[min(i + 1, len(cols) - 1)]
                else:
                    c = cols[1] if len(cols) > 1 else "final"
                out.append((p.pid, idx, c))
        if s.p.sessions and not any(q.source is None for q in s.paths) and b in wfl.IO and \
                wfl.palette_problem([], "frames", b) is None:     # sessions without a workflow: the block starts one
            i = cols.index("sources") + 1 if "sources" in cols else 0
            out.append((NEW_FRAMES, 0, cols[min(i, len(cols) - 1)]))
        return out

    def show_wait(self):
        """The line over the graph while the worker reads the project (or says why it could not)."""
        s = self.s
        if s.error:
            text = _("Project unreadable: {why}").format(why=s.error)
        else:
            text = "" if s.ready else _("Reading the project…")
        self.wait.setText(text)
        self.wait.setVisible(bool(text))
        loading = not s.error and not s.ready
        self.veil.setVisible(loading)
        self.spinner.setVisible(loading)
        if loading != getattr(self, "_loading", False):  # the system's busy cursor, as long as it lasts
            self._loading = loading
            if loading:
                self.viewport().setCursor(Qt.BusyCursor)
            else:
                self.viewport().unsetCursor()
        self._place_wait()

    def _place_wait(self):
        self.veil.setGeometry(self.rect())
        self.spinner.move((self.width() - self.spinner.width()) // 2, self.height() // 2 - 90)
        self.wait.setGeometry(0, self.height() // 2 - 40, self.width(), 30)
        self.veil.raise_()
        self.spinner.raise_()
        self.wait.raise_()

    # ------------------------------------------------------------ pulling an output to the post
    def start_wire(self, item, port=None):
        """A port is pulled: its wire follows the mouse; a Stars block's port shows where a Stars and background block
        can go (the Composition section, on its workflow's lane)."""
        self._pull = item
        self._pull_port = port
        self._pull_path = self.scene().addPath(QPainterPath(), QPen(QColor(ACCENT), 2, Qt.DashLine))
        self._pull_path.setZValue(9)
        self.grabKeyboard()                              # Escape cancels, wherever the focus was (keyPressEvent)
        self._comp_ghost = None
        if item.node.key == "etoiles" and "composition" in getattr(self, "_cx", {}):
            x, w = self._cx["composition"], self._cw["composition"]
            y = item.node.y - BH / 2
            self._comp_ghost = GhostItem(QRectF(x + 4, y + 4, w - 8, BH - 8), _("Combine layers"), role="add")
            self._comp_ghost.setZValue(8)
            self.scene().addItem(self._comp_ghost)

    def _on_post(self, pos):
        fi = self.final_item
        return fi is not None and fi.sceneBoundingRect().adjusted(-16, -16, 16, 16).contains(pos)

    def _on_composition(self, pos):
        g = self._comp_ghost
        return g is not None and g.rect.adjusted(-12, -12, 12, 12).contains(pos)

    def move_wire(self, pos):
        y = next((py for what, py in self._pull.ports if what == self._pull_port), BH / 2)
        a = self._pull.mapToScene(QPointF(self._pull.w, y))
        path = QPainterPath(a)
        dx = max(30.0, (pos.x() - a.x()) * 0.5)
        path.cubicTo(QPointF(a.x() + dx, a.y()), QPointF(pos.x() - dx, pos.y()), pos)
        self._pull_path.setPath(path)
        hot = self._on_post(pos)
        if self.final_item.target != hot:
            self.final_item.target = hot
            self.final_item.update()
        if self._comp_ghost is not None and self._comp_ghost.hot != self._on_composition(pos):
            self._comp_ghost.hot = not self._comp_ghost.hot
            self._comp_ghost.update()
        self._wiring = True
        self._edge_scroll(self.mapFromScene(pos))         # held near the edge, the view scrolls (Post off view)

    def end_wire(self, pos):
        self.releaseKeyboard()
        self._wiring = False
        self._edge_scroll(None)
        node, port = self._pull.node, self._pull_port
        self.scene().removeItem(self._pull_path)
        self.final_item.target = False
        self.final_item.update()
        comp = self._on_composition(pos)
        if self._comp_ghost is not None:
            self.scene().removeItem(self._comp_ghost)
            self._comp_ghost = None
        if self._on_post(pos):
            if port is None:
                QTimer.singleShot(0, lambda: self.sendToPost.emit(node))
            else:
                QTimer.singleShot(0, lambda: self.partToPost.emit(node, port))
        elif comp:
            QTimer.singleShot(0, lambda: self.newComposition.emit(node))

    def event(self, e):                                               # noqa: N802
        if e.type() == QEvent.ShortcutOverride and e.key() == Qt.Key_Escape and self._pulling():
            e.accept()                                   # Escape is the pull's, not a shortcut's
            return True
        if e.type() == QEvent.ShortcutOverride and e.key() == Qt.Key_1 and e.modifiers() == Qt.ControlModifier:
            e.accept()                                   # here Ctrl+1 is the readable size (issue #44), not the
            return True                                  # window's "Graph" (a shortcut of both would do neither)
        return super().event(e)

    def keyPressEvent(self, e):                                       # noqa: N802
        if e.key() == Qt.Key_Escape and self.cancel_wire():
            e.accept()
            return
        if e.key() == Qt.Key_1 and e.modifiers() == Qt.ControlModifier:
            self.zoom_to(1.0)
            e.accept()
            return
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = True
            self.viewport().setCursor(Qt.OpenHandCursor)
            e.accept()
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):                                     # noqa: N802
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            self.viewport().unsetCursor()
            e.accept()
            return
        super().keyReleaseEvent(e)

    def wheelEvent(self, e):                                          # noqa: N802
        """As the Post image: the wheel zooms at the pointer; two fingers on a touchpad pan, a pinch zooms."""
        if self.scene().sceneRect().isEmpty():
            return
        kind, a, b = wheel_gesture(e)
        if kind == "zoom":
            if a:
                self.zoom_by(ZOOM_STEP ** a, e.position())
        else:
            self.pan_by(a, b)
        e.accept()

    def viewportEvent(self, e):                                       # noqa: N802
        f = pinch_factor(e)
        if f is not None:                                # a native pinch (macOS, Wayland)
            self.zoom_by(f, e.position())
            return True
        return super().viewportEvent(e)

    def _pulling(self) -> bool:
        return getattr(getattr(self, "_pull", None), "_wire", False)

    def cancel_wire(self) -> bool:
        """Escape while a port is pulled: the wire goes and its release sends nothing."""
        item = getattr(self, "_pull", None)
        if item is None or not getattr(item, "_wire", False):
            return False
        item._wire = False
        self.releaseKeyboard()
        self._wiring = False
        self._edge_scroll(None)
        self.scene().removeItem(self._pull_path)
        self.final_item.target = False
        self.final_item.update()
        if self._comp_ghost is not None:
            self.scene().removeItem(self._comp_ghost)
            self._comp_ghost = None
        return True

    def _keep_selected(self, it, key):
        """A wire drawn again (the graph is rebuilt at each change) stays selected when it was (2nd round)."""
        it.key = key
        if key == getattr(self, "_wire_key", None):
            self.sel_wire = it
            it.selected = True
            it._look(True)

    def select_wire(self, w):
        """A removable wire selected (issue #52); a block's selection goes."""
        self._wire_key = getattr(w, "key", None)
        old = getattr(self, "sel_wire", None)
        if old is not None and old is not w:
            try:
                old.selected = False
                old._look(False)
            except RuntimeError:                          # rebuilt meanwhile
                pass
        self.sel_wire = w
        if w is not None:
            w.selected = True
            w._look(True)
            self.deselected.emit()                        # one thing selected at a time, as in editors (2nd round)
            self.wireSelected.emit(w)

    def open_layer(self, node):
        """Post opened on a Combine layers block's layer; the one Post gives by default made a layer first, without an
        undo step (decision 7)."""
        lid = node.lid
        if lid is None and node.paths and hasattr(self.s, "default_post_layer"):
            lid = self.s.default_post_layer(node.paths[0].pid)
        if lid is not None:
            self.openLayer.emit(lid)

    def clear_selection(self):
        """Nothing selected (decision 5): an empty click or Escape; Del and D then do nothing."""
        self.select_wire(None)
        self.deselected.emit()

    def mousePressEvent(self, e):                                     # noqa: N802
        pos = e.position().toPoint()
        left = e.button() == Qt.LeftButton
        it = self.itemAt(pos) if left else None
        empty = left and not isinstance(it, (BlockItem, WireItem, GhostItem, LinkItem, ProjectItem, AlignItem))  # bands, chain wires
        # issue #44: the middle button, Space + drag, or a left drag on an empty area moves the graph; a press on a
        # chain wire says what it says, a left click on a block still selects it
        if e.button() == Qt.MiddleButton or (left and (self._space or (empty and not isinstance(it, ChainWireItem)))):
            self._pan = (pos, self.horizontalScrollBar().value(), self.verticalScrollBar().value(), False,
                         empty and not self._space)
            self.viewport().setCursor(Qt.ClosedHandCursor)
            e.accept()
            return
        if empty:                                         # a chain wire: it selects nothing, as an empty click
            self.clear_selection()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):                                      # noqa: N802
        if self._pan is not None:
            start, h0, v0, moved, click = self._pan
            d = e.position().toPoint() - start
            if not moved and d.manhattanLength() >= QApplication.startDragDistance():
                moved = True
            self.horizontalScrollBar().setValue(h0 - d.x())
            self.verticalScrollBar().setValue(v0 - d.y())
            self._pan = (start, h0, v0, moved, click)
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):                                   # noqa: N802
        if self._pan is not None:
            _start, _h, _v, moved, click = self._pan
            self._pan = None
            if self._space:
                self.viewport().setCursor(Qt.OpenHandCursor)
            else:
                self.viewport().unsetCursor()
            if moved:
                self._now_manual()
            elif click:                                   # a click on the empty area: nothing selected (decision 5)
                self.clear_selection()
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def remove_selected_wire(self) -> bool:
        w = getattr(self, "sel_wire", None)
        if w is None or w.scene() is not self.scene():
            return False
        self.sel_wire = self._wire_key = None
        w.on_remove()
        return True

    def _chain_wire(self, path, pen, tip: str = "") -> ChainWireItem:
        """A wire of the chain in the scene (issue #52: it says it cannot be removed)."""
        it = ChainWireItem(self, path, pen, tip)
        self.scene().addItem(it)
        return it

    def _wire(self, a: Node, b: Node, strong: bool, on_remove=None, tip: str = "", pid: int | None = None,
              via_y: float | None = None, waiting: bool = False):
        """A wire from a block to the next block of its workflows. A branch bends in the gap right after the block it
        leaves (the last one its workflows share), then runs straight on its own lane past the columns it skips."""
        x1, y1 = a.col + a.w, a.y
        x2, y2 = b.col, b.y
        path = QPainterPath(QPointF(x1, y1))
        if via_y is not None and x2 - x1 > 4 * BEND:  # a workflow's image to Post: on its own lane past the blocks
            if abs(via_y - y1) > 1:                   # of the others, bent only at both ends (control round: it
                path.cubicTo(QPointF(x1 + BEND / 2, y1), QPointF(x1 + BEND / 2, via_y), QPointF(x1 + BEND, via_y))
            path.lineTo(QPointF(x2 - BEND, via_y))    # crossed the Combine layers blocks on its way)
            path.cubicTo(QPointF(x2 - BEND / 2, via_y), QPointF(x2 - BEND / 2, y2), QPointF(x2, y2))
            dx = xb = None
        else:
            xb = min(x2, self._next_x.get(a.key, x2)) if b.key != "final" else x2     # where the bend ends
            dx = max(30.0, (xb - x1) * 0.5) if abs(y2 - y1) > 1 else 0
        if dx:
            path.cubicTo(QPointF(x1 + dx, y1), QPointF(xb - dx, y2), QPointF(xb, y2))
        if xb is not None and (xb < x2 or not dx):
            path.lineTo(QPointF(x2, y2))
        pen = QPen(QColor("#9aa3b2" if strong else "#3d4452"), 2.2 if strong else 1.6)
        if waiting:                                        # its image comes to Post with the lights
            self._chain_wire(path, QPen(QColor("#6b7382"), 1.6, Qt.DashLine), tip).setZValue(1)
            return
        if on_remove is not None:                          # issue #52: a link to Post the user can take away
            it = WireItem(self, path, pen, on_remove, tip, pid)
            self.scene().addItem(it)
            self._keep_selected(it, ("wf", pid, a.key))
        else:
            it = self._chain_wire(path, pen)
        it.setZValue(1)

    def _join(self, sources: list[Node], targets: list[Node], cur):
        """The sessions' Calibration blocks into the first blocks of the chain (issue #43): a wire from each to one
        vertical bar in the gap before them, then one wire from the bar to each first block."""
        if not sources or not targets:
            return
        x0 = max(n.col + n.w for n in sources)
        x2 = min(n.col for n in targets)
        xb = x0 + max(6.0, (x2 - x0) / 2)
        pen = QPen(QColor("#9aa3b2"), 2.0)
        ys = [n.y for n in sources] + [n.y for n in targets]
        for n in sources:
            path = QPainterPath(QPointF(n.col + n.w, n.y))
            path.lineTo(QPointF(xb, n.y))
            self._chain_wire(path, pen).setZValue(1)
        bar = self._chain_wire(_line(xb, min(ys), xb, max(ys)), QPen(QColor("#9aa3b2"), 3.0),
                               _("The sessions join here: they are registered and stacked together"))
        bar.setZValue(1)
        for t in targets:
            strong = cur is not None and cur in t.paths
            path = QPainterPath(QPointF(xb, t.y))
            path.lineTo(QPointF(t.col, t.y))
            self._chain_wire(path, QPen(QColor("#9aa3b2" if strong else "#3d4452"), 2.2 if strong else 1.6)) \
                .setZValue(1)

    def _image_wire(self, a: Node, ya: float, b: Node, yb: float, what: str, word: str, strong: bool,
                    direct: bool = False):
        """A wire of one of the Stars block's images, from its port to a Stars and background block or straight to
        Post (dashed), in the image's colour: it arcs over (the starless image) or under (the stars) the blocks on its
        way; its name in its tooltip."""
        above = what == "starless"
        x1, y1 = a.col + a.w, a.y - BH / 2 + ya
        x2, y2 = b.col, b.y - BH / 2 + yb
        path = QPainterPath(QPointF(x1, y1))
        if x2 - x1 < 80:                                   # the next block (#17: no Output block between): straight
            mid = (x1 + x2) / 2
            path.cubicTo(QPointF(mid, y1), QPointF(mid, y2), QPointF(x2, y2))
        else:
            bend = (min(y1, y2) - BH / 2 - 8) if above else (max(y1, y2) + BH / 2 + 8)
            path.cubicTo(QPointF(x1 + 40, bend), QPointF(x2 - 40, bend), QPointF(x2, y2))
        col = QColor(IMAGE_COL[what])
        if not strong:
            col.setAlpha(110)
        pen = QPen(col, 2.0 if strong else 1.5, Qt.DashLine if direct else Qt.SolidLine)
        if direct:                                         # issue #52: sent to Post as it is: it can be taken away
            it = WireItem(self, path, pen, lambda: self.partOutOfPost.emit(a, what),
                          _("{image} → Post").format(image=word))
            it.part = (a.paths[0].pid if a.paths else None, what)     # the layer it makes (its panel)
            self.scene().addItem(it)
            self._keep_selected(it, ("part", a.paths[0].pid if a.paths else None, what))
        else:
            it = self._chain_wire(path, pen, word)
        it.setZValue(4 if direct else 2)                  # behind the blocks (their text stays readable)

    def _post_sources(self, g: Graph) -> list[Node]:
        """The blocks whose image is a layer of Post (``project.post["layers"]``, image layers: the key Inspect gave,
        ``services/inspect.post_key``), each once."""
        import json
        layers = ((getattr(self.s.p, "post", None) or {}).get("layers") or []) if self.s.p is not None else []
        by_stage = {st: b["key"] for b in STEPS for st in b["stages"]}
        out = []
        for lay in layers:
            key = str(lay.get("key") or "") if isinstance(lay, dict) and lay.get("kind") == "image" else ""
            wid, stage, skey = None, None, None
            if key.startswith("inspect:"):
                try:
                    d = json.loads(key[len("inspect:"):])
                except ValueError:
                    continue
                wid = d.get("wid")
                stage = (d.get("stages") or [None])[0] or ("calibrate" if d.get("kind") == "calframe" else None)
            elif key and not key.startswith("file:") and key.count("/") >= 2:
                stage, skey = key.split("/")[:2]
            block = by_stage.get(stage)
            if block is None:
                continue
            paths = [q for q in self.s.paths if wid is None or q.pid == wid]
            if skey is not None and stage != "calibrate":       # the workflow whose result it is
                paths = [q for q in paths if self.s.stage_key(q, stage) == skey] or paths
            n = next((g.node_of(q, block) for q in paths if g.node_of(q, block) is not None), None)
            if n is not None and n not in out:
                out.append(n)
        return out

    def _post_wire(self, a: Node, b: Node):
        """An image of this block sent to Post: a dashed arrow from the block's top over the graph to Post."""
        x1, y1 = a.col + a.w / 2, a.y - BH / 2
        x2, y2 = b.col + b.w / 2, b.y - BH / 2
        high = min(y1, y2) - 30
        path = QPainterPath(QPointF(x1, y1))
        path.cubicTo(QPointF(x1, high), QPointF(x2, high), QPointF(x2, y2))
        col = QColor("#c9a227")
        it = self.scene().addPath(path, QPen(col, 1.6, Qt.DashLine))
        it.setZValue(2)
        it.setToolTip(_("An image of {block} is a layer of Post").format(block=block_title(a.paths[0] if a.paths
                                                                                          else None, a.key)))
        head = QPainterPath(QPointF(x2, y2))                     # the arrow's head, pointing down into Post
        head.lineTo(QPointF(x2 - 5, y2 - 9))
        head.lineTo(QPointF(x2 + 5, y2 - 9))
        head.closeSubpath()
        h = self.scene().addPath(head, QPen(col, 1.0), col)
        h.setZValue(2)

    def _side_wire(self, a: Node, b: Node, lab: str):
        x1, y1 = a.col + a.w / 2, a.y + BH / 2
        x2, y2 = b.col + b.w / 2, b.y + BH / 2
        low = max(y1, y2) + 34
        path = QPainterPath(QPointF(x1, y1))
        path.cubicTo(QPointF(x1, low), QPointF(x2, low), QPointF(x2, y2))
        it = self.scene().addPath(path, QPen(QColor(ACCENT), 1.5, Qt.DashLine))
        it.setZValue(2)
        t = self.scene().addSimpleText(lab, QFont("Segoe UI", 8))
        t.setBrush(QColor(ACCENT))
        m = path.pointAtPercent(0.5)
        t.setPos(m.x() - t.boundingRect().width() / 2, m.y() + 2)
        t.setZValue(3)

    def _show_all(self, on: bool):
        self._keep_show_all(on)
        self._zoomed = False                              # today's view (issue #44: Show all is F / Ctrl+0)
        self._view_timer.stop()
        self._write_view()
        self.fit()

    def fit_all(self):
        """F, Ctrl+0: the whole graph (today's Show all)."""
        if self.b_all.isChecked():
            self._show_all(True)
        else:
            self.b_all.setChecked(True)

    def fit(self):
        """The default view shows the whole chain, from Frames to Post, at any window width: the scale that fits (at most
        1.3), lowered down to MIN_SCALE; when even that does not fit, the left end (Frames) is shown, top left, and the
        user scrolls right (or down). It is done again whenever the view changes size (the window, the panel)."""
        r = self.scene().sceneRect()
        if r.isEmpty() or self._fitting:
            return
        if self._zoomed:                                 # the user's zoom and position: kept (issue #44)
            self._apply_view()
            return
        self._fitting = True
        try:
            vw, vh = self.viewport().width(), self.viewport().height()
            k = min(1.3, (vw - 4) / r.width(), (vh - 10) / r.height())
            if not self.b_all.isChecked():
                k = max(k, MIN_SCALE)
            self.resetTransform()
            self.scale(k, k)
            # the middle of the view, in the scene: the scene's own middle when it fits, else its left / top edge at the
            # view's edge
            x = r.center().x() if r.width() * k <= vw else r.left() + vw / (2 * k)
            y = r.center().y() if r.height() * k <= vh else r.top() + vh / (2 * k)
            self.centerOn(QPointF(x, y))
            it = self._sel_item
            if it is not None and it.scene() is self.scene() and r.height() * k > vh:
                seen = self.mapToScene(self.viewport().rect()).boundingRect()
                box = it.sceneBoundingRect()                    # the chosen lane stays in view when the graph scrolls down
                self.ensureVisible(QRectF(seen.left(), box.top(), 1, box.height()), 0, 40)
        finally:
            self._fitting = False
        self._after_view()

    # ------------------------------------------------------------ zoom and position (issue #44)
    def zoom(self) -> float:
        """The scale shown: 1.0 is the blocks at their designed size (100 %)."""
        return self.transform().m11()

    def _apply_view(self):
        """The user's zoom and centre, on the view as it is now."""
        self._fitting = True
        try:
            self.resetTransform()
            self.scale(self._k, self._k)
            self.centerOn(self._c)
        finally:
            self._fitting = False
        self._after_view()

    def _manual(self, k: float, c: QPointF):
        """From now on the view is the user's: ``k`` around ``c``; "Show all" goes off."""
        self._zoomed = True
        self._k, self._c = k, c
        if self.b_all.isChecked():
            self.b_all.blockSignals(True)
            self.b_all.setChecked(False)
            self.b_all.blockSignals(False)
            self._keep_show_all(False)
        self._apply_view()
        self._store_view()

    def _now_manual(self):
        """The view moved by hand (a pan, a scroll bar): what it shows becomes the user's view."""
        vp = self.viewport().rect()
        self._manual(self.zoom(), self.mapToScene(vp.center()))

    def _keep_show_all(self, on: bool):
        p = getattr(self.s, "p", None)
        if p is not None and hasattr(p, "show_all"):
            p.show_all = bool(on)

    def zoom_to(self, k: float, at: QPointF | None = None):
        """The zoom ``k`` (1.0 is 100 %), the scene point under ``at`` (default: the view's centre) kept in place."""
        vp = self.viewport()
        mid = QPointF(vp.width() / 2, vp.height() / 2)
        at = at if at is not None else mid
        k = min(max(k, MIN_ZOOM), MAX_ZOOM)
        pt = self.mapToScene(at.toPoint())
        self._manual(k, QPointF(pt + (mid - at) / k))

    def zoom_by(self, f: float, at: QPointF | None = None):
        self.zoom_to(self.zoom() * f, at)

    def pan_by(self, dx: float, dy: float):
        """The graph moved by (dx, dy) pixels of the view (two fingers on a touchpad)."""
        h, v = self.horizontalScrollBar(), self.verticalScrollBar()
        h.setValue(h.value() - int(dx))
        v.setValue(v.value() - int(dy))
        self._now_manual()

    def center_on(self, c: QPointF):
        """The view centred on this scene point (the minimap); the zoom kept."""
        self._manual(self.zoom(), c)

    def reveal(self, it):
        """Scrolls the view so that the block is seen, only when it is off-screen; the zoom is kept."""
        try:
            if it is None or it.scene() is not self.scene():
                return
        except RuntimeError:                              # the item went with the scene
            return
        if self.viewport().rect().contains(self.mapFromScene(it.sceneBoundingRect()).boundingRect()):
            return
        self.ensureVisible(it.sceneBoundingRect(), 30, 30)
        if self._zoomed:
            self._store_view()

    # the user's zoom and position: kept per project in the settings (as the block shown, ProjectModel.remember_selection:
    # opening a project and looking at it writes nothing into it)
    GRAPH_VIEW_KEY = "graph_view"

    def _view_key(self) -> str | None:
        root = getattr(self.s, "root", None)
        return Path(root).as_posix().lower() if root is not None else None

    def _saved_view(self):
        key = self._view_key()
        got = (prefs.get(self.GRAPH_VIEW_KEY) or {}).get(key) if key else None
        if not isinstance(got, dict) or "k" not in got:
            return None
        return float(got["k"]), QPointF(float(got.get("x", 0.0)), float(got.get("y", 0.0)))

    def _store_view(self):
        """The user's view changed: taken, and written to the settings a moment later."""
        if self._rebuilding or self._fitting or not self._zoomed:
            return
        self._k, self._c = self.zoom(), self.mapToScene(self.viewport().rect().center())
        self._view_timer.start()

    def _write_view(self):
        key = self._view_key()
        if key is None:
            return
        cur = dict(prefs.get(self.GRAPH_VIEW_KEY) or {})
        cur.pop(key, None)
        if self._zoomed:
            cur[key] = {"k": round(self._k, 4), "x": round(self._c.x(), 1), "y": round(self._c.y(), 1)}
        prefs.set(self.GRAPH_VIEW_KEY, dict(list(cur.items())[-50:]))   # the oldest go beyond 50 projects

    def hideEvent(self, e):                                           # noqa: N802
        if self._view_timer.isActive():                  # closing the project or leaving the page: not lost
            self._view_timer.stop()
            self._write_view()
        super().hideEvent(e)

    def _scrolled(self):
        if self._rebuilding:
            return
        self._place_post_button()
        if self._zoomed and not self._fitting:
            self._store_view()
        self._place_map()

    def _after_view(self):
        self._place_post_button()
        k = self.zoom()
        self.b_zoom.setText(number(k * 100, 0 if k >= 0.1 else 1) + " %")
        self._place_map()

    def _whole_visible(self) -> bool:
        return self.horizontalScrollBar().maximum() <= 0 and self.verticalScrollBar().maximum() <= 0

    def _place_map(self):
        """The minimap in the view's corner, while part of the graph is out of view (as the Post image's)."""
        m = getattr(self, "map", None)
        if m is None:
            return
        whole = self._whole_visible() or self.scene().sceneRect().isEmpty()
        m.setVisible(not whole)
        if not whole:
            m.place()
            m.update()

    def _place_post_button(self):
        """"Post ›" at the view's top right corner, under "Show all", while the Post block is out of view (hidden while a
        block is dragged: the drag scrolls the view itself)."""
        b, fi = getattr(self, "b_post", None), self.final_item
        if b is None:
            return
        try:
            gone = fi is None or fi.scene() is not self.scene()
        except RuntimeError:                              # the item was deleted with the scene
            gone = True
        vp = self.viewport().rect()
        box = self.mapFromScene(fi.sceneBoundingRect()).boundingRect() if not gone else None
        if gone or self.drag_block is not None or vp.contains(box):
            b.hide()
            return
        b.setText(_("Post"))
        set_icon(b, "forward", 16)
        b.setLayoutDirection(Qt.RightToLeft)              # its arrow after its name
        b.setToolTip(_("Post is further on: show it"))
        b.adjustSize()
        b.move(self.width() - b.width() - 14, self.b_all.geometry().bottom() + 6)   # under "Show all", over no block
        b.raise_()
        b.show()

    def show_post(self):
        """Scrolls the view to the Post block."""
        if self.final_item is not None:
            self.ensureVisible(self.final_item, 30, 30)

    def _edge_scroll(self, pos):
        """A drag at ``pos`` (in the view): held near the view's edge (after a short moment, so that a drag only
        crossing the edge does not move the view), the view scrolls that way, faster nearer the edge, as reference
        editors do (GUI exploration 2: at 1366 x 768 most places, and Post, were out of view). None: the drag ended."""
        was = self._edge_pos
        self._edge_pos = pos
        if pos is None or self._edge_delta(pos) == (0, 0):
            self._edge_timer.stop()
            return
        if was is None or self._edge_delta(was) == (0, 0):
            self._edge_since = time.monotonic()
        if not self._edge_timer.isActive():
            self._edge_timer.start()

    def _edge_delta(self, pos) -> tuple[int, int]:
        vp = self.viewport().rect()

        def step(v, size):
            if v < EDGE:
                return -int(4 + 16 * min(1.0, (EDGE - v) / EDGE))
            if v > size - EDGE:
                return int(4 + 16 * min(1.0, (v - size + EDGE) / EDGE))
            return 0
        return step(pos.x(), vp.width()), step(pos.y(), vp.height())

    def _edge_tick(self):
        pos = self._edge_pos
        if pos is None:
            self._edge_timer.stop()
            return
        if time.monotonic() - self._edge_since < EDGE_WAIT:
            return
        dx, dy = self._edge_delta(pos)
        hb, vb = self.horizontalScrollBar(), self.verticalScrollBar()
        before = (hb.value(), vb.value())
        hb.setValue(hb.value() + dx)
        vb.setValue(vb.value() + dy)
        if (hb.value(), vb.value()) == before:
            return                                        # at the end: nothing more to show that way
        if self._wiring:                                  # what is under the cursor now: lit as it would be by a move
            self.move_wire(self.mapToScene(pos))
        else:
            self._light_ghost(pos)

    def _light_ghost(self, pos):
        """The place under ``pos`` (in the view) lit, the others not, another workflow's shown on its row; the place
        (or None)."""
        hit = self._ghost_at(pos)
        y = self.mapToScene(pos).y()
        for g in self.ghosts:
            if g.far:
                g.setVisible(abs(g.rect.center().y() - y) <= LANE / 2)
            if g.hot != (g is hit):
                g.hot = g is hit
                g.update()
        return hit

    def resizeEvent(self, e):                                         # noqa: N802
        super().resizeEvent(e)
        self.fit()
        self.toast.move(16, self.height() - self.toast.height() - 16)
        self.b_all.adjustSize()
        self.b_all.move(self.width() - self.b_all.width() - 14, 10)
        self.b_zoom.adjustSize()
        self.b_zoom.move(self.b_all.x() - self.b_zoom.width() - 6, 10)
        self.b_zoom.raise_()
        self._place_wait()
        self._place_post_button()

    def show_toast(self, text):
        self.toast.setText(text)
        self.toast.adjustSize()
        self.toast.move(16, self.height() - self.toast.height() - 16)
        self.toast.show()
        self.toast.raise_()
        self._toast_timer.start(5000)

    def can_exit(self, node) -> bool:
        """Whether a block's image can go to Post as a layer of its own: in a graph, a block giving an image before
        the last one (issue #47); in a workflow of options, Stacking."""
        if not node.paths or node.sid is not None:
            return False
        s = self.s
        if node.key == "image":                           # blocks after it: its own image as it is, as a layer (2nd round)
            return len(node.paths) == 1 and bool(s.graph(node.paths[0]))
        return any(s.can_send_to_post(q, node.key) for q in node.paths) or (
            node.key in EXITS and all(s.p.workflow(q.pid).blocks is None for q in node.paths))

    def has_port(self, node) -> bool:
        """Whether a block shows its port ● and "To Post": every block of a graph giving an image, also when its image
        is a layer already or it is its workflow's last one (control round: the port went away, a second pull or a
        pull after its wire was taken away did nothing, without a word; ``GraphPage.send_to_post`` says so, or gives
        the wire back)."""
        if self.can_exit(node):
            return True
        if not node.paths or node.sid is not None or node.cut or node.grey or \
                node.key in ("sources", "final", "session", "composition", "etoiles", "fusion", "image"):
            return False
        s = self.s
        q = node.paths[0]
        if "fusion" in (q.source or {}) or s.p.workflow(q.pid).blocks is None:
            return False
        i = s.node_index(q, node.key)
        g = s.graph(q)
        return i is not None and not g[i].get("off") and g[i]["block"] in wfl.IO and \
            wfl.IO[g[i]["block"]][1] == wfl.IMAGE

    # ------------------------------------------------------------ menus
    def block_menu(self, node: Node, pos):
        m = QMenu(self)
        if node.key == "final":                              # the Post block (2nd round): what its double-click does
            m.addAction(_("Open Post")).triggered.connect(lambda: self.postOpened.emit())
            m.exec(pos)
            return
        # "To Post" at one place in every block's menu: after Inspect, between separators (control round)
        to_post: list = []
        if node.key in ("image", "fusion") and node.paths:   # a source block: a block like the others
            p = node.paths[0]
            if self.graph is not None and p.pid not in self.graph.to_post and self.s.gives_post(p):
                to_post.append((_("To Post"), lambda _c=False, q=p: self.workflowMenu.emit("to_post", q.pid)))
            elif node.key == "fusion" or self.can_exit(node):   # question 3: an Image's own image, as other blocks;
                to_post.append((_("To Post"), lambda: self.sendToPost.emit(node)))     # in Post already: said
            m.addAction(_("Delete") + "\t" + _("Del")).triggered.connect(lambda: self.remove_block(node, None))
            if node.key == "image":
                for text, fn in to_post:
                    m.addSeparator()
                    m.addAction(text).triggered.connect(fn)
                m.exec(pos)
                return
        elif node.key == "etoiles" and len(node.paths) == 1:
            for what, text in (("starless", "Starless to Post"), ("stars", "Stars to Post")):
                if self.graph is not None and (node, what) in self.graph.direct:      # sent: it can be taken back (#29)
                    out = "Starless out of Post" if what == "starless" else "Stars out of Post"
                    to_post.append((_(out), lambda _c=False, w=what: self.partOutOfPost.emit(node, w)))
                else:
                    to_post.append((_(text), lambda _c=False, w=what: self.partToPost.emit(node, w)))
            to_post.append((_("Combine layers"), lambda: self.newComposition.emit(node), "add"))
        elif self.has_port(node) and self.graph is not None and any(
                self.graph.leaves.get(q.pid) is node and q.pid in self.graph.awaits_post for q in node.paths):
            to_post.append((_("To Post") + "  —  " + _("lights first"), None))        # said, not hidden
        elif self.has_port(node):
            to_post.append((_("To Post"), lambda: self.sendToPost.emit(node)))
        wf = None
        if node.key == "post" or node.handle:                # the workflow's handle (#17: its Stars block): its own
            p = node.paths[0]                                # titled submenu, apart from the block's actions
            wf = QMenu(_("Workflow “{name}”").format(name=p.name).replace("&", "&&"), m)
            wf.addAction(_("Rename")).triggered.connect(lambda: self.workflowMenu.emit("rename", p.pid))
            wf.addAction(_("Duplicate")).triggered.connect(lambda: self.workflowMenu.emit("duplicate", p.pid))
            wf.addAction(_("Save template…")).triggered.connect(lambda: self.workflowMenu.emit("template", p.pid))
            # its image to Post: the block's own "To Post" (control round: here too, a Stars/background workflow's
            # made a layer of the image before star removal named only after the workflow); a block without one
            if not to_post and node.key != "etoiles" and self.graph is not None and \
                    p.pid not in self.graph.to_post and self.s.gives_post(p):
                wf.addAction(_("To Post")).triggered.connect(
                    lambda: self.workflowMenu.emit("to_post", p.pid))
            wf.addSeparator()
            wf.addAction(_("Delete workflow")).triggered.connect(lambda: self.workflowMenu.emit("delete", p.pid))
        elif node.key == "composition":
            m.addAction(_("Open Post")).triggered.connect(lambda: self.open_layer(node))
            m.addSeparator()
            m.addAction(_("Delete") + "\t" + _("Del")).triggered.connect(lambda: self.remove_block(node, None))
            m.exec(pos)
            return
        if node.key not in ("sources", "final", "session", "composition", "fusion", "image") and \
                (node.sid is None or node.key == "calibration") and node.paths:   # #47, a session's Calibration (5.1)
            if node.shared:
                m.addAction(_("Delete in {n} workflows").format(n=len(node.paths))).triggered.connect(
                    lambda: self.remove_block(node, None))
                for p in node.paths:
                    m.addAction(_("Delete in {name}").format(name=p.name).replace("&", "&&")).triggered.connect(
                        lambda _x=False, q=p: self.remove_block(node, [q.pid]))
            else:
                m.addAction(_("Delete") + "\t" + _("Del")).triggered.connect(lambda: self.remove_block(node, None))
        if node.key == "session" or (node.key == "sources" and len(self.s.p.sessions) == 1):   # a lone Session too
            sid = node.sid if node.sid is not None else self.s.p.sessions[0].id                    # (5.2)
            m.addAction(_("Rename session")).triggered.connect(lambda: self.sessionMenu.emit("rename", sid))
            m.addAction(_("Delete") + "\t" + _("Del")).triggered.connect(lambda: self.sessionMenu.emit("remove", sid))
            m.addSeparator()
            m.addAction(_("Add session…")).triggered.connect(lambda: self.newSession.emit())
            m.exec(pos)
            return
        offs = {q.pid: self.s.can_switch_off(q, node.key) for q in node.paths} if node.paths and \
            (node.sid is None or node.key == "calibration") \
            and node.key not in ("sources", "final", "composition", "fusion", "image", "session") else {}
        why_off = next((w for w in offs.values() if w), "")    # every workflow, not only the first (3.7)
        if why_off and not node.cut:                      # issue #47: said, not hidden
            if not m.isEmpty():
                m.addSeparator()
            a = m.addAction((_("Switch off") + "  —  " + why_off).replace("&", "&&"))
            a.setEnabled(False)
        if (node.sid is None or node.key == "calibration") and node.paths and \
                node.key not in ("sources", "final", "composition", "fusion", "image") and (node.cut or not why_off):
            if not m.isEmpty():
                m.addSeparator()
            word = _("Switch on") if node.cut else _("Switch off")
            if len(node.paths) > 1:
                m.addAction(_("{action} in {n} workflows").format(action=word, n=len(node.paths))).triggered.connect(
                    lambda: self.cutToggled.emit(node, None))
                for p in node.paths:
                    m.addAction(_("{action} in {name}").format(action=word, name=p.name).replace("&", "&&")).triggered.connect(
                        lambda _x=False, q=p: self.cutToggled.emit(node, [q.pid]))
            else:
                a = m.addAction(word + "\tD")
                a.triggered.connect(lambda: self.cutToggled.emit(node, None))
        if node.key not in ("sources", "final"):
            if not m.isEmpty():
                m.addSeparator()
            m.addAction(_("Inspect")).triggered.connect(lambda: self.blockOpened.emit(node))
            if to_post:
                m.addSeparator()
            for text, fn, *role in to_post:
                a = m.addAction(text)
                if role:
                    a.setIcon(icon(role[0], 16))
                if fn is None:
                    a.setEnabled(False)
                else:
                    a.triggered.connect(fn)
            m.addSeparator()
            bp = self.graph.breakpoint(node) if self.graph is not None else None
            a = m.addAction(_("Review pause"))
            a.setCheckable(True)
            a.setChecked(bp is not None and bp[1] == bp[2])
            a.setToolTip(_("The run stops after this block, so that you can check it before it goes on."))
            a.triggered.connect(lambda: self.pauseToggled.emit(node))
            if bp is not None and bp[0] == "paused":
                m.addAction(_("Mark reviewed")).triggered.connect(lambda: self.reviewMarked.emit(node))
        if wf is not None:
            if not m.isEmpty():
                m.addSeparator()
            m.addMenu(wf)
        if not m.isEmpty():
            m.exec(pos)

    def drag_out(self, node: Node):
        """A placed block dragged (issue #54): dropped on the palette, it is removed (the same rules, Undo)."""
        self._dragged = node
        drag = QDrag(self)
        mime = QMimeData()
        mime.setText("placed:" + node.key)
        drag.setMimeData(mime)
        it = next((x for x in self.scene().items() if isinstance(x, BlockItem) and x.node is node), None)
        if it is not None:
            pm = QPixmap(int(it.boundingRect().width()), int(it.boundingRect().height()))
            pm.fill(Qt.transparent)
            pa = QPainter(pm)
            it.paint(pa, None)
            pa.end()
            drag.setPixmap(pm)
        drag.exec(Qt.MoveAction)
        self._dragged = None

    def remove_block(self, node: Node, pids):
        if node.key == "composition":
            self.layerRemoved.emit(node.lid)
            return
        self.removeTool.emit(node, pids)

    # ------------------------------------------------------------ drag and drop of tools
    def _tool(self, e):
        """The palette block a drag carries (its engine name; an older tool key is read as its block)."""
        t = e.mimeData().text() if e.mimeData().hasText() else ""
        if t.startswith("block:"):
            return t[6:]
        if t.startswith("tool:"):
            return {"debruitage": "denoise", "etoiles": "stars"}.get(t[5:], t[5:])
        return None

    def _files(self, e) -> list[str]:
        """The local files or folders a drag from the Explorer carries (decision 9)."""
        md = e.mimeData()
        if not md.hasUrls():
            return []
        return [u.toLocalFile() for u in md.urls() if u.isLocalFile() and u.toLocalFile()]

    def _session_at(self, pos) -> int | None:
        """The session of the Session block under ``pos`` (the lone Session too, drawn as the frames' block)."""
        sp = self.mapToScene(pos)
        for it in self.scene().items(sp):
            if isinstance(it, BlockItem):
                n = it.node
                if n.key == "session" and n.sid is not None:
                    return n.sid
                if n.key == "sources" and len(self.s.p.sessions) == 1:
                    return self.s.p.sessions[0].id
                return None
        return None

    def dragEnterEvent(self, e):                                      # noqa: N802
        if self._files(e) and self.s.p.sessions:         # files from the Explorer: onto a Session (decision 9)
            e.acceptProposedAction()
            return
        k = self._tool(e)
        if k in ("session", "image"):                    # a source starts a flow: the graph takes it
            self.drag_block = k
            self.rebuild()                               # its places light up (8.2)
            e.acceptProposedAction()
            return
        if k in ("fusion", "composition"):               # issue #53: in the Composition column (Combine layers: after a split)
            self.drag_block = k
            self.rebuild()
            e.acceptProposedAction()
            return
        if not k or k not in wfl.IO or not (self.s.paths or self.s.p.sessions):   # sessions alone: after them (8.3)
            e.ignore()
            return
        self.drag_block = k
        self.rebuild()
        e.acceptProposedAction()

    def _ghost_at(self, pos):
        sp = self.mapToScene(pos)
        for g in self.ghosts:
            if g.rect.adjusted(-10, -10, 10, 10).contains(sp):
                return g
        return None

    def _source_place(self, pos) -> str:
        """Where a Session or Image dragged would go (question 5, 8.2): "new" on the new-flow zone, "join" on
        "+ Session" (a Session joins the flow); "" elsewhere: refused."""
        at = self.mapToScene(pos)
        nf = getattr(self, "new_flow_ghost", None)
        if nf is not None and nf.scene() is self.scene() and nf.rect.adjusted(-10, -10, 10, 10).contains(at):
            return "new"
        ps = getattr(self, "plus_session_ghost", None)
        if self.drag_block == "session" and ps is not None and ps.scene() is self.scene() and \
                ps.rect.adjusted(-6, -6, 6, 6).contains(at):
            return "join"
        return ""

    def dragMoveEvent(self, e):                                       # noqa: N802
        self._edge_scroll(e.position().toPoint())        # near the edge: the places out of view come in
        if self._files(e):                               # files: taken over a Session block only
            if self._session_at(e.position().toPoint()) is not None:
                e.acceptProposedAction()
            else:
                e.ignore()
            return
        if self.drag_block in ("session", "image"):
            where = self._source_place(e.position().toPoint())    # its places are lit while it is dragged
            if where:
                e.acceptProposedAction()
            else:
                e.ignore()
            return
        hit = self._light_ghost(e.position().toPoint())
        if hit:
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragLeaveEvent(self, e):                                      # noqa: N802
        self._edge_scroll(None)
        self.drag_block = None
        self.rebuild()

    def dropEvent(self, e):                                           # noqa: N802
        self._edge_scroll(None)
        files = self._files(e)
        if files:                                        # decision 9: added to the Session dropped on
            sid = self._session_at(e.position().toPoint())
            if sid is None:
                e.ignore()
                return
            e.acceptProposedAction()
            QTimer.singleShot(0, lambda: self.filesDropped.emit(sid, files))
            return
        hit = self._ghost_at(e.position().toPoint())
        k = self.drag_block
        self.drag_block = None
        if k == "fusion":                                 # issue #53: on its place in the Composition column
            self.rebuild()
            if hit:
                e.acceptProposedAction()
                QTimer.singleShot(0, lambda: self.sourceDropped.emit("fusion"))
            return
        if k in ("session", "image"):
            self.drag_block = k
            where = self._source_place(e.position().toPoint())
            self.drag_block = None
            if not where:                                 # question 5: not on Post, a block or the void
                e.ignore()
                self.rebuild()
                return
            e.acceptProposedAction()
            QTimer.singleShot(0, lambda: self.sourceDropped.emit(k + (":new" if where == "new" else ":join")))
            return
        if hit and getattr(hit, "branch_key", None) and k:      # a new branch from that block
            e.acceptProposedAction()
            pid, key = hit.pid, hit.branch_key
            QTimer.singleShot(0, lambda: self.branchDropped.emit(k, pid, key))
            return
        if k == "composition":
            if hit:
                e.acceptProposedAction()
                split = hit.node
                QTimer.singleShot(0, lambda: self.newComposition.emit(split))
            else:
                self.rebuild()
            return
        if hit and k:
            e.acceptProposedAction()
            pid, idx = hit.pid, getattr(hit, "idx", None)
            QTimer.singleShot(0, lambda: self.blockDropped.emit(k, pid, -1 if idx is None else idx))
        else:
            self.rebuild()


class GraphMap(QWidget):
    """The graph's overview in the view's corner, as the Inspect and Post viewers' (issue #44): shown while part of the
    graph is out of view; a click or a drag in it moves the view there."""

    def __init__(self, view):
        super().__init__(view.viewport())
        self.view = view
        self.rows: list[tuple[QRectF, QColor]] = []       # each block: its rectangle in the scene, the colour of its state
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip(_("Overview: click or drag to move the view"))

    def _frame(self) -> tuple[QRectF, float]:
        """The map's rectangle in this widget, and pixels per scene unit."""
        r = self.view.scene().sceneRect()
        f = min(MAP_W / max(r.width(), 1.0), MAP_H / max(r.height(), 1.0))
        return QRectF(4, 4, r.width() * f, r.height() * f), f

    def place(self):
        m, _f = self._frame()
        vp = self.view.viewport().rect()
        self.setGeometry(vp.width() - int(m.width()) - 16, vp.height() - int(m.height()) - 16, int(m.width()) + 8,
                         int(m.height()) + 8)
        self.raise_()

    def paintEvent(self, e):                                         # noqa: N802
        m, f = self._frame()
        sr = self.view.scene().sceneRect()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(15, 17, 21, 220))
        p.drawRoundedRect(QRectF(self.rect()), 5, 5)

        def place(r: QRectF) -> QRectF:
            return QRectF(m.x() + (r.x() - sr.x()) * f, m.y() + (r.y() - sr.y()) * f, r.width() * f, r.height() * f)
        for r, col in self.rows:
            p.setBrush(col)
            p.drawRoundedRect(place(r), 1.5, 1.5)
        v = place(self.view.mapToScene(self.view.viewport().rect()).boundingRect()).intersected(m)
        p.save()
        p.setClipRect(m)
        out = QPainterPath()
        out.addRect(m)
        inner = QPainterPath()
        inner.addRect(v)
        p.setBrush(QColor(0, 0, 0, 110))
        p.drawPath(out.subtracted(inner))
        p.restore()
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(ACCENT), 1.5))
        p.drawRect(v)
        p.end()

    def _go(self, pos: QPointF):
        m, f = self._frame()
        sr = self.view.scene().sceneRect()
        self.view.center_on(QPointF(sr.x() + (pos.x() - m.x()) / f, sr.y() + (pos.y() - m.y()) / f))

    def mousePressEvent(self, e):                                    # noqa: N802
        if e.button() == Qt.LeftButton:
            self._go(e.position())
        e.accept()

    def mouseMoveEvent(self, e):                                     # noqa: N802
        if e.buttons() & Qt.LeftButton:
            self._go(e.position())
        e.accept()

    def wheelEvent(self, e):                                         # noqa: N802
        e.ignore()                                       # the wheel still zooms the graph


class ToolCard(QFrame):
    """A tool: dragged onto a workflow of the graph, or added to the current workflow by a double-click."""

    def __init__(self, page, key, state, text, new=False, parent=None):
        super().__init__(parent)
        self.page, self.key, self.state = page, key, state
        self._press = None
        on = state in ("in", "off")
        self.setObjectName("tool")
        self.setCursor(Qt.ArrowCursor if on else Qt.OpenHandCursor)
        self.setStyleSheet(f"QFrame#tool {{ background: {'transparent' if on else '#232831'}; border: 1px "
                           f"{'dashed #3a4150' if on else 'solid #353b47'}; border-radius: 6px; }}"
                           f"QFrame#tool:hover {{ border-color: {'#3a4150' if on else ACCENT}; }}"
                           " QLabel { background: transparent; }")
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 6, 8, 6)
        v.setSpacing(1)
        t = QLabel(("⋮⋮  " if not on else "") + block_title(None, TOOL_COLUMN[key]))
        t.setStyleSheet("font-weight: 600;" + (" color: #6b7280;" if on else ""))
        if new:
            t.setText(t.text() + f"  <span style='color:{NEW}'>◇</span>")
            t.setTextFormat(Qt.RichText)
        v.addWidget(t)
        v.addWidget(label(text, "small"))
        self.setToolTip(_(TOOLS[key]["what"]) + (("\n◇ " + _("Not in the engine yet.")) if new else ""))
        t.setMinimumWidth(t.sizeHint().width())          # a long name (#36) is never cut by a narrow bar
        self.setMinimumWidth(140)

    def mousePressEvent(self, e):                                     # noqa: N802
        self._press = e.position().toPoint()

    def mouseMoveEvent(self, e):                                      # noqa: N802
        if self._press is None or not (e.buttons() & Qt.LeftButton) or self.state != "add":
            return
        if (e.position().toPoint() - self._press).manhattanLength() < QApplication.startDragDistance():
            return
        self._press = None
        drag = QDrag(self)
        mime = QMimeData()
        mime.setText("tool:" + self.key)
        drag.setMimeData(mime)
        drag.setPixmap(self.grab())
        drag.setHotSpot(QPoint(20, 14))
        drag.exec(Qt.CopyAction)

    def mouseDoubleClickEvent(self, e):                               # noqa: N802
        if self.state == "add" and self.page.s.path is not None:
            self.page.add_tool(self.key, self.page.s.path.pid)


#: the palette's width (issue #47): a panel on the left of the graph; folded (issue #44), a column of icons
PALETTE_W = 140
PALETTE_FOLDED_W = 46
PALETTE_UNFOLD_MS, PALETTE_FOLD_MS = 200, 300   # automatic: the mouse only crossing it opens nothing, a slip closes nothing
NEW_FRAMES = -1        # a block placed "in the workflow from the frames" of sessions that have none yet: it starts one
#: the palette (issue #47): every block, by group; the sources and the composition blocks are not graph nodes. The
#: headings are one word, shown in capitals ("FRAMES": its own entry, "Frames" being the frames' block, "Images")
PALETTE_GROUPS = [("Sources", ["session", "image", "align_images"]),
                  ("FRAMES", ["calibration", "registration", "selection", "crop", "stacking"]),
                  ("Image", ["astrometry", "gradient", "deconv", "denoise", "color"]),
                  ("Post", ["stars", "synth"]),
                  ("Composition", ["fusion", "composition"])]
PALETTE_TITLES = {"session": "Session", "image": "Image", "fusion": "Combine channels", "align_images": "Align images"}


class PaletteCard(QFrame):
    """A block of the palette (issue #47): dragged onto a workflow (only its valid places light up), or placed in the
    current workflow at its usual place by a double-click. Greyed when it cannot be placed: its tooltip says why.
    ``branch``: it cannot be placed in the current workflow's chain, but can go elsewhere: available, by a drag only;
    the text says where. ``auto``: in the palette that unfolds by itself: the same card folded or not (``set_narrow``)."""

    def __init__(self, page, block: str, reason: str, parent=None, folded: bool = False, branch: str = "",
                 auto: bool = False):
        super().__init__(parent)
        self.page, self.block, self.reason, self.branch = page, block, reason, branch
        self._press = None
        # inside the palette, its scroll bar beside it; folded (issue #44): its icon alone, the name in the tooltip
        self.setMaximumWidth(PALETTE_FOLDED_W - 12 if folded else PALETTE_W - 26)
        on = not reason or branch
        self.setObjectName("tool")
        self.setCursor(Qt.ForbiddenCursor if not on else Qt.ArrowCursor if block == "align_images" else
                       Qt.OpenHandCursor)
        # available: a filled card; not available: the same card faint (a disabled tool), never dashed: dashed is
        # for the places a block can be dropped on
        self.setStyleSheet(f"QFrame#tool {{ background: {'#232831' if on else '#1e2128'}; border: 1px solid "
                           f"{'#353b47' if on else '#262a31'}; border-radius: 6px; }}"
                           f"QFrame#tool:hover {{ border-color: {ACCENT if on else '#3a4150'}; }}"
                           " QLabel { background: transparent; }")
        self.title = _(PALETTE_TITLES[block]) if block in PALETTE_TITLES else block_title(None, view_key(block))
        role = BLOCK_ROLES[block]
        if folded:
            v = QVBoxLayout(self)
            v.setContentsMargins(0, 6, 0, 6)
            v.addWidget(IconLabel(role, 20, "normal" if on else "disabled"), 0, Qt.AlignHCenter)
        else:
            v = QHBoxLayout(self)
            v.setContentsMargins(*((9, 5, 5, 5) if auto else (7, 5, 7, 5)))   # auto: the icon centred in the strip
            v.setSpacing(5)
            v.addWidget(IconLabel(role, 16, "normal" if on else "disabled"), 0, Qt.AlignTop)
            t = QLabel(self.title)
            t.setWordWrap(True)                          # a long name wraps in the palette's width
            t.setStyleSheet("font-weight: 600; font-size: 9pt;" + ("" if on else " color: #555b66;"))
            t.ensurePolished()                           # a word longer than the width (the night test pass:
            if max(t.fontMetrics().horizontalAdvance(w) for w in self.title.split() or [""]) > PALETTE_W - 64:
                t.setStyleSheet(t.styleSheet().replace("9pt", "8pt"))      # "Déconvolutior"): a size smaller
            v.addWidget(t, 1)
            self.text = t
            if auto:                                     # its height whether its name shows or not: nothing moves
                v.addStretch()
                t.ensurePolished()
                self.setFixedHeight(max(t.heightForWidth(PALETTE_W - 61), 16) + 12)
        key = view_key(block)
        what = textwrap.fill(_(BY_KEY[key]["what"]), 70) if key in BY_KEY else ""
        if branch:
            tip = "\n".join(t for t in (what, reason[:1].upper() + reason[1:], branch) if t)
        elif reason:                                     # not available: why, the sentence begun with a capital
            tip = reason[:1].upper() + reason[1:]
        else:
            tip = what
        self.setToolTip((self.title + ("\n" + tip if tip else "")) if folded else tip)
        self.setAccessibleName(self.title)

    def set_narrow(self, narrow: bool):
        self.text.setVisible(not narrow)
        self.setMaximumWidth(PALETTE_FOLDED_W - 12 if narrow else PALETTE_W - 26)

    def mousePressEvent(self, e):                                     # noqa: N802
        self._press = e.position().toPoint()

    def mouseMoveEvent(self, e):                                      # noqa: N802
        if self._press is None or not (e.buttons() & Qt.LeftButton) or (self.reason and not self.branch) or \
                self.block not in wfl.IO and self.block not in ("session", "image", "composition", "fusion"):
            return
        if (e.position().toPoint() - self._press).manhattanLength() < QApplication.startDragDistance():
            return
        self._press = None
        drag = QDrag(self)
        mime = QMimeData()
        mime.setText("block:" + self.block)
        drag.setMimeData(mime)
        drag.setPixmap(self.grab())
        drag.setHotSpot(QPoint(20, 12))
        self.page.unfold_palette(False)                  # the places it covers are the first ones
        self.page.hide_templates()
        drag.exec(Qt.CopyAction)

    def mouseDoubleClickEvent(self, e):                               # noqa: N802
        if not self.reason:
            self.page.palette_action(self.block)


class CombineCard(QFrame):
    """The Combine layers block in the toolbar (issue #36): a click adds one for the current
    workflow's split block (a layer of Post that puts its stars back on its starless image); greyed without one."""

    def __init__(self, page, split, parent=None):
        super().__init__(parent)
        self.page, self.split = page, split
        on = split is not None
        self.setObjectName("tool")
        self.setStyleSheet(f"QFrame#tool {{ background: {'#232831' if on else 'transparent'}; border: 1px "
                           f"{'solid #353b47' if on else 'dashed #3a4150'}; border-radius: 6px; }}"
                           f"QFrame#tool:hover {{ border-color: {ACCENT if on else '#3a4150'}; }}"
                           " QLabel { background: transparent; }")
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 6, 8, 6)
        v.setSpacing(1)
        t = QLabel(("+  " if on else "") + block_title(None, "composition"))
        t.setStyleSheet("font-weight: 600;" + ("" if on else " color: #6b7280;"))
        v.addWidget(t)
        v.addWidget(label(_("Post layer") if on else _("after Stars/background"), "small"))
        self.setToolTip(_(BY_KEY["composition"]["what"]))
        t.setMinimumWidth(t.sizeHint().width())
        self.setMinimumWidth(140)
        self.setEnabled(on)

    def mouseReleaseEvent(self, e):                                   # noqa: N802
        if self.split is not None and e.button() == Qt.LeftButton:
            self.page.new_composition(self.split)


class GraphPage(QWidget):
    """Graph mode: the toolbar of optional blocks, the graph and the panel of the selected block."""
    openBlock = Signal()                   # double-click or Enter on a block: Inspect (Post when sel_step is "final")
    openPost = Signal()                    # "Open Post" of the Post block
    openLayer = Signal(int)                # Open in Post: Post, with this layer selected
    addSession = Signal()
    variantMade = Signal(int, int)         # a variant (id) was made from this workflow (id): Inspect compares them
    changed = Signal()                     # the page was built again: the window updates its Run button and status
    toasted = Signal(str)
    sideChosen = Signal(str, str)          # issue #44: the user folded or pinned a side: "right" | "left", its state
    sidesShown = Signal()                  # the arrow of a hidden side: the panels come back

    def __init__(self, project, steps=None, parent=None):
        super().__init__(parent)
        self.s = project
        self._pending = False
        self._last: tuple | None = None                  # (seconds, count of workflows) at the last refresh
        ph = QHBoxLayout(self)
        ph.setContentsMargins(0, 0, 0, 0)
        ph.setSpacing(0)
        centre = QWidget()
        cv = QVBoxLayout(centre)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        tools = QFrame()
        tools.setObjectName("tools")
        tools.setStyleSheet("QFrame#tools { background: #1c1f26; }")
        self.tl = QVBoxLayout(tools)
        self.tl.setContentsMargins(10, 10, 14, 10)
        self.tl.setSpacing(4)
        # the palette (issue #47): a panel on the left, its groups one under the other, as the reference apps' tool
        # and library panels; it scrolls in a short window
        bar = QScrollArea(self)                          # over the page: unfolded by itself, it covers the graph
        bar.setWidget(tools)
        bar.setWidgetResizable(True)
        bar.setFrameShape(QFrame.NoFrame)
        bar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        bar.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        bar.setFixedWidth(PALETTE_W)
        bar.setStyleSheet("QScrollArea { background: #1c1f26; border-right: 1px solid #2a2f38; } "
                          "QScrollBar:vertical { width: 8px; background: #1c1f26; } QScrollBar::handle:vertical "
                          "{ background: #3a4150; border-radius: 4px; min-height: 40px; } QScrollBar::add-line, "
                          "QScrollBar::sub-line { height: 0; }")
        self.palette_bar = bar
        bar.setAcceptDrops(True)
        bar.viewport().setAcceptDrops(True)
        bar.viewport().installEventFilter(self)          # issue #54: a placed block dropped here is removed
        bar.installEventFilter(self)                     # the mouse coming and going: the automatic mode
        self.palette_slot = QWidget()                    # the room it takes in the page
        self.palette_slot.setFixedWidth(PALETTE_W)
        ph.addWidget(self.palette_slot)
        self._unfolded = False
        self._only_unfolded_items: list[QWidget] = []
        self._unfold_timer = QTimer(self, singleShot=True, interval=PALETTE_UNFOLD_MS)
        self._unfold_timer.timeout.connect(lambda: self.unfold_palette(True))
        self._fold_timer = QTimer(self, singleShot=True, interval=PALETTE_FOLD_MS)
        self._fold_timer.timeout.connect(lambda: self.unfold_palette(False))
        self.graph = GraphView(project)
        self.graph.palette_flow = self.palette_path       # Q11: a palette block dragged shows its workflow's places first
        self.graph.project_selected = lambda: self.project_node is self.current_node()
        cv.addWidget(self.graph, 1)
        self.project_node = Node("project")              # the Project block's, in the panel (never in the graph's flow)
        self._project_sel = False
        self.templates = TemplatePage()                  # in place of the blank graph, behind its link
        self.templates.hide()
        self.templates.back.connect(self.hide_templates)
        self.templates.chosen.connect(self.template_chosen)
        cv.addWidget(self.templates, 1)
        ph.addWidget(centre, 1)
        self.panel = Panel(project, steps)
        ph.addWidget(self.panel)
        # issue #44: the right panel folds to icons (it opens over the graph), or hides leaving a small arrow
        self.dock = Dock(self, self.panel, lambda: ph.addWidget(self.panel), lambda: ph.removeWidget(self.panel),
                         self.panel.folds.items, _("Block settings"))
        ph.insertWidget(ph.indexOf(self.panel), self.dock.strip)
        self.dock.chosen.connect(lambda st: self.sideChosen.emit("right", st))
        self.dock.arrow.clicked.connect(self.sidesShown.emit)
        self.panel.sectionsChanged.connect(self.dock.refresh_icons)
        self.panel.corner = fold_button(self.panel, self.dock.fold, _("Fold to icons"))
        self.left_state = "docked"
        self.left_arrow = QToolButton(self)
        set_icon(self.left_arrow, "chevron_right", 16, "dim")
        self.left_arrow.setFixedSize(ARROW_W, ARROW_H)
        self.left_arrow.setToolTip(_("Show the blocks again"))
        self.left_arrow.setStyleSheet("QToolButton { background: #1c1f26; border: 1px solid #2a2f38; padding: 0; "
                                      "border-top-right-radius: 5px; border-bottom-right-radius: 5px; }")
        self.left_arrow.clicked.connect(self.sidesShown.emit)
        self.left_arrow.hide()
        gv = self.graph
        gv.blockClicked.connect(self.select_node)
        gv.blockOpened.connect(lambda n: (self.select_node(n), self.openBlock.emit()))
        gv.newWorkflow.connect(lambda: self.new_workflow())
        gv.toolDropped.connect(self.add_tool)
        gv.blockDropped.connect(lambda b, pid, idx: self.place_block(b, pid, None if idx < 0 else idx))
        gv.sourceDropped.connect(self.palette_action)
        gv.postLinkRemoved.connect(self.remove_post_link)
        gv.branchDropped.connect(self.branch_block)
        gv.removeTool.connect(self.remove_tool)
        gv.workflowMenu.connect(self.workflow_action)
        gv.sendToPost.connect(self.send_to_post)
        gv.pauseToggled.connect(self.toggle_pause)
        gv.reviewMarked.connect(self.mark_reviewed)
        gv.partToPost.connect(self.part_to_post)
        gv.partOutOfPost.connect(self.part_out_of_post)
        gv.newComposition.connect(self.new_composition)
        gv.layerRemoved.connect(self.remove_layer)
        gv.openLayer.connect(self.openLayer.emit)
        gv.cutToggled.connect(self.toggle_cut)
        gv.newSession.connect(self.addSession.emit)
        gv.templatesAsked.connect(self.show_templates)
        gv.projectClicked.connect(self.select_project)
        gv.sessionMenu.connect(self.session_action)
        gv.postOpened.connect(self.open_post)
        gv.deselected.connect(self.deselect)
        gv.wireSelected.connect(self.show_wire)
        gv.filesDropped.connect(self.add_dropped_files, Qt.QueuedConnection)   # its menu after the drop is over
        pn = self.panel
        pn.changed.connect(self.request_refresh)
        pn.openTune.connect(self.openBlock.emit)
        pn.openPost.connect(self.openPost.emit)
        pn.variantMade.connect(self.variant_made)
        pn.addSession.connect(self.addSession.emit)
        pn.renamed.connect(self.rename)
        if hasattr(pn, "selectBlock"):
            pn.selectBlock.connect(lambda key: self.select(step=key))
        if hasattr(pn, "restoreBlock"):
            pn.restoreBlock.connect(lambda key: self.toggle_cut(Graph(self.s).node_of(self.s.path, key), None))
        pn.toast.connect(self.toast)
        pn.removeTool.connect(lambda: self.remove_tool(self.current_node(), None))
        if hasattr(pn, "saveTemplate"):
            pn.saveTemplate.connect(lambda pid: self.workflow_action("template", pid))
        pn.openLayer.connect(self.openLayer.emit)
        if hasattr(pn, "cutToggled"):
            pn.cutToggled.connect(lambda: self.toggle_cut(self.current_node(), None))
        for sig in (project.pathsChanged, project.currentChanged, project.costChanged, project.sourcesChanged,
                    project.readyChanged):
            sig.connect(self.request_refresh)
        if hasattr(project, "postChanged"):
            project.postChanged.connect(self._post_changed)
        for key in ("Return", "Enter"):
            ret = QShortcut(QKeySequence(key), self.graph)
            ret.setContext(Qt.WidgetWithChildrenShortcut)
            ret.activated.connect(lambda: self.current_node() is not None and self.openBlock.emit())
        esc = QShortcut(QKeySequence(Qt.Key_Escape), self.graph)    # decision 5: Escape selects nothing
        esc.setContext(Qt.WidgetWithChildrenShortcut)
        esc.activated.connect(lambda: self.graph.cancel_wire() or self.graph.clear_selection())   # a pulled wire first
        cut = QShortcut(QKeySequence("D"), self.graph)          # switch the selected block off or on (issue #47)
        cut.setContext(Qt.WidgetWithChildrenShortcut)
        cut.activated.connect(lambda: self.toggle_cut(self.current_node(), None))
        dele = QShortcut(QKeySequence(QKeySequence.Delete), self.graph)      # delete the selected block (issue #47)
        dele.setContext(Qt.WidgetWithChildrenShortcut)
        dele.activated.connect(self._delete_selected)
        # GUI exploration 4, Q5: F2 renames, Ctrl+D duplicates, as in the reference apps
        f2 = QShortcut(QKeySequence(Qt.Key_F2), self.graph)
        f2.setContext(Qt.WidgetWithChildrenShortcut)
        f2.activated.connect(self._rename_selected)
        dup = QShortcut(QKeySequence("Ctrl+D"), self.graph)
        dup.setContext(Qt.WidgetWithChildrenShortcut)
        dup.activated.connect(self._duplicate_selected)
        # issue #44: the same keys as the Post image; D, Del, F2, Ctrl+D and Enter keep their meaning
        for seqs, fn in ((("F", "Ctrl+0"), gv.fit_all), (("1",), lambda: gv.zoom_to(1.0)),
                         (("+", "=", "Ctrl++"), lambda: gv.zoom_by(ZOOM_STEP)),
                         (("-", "Ctrl+-"), lambda: gv.zoom_by(1 / ZOOM_STEP))):
            for seq in seqs:
                sc = QShortcut(QKeySequence(seq), self.graph)
                sc.setContext(Qt.WidgetWithChildrenShortcut)
                sc.activated.connect(fn)
        self.refresh()

    # ------------------------------------------------------------ refresh
    def _post_changed(self, source):
        """Post's composition changed: the Composition section and the wires to Post follow (not for a slider of the
        panel here: the panel stays under the mouse)."""
        if source is not self.panel:
            self.request_refresh()

    def request_refresh(self, *_a):
        if not self._pending:
            self._pending = True
            QTimer.singleShot(0, self.refresh)

    def refresh(self):
        self._pending = False
        s = self.s
        if s.path is None and s.paths:
            s.current = 0
        self.graph.rebuild()
        node = self.current_node(self.graph.graph)
        self.panel.refresh(node)
        w = getattr(self.graph, "sel_wire", None)
        if node is None and w is not None:                # a wire still chosen: its layer stays in the panel
            self.show_wire(w)
        self.build_tools()
        if not self.is_blank():
            self.hide_templates()
        self.panel.corner.setToolTip(_("Fold to icons"))     # issue #44: its words follow the language as the page's
        self.left_arrow.setToolTip(_("Show the blocks again"))
        self.dock.retranslate()
        if s.ready:
            sec = s.total_seconds()
            last = self._last
            said = time.monotonic() - getattr(self, "_said_at", -9.0) < 1.5   # the gesture's own message stays
            refused = bool(getattr(s, "run_problems", lambda: [])())         # no time for a run that is refused
            if last is not None and s.paths and len(s.paths) == last[1] and sec - last[0] >= 30 and \
                    s.has_data() and not said and not refused:   # no estimate without frames or an image
                t = hm(sec - last[0])                    # "≤ 1 min" is already an estimate: one sign only
                self.toast(_("+ {t} to run").format(t=("" if s.cost().measured or t.startswith("≤") else "≈ ") + t))
            self._last = (sec, len(s.paths))
        self.changed.emit()

    def toast(self, text: str):
        self.graph.show_toast(text)
        self.toasted.emit(text)
        if "Ctrl+Z" in text and hasattr(self.s, "name_step"):
            self.s.name_step(text)                       # what Ctrl+Z / Ctrl+Y will say they undid or redid
            self._said_at = time.monotonic()             # not covered by the "+ ≈ 1 min to run" that follows

    def select_project(self):
        """The Project block chosen: its panel, no block selected."""
        s = self.s
        s.pending_change = None
        s.scope = None
        s.sel_step = ""
        self._project_sel = True
        s.select()
        self.request_refresh()

    def current_node(self, g=None):
        if self.is_blank() or (self._project_sel and not self.s.sel_step):
            return self.project_node
        g = g or Graph(self.s)
        k = self.s.sel_step
        q = self.s.path
        if k == "sources" and not self.s.p.sessions and q is not None and q.source is not None:
            # no frames in the project: its Image (or Combine channels) is what is shown and what the palette targets, never
            # the frames' import (decision 5)
            return g.node_of(q, "fusion" if "fusion" in q.source else "image") or g.root
        if k == "sources":
            return g.root
        if k == "final":
            return g.final
        if k == "composition":
            return g.node_of(self.s.path, k, getattr(self.s, "sel_layer", None))
        return g.node_of(self.s.path, k)

    def build_tools(self):
        """The palette (issue #47): every block, greyed where it cannot be placed in the current workflow."""
        clear(self.tl)
        s = self.s
        p = self.palette_path()
        folded = self.left_state == "folded"
        auto = self.left_state == "auto"
        self._only_unfolded_items = []
        self.tl.setContentsMargins(*((4, 6, 8, 6) if folded else (4, 10, 14, 10) if auto else (10, 10, 14, 10)))
        toggle = QToolButton()
        icon_only(toggle, "more", _("Blocks display"), 16, "dim")
        toggle.setAutoRaise(True)
        toggle.setFixedSize(20, 18)
        toggle.setStyleSheet("QToolButton { border: none; padding: 0; } "
                             "QToolButton:hover { background: #2b3446; border-radius: 3px; }")
        toggle.clicked.connect(lambda: self.palette_menu(toggle))
        self.palette_button = toggle
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        head = QVBoxLayout()
        head.setSpacing(0)
        if folded:
            top.addStretch(1)
        else:
            name = label(_("Blocks"), "h2")
            text = _('for "{name}"').format(name=p.name) if p is not None else _("new workflow")
            forw = label(text, "small", wrap=not auto)
            if auto:                                      # one line: the blocks under it never move
                forw.setToolTip(text)
                forw.setText(forw.fontMetrics().elidedText(text, Qt.ElideRight, PALETTE_W - 44))
                self._only_unfolded(name)
                self._only_unfolded(forw)
            head.addWidget(name)
            head.addWidget(forw)
            top.addLayout(head, 1)
        top.addWidget(toggle, 0, Qt.AlignTop)
        self.tl.addLayout(top)
        self.tl.addSpacing(4)
        g = getattr(self.graph, "graph", None)
        split = g.node_of(p, "etoiles") if (g is not None and p is not None and s.ready) else None
        self.cards: dict[str, PaletteCard] = {}
        for title, blocks in PALETTE_GROUPS:
            self.tl.addSpacing(6)
            head_l = QLabel(_(title).upper())                 # a group's header, as the graph's phase bands
            head_l.setStyleSheet("color: #8b93a1; font-size: 8pt; font-weight: 600; letter-spacing: 1px; "
                                 "background: transparent;")
            head_l.setVisible(not folded)                     # folded: only the blocks' first letters
            if auto:
                self._only_unfolded(head_l)
            self.tl.addWidget(head_l)
            for b in blocks:
                reason = self.palette_reason(b, p, split)
                card = PaletteCard(self, b, reason, folded=folded, branch=self.drag_only(b) if reason else "", auto=auto)
                self.cards[b] = card
                self.tl.addWidget(card)
        self.combine_card = self.cards["composition"]
        self.tl.addStretch(1)
        self._fit_palette()

    def _only_unfolded(self, w: QWidget):
        """Shown only unfolded: its place kept when it is not, and no width asked of the strip."""
        sp = w.sizePolicy()
        sp.setHorizontalPolicy(QSizePolicy.Ignored)
        sp.setRetainSizeWhenHidden(True)
        w.setSizePolicy(sp)
        self._only_unfolded_items.append(w)

    def _fit_palette(self):
        """The palette's width and, automatic, what shows in it: not built again for a hover (the cards stay under the
        mouse, the one dragged alive)."""
        st = self.left_state
        auto = st == "auto"
        wide = not (st == "folded" or auto and not self._unfolded)
        self.palette_bar.setFixedWidth(PALETTE_W if wide else PALETTE_FOLDED_W)
        self.palette_slot.setFixedWidth(PALETTE_FOLDED_W if st in ("folded", "auto") else PALETTE_W)
        if auto:
            m = self.tl.contentsMargins()
            m.setRight(14 if wide else 8)
            self.tl.setContentsMargins(m)
            for w in self._only_unfolded_items:
                w.setVisible(wide)
            for card in self.cards.values():
                card.set_narrow(not wide)
        self._place_palette()

    def _place_palette(self):
        self.palette_bar.setGeometry(0, 0, self.palette_bar.width(), self.height())
        self.palette_bar.raise_()

    def unfold_palette(self, on: bool):
        """The automatic palette opened over the graph, or back to its icons."""
        if not on and QApplication.activePopupWidget() is not None:
            self._fold_timer.start()                      # its menu is open
            return
        self._unfold_timer.stop()
        self._fold_timer.stop()
        on = on and self.left_state == "auto"
        if on != self._unfolded:
            self._unfolded = on
            self._fit_palette()

    def palette_menu(self, button):
        """The palette's three ways of showing, as the reference apps' ribbon options."""
        m = QMenu(self)
        for state, word in (("auto", "Automatic"), ("docked", "Always open"), ("folded", "Always folded")):
            a = m.addAction(_(word))
            a.setCheckable(True)
            a.setChecked(state == self.left_state)
            a.setData(state)
        a = m.exec(button.mapToGlobal(button.rect().bottomLeft()))
        state = a.data() if a is not None else None
        m.deleteLater()
        if state and state != self.left_state:
            self.sideChosen.emit("left", state)

    def set_layout(self, right: str, left: str, solo: bool):
        """Issue #44: the right panel (docked | icons | hidden), the blocks' palette (docked | folded | hidden) and the
        sections of the panel (one at a time on a small window)."""
        self.panel.folds.set_solo(solo)
        self.dock.set_state(right)
        if left != self.left_state:
            self.left_state = left
            self._unfolded = False
            self._unfold_timer.stop()
            self._fold_timer.stop()
            self.palette_bar.setVisible(left != "hidden")
            self.palette_slot.setVisible(left != "hidden")
            self.left_arrow.setVisible(left == "hidden")
            self.build_tools()
        self._place_arrows()

    def _place_arrows(self):
        self.left_arrow.move(0, (self.height() - ARROW_H) // 2)
        self.left_arrow.raise_()

    def resizeEvent(self, e):                                         # noqa: N802
        super().resizeEvent(e)
        self._place_palette()
        self._place_arrows()

    def eventFilter(self, obj, e):                                    # noqa: N802
        """The palette takes a placed block dragged onto it: removed, as dragging out of a toolbar (issue #54)."""
        bar = getattr(self, "palette_bar", None)
        if bar is not None and obj is bar:
            if e.type() == QEvent.Enter and self.left_state == "auto":
                self._fold_timer.stop()
                if not self._unfolded:
                    self._unfold_timer.start()
            elif e.type() == QEvent.Leave:
                self._unfold_timer.stop()
                if self._unfolded:
                    self._fold_timer.start()
            elif e.type() == QEvent.Hide:                 # another page, Tab
                self.unfold_palette(False)
            return False
        if bar is not None and obj is bar.viewport() and e.type() in (QEvent.DragEnter, QEvent.DragMove, QEvent.Drop):
            t = e.mimeData().text() if e.mimeData().hasText() else ""
            node = getattr(self.graph, "_dragged", None)
            if not t.startswith("placed:") or node is None:
                return False
            if e.type() == QEvent.Drop:
                e.acceptProposedAction()
                QTimer.singleShot(0, lambda n=node: self.remove_tool(n, None))
            else:
                e.acceptProposedAction()
            return True
        return super().eventFilter(obj, e)

    def drag_only(self, block: str) -> str:
        """Where a block the current workflow cannot take can still be dragged, in words; "" when nowhere."""
        s = self.s
        if not s.ready:
            return ""
        flows = self.graph._flows()
        g = self.graph.graph
        if block == "composition":
            there = g is not None and any(n.key == "etoiles" and n.paths for n in g.nodes)
        else:
            there = block in wfl.IO and any(s.place_indices(q, block) for q in flows)
        if there:
            return _("Drag it onto another workflow")
        if block in wfl.FRAME_BLOCKS and any(s.can_branch(q, key, block) for q in flows for key in s.branch_keys(q)):
            return _("Drag it between blocks for a new branch")
        return ""

    def palette_reason(self, block: str, p, split) -> str:
        """Why a palette block cannot be placed now (translated), "" when it can."""
        s = self.s
        if not s.ready:
            return _("reading the project…")
        if block == "align_images":
            return s.align_problem()
        if p is None and s.p.sessions and block not in ("session", "image", "fusion", "composition"):
            return s.palette_reason(None, block)          # sessions without a workflow: the block starts one
        if block in ("session", "image"):
            return ""
        if block == "fusion":                             # question 6: something to combine, said what to do (8.5)
            ok = len([q for q in s.paths if s.is_mono(q)]) >= 2 or any(
                q.source is None and not s.is_mono(q) and any(n["block"] == "stacking" for n in s.graph(q))
                for q in s.paths) or bool(self.linear_images())
            if not ok:
                return _("needs 2 filter workflows, a colour stack or image")
            got = self.fusion_choice()                    # decision 2: greyed until each channel has what it needs
            if got is not None:                           # its frames (control round: a template's colour project
                empty = []                                # without frames made a Combine channels of channels with nothing)
                for i in got[1]:
                    q = s.view(i["pid"])
                    if q.source is None and not any(x.lights for x in s.p.sessions_of(q.pid)):
                        empty.append((i["role"], q))
                if empty and len({i["pid"] for i in got[1]}) == 1:       # a colour image's channels: its workflow
                    return _("lights first in “{name}”").format(name=empty[0][1].name)
                if empty:
                    return _("lights first in {channels}").format(channels=", ".join(r for r, _q in empty))
            if got is not None and len({i["pid"] for i in got[1]}) > 1:
                miss = [i["role"] for i in got[1] if s.is_graph(s.view(i["pid"])) and not any(
                    n["block"] == "astrometry" and not n["off"] for n in s.graph(s.view(i["pid"])))]
                if miss:
                    return _("Astrometry first in {channels}").format(channels=", ".join(miss))
            return ""
        if block == "composition":
            return "" if split is not None else _("needs Stars/background")
        if p is None:
            return _("place Session/Image first")
        if getattr(p, "source", None) and "fusion" in (p.source or {}) and block not in ("output", "stars"):
            return s.palette_reason(p, block)
        return s.palette_reason(p, block)

    def add_dropped_files(self, sid: int, paths: list, kind: str | None = None):
        """Files or folders dragged from the Explorer onto a Session block (decision 9, as reference apps import by a
        drop): nothing is guessed from their names, a menu asks what they are (``kind``: already said)."""
        s = self.s
        kind = kind or self.ask_dropped_kind()
        if kind is None:
            return
        notes: list = []
        with s.batch():                                   # one drop, one undo step
            n = s.add_session_entries(sid, kind, paths, notes)
        said = [_(t).format(name=nm) for t, nm in notes]
        if n:
            said.insert(0, plural(n, "{n} file added to “{session}” · Ctrl+Z", "{n} files added to “{session}” · Ctrl+Z",
                                  session=s.session(sid).name))     # one undo step (control round)
        self.toast(" · ".join(said) if said else _("Nothing added: no frame in what was dropped"))

    def ask_dropped_kind(self) -> str | None:
        """What the files dropped on a Session block are, as the user says; None: nothing."""
        from .sources import GROUPS
        m = QMenu(self)
        for key, title, _sec in GROUPS:
            m.addAction(_(title)).setData(key)
        a = m.exec(QCursor.pos())
        kind = a.data() if a is not None else None
        m.deleteLater()
        return kind

    def palette_action(self, block: str):
        """A double-click on a palette block: placed in the current workflow at its usual place; a source starts a
        workflow. "<source>:new" (dropped on the new-flow zone): a parallel flow of its own; "session:join" (dropped
        on "+ Session"): it joins the flow "+ Session" is drawn with."""
        s = self.s
        block, _sep, where = block.partition(":")
        new = where == "new"
        if block == "align_images":
            self.align_images()
            return
        if block == "session" and new and any(q.source is None for q in s.paths):
            q = s.add_parallel_flow()                      # issue #47: a second flow, its own night
            sid = s.sessions_of(q)[0]
            self.select(index=s.paths.index(q), step=f"session@{sid}")
            self.toast(_("“{session}”: workflow of its own · Ctrl+Z").format(session=s.session(sid).name))
            return
        if block == "session":                             # issue #48: placed empty, its files in its Inspect
            target = self.palette_path()
            joined = s.view(self.graph.plus_session_pid) if where == "join" and \
                getattr(self.graph, "plus_session_pid", None) is not None else None
            target = joined or target
            with s.batch():                           # one Ctrl+Z takes it away (2.2)
                sid = s.add_filter_session("")
                made = s.ensure_frames_workflow()          # a blank page: the Session starts a workflow
                q = made or s.join_session(target, sid)    # else it joins the chosen flow (decision 1)
            if q is not None:
                q = s.view(q.pid)
                s.current = s.paths.index(q)
            self.select(step=f"session@{sid}" if s.separate_sessions() else "sources")
            name = s.session(sid).name
            if q is not None and not made and len([x for x in s.paths if x.source is None]) > 1 and \
                    s.sessions_of(q) == [sid]:
                self.toast(_("“{session}”: workflow of its own · Ctrl+Z").format(session=name))
            elif q is not None and not made:              # the scope is visible: which flow it joined
                self.toast(_("“{session}” joins “{name}” · Ctrl+Z").format(session=name, name=q.name))
        elif block == "image":                             # placed like the Session: its file chosen in its panel
            with s.batch():                           # one Ctrl+Z takes it away (2.2)
                q = s.add_image_workflow(s.unique_name(_("Image")), "")
                s.set_graph(q, [])
            q = s.view(q.pid)
            s.log('Workflow "{name}" created', name=q.name)
            self.select(index=s.paths.index(q), step="image")
        elif block == "fusion":                            # placed at once: its palette and channels in its panel
            self.place_fusion()
        elif block == "composition":
            g = Graph(s)
            split = g.node_of(s.path, "etoiles") if s.path is not None else None
            if split is not None:
                self.new_composition(split)
        else:
            p = self.palette_path()
            self.place_block(block, p.pid if p is not None else NEW_FRAMES)

    def place_fusion(self):
        """A Combine channels block placed from the palette (no template window once the project is made): the filters'
        workflows combined with the first palette they make (SHO, HOO, LRGB, RGB...), else a colour workflow's
        channels as H-alpha and OIII (dual band); its palette and channels are changed in its panel. Refused, with
        what to do, until each channel has what it needs; what it adds to the channels is said (decision 2)."""
        s = self.s
        why = self.palette_reason("fusion", s.path, None)
        got = self.fusion_choice()
        if why or got is None:
            self.toast(_("Combine channels: {why}").format(why=why or _("place its workflows first")))
            return
        pal, inputs, stars = got
        chans = {i["pid"]: s.view(i["pid"]) for i in inputs}
        before = {pid: [n["block"] for n in s.graph(q)] for pid, q in chans.items()}
        q = s.add_fusion(pal, pal, inputs, stars=stars)
        s.log('Combine channels "{name}" placed', name=q.name)
        self.select(index=s.paths.index(q), step="fusion")
        added, where = [], []
        for pid, old in before.items():
            new = [n["block"] for n in s.graph(s.view(pid))]
            more = [b for b in new if b not in old]
            if more:
                added += [b for b in more if b not in added]
                where.append(s.view(pid).name)
        if added:                                         # the blocks it put in the channels: said, Ctrl+Z
            self.toast(_("Combine channels “{block}” placed · {blocks} added to {where} · Ctrl+Z").format(
                block=q.name, blocks=", ".join(block_title(None, view_key(b)) for b in added),
                where=", ".join(where)))
        else:
            self.toast(_("Combine channels “{block}” placed · Ctrl+Z").format(block=q.name))

    def linear_images(self) -> list:
        """The Image workflows a Combine channels can read (issue #55): a linear image whose file is chosen."""
        return [q for q in self.s.paths if self.s.source_kind(q) == "image" and q.source.get("path")]

    def fusion_choice(self):
        """What a Combine channels placed now combines: (palette, inputs, stars), or None when there is nothing to combine."""
        from ..stages.fusion import PALETTES
        s = self.s
        by_filter = {}
        for f, sids in s.filters().items():
            q = next((q for q in s.paths if q.source is None and q.sessions is not None and not q.end
                      and sorted(q.sessions) == sorted(sids)), None)
            if q is not None:
                by_filter[f] = q
        order = ["SHO", "HSO", "HOO", "LRGB", "RGB"] + [x for x in PALETTES if x not in ("SHO", "HSO", "HOO", "LRGB",
                                                                                        "RGB")]
        pal = next((x for x in order if x in PALETTES and set(PALETTES[x][0]) <= set(by_filter)), None)
        # #39: the stars as its template makes them (option A): the channels' for RGB and LRGB, else H-alpha's;
        # a dual band's, of the colour image itself
        if pal is not None:
            inputs = [{"pid": by_filter[f].pid, "role": f, "channel": "L"} for f in PALETTES[pal][0]]
            stars = {"mode": "palette"} if pal in ("RGB", "LRGB") or "Ha" not in PALETTES[pal][0] else \
                {"mode": "channel", "role": "Ha"}
        else:
            colour = next((q for q in s.paths if q.source is None and not s.is_mono(q) and not q.end), None)
            image = next(iter(self.linear_images()), None) if colour is None else None     # issue #55: no stack to read
            if colour is None and image is None:
                return None
            pal = "HOO"                                   # a dual-band filter on a colour camera
            src = colour or image
            inputs = [{"pid": src.pid, "role": "Ha", "channel": "R"}, {"pid": src.pid, "role": "OIII", "channel": "GB"}]
            stars = {"mode": "workflow", "pid": src.pid} if image is None else None      # an image keeps its stars
        return pal, inputs, stars

    def place_block(self, block: str, pid=None, index=None):
        """Place a block in a workflow (issue #47): at ``index`` of its graph, or at its usual place. ``NEW_FRAMES``:
        in the workflow from the frames, made when there is none yet."""
        s = self.s
        p = None if pid == NEW_FRAMES else s.view(pid) if pid is not None else s.path
        if p is None and not s.p.sessions:
            self.toast(_("Place Session/Image first"))
            return
        try:
            with s.batch():                               # the workflow it starts and the block: one Ctrl+Z (2.2)
                if p is None:                             # sessions, no workflow yet: the block starts one
                    p = s.ensure_frames_workflow() or next((q for q in s.paths if q.source is None), None)
                if p is None:
                    raise ValueError(_("Place Session/Image first"))
                key = s.insert_block(p, block, index)
        except ValueError as e:
            self.toast(str(e))
            return
        s.log('{block} added to "{name}"', block=block_title(p, key), name=p.name)
        q = s.view(p.pid)
        if key == "calibration" and q.source is None and s.separate_sessions():
            mine = [x for x in s.separate_sessions() if x in s.sessions_of(q)] or s.separate_sessions()
            key = f"calibration@{mine[0]}"                # drawn beside each session: one of this workflow's (1.6)
        s.select(index=s.paths.index(q), step=key)
        node = Graph(s).node_of(q, key)
        s.scope = [x.pid for x in node.paths] if node is not None else [p.pid]

    def branch_block(self, block: str, pid: int, key: str):
        """A block dropped under a block (issue #47): a new branch of the workflow from there, with that block."""
        s = self.s
        p = s.view(pid)
        try:
            q, new_key = s.add_branch(p, key, block)
        except ValueError as e:
            self.toast(str(e))
            return
        s.log('Branch "{name}" made', name=q.name)
        self.select(index=s.paths.index(q), step=new_key)
        s.scope = [q.pid]                                 # its settings change it alone
        self.toast(_('Branch "{name}" made · Ctrl+Z').format(name=q.name))    # its name says the block placed

    def image_as_is(self, p):
        """An Image block's file sent to Post as it is (question 3): an Image workflow of its own, without blocks."""
        s = self.s
        src = p.source or {}
        with s.batch():
            q = s.add_image_workflow(s.unique_name(_("{name} as it is").format(name=p.name)), src.get("path", ""),
                                     stretched=bool(src.get("stretched")))
            s.set_graph(q, [])
        q = s.view(q.pid)
        s.log('"{name}" sent to Post', name=q.name)
        self.toast(_('"{name}" to Post · Ctrl+Z').format(name=q.name))

    def remove_post_link(self, pid: int):
        """A workflow's wire to Post taken away (issue #52): its image is no longer a layer; an output sent to Post
        from a block (a workflow of its own) goes with it."""
        s = self.s
        p = s.view(pid)
        if p is None:
            return
        only_sent = bool(p.end) or s.is_sent_branch(p)     # question 1: a branch made only to send that image goes
        if only_sent:                                     # with its wire (a branch with blocks of its own stays)
            s.remove_path(s.paths.index(p))
            s.log('Workflow "{name}" deleted', name=p.name)
        else:
            s.remove_post_part(pid, None, self)
        self.toast(_("“{name}” out of Post · Ctrl+Z").format(name=p.name)       # undone as said (decision 11)
                   if not only_sent else _("“{name}” deleted · Ctrl+Z").format(name=p.name))
        self.request_refresh()

    def _delete_selected(self):
        if self.graph.remove_selected_wire():             # issue #52: a selected wire first
            return
        node = self.current_node()
        if node is not None and node.key not in ("final", "project"):
            self.remove_tool(node, None)

    def _rename_selected(self):
        """F2: a Session selected, the Session (the only one, when its block stands for the frames); a block of a
        workflow, its workflow; Post or nothing, nothing."""
        s = self.s
        k = s.sel_step or ""
        if k.startswith("session@"):
            self.session_action("rename", int(k.split("@")[1]))
        elif k == "sources":
            if len(s.p.sessions) == 1:
                self.session_action("rename", s.p.sessions[0].id)
        elif k and k != "final" and "@" not in k and s.path is not None:     # a session's Calibration: nothing
            self.workflow_action("rename", s.path.pid)

    def _duplicate_selected(self):
        """Ctrl+D: the workflow of the selected block, as its menu's Duplicate."""
        s = self.s
        k = s.sel_step or ""
        if k and k not in ("sources", "final") and "@" not in k and s.path is not None:
            self.workflow_action("duplicate", s.path.pid)

    # ------------------------------------------------------------ selection
    def deselect(self):
        """Nothing selected (decision 5): an empty click, Escape, or a wire chosen; the panel empties, Del and D do
        nothing, the palette places in the current workflow."""
        s = self.s
        if self._project_sel:
            self._project_sel = False
            self.request_refresh()
        if not s.sel_step:
            return
        s.pending_change = None
        s.scope = None
        s.sel_step = ""
        s.select()

    def show_wire(self, w):
        """A wire to Post chosen (control round: the panel went empty): the layer it makes, by its name, what it is,
        Rename (a workflow's), Out of Post (as Del), Open Post."""
        s = self.s
        from .post3 import tr_name
        pid, part = (w.pid, None) if w.pid is not None else (w.part or (None, None))
        p = s.view(pid) if pid is not None else None
        if p is None:
            return
        lay = next((x for x in s.post_layers() if x.get("kind") == "workflow" and x.get("pid") == pid
                    and x.get("part") == part), None)
        if part is None:
            name = tr_name(str((lay or {}).get("name") or p.name))
            note = _("The image of “{name}”, a layer of Post. Del or Out of Post takes it out; Ctrl+Z brings it "
                     "back.").format(name=p.name)
            actions = [(_("Rename"), lambda: self.workflow_action("rename", pid), _("The workflow and its layer")),
                       (_("Out of Post"), lambda: self.remove_post_link(pid), _("Its image is no longer a layer"))]
        else:
            word = _(dict(STAR_OUTPUTS)[part])
            name = tr_name(str((lay or {}).get("name") or f"{p.name} · {word}"))
            note = _("{image} of “{name}”, sent to Post as it is: a layer. Del or Out of Post takes it out; Ctrl+Z "
                     "brings it back.").format(image=word, name=p.name)
            node = Graph(s).node_of(p, "etoiles")
            actions = [(_("Out of Post"), lambda: node is not None and self.part_out_of_post(node, part),
                        _("Its image is no longer a layer"))]
        actions.append((_("Open Post"), self.openPost.emit, _("Post, where the layer is set")))
        self.panel.show_wire(name, note, actions)

    def open_post(self):
        """Open Post (the Post block's menu, as its double-click): Post selected, then opened; not an undo step."""
        self.s.select(step="final")
        self.openBlock.emit()

    def select_node(self, node):
        s = self.s
        self._project_sel = False
        if node.key == "composition" and node.lid is None and node.grey:   # a Combine layers to come (Q10): said, not chosen
            self.toast(_("Combine layers comes with the lights: add them to the Session"))
            return
        s.pending_change = None
        self.graph.select_wire(None)                      # a block chosen: the wire chosen before is not any more
        if node.key not in ("sources", "final") and node.paths:
            if s.path not in node.paths:
                s.current = s.paths.index(node.paths[0])
            s.scope = [q.pid for q in node.paths]
        else:
            s.scope = None
        if node.key == "composition":
            s.sel_layer = node.lid
        if node.key in ("sources", "session") or node.sid is not None:
            frames = [q for q in node.paths if q.source is None]  # a Session chosen: the palette places in its frames'
            if frames and s.path not in frames:                   # workflow (not in the Image chosen before)
                s.current = s.paths.index(frames[0])
        s.select(step=node.ref)

    def palette_path(self):
        """The workflow the palette places in: the chosen block's. A Session (or the frames) chosen without a workflow
        from its frames yet: None (a block placed starts one)."""
        s = self.s
        node = self.current_node() if s.ready and s.sel_step else None
        if node is not None and (node.key in ("sources", "session") or node.sid is not None) and node is not self.project_node:
            return s.path if s.path is not None and s.path.source is None else None
        return s.path

    def select(self, index=None, step=None):
        s = self.s
        s.pending_change = None
        s.select(index, step)
        self._project_sel = False
        node = self.current_node()
        s.scope = [q.pid for q in node.paths] if node is not None and node.key not in ("sources", "final", "project") else None

    # ------------------------------------------------------------ actions
    def is_blank(self) -> bool:
        return not self.s.paths and not self.s.p.sessions

    def show_templates(self):
        """The templates in place of the blank graph."""
        self.templates.build()
        self.graph.hide()
        self.templates.show()

    def hide_templates(self):
        if self.templates.isVisibleTo(self):
            self.templates.hide()
            self.graph.show()

    def template_chosen(self, t: dict, files: list):
        from .gallery import _view_opts
        self.make_from_template(t, self.s.unique_name(t["palette"] if t.get("kind") == "fusion" else _(t["name"])),
                                _view_opts(t), files)

    def new_workflow(self, start=None, tab=None):
        """New workflow: on a blank page its templates in place of the graph; else the template gallery (issue #43), or
        a copy of a workflow (``start``); ``tab``: the tab it opens on."""
        if start is None and self.is_blank():
            self.show_templates()
            return None
        d = TemplateGallery(self.s, start, self)
        if tab and tab in getattr(d, "tab_keys", []):
            d.tabs.setCurrentIndex(d.tab_keys.index(tab))
        d.accepted.connect(lambda: self.make_workflow(d))
        d.open()                                           # window-modal, without blocking (a new project opens it)
        self._gallery = d
        return d

    def make_workflow(self, d):
        """The workflow the gallery's choice makes: a copy, from an image, or from the frames."""
        if d.choice is not None:
            self.make_from_template(d.choice, d.name.text().strip() or _("New workflow"), d.opts, d.images)

    def align_images(self) -> bool:
        """Every image in the Align images block (the palette, Post's button), which is selected; False when it cannot."""
        s = self.s
        first = s.align_images()
        if first is None:
            self.toast(_("Align images: {why}").format(why=s.align_problem()))
            return False
        moved = s.align_group(first)[1]                   # the first that moves: what Inspect shows, and computes
        self.select(index=s.paths.index(moved), step="align_images")
        return True

    def place_images(self, files: list):
        """One Image workflow per file, each a layer of Post, as a file's Inspect sends it; the first, or None."""
        from ..stages.image import image_problem, stretched_by_nature
        s = self.s
        for f in files:
            why = image_problem(f)
            if why:
                self.toast(_(why).format(name=Path(f).name))
                return None
        made = []
        with s.batch():
            for f in files:
                q = s.add_image_workflow(s.unique_name(Path(f).stem), f, stretched=stretched_by_nature(f))
                s.set_graph(q, [])
                made.append(s.view(q.pid))
        return made[0] if made else None

    def make_from_template(self, t: dict, name: str, opts: dict, files: list):
        """The workflow a template makes: a copy, from images, or from the frames."""
        s = self.s
        from .gallery import template_blocks, template_source
        empty = [q for q in s.paths if q.source is None and s.p.workflow(q.pid).blocks == []]
        frames_template = not t.get("copy") and template_source(t) == "frames"
        if empty and frames_template:                      # an empty frames workflow (#47): a frames template takes
            # its place; an image template starts a flow of its own beside it (issue #55)
            with s.batch():
                for q in empty:
                    s.remove_path(s.paths.index(q))
        if t.get("copy") is not None:                      # a copy keeps its options, settings and exit
            p = s.duplicate(t["copy"], name)
        elif t.get("kind") in ("filters", "fusion"):
            p = self.make_filters(t, name, opts)
            if p is None:
                return
        elif t.get("plain"):
            p = self.place_images(files)
            if p is None:
                return
        elif template_source(t) in ("image", "stretched"):
            with s.batch():
                p = s.add_image_workflow(name, files[0], opts, stretched=template_source(t) == "stretched")
                s.set_graph(p, template_blocks(t))
        else:
            with s.batch():
                # GUI exploration 7.2: it starts from a Session block, empty when the project has no frames yet
                if not s.p.sessions:
                    s.add_filter_session("")
                p = s.add_path(name, opts, None)
                s.set_graph(p, template_blocks(t))         # issue #47: only the blocks it uses
        s.log('Workflow "{name}" created', name=p.name)
        todo = s.first_to_fill(p)                          # the block that waits for the user: its empty Session,
        if todo is not None:                               # its Image without a file (GUI exploration 2)
            self.select(index=s.paths.index(todo[0]), step=todo[1])
        else:
            self.select(index=s.paths.index(p), step="post")

    def make_filters(self, t: dict, name: str, opts: dict):
        """A filters template (issue #43): one workflow per filter (made when the filter has none yet), then for the
        full one the Combine channels block of their outputs. Returns the workflow shown (the fusion, or the first
        filter's)."""
        s = self.s
        from .gallery import template_blocks
        with s.batch():
            # One workflow per filter: the project's filters, else those chosen in the gallery (GUI exploration 7.2)
            filters = (sorted(s.filters()) or list(t["filters"])) if t["kind"] == "filters" \
                else list(t["filters"])
            for f in filters:                               # a missing filter: an empty Session block (issue #47)
                if f not in s.filters():
                    s.add_filter_session(f)
            made = [(f, s.filter_workflow(f, opts)) for f in filters]
            made = [(f, q) for f, q in made if q is not None]
            for _f, q in made:                              # a filter's workflow: no Color block (issue #47)
                if s.p.workflow(q.pid).blocks is None:
                    s.set_graph(q, template_blocks(t, mono=True))
            if not made:
                self.toast(_("No filter: set it in Session"))
                return None
            if t["kind"] == "filters":
                return made[0][1]
            # option A: each channel's stars taken off, the starless palette; the stars: of the channels (RGB, LRGB:
            # their own colours), else of H-alpha, neutral (narrowband stars in a palette come out magenta)
            roles = [f for f, _q in made]
            stars = {"mode": "palette"} if t["palette"] in ("RGB", "LRGB") or "Ha" not in roles else \
                {"mode": "channel", "role": "Ha"}
            return s.add_fusion(name, t["palette"], [{"pid": q.pid, "role": f, "channel": "L"} for f, q in made],
                                stars=stars)

    def toggle_cut(self, node, pids):
        """Switch a block off, or on again (issue #47): in ``pids`` only, else in every workflow going through it. Its
        settings and results stay, the data passes through it; Undo takes it back."""
        s = self.s
        why = {"session": "a source: Del removes it", "sources": "a source: Del removes it",
               "align_images": "Del removes it",
               "image": "a source: Del removes it",      # a Combine channels reads its channels (control round: not a source)
               "fusion": "it combines its channels: Del removes it; to leave it out of Post, remove its wire to Post",
               "composition": "hide its layer in Post, or Del removes it"}.get(node.key if node is not None else "")
        if why and (node.key != "sources" or s.p.sessions):     # refused, and why (decision 11)
            self.toast(_("{block} cannot be switched off: {why}").format(
                block=node_title(s, node) if node.key in ("session", "sources") else block_title(None, node.key),
                why=_(why)))
            return
        if node is None or node.key == "project" or (node.sid is not None and node.key != "calibration") or not node.paths or \
                node.key in ("sources", "final", "session", "composition", "fusion", "image"):
            return
        paths = [q for q in node.paths if pids is None or q.pid in pids]
        on = not node.cut
        name = block_title(paths[0], node.key) if paths else block_title(None, node.key)   # as the graph names it
        try:
            done = s.set_cut(paths, node.key, on)
        except ValueError as e:                           # the block named, as Del's refusal (control round)
            self.toast(_("{block} cannot be switched off: {why}").format(block=name, why=str(e))
                       if on and not str(e).startswith(name) else str(e))
            return
        if done:
            where = paths[0].name if len(paths) == 1 else plural(len(paths), "{n} workflow", "{n} workflows")
            if node.sid is not None:                      # one Calibration for every session: the scope said
                where = _("all sessions · {where}").format(where=where)
            text = "{block} off · {where}" if on else "{block} on · {where}"
            self.toast(_(text).format(block=name, where=where) + " · Ctrl+Z")
            s.log(text, block=name, where=where)          # the Log: the block's name, never its key (9.4)

    def session_action(self, action: str, sid: int):
        s = self.s
        if action == "rename":
            name = ask_name(self, _("Rename session"), s.session(sid).name)
            if name is not None:
                why = s.name_problem(name, sid=sid)       # decision 8: never empty, never another's name
                if why:
                    self.toast(_("Name refused: {why}").format(why=why))
                else:
                    s.rename_session(sid, name)
        elif action == "remove":
            name = s.session(sid).name
            why = s.session_removal_problem(sid)          # a fusion's channel: refused with the reason (decision 3)
            if why:
                self.toast(_("Session “{name}” kept: {why}").format(name=name, why=why))
                return
            flows = [q.name for q in s.flows_of_session(sid)]    # its flow goes with it: said (the scope visible)
            if s.remove_session(sid):                     # as any block: at once, Ctrl+Z brings it back
                s.log('Session "{name}" removed', name=name)
                s.select(step="sources")
                self.toast(_("“{name}” and its workflow “{flow}” removed · files kept · Ctrl+Z").format(
                    name=name, flow=", ".join(flows)) if flows else
                    _("“{name}” removed · files kept · Ctrl+Z").format(name=name))

    def add_tool(self, key, pid):
        """An older tool key (Denoise, Split stars) placed in a workflow: its block at its usual place."""
        self.place_block({"debruitage": "denoise", "etoiles": "stars"}.get(key, key), pid)

    def remove_tool(self, node, pids):
        """Delete a block (issue #47): from ``pids`` only, else from every workflow going through it; refused, with
        the reason, where the blocks after it need it."""
        s = self.s
        if node is not None and node.key == "sources" and len(s.p.sessions) == 1:
            self.session_action("remove", s.p.sessions[0].id)   # a lone Session: this block stands for it
            return
        if node is None or node.key in ("sources", "final"):
            return
        if node.key == "session":
            self.session_action("remove", node.sid)
            return
        if node.key == "composition":
            self.remove_layer(node.lid, node)
            return
        if node.key == "align_images":
            s.remove_alignment(node.paths[0])
            s.log("Align images removed")
            s.select(step="image")
            self.toast(_("Align images removed · Ctrl+Z"))
            return
        if node.key in ("fusion", "image"):              # a source block: it goes with the blocks it feeds
            gone = [q for q in node.paths if pids is None or q.pid in pids]
            for q in gone:                                # an Image a fusion reads: refused (decision 3)
                why = s.channel_problem(q) if node.key == "image" else ""
                if why:
                    self.toast(_("{block} kept: {why}").format(block=block_title(None, node.key), why=why))
                    return
            with s.batch():
                for q in gone:                            # a Combine channels takes back what it added (decision 2)
                    if node.key == "fusion":
                        s.remove_fusion(s.view(q.pid))
                    else:
                        s.remove_path(s.paths.index(q))
            s.log("{block} and next removed", block=block_title(None, node.key))
            s.select(step="sources")
            self.toast(_("{block} and next removed · Ctrl+Z").format(
                block=block_title(None, node.key)))
            return
        key = node.key
        targets = [q for q in node.paths if pids is None or q.pid in pids]
        night = None if key != "calibration" or not targets else s.own_calibration_night(targets[0], node.sid)
        for p in targets:                                 # all or none (GUI exploration 3.7): refused, the block
            why = "" if night is not None else s.removal_problem(p, key)     # stays, selected, in every workflow
            if why:
                self.toast(_("{block} kept: {why}").format(
                    block=block_title(p, key), name=p.name, why=why))
                return
        title = block_title(targets[0], key) if targets else block_title(None, key)    # its name before it goes
        # a workflow with no block of its own already (a block's image sent to Post, a copy): stays so
        sent = {p.pid for p in targets if s.is_emptied_branch(p)}
        names = {p.pid: p.name for p in targets}           # as the user saw it (its name follows its last block)
        gone: list[str] = []
        with s.batch():
            for p in targets:
                s.log('{block} removed from "{name}"', block=block_title(p, key), name=p.name)
                if night is not None:                     # beside the Session's: the workflow takes that one again
                    s.remove_own_calibration(p, night)
                else:
                    s.remove_node(p, key)
            for p in targets:                             # a branch left with no block of its own goes with it, in
                q = s.view(p.pid)                         # the same undo step (control round)
                if q is not None and q.pid not in sent and s.is_emptied_branch(q):
                    gone.append(names[q.pid])
                    s.remove_path(s.paths.index(q))
                    s.log('Workflow "{name}" deleted', name=names[q.pid])
        if gone:                                          # its trunk chosen: the block before it shown there
            trunk = next((q for q in s.paths if q.pid in {x.pid for x in node.paths}), None)
            if trunk is not None:
                s.current = s.paths.index(trunk)
        s.select(step=node.parent.key if node.parent is not None else "sources")
        if gone:
            self.toast(_("{block} removed with its branch “{name}” · Ctrl+Z").format(block=title,
                                                                                    name=", ".join(gone)))
        elif night is not None:
            self.toast(_("{block} removed · Ctrl+Z").format(block=title))
        elif node.sid is not None:                        # a session's Calibration: the one of every session goes
            self.toast(_("{block} removed · all sessions · Ctrl+Z").format(block=title))
        else:
            self.toast(_("{block} removed · Ctrl+Z").format(block=title))

    def rename(self, pid, name):
        why = self.s.name_problem(name, pid=pid)          # decision 8: never empty, never another's name
        if why:
            self.toast(_("Name refused: {why}").format(why=why))
            return
        self.s.rename(pid, name)

    def workflow_action(self, action, pid):
        s = self.s
        p = s.view(pid)
        if p is None:
            return
        if action == "rename":
            t = ask_name(self, "Rename", p.name)
            if t is not None:
                self.rename(pid, t)
        elif action == "duplicate":                        # at once: no template window after the project's start
            q = s.duplicate(p, _("{name} (copy)").format(name=p.name))
            s.log('Workflow "{name}" created', name=q.name)
            self.select(index=s.paths.index(q), step="post")
            self.toast(_("“{name}” duplicated: “{copy}” · Ctrl+Z").format(name=p.name, copy=q.name))   # (decision 11)
        elif action == "template":
            self.save_as_template(p)
        elif action == "to_post":                          # issue #52: a wire taken away, given back
            s.add_post_layer(pid, None, self)
            s.log('"{name}" sent to Post', name=p.name)
            self.toast(_('"{name}" to Post · Ctrl+Z').format(name=p.name))      # said, as the others (decision 11)
            self.request_refresh()
        elif action == "as_is":
            self.image_as_is(p)
        elif action == "delete":                           # at once, no blocking question: said, Ctrl+Z (10.1)
            why = s.channel_problem(p)                    # a fusion's channel: refused with the reason (decision 3)
            if why:
                self.toast(_('Workflow "{name}" kept: {why}').format(name=p.name, why=why))
                return
            gone = s.delete_workflow(p)                   # with the Sessions only it read (none left unwired)
            s.log('Workflow "{name}" deleted', name=p.name)
            s.select(step="sources")
            self.toast(_('Workflow "{name}" deleted with “{sessions}” · files kept · Ctrl+Z').format(
                name=p.name, sessions=", ".join(gone)) if gone else
                _('Workflow "{name}" deleted · Ctrl+Z').format(name=p.name))

    def save_as_template(self, p):
        """Save as template… (issue #47): the workflow's graph, under a name, in a tab of the gallery, kept in the
        preferences (Export… in the gallery shares it as a file)."""
        from .gallery import TABS, make_template, save_user_template, template_dialog
        s = self.s
        src = s.source_kind(p)
        if src == "fusion":
            self.toast(_("Combine channels: no template"))
            return
        got = template_dialog(self, p.name)
        if got is None:
            return
        name, tab = got
        what = _(dict(TABS)[tab])
        t = save_user_template(make_template(name, tab, _("Made from the workflow “{name}”.").format(name=p.name),
                                             src, s.graph(p)))
        s.log('Template "{name}" saved', name=t["name"])
        self.toast(_("Template “{name}” saved · {tab}").format(name=t["name"], tab=what))

    def send_to_post(self, node):
        s = self.s
        if not node.paths:
            return
        # named after the workflow that owns the block: the first going through it (control round: a shared Stacking
        # sent after a Gradient branch was chosen was named "Workflow · Gradient · Raw stack")
        p = node.paths[0]
        if node.key == "fusion":                          # its image is its workflow's: given back, or said
            if p.pid in Graph(s).to_post:
                self.toast(_("{block}: its image is already a layer of Post").format(block=p.name))
            else:
                self.workflow_action("to_post", p.pid)
            return
        if node.key == "image":                           # its own image, as it is (question 3, 2nd round)
            if not (p.source or {}).get("path"):
                self.toast(_("Image: choose its file first"))
            else:
                self.image_as_is(p)
            return
        name = s.branch_name(p.name, node.key)            # "Ha · Denoise": which workflow, which block (expl. 2)
        # a branch up to the block, for a workflow of options too (GUI exploration 3.1: no "Output" block of an exit)
        ok = next((q for q in [p] + list(node.paths) if s.can_send_to_post(q, node.key)), None)
        if ok is None and all(s.p.workflow(q.pid).blocks is not None for q in node.paths):
            # a graph's block whose image is a layer already (sent before, or its workflow's last one): said (2nd round)
            g = Graph(s)                                  # a graph of now: its own nodes (control round: compared to
            last = next((q for q in node.paths if g.leaves.get(q.pid) is not None   # the view's, never the same)
                         and g.leaves.get(q.pid) is g.node_of(q, node.key)), None)
            if last is not None and last.pid not in g.to_post and s.gives_post(last):
                self.workflow_action("to_post", last.pid)            # its wire taken away before: given back
            elif last is not None and last.pid in g.awaits_post:     # no layer yet: never "already a layer"
                self.toast(_("{block}: lights first").format(block=block_title(p, node.key)))
            else:
                self.toast(_("{block}: its image is already a layer of Post").format(
                    block=block_title(p, node.key)))
            return
        if ok is not None:                                # issue #47: a branch up to the block, its own Output
            p = ok
            try:
                q = s.send_to_post(p, node.key, name)
            except ValueError as e:
                self.toast(str(e))
                return
        else:
            q = s.add_exit(p, node.key, name)
        s.log('Layer "{name}" from {block}', name=q.name, block=block_title(p, node.key))
        self.select(index=s.paths.index(q), step="post")
        self.toast(_('Layer "{name}" ({block}) to Post').format(
            name=q.name, block=block_title(p, node.key)) + " · Ctrl+Z")

    def part_to_post(self, node, part):
        """One of the Stars block's images sent to Post as it is: a layer of its own (the stars in Screen mode)."""
        s = self.s
        p = s.path if s.path in node.paths else node.paths[0]
        word = _(dict(STAR_OUTPUTS)[part])
        if any(x.get("pid") == p.pid and x.get("part") == part for x in s.post_layers()):
            self.toast(_('{image} · "{name}": already in Post').format(image=word, name=p.name))
            return
        s.add_post_layer(p.pid, part, self)
        s.log('{image} of "{name}" sent to Post', image=dict(STAR_OUTPUTS)[part], name=p.name)
        self.toast(_('{image} · "{name}" to Post').format(image=word, name=p.name) + " · Ctrl+Z")

    def part_out_of_post(self, node, part):
        """One of the Stars/background block's images sent as it is, taken out of Post (its layers, with their filters; #29)."""
        s = self.s
        p = s.path if s.path in node.paths else node.paths[0]
        word = _(dict(STAR_OUTPUTS)[part])
        if s.remove_post_part(p.pid, part, self):
            s.log('{image} of "{name}" taken out of Post', image=dict(STAR_OUTPUTS)[part], name=p.name)
            self.toast(_('{image} of "{name}" taken out of Post').format(image=word, name=p.name) + " · Ctrl+Z")

    def new_composition(self, node):
        """A Stars and background block for this Stars block: a layer of Post that recombines its two images."""
        s = self.s
        p = s.path if s.path in node.paths else node.paths[0]
        lid = s.add_post_layer(p.pid, "combined", self)
        s.log('Combine layers for "{name}"', name=p.name)
        s.sel_layer = lid
        s.select(index=s.paths.index(p), step="composition")
        s.scope = [p.pid]
        from .post3 import tr_name
        lay = s.post_layer(lid) or {}                     # said, with its name (2nd round, decision 11)
        self.toast(_('Layer "{name}" ({block}) to Post').format(
            name=tr_name(str(lay.get("name") or p.name)), block=block_title(None, "composition")) + " · Ctrl+Z")

    def remove_layer(self, lid, node=None):
        """A Stars and background block taken out of Post (its layer, with its settings); the one Post gives by
        default (no layer yet) is taken out too (GUI exploration 3.6). Said, with Ctrl+Z."""
        s = self.s
        if lid is None and node is not None and node.paths:
            s.remove_post_part(node.paths[0].pid, "combined", self)
        else:
            s.remove_post_layer(lid, self)
        s.log("Combine layers removed")
        self.toast(_("{block} removed · Ctrl+Z").format(block=block_title(None, "composition")))
        if s.sel_step == "composition" and getattr(s, "sel_layer", None) == lid:
            s.select(step="etoiles")

    def toggle_pause(self, node):
        """Review pause: set on every workflow going through the block, or taken off when they all have it."""
        s = self.s
        paths = list(node.paths)
        if not paths or node.key in ("sources", "final"):
            return
        ref = node.ref                                    # a session's Calibration: its own ("calibration@2")
        on = not all(s.has_breakpoint(q, ref) for q in paths)
        s.set_breakpoint(paths, ref, on)
        # as the graph names it: "Gradient 1", "Calibration · Session 2" (GUI exploration round 2)
        title = s.pause_name(paths[0].pid, ref) if hasattr(s, "pause_name") else block_title(paths[0], node.key)
        if on and not all(s.has_breakpoint(q, ref) for q in paths):
            # GUI exploration 2: a pause that is not kept (a session's Calibration) is never announced: said why
            self.toast(_("No pause after a session's {block}: put it on Registration").format(block=title)
                       if node.sid is not None else _("No pause possible after {block}").format(block=title))
            return
        if on:
            s.log("Pause set after {block}", block=title)
            self.toast(_("Review pause after {block}").format(block=title) if len(paths) == 1
                       else plural(len(paths), "Review pause after {block} · {n} workflow",
                                   "Review pause after {block} · {n} workflows", block=title))
        else:
            s.log("Pause removed after {block}", block=title)
            self.toast(_("No more pause after {block}").format(block=title))

    def mark_reviewed(self, node):
        """Mark reviewed: the results are checked, the run will go through this block."""
        s = self.s
        pids = [q.pid for q in node.paths if s.review_state(q, node.ref) == "paused"]
        if not pids:
            return
        from .model import Pause
        s.mark_reviewed(Pause(node.ref, pids))
        title = s.pause_name(pids[0], node.ref) if hasattr(s, "pause_name") else block_title(node.paths[0], node.key)
        s.log("{block} reviewed", block=title)
        self.toast(_("{block} reviewed").format(block=title))

    def variant_made(self, pid, orig):
        """The variant is shown at the same block, side by side with the workflow it comes from."""
        self.select(index=self.s.index_of(pid))
        q = self.s.view(pid)
        self.toast(_('Variant "{name}" created').format(name=q.name) + " · Ctrl+Z")
        self.variantMade.emit(pid, orig)
