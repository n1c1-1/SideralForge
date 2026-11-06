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
and ``openPost`` (Inspect or Post), ``addSources`` and ``addSession`` (import frames), ``variantMade`` (Inspect opens
the variant compared with its original), ``changed`` (the Run button and the status line).

Issue #43: the sessions calibrated each on its own sit as rows of a Session block and its Calibration block before the
chain; their wires join into one bar before Registration (``_join``). A block the user cut is drawn hatched in grey
with its wire across it; a block greyed by a cut has a dashed amber outline, ⊘ and its reason: the image passes
through both. A block is cut or restored by its menu, its panel's switch or the D key. A workflow from an image starts
with an Image block on its own lane.

The scene's items: ``BlockItem`` (a block), ``GhostItem`` (where a dragged tool can go), ``BandItem`` (a phase band);
``ToolCard`` is a card of the toolbar.
"""
from __future__ import annotations

from PySide6.QtCore import QMimeData, QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDrag,
    QFont,
    QFontMetrics,
    QFontMetricsF,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core import workflow as wfl
from . import prefs
from .blocks import (
    BREAK_COL,
    BY_KEY,
    COLUMNS,
    OPTIONAL,
    PAUSE_COL,
    PHASES,
    REVIEWED_COL,
    STAR_OUTPUTS,
    STATE_WORD,
    STEPS,
    TOOL_COLUMN,
    TOOLBOX,
    TOOLS,
    Graph,
    Node,
    block_title,
    node_title,
)
from .dialogs import ask_name
from .gallery import TemplateGallery
from .i18n import _, hm, plural
from .model import NotInEngine, cuttable
from .panel import Panel
from .widgets import ACCENT, BAD, NEW, OK, WARN, clear, label

BW, BH = 120, 54            # block size (a column is wider when its title needs it: TITLE_PAD)
GAP = 12                    # between two columns: where the branches split
LANE = 80                   # lane pitch
SROW = BH + 14              # the pitch of the sessions' rows (issue #43)
CUT_COL, GREY_COL = "#6b7280", "#c9a227"
TOP = 74                    # room for the phase labels
MARGIN = 8                  # around the graph, left and right
MIN_SCALE = 0.6             # the smallest the default view goes to show the whole chain; below, the text cannot be read:
#                             the view keeps this scale, shows the left end (Frames) and scrolls right. Show all goes lower
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


class BlockItem(QGraphicsItem):
    def __init__(self, view, node: Node, title, summary, state, selected, dim, removable, w: float = BW, bp=None):
        super().__init__()
        self.view, self.node, self.w = view, node, w
        self.title, self.summary, self.state = title, summary, state
        self.selected, self.dim, self.removable = selected, dim, removable
        self.bp = bp                           # None, or Graph.breakpoint: (state, workflows that pause here, of how many)
        self.hover = False
        self.target = False                    # the Post block, while an output is pulled over it
        self._wire = False
        self.ports = PORTS.get(node.key, ())
        if node.key == "fusion" and node.paths and ((node.paths[0].source or {}).get("fusion") or {}).get("stars"):
            self.ports = PORTS["etoiles"]                 # a starless fusion: its two images, as the Split block's
        self.port = bool(self.ports)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        tip = f"{title}\n{summary}\n" + (_(STATE_WORD[state]) if node.key not in ("sources", "final", "session") else "")
        if node.cut:
            tip += "\n" + _("Cut: the image passes through. D or the menu restores it.")
        elif node.grey:
            tip += "\n" + _("Cannot run: {why}. The image passes through.").format(why=node.grey)
        if node.key in EXITS:
            tip += "\n" + _("Output ●: drag it onto Post to make a layer")
        elif node.key == "etoiles":
            tip += "\n" + _("● top: the starless image; ● bottom: the stars alone. Drag one onto Post to make it a "
                            "layer, or onto Composition for a Combine stars & background layers block")
        if bp is not None:
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
        tw = w - 18 - (18 if self.removable and self.hover else 0)
        # a workflow's name loses its middle (two variants differ at their end; the whole name is in the tooltip), the
        # engine's words their end
        mode = Qt.ElideMiddle if self.node.key == "post" else Qt.ElideRight
        title = ("⊘ " if self.node.grey else "") + self.title
        p.drawText(QRectF(11, 6, tw, 20), Qt.AlignLeft | Qt.AlignVCenter,
                   QFontMetrics(f).elidedText(title, mode, int(tw)))
        f2 = QFont("Segoe UI", 8.5)
        p.setFont(f2)
        p.setPen(QColor("#8b93a1"))
        p.drawText(QRectF(11, 28, w - 15, 18), Qt.AlignLeft | Qt.AlignVCenter, elide(self.summary, f2, w - 15))
        if self.removable and self.hover:
            p.setPen(QPen(QColor("#e06c6c"), 1.6))
            xr = self._x_rect().adjusted(4, 4, -4, -4)
            p.drawLine(xr.topLeft(), xr.bottomRight())
            p.drawLine(xr.topRight(), xr.bottomLeft())
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
        self.setCursor(Qt.OpenHandCursor if self._on_port(e.pos()) != NO_PORT else Qt.PointingHandCursor)

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
                self.view.blockClicked.emit(self.node)
        e.accept()

    def mouseMoveEvent(self, e):                                      # noqa: N802
        if self._wire:
            self.view.move_wire(e.scenePos())

    def mouseReleaseEvent(self, e):                                   # noqa: N802
        if self._wire:
            self._wire = False
            self.view.end_wire(e.scenePos())
        e.accept()

    def mouseDoubleClickEvent(self, e):                               # noqa: N802
        if self.node.key == "composition":
            self.view.openLayer.emit(self.node.lid)
        else:
            self.view.blockOpened.emit(self.node)
        e.accept()

    def contextMenuEvent(self, e):                                    # noqa: N802
        self.view.block_menu(self.node, e.screenPos())
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
    """A dashed place: where a dragged tool can go, or the next workflow."""

    def __init__(self, rect: QRectF, text, on_click=None, pid=None):
        super().__init__()
        self.rect, self.text, self.on_click, self.pid = rect, text, on_click, pid
        self.hot = False
        self.setAcceptHoverEvents(True)
        if on_click:
            self.setCursor(Qt.PointingHandCursor)

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
        p.drawText(self.rect.adjusted(12, 8, -8, 0), Qt.AlignLeft | Qt.AlignTop, self.text.upper())


class GraphView(QGraphicsView):
    blockClicked = Signal(object)          # Node
    blockOpened = Signal(object)           # Node: double-click, to Inspect
    newWorkflow = Signal()
    toolDropped = Signal(str, int)         # tool key, workflow pid
    removeTool = Signal(object, object)    # Node, list of pids or None (all the workflows of the block)
    workflowMenu = Signal(str, int)        # action, pid
    sendToPost = Signal(object)            # Node whose output was pulled to the Post block
    pauseToggled = Signal(object)          # Node: "Pause here for review" was chosen (set it, or take it off)
    reviewMarked = Signal(object)          # Node: "Mark as reviewed" was chosen
    partToPost = Signal(object, str)       # a Stars node's image (starless | stars) pulled onto Post
    partOutOfPost = Signal(object, str)    # a Stars node's image sent as it is: taken out of Post (#29)
    newComposition = Signal(object)        # a Stars node's port pulled onto Composition: a new Stars and background block
    layerRemoved = Signal(int)             # a Stars and background block, or an image sent as it is: its Post layer
    openLayer = Signal(int)                # Open in Post (a Post layer's id)
    cutToggled = Signal(object, object)    # Node, list of pids (None: every workflow of the block): cut or restore it
    newSession = Signal()                  # "+ Session" (issue #43)
    sessionMenu = Signal(str, int)         # action (rename, remove), session id

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
        self.b_all = QPushButton(_("Show all"), self)
        self.b_all.setCheckable(True)
        self.b_all.setObjectName("toggle")
        self.b_all.setToolTip(_("The whole graph in the window, smaller; off: readable size, scrolling"))
        self.b_all.toggled.connect(lambda _x: self.fit())

    # ------------------------------------------------------------ layout
    def columns(self, g: Graph):
        used = {n.key for n in g.nodes}
        used.discard("session")
        if self.drag_tool:
            used.add(TOOL_COLUMN[self.drag_tool])
        if "etoiles" in used or "fusion" in used:   # the Composition section: the stars recombined, the fusions
            used.add("composition")
        return [c for c in COLUMNS if c not in OPTIONAL or c in used]

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
        a block never runs behind it."""
        out: list[int] = []
        todo = [g.root]
        while todo:
            n = todo.pop()
            if not n.children:
                out += [p.pid for p in n.paths if p.pid not in out]
            todo += reversed(n.children)
        return out + [p.pid for p in g.s.paths if p.pid not in out]

    def rebuild(self):
        sc = self.scene()
        # issue #24: a change (or a stage event of a run) builds the graph again; where the user had scrolled stays,
        # when the graph keeps its size
        was = (sc.sceneRect(), self.horizontalScrollBar().value(), self.verticalScrollBar().value())
        sc.clear()
        self.ghosts = []
        self._sel_item = None
        self.final_item = None
        self.b_all.setText(_("Show all"))
        self.b_all.setToolTip(_("The whole graph in the window, smaller; off: readable size, scrolling"))
        self.b_all.adjustSize()
        s = self.s
        g = Graph(s)
        self.graph = g
        cols = self.columns(g)
        cw = {c: self.column_width(c, [p.name for p in s.paths]) for c in cols}
        cx, x = {}, 0.0
        for c in cols:
            cx[c] = x
            x += cw[c] + GAP
        self._next_x = {c: cx[c] + cw[c] + GAP for c in cols}       # where the column after it starts
        order = self.lane_order(g)
        frames = [pid for pid in order if getattr(s.view(pid), "source", None) is None]
        order = frames + [pid for pid in order if pid not in frames]      # the image workflows' lanes last
        n_lanes = max(1, len(s.paths))
        # issue #43: the sessions' rows, centred on the lanes of the workflows from the frames
        ns = len(g.sessions)
        f_mid = (len(frames) - 1) * LANE / 2 if frames else 0.0
        top = TOP + max(0.0, (ns - 1) * SROW / 2 - f_mid)
        lanes = {pid: top + i * LANE for i, pid in enumerate(order)}
        mid = top + (n_lanes - 1) * LANE / 2 if s.paths else top + LANE / 2
        fmid = top + f_mid
        for i, (sn, cn) in enumerate(g.sessions):
            sn.y = cn.y = fmid + (i - (ns - 1) / 2) * SROW
        for n in g.nodes:
            if n.key == "session" or n.sid is not None:
                pass
            elif n.key in ("sources", "final") or not n.paths:
                n.y = fmid if n.key == "sources" else mid
            else:
                ys = [lanes[p.pid] for p in n.paths]
                n.y = (min(ys) + max(ys)) / 2
            col = "sources" if n.key in ("session", "image") else n.key
            n.col = cx[col]
            n.w = cw[col]
        seen: dict[int, int] = {}
        for n in g.compositions:                   # two blocks of one workflow: one a little below the other
            k = seen.get(n.paths[0].pid, 0)
            seen[n.paths[0].pid] = k + 1
            n.y += k * (BH + 6)
        self._lanes, self._cx, self._cw = lanes, cx, cw
        bottom = max(top + n_lanes * LANE, fmid + (ns + 1) / 2 * SROW) + LANE
        # phase bands
        for title, a, b in PHASES:
            span = [c for c in COLUMNS[COLUMNS.index(a): COLUMNS.index(b) + 1] if c in cx]
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
            self._wire(sn, cn, True)
        for t in (firsts if g.sessions else []):   # each first block from the sessions its workflows stack
            src = [cn for _sn, cn in g.sessions if set(map(id, cn.paths)) & set(map(id, t.paths))]
            if len(src) > 1:
                self._join(src, [t], cur)
            elif src:
                self._wire(src[0], t, cur is not None and cur in t.paths)
        # issue #43: the fusions, from the outputs of the workflows they combine
        for n in g.nodes:
            for a in g.fusion_inputs.get(id(n), []):
                self._wire(a, n, cur is not None and (cur in n.paths or cur in a.paths))
        # issue #43: "+ Session" under the sessions (or under the one Session block), where a session goes
        if s.sources_loaded and g.root.paths:
            last = g.sessions[-1][0] if g.sessions else g.root
            gh = GhostItem(QRectF(cx["sources"], last.y + BH / 2 + 8, cw["sources"], BH * 0.6), "+  " + _("Session"),
                           self.newSession.emit)
            gh.setToolTip(_("Add a session: a night of lights, with its own calibration frames"))
            sc.addItem(gh)
        for p in s.paths:                          # a workflow's Output image, when it is a layer of Post
            if p.pid in g.to_post:
                self._wire(g.leaves[p.pid], g.final, cur is not None and p is cur)
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
        for n in g.nodes:
            if n.key not in ("sources", "final") and not n.paths:
                continue
            if n.key == "sources" and (g.sessions or (s.paths and not g.root.paths)):
                continue                                  # the sessions' blocks stand for it; or only image workflows
            title = node_title(s, n)
            removable = n.key in OPTIONAL_BLOCKS or n.key == "composition"
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
        y = TOP + len(s.paths) * LANE - BH / 2 if s.paths else mid - BH / 2
        x0 = cx["calibration"]
        last = next(k for k in ("post", "etoiles", "debruitage", "couleur") if k in cx)   # the outputs' column
        x1 = cx[last] + cw[last]
        gh = GhostItem(QRectF(x0, y, x1 - x0, BH), "+  " + _("New workflow"), self.newWorkflow.emit)
        sc.addItem(gh)
        # where the dragged tool can go
        if self.drag_tool:
            key = self.drag_tool
            for p in s.paths:
                if s.has_tool(p.opts, key) or (p.end and COLUMNS.index(TOOL_COLUMN[key]) > COLUMNS.index(p.end)):
                    continue
                x = cx[TOOL_COLUMN[key]]
                short = block_title(None, TOOL_COLUMN[key])
                gh = GhostItem(QRectF(x + 4, lanes[p.pid] - BH / 2 + 4, cw[TOOL_COLUMN[key]] - 8, BH - 8), short,
                               None, p.pid)
                gh.setZValue(8)
                sc.addItem(gh)
                self.ghosts.append(gh)
        r = sc.itemsBoundingRect().adjusted(-MARGIN, -10, MARGIN, 24)
        sc.setSceneRect(r)
        self.show_wait()
        self.fit()
        old, h0, v0 = was
        if not old.isEmpty() and abs(old.width() - r.width()) < 1 and abs(old.height() - r.height()) < 1:
            self.horizontalScrollBar().setValue(h0)
            self.verticalScrollBar().setValue(v0)

    def show_wait(self):
        """The line over the graph while the worker reads the project (or says why it could not)."""
        s = self.s
        if s.error:
            text = _("This project cannot be read: {why}").format(why=s.error)
        else:
            text = "" if s.ready else _("Reading the project…")
        self.wait.setText(text)
        self.wait.setVisible(bool(text))
        self.wait.setGeometry(0, self.height() // 2 - 40, self.width(), 30)
        self.wait.raise_()

    # ------------------------------------------------------------ pulling an output to the post
    def start_wire(self, item, port=None):
        """A port is pulled: its wire follows the mouse; a Stars block's port shows where a Stars and background block
        can go (the Composition section, on its workflow's lane)."""
        self._pull = item
        self._pull_port = port
        self._pull_path = self.scene().addPath(QPainterPath(), QPen(QColor(ACCENT), 2, Qt.DashLine))
        self._pull_path.setZValue(9)
        self._comp_ghost = None
        if item.node.key == "etoiles" and "composition" in getattr(self, "_cx", {}):
            x, w = self._cx["composition"], self._cw["composition"]
            y = item.node.y - BH / 2
            self._comp_ghost = GhostItem(QRectF(x + 4, y + 4, w - 8, BH - 8), "+  " + _("Combine stars & background layers"))
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
        x, y = self.mapFromScene(pos).x(), self.mapFromScene(pos).y()
        if x > self.viewport().width() - 20 or y > self.viewport().height() - 20:
            self.ensureVisible(QRectF(pos, pos), 20, 20)

    def end_wire(self, pos):
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

    def _wire(self, a: Node, b: Node, strong: bool):
        """A wire from a block to the next block of its workflows. A branch bends in the gap right after the block it
        leaves (the last one its workflows share), then runs straight on its own lane past the columns it skips."""
        x1, y1 = a.col + a.w, a.y
        x2, y2 = b.col, b.y
        path = QPainterPath(QPointF(x1, y1))
        xb = min(x2, self._next_x.get(a.key, x2)) if b.key != "final" else x2     # where the bend ends
        dx = max(30.0, (xb - x1) * 0.5) if abs(y2 - y1) > 1 else 0
        if dx:
            path.cubicTo(QPointF(x1 + dx, y1), QPointF(xb - dx, y2), QPointF(xb, y2))
        if xb < x2 or not dx:
            path.lineTo(QPointF(x2, y2))
        it = self.scene().addPath(path, QPen(QColor("#9aa3b2" if strong else "#3d4452"), 2.2 if strong else 1.6))
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
            self.scene().addPath(path, pen).setZValue(1)
        bar = self.scene().addPath(_line(xb, min(ys), xb, max(ys)), QPen(QColor("#9aa3b2"), 3.0))
        bar.setZValue(1)
        bar.setToolTip(_("The sessions join here: they are registered and stacked together"))
        for t in targets:
            strong = cur is not None and cur in t.paths
            path = QPainterPath(QPointF(xb, t.y))
            path.lineTo(QPointF(t.col, t.y))
            self.scene().addPath(path, QPen(QColor("#9aa3b2" if strong else "#3d4452"), 2.2 if strong else 1.6)) \
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
        it = self.scene().addPath(path, QPen(col, 2.0 if strong else 1.5, Qt.DashLine if direct else Qt.SolidLine))
        it.setZValue(4 if direct else 2)                  # behind the blocks (their text stays readable)
        it.setToolTip(_("{image}: to Post as a layer").format(image=word) if direct else word)

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

    def fit(self):
        """The default view shows the whole chain, from Frames to Post, at any window width: the scale that fits (at most
        1.3), lowered down to MIN_SCALE; when even that does not fit, the left end (Frames) is shown, top left, and the
        user scrolls right (or down). It is done again whenever the view changes size (the window, the panel)."""
        r = self.scene().sceneRect()
        if r.isEmpty() or self._fitting:
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

    def resizeEvent(self, e):                                         # noqa: N802
        super().resizeEvent(e)
        self.fit()
        self.toast.move(16, self.height() - self.toast.height() - 16)
        self.b_all.adjustSize()
        self.b_all.move(self.width() - self.b_all.width() - 14, 10)
        self.wait.setGeometry(0, self.height() // 2 - 40, self.width(), 30)

    def show_toast(self, text):
        self.toast.setText(text)
        self.toast.adjustSize()
        self.toast.move(16, self.height() - self.toast.height() - 16)
        self.toast.show()
        self.toast.raise_()
        self._toast_timer.start(5000)

    # ------------------------------------------------------------ menus
    def block_menu(self, node: Node, pos):
        m = QMenu(self)
        if node.key == "post" or node.handle:                # the workflow's handle (#17: its Stars block)
            p = node.paths[0]
            m.addAction(_("Rename")).triggered.connect(lambda: self.workflowMenu.emit("rename", p.pid))
            m.addAction(_("Duplicate")).triggered.connect(lambda: self.workflowMenu.emit("duplicate", p.pid))
            m.addSeparator()
            m.addAction(_("Delete workflow")).triggered.connect(lambda: self.workflowMenu.emit("delete", p.pid))
            if node.handle:
                m.addSeparator()
        elif node.key == "composition":
            m.addAction(_("Open in Post")).triggered.connect(lambda: self.openLayer.emit(node.lid))
            m.addSeparator()
            m.addAction(_("Delete block")).triggered.connect(lambda: self.remove_block(node, None))
            m.exec(pos)
            return
        if node.key == "etoiles" and len(node.paths) == 1:
            for what, text in (("starless", "Send the starless image to Post"), ("stars", "Send the stars alone to Post")):
                if self.graph is not None and (node, what) in self.graph.direct:      # sent: it can be taken back (#29)
                    out = "Take the starless image out of Post" if what == "starless" else "Take the stars alone out of Post"
                    m.addAction(_(out)).triggered.connect(lambda _c=False, w=what: self.partOutOfPost.emit(node, w))
                else:
                    m.addAction(_(text)).triggered.connect(lambda _c=False, w=what: self.partToPost.emit(node, w))
            m.addAction(_("New Combine stars & background layers block")).triggered.connect(
                lambda: self.newComposition.emit(node))
            m.addSeparator()
        if node.key in OPTIONAL_BLOCKS:
            if node.shared:
                m.addAction(_("Remove from the {n} workflows").format(n=len(node.paths))).triggered.connect(
                    lambda: self.remove_block(node, None))
                for p in node.paths:
                    m.addAction(_('Remove from "{name}" only').format(name=p.name)).triggered.connect(
                        lambda _x=False, q=p: self.remove_block(node, [q.pid]))
            else:
                m.addAction(_("Delete block")).triggered.connect(lambda: self.remove_block(node, None))
        if node.key == "session":
            m.addAction(_("Rename session")).triggered.connect(lambda: self.sessionMenu.emit("rename", node.sid))
            m.addAction(_("Remove from the project")).triggered.connect(lambda: self.sessionMenu.emit("remove", node.sid))
            m.addSeparator()
            m.addAction(_("Add a session…")).triggered.connect(lambda: self.newSession.emit())
            m.exec(pos)
            return
        if node.sid is None and cuttable(node.key) and node.paths:
            if not m.isEmpty():
                m.addSeparator()
            word = _("Restore") if node.cut else _("Cut")
            if len(node.paths) > 1:
                m.addAction(_("{action} in the {n} workflows").format(action=word, n=len(node.paths))).triggered.connect(
                    lambda: self.cutToggled.emit(node, None))
                for p in node.paths:
                    m.addAction(_('{action} in "{name}" only').format(action=word, name=p.name)).triggered.connect(
                        lambda _x=False, q=p: self.cutToggled.emit(node, [q.pid]))
            else:
                a = m.addAction(_("{action} this block").format(action=word) + "\tD")
                a.triggered.connect(lambda: self.cutToggled.emit(node, None))
        if node.key not in ("sources", "final"):
            if not m.isEmpty():
                m.addSeparator()
            m.addAction(_("Inspect")).triggered.connect(lambda: self.blockOpened.emit(node))
            if node.key in EXITS:
                m.addAction(_("Send the output to Post")).triggered.connect(lambda: self.sendToPost.emit(node))
            m.addSeparator()
            bp = self.graph.breakpoint(node) if self.graph is not None else None
            a = m.addAction(_("Pause here for review"))
            a.setCheckable(True)
            a.setChecked(bp is not None and bp[1] == bp[2])
            a.setToolTip(_("The run stops after this block, so that you can check it before it goes on."))
            a.triggered.connect(lambda: self.pauseToggled.emit(node))
            if bp is not None and bp[0] == "paused":
                m.addAction(_("Mark as reviewed")).triggered.connect(lambda: self.reviewMarked.emit(node))
        if not m.isEmpty():
            m.exec(pos)

    def remove_block(self, node: Node, pids):
        if node.key == "composition":
            self.layerRemoved.emit(node.lid)
            return
        self.removeTool.emit(node, pids)

    # ------------------------------------------------------------ drag and drop of tools
    def _tool(self, e):
        t = e.mimeData().text() if e.mimeData().hasText() else ""
        return t[5:] if t.startswith("tool:") else None

    def dragEnterEvent(self, e):                                      # noqa: N802
        k = self._tool(e)
        if not k or not self.s.paths:
            e.ignore()
            return
        self.drag_tool = k
        self.rebuild()
        e.acceptProposedAction()

    def _ghost_at(self, pos):
        sp = self.mapToScene(pos)
        for g in self.ghosts:
            if g.rect.adjusted(-10, -10, 10, 10).contains(sp):
                return g
        return None

    def dragMoveEvent(self, e):                                       # noqa: N802
        hit = self._ghost_at(e.position().toPoint())
        for g in self.ghosts:
            if g.hot != (g is hit):
                g.hot = g is hit
                g.update()
        if hit:
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragLeaveEvent(self, e):                                      # noqa: N802
        self.drag_tool = None
        self.rebuild()

    def dropEvent(self, e):                                           # noqa: N802
        hit = self._ghost_at(e.position().toPoint())
        k = self.drag_tool
        self.drag_tool = None
        if hit and k:
            e.acceptProposedAction()
            pid = hit.pid
            QTimer.singleShot(0, lambda: self.toolDropped.emit(k, pid))
        else:
            self.rebuild()


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


class CombineCard(QFrame):
    """The Combine stars & background layers block in the toolbar (issue #36): a click adds one for the current
    workflow's split block (a layer of Post that puts its stars back on its starless image); greyed without one."""

    def __init__(self, page, split, parent=None):
        super().__init__(parent)
        self.page, self.split = page, split
        on = split is not None
        self.setObjectName("tool")
        self.setCursor(Qt.PointingHandCursor if on else Qt.ArrowCursor)
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
        v.addWidget(label(_("a layer of Post") if on else _("after a split block"), "small"))
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
    addSources = Signal()                  # "Import folder…" of the Frames block
    addSession = Signal()
    variantMade = Signal(int, int)         # a variant (id) was made from this workflow (id): Inspect compares them
    changed = Signal()                     # the page was built again: the window updates its Run button and status
    toasted = Signal(str)

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
        tools.setStyleSheet("QFrame#tools { background: #1c1f26; border-bottom: 1px solid #2a2f38; }")
        self.tl = QHBoxLayout(tools)
        self.tl.setContentsMargins(12, 8, 12, 8)
        self.tl.setSpacing(8)
        cv.addWidget(tools)
        self.graph = GraphView(project)
        cv.addWidget(self.graph, 1)
        ph.addWidget(centre, 1)
        self.panel = Panel(project, steps)
        ph.addWidget(self.panel)
        gv = self.graph
        gv.blockClicked.connect(self.select_node)
        gv.blockOpened.connect(lambda n: (self.select_node(n), self.openBlock.emit()))
        gv.newWorkflow.connect(lambda: self.new_workflow())
        gv.toolDropped.connect(self.add_tool)
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
        gv.sessionMenu.connect(self.session_action)
        pn = self.panel
        pn.changed.connect(self.request_refresh)
        pn.openTune.connect(self.openBlock.emit)
        pn.openPost.connect(self.openPost.emit)
        pn.variantMade.connect(self.variant_made)
        pn.addSources.connect(self.addSources.emit)
        pn.addSession.connect(self.addSession.emit)
        pn.renamed.connect(self.rename)
        if hasattr(pn, "selectBlock"):
            pn.selectBlock.connect(lambda key: self.select(step=key))
        if hasattr(pn, "restoreBlock"):
            pn.restoreBlock.connect(lambda key: self.toggle_cut(Graph(self.s).node_of(self.s.path, key), None))
        pn.toast.connect(self.toast)
        pn.removeTool.connect(lambda: self.remove_tool(self.current_node(), None))
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
            ret.activated.connect(self.openBlock.emit)
        cut = QShortcut(QKeySequence("D"), self.graph)          # cut or restore the selected block (issue #43)
        cut.setContext(Qt.WidgetWithChildrenShortcut)
        cut.activated.connect(lambda: self.toggle_cut(self.current_node(), None))
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
        self.panel.refresh(self.current_node(self.graph.graph))
        self.build_tools()
        if s.ready:
            sec = s.total_seconds()
            last = self._last
            if last is not None and s.paths and len(s.paths) == last[1] and sec - last[0] >= 30:
                self.toast(_("+ {t} to run").format(t=("" if s.cost().measured else "≈ ") + hm(sec - last[0])))
            self._last = (sec, len(s.paths))
        self.changed.emit()

    def toast(self, text: str):
        self.graph.show_toast(text)
        self.toasted.emit(text)

    def current_node(self, g=None):
        g = g or Graph(self.s)
        k = self.s.sel_step
        if k == "sources":
            return g.root
        if k == "final":
            return g.final
        if k == "composition":
            return g.node_of(self.s.path, k, getattr(self.s, "sel_layer", None))
        return g.node_of(self.s.path, k)

    def build_tools(self):
        clear(self.tl)
        s = self.s
        p = s.path
        head = QVBoxLayout()
        head.setSpacing(0)
        head.addWidget(label(_("Add step"), "h2"))
        head.addWidget(label(_('for "{name}"').format(name=p.name) if p is not None else "", "small"))
        self.tl.addLayout(head)
        self.tl.addSpacing(6)
        spec = prefs.spec()
        for k in TOOLBOX:
            if k == "deconv" and not spec:
                continue                                    # ◇: in Spec mode only
            new = k == "deconv"
            if p is None or not s.ready:
                state, text = "off", ""
            elif new:
                state, text = "off", _("not in the engine yet")
            elif getattr(p, "source", None) is not None:     # an image has no half-stacks (issue #43)
                state, text = "off", _("needs the half-stacks of a stack")
            elif p.end and COLUMNS.index(TOOL_COLUMN[k]) > COLUMNS.index(p.end):
                state, text = "in", _("after the output")
            elif s.has_tool(p.opts, k):
                state, text = "in", _("already there")
            else:
                dt = s.tool_cost(p.opts, k)
                sure = s.opts_measured(s.with_tool(p.opts, k)[0]) if hasattr(s, "opts_measured") else True
                state, text = "add", ((f"+{hm(dt)}" if sure else f"≈ +{hm(dt)}") if dt >= 30 else _("same duration"))
            card = ToolCard(self, k, state, text, new)
            card.setEnabled(p is not None and not new and not (state == "off" and getattr(p, "source", None)))
            self.tl.addWidget(card)
        # issue #36: the Combine block is added here too, not only from the split block's menu or its dots
        g = getattr(self.graph, "graph", None)
        split = g.node_of(p, "etoiles") if (g is not None and p is not None and s.ready) else None
        self.combine_card = CombineCard(self, split)
        self.tl.addWidget(self.combine_card)
        self.tl.addStretch(1)

    # ------------------------------------------------------------ selection
    def select_node(self, node):
        s = self.s
        s.pending_change = None
        if node.key not in ("sources", "final") and node.paths:
            if s.path not in node.paths:
                s.current = s.paths.index(node.paths[0])
            s.scope = [q.pid for q in node.paths]
        else:
            s.scope = None
        if node.key == "composition":
            s.sel_layer = node.lid
        s.select(step=node.ref)

    def select(self, index=None, step=None):
        s = self.s
        s.pending_change = None
        s.select(index, step)
        node = self.current_node()
        s.scope = [q.pid for q in node.paths] if node is not None and node.key not in ("sources", "final") else None

    # ------------------------------------------------------------ actions
    def new_workflow(self, start=None):
        """New workflow: the template gallery (issue #43), or a copy of a workflow (``start``)."""
        d = TemplateGallery(self.s, start, self)
        d.accepted.connect(lambda: self.make_workflow(d))
        d.open()                                           # window-modal, without blocking (a new project opens it)
        self._gallery = d
        return d

    def make_workflow(self, d):
        """The workflow the gallery's choice makes: a copy, from an image, or from the frames."""
        s = self.s
        t = d.choice
        if t is None:
            return
        name = d.name.text().strip() or _("New workflow")
        if t.get("copy") is not None:                      # a copy keeps its options, settings and exit
            p = s.duplicate(t["copy"], name)
        elif t.get("kind") in ("filters", "fusion", "dualband"):
            p = self.make_filters(t, name, d.opts)
            if p is None:
                return
        elif t.get("source") == "image":
            p = s.add_image_workflow(name, d.image, d.opts, stretched=bool(t.get("stretched")))
        else:
            with s.batch():
                if t.get("sessions") and s.stack_mode == "together" and len(s.p.sessions) <= 1:
                    s.set_stack_mode("per_session")
                p = s.add_path(name, d.opts, None, end=t.get("end"))
        s.log('Workflow "{name}" created', name=p.name)
        self.select(index=s.paths.index(p), step="post")

    def make_filters(self, t: dict, name: str, opts: dict):
        """A filters template (issue #43): one workflow per filter (made when the filter has none yet), then for a
        palette the Fusion block of their outputs; a dual-band one: a colour workflow, then its fusion. Returns the
        workflow shown (the fusion, or the first filter's)."""
        s = self.s
        with s.batch():
            if t["kind"] == "dualband":
                colour = next((q for q in s.paths if q.source is None and not s.is_mono(q) and not q.end), None)
                if colour is None:
                    colour = s.add_path(s.unique_name(_("Dual band")), opts)
                # option A: the starless channels, the stars of the colour image itself (their own colours)
                return s.add_fusion(name, t["palette"], [{"pid": colour.pid, "role": "Ha", "channel": "R"},
                                                         {"pid": colour.pid, "role": "OIII", "channel": "GB"}],
                                    stars={"mode": "workflow", "pid": colour.pid})
            filters = sorted(s.filters()) if t["kind"] == "filters" else list(t["filters"])
            made = [(f, s.filter_workflow(f, opts)) for f in filters]
            made = [(f, q) for f, q in made if q is not None]
            if not made:
                self.toast(_("No session with a filter: set the filter of your sessions first (their Session block)"))
                return None
            if t["kind"] == "filters":
                return made[0][1]
            # option A: each channel's stars taken off, the starless palette; the stars: of the channels (RGB, LRGB:
            # their own colours), else of H-alpha, neutral (narrowband stars in a palette come out magenta)
            roles = [f for f, _q in made]
            stars = {"mode": "palette"} if t["palette"] in ("RGB", "LRGB") or "Ha" not in roles else                 {"mode": "channel", "role": "Ha"}
            return s.add_fusion(name, t["palette"], [{"pid": q.pid, "role": f, "channel": "L"} for f, q in made],
                                stars=stars)

    def toggle_cut(self, node, pids):
        """Cut a block, or restore it (issue #43): in ``pids`` only, else in every workflow going through it. Its
        settings and results stay; Undo takes it back."""
        s = self.s
        if node is None or node.sid is not None or not cuttable(node.key) or not node.paths:
            if node is not None and node.key in ("empilement", "post"):
                self.toast(_("Stacking and Output cannot be cut"))
            return
        paths = [q for q in node.paths if pids is None or q.pid in pids]
        on = not node.cut
        try:
            done = s.set_cut(paths, node.key, on)
        except ValueError as e:
            self.toast(str(e))
            return
        if done:
            name = block_title(None, node.key)
            where = paths[0].name if len(paths) == 1 else plural(len(paths), "{n} workflow", "{n} workflows")
            self.toast((_("{block} cut in {where}: the image passes through") if on else
                        _("{block} restored in {where}")).format(block=name, where=where) + " · " +
                       _("Ctrl+Z undoes it"))
            s.log("Block {block} cut" if on else "Block {block} restored", block=node.key)

    def session_action(self, action: str, sid: int):
        s = self.s
        if action == "rename":
            name = ask_name(self, _("Rename session"), s.session(sid).name)
            if name:
                s.rename_session(sid, name)
        elif action == "remove":
            if len(s.p.sessions) <= 1:
                self.toast(_("The last session stays"))
                return
            if QMessageBox.question(self, _("Remove from the project"), _(
                    "Take the session “{name}” out of the project? Its files are not touched.").format(
                    name=s.session(sid).name)) == QMessageBox.Yes:
                s.remove_session(sid)

    def add_tool(self, key, pid):
        s = self.s
        p = s.view(pid)
        if p is None or s.has_tool(p.opts, key):
            return
        try:
            new, notes = s.with_tool(p.opts, key)
        except (wfl.OptionRefused, NotInEngine, ValueError) as e:
            self.toast(_(str(e)))
            return
        if new == p.opts and notes:                         # ◇: nothing to add
            self.toast("◇ " + _("Not in the engine yet."))
            return
        s.set_opts(p, new)
        s.log('{block} added to "{name}"', block=block_title(p, TOOL_COLUMN[key]), name=p.name)
        s.select(index=s.paths.index(s.view(pid)), step=TOOL_COLUMN.get(key, key))
        node = Graph(s).node_of(s.view(pid), s.sel_step)
        s.scope = [q.pid for q in node.paths] if node is not None else [pid]
        if notes:
            self.toast(" · ".join(_(n) for n in notes))

    def remove_tool(self, node, pids):
        s = self.s
        if node is None:
            return
        if node.key == "composition":
            self.remove_layer(node.lid)
            return
        key = node.key
        with s.batch():
            for p in [q for q in node.paths if pids is None or q.pid in pids]:
                new, notes = s.without_tool(p.opts, key)
                if new is None:
                    self.toast(_('{block} cannot be removed from "{name}": {why}').format(
                        block=block_title(p, key), name=p.name, why=_(notes[0])))
                    continue
                s.set_opts(p, new)
                s.log('{block} removed from "{name}"', block=block_title(p, key), name=p.name)
                if notes:
                    self.toast(" · ".join(_(n) for n in notes))
        s.select(step=node.parent.key if node.parent is not None else "sources")

    def rename(self, pid, name):
        self.s.rename(pid, name)

    def workflow_action(self, action, pid):
        s = self.s
        p = s.view(pid)
        if p is None:
            return
        if action == "rename":
            t = ask_name(self, "Rename", p.name)
            if t:
                self.rename(pid, t)
        elif action == "duplicate":
            self.new_workflow(p)
        elif action == "delete":
            if QMessageBox.question(self, _("Delete"), _('Delete the workflow "{name}"?').format(name=p.name)) \
                    == QMessageBox.Yes:
                s.remove_path(s.paths.index(p))
                s.log('Workflow "{name}" deleted', name=p.name)
                s.select(step="sources")

    def send_to_post(self, node):
        s = self.s
        if not node.paths:
            return
        p = s.path if s.path in node.paths else node.paths[0]
        name = {"empilement": _("Raw stack")}.get(node.key, _("{block} output").format(block=block_title(p, node.key)))
        q = s.add_exit(p, node.key, name)
        s.log('Layer "{name}" from {block}', name=q.name, block=block_title(p, node.key))
        self.select(index=s.paths.index(q), step="post")
        self.toast(_('Layer "{name}": output of {block}, sent to Post').format(
            name=q.name, block=block_title(p, node.key)))

    def part_to_post(self, node, part):
        """One of the Stars block's images sent to Post as it is: a layer of its own (the stars in Screen mode)."""
        s = self.s
        p = s.path if s.path in node.paths else node.paths[0]
        word = _(dict(STAR_OUTPUTS)[part])
        if any(x.get("pid") == p.pid and x.get("part") == part for x in s.post_layers()):
            self.toast(_('{image} of "{name}" is already a layer of Post').format(image=word, name=p.name))
            return
        s.add_post_layer(p.pid, part, self)
        s.log('{image} of "{name}" sent to Post', image=dict(STAR_OUTPUTS)[part], name=p.name)
        self.toast(_('{image} of "{name}" sent to Post as a new layer').format(image=word, name=p.name))

    def part_out_of_post(self, node, part):
        """One of the split block's images sent as it is, taken out of Post (its layers, with their filters; #29)."""
        s = self.s
        p = s.path if s.path in node.paths else node.paths[0]
        word = _(dict(STAR_OUTPUTS)[part])
        if s.remove_post_part(p.pid, part, self):
            s.log('{image} of "{name}" taken out of Post', image=dict(STAR_OUTPUTS)[part], name=p.name)
            self.toast(_('{image} of "{name}" taken out of Post').format(image=word, name=p.name))

    def new_composition(self, node):
        """A Stars and background block for this Stars block: a layer of Post that recombines its two images."""
        s = self.s
        p = s.path if s.path in node.paths else node.paths[0]
        lid = s.add_post_layer(p.pid, "combined", self)
        s.log('Combine stars & background layers block for "{name}"', name=p.name)
        s.sel_layer = lid
        s.select(index=s.paths.index(p), step="composition")
        s.scope = [p.pid]

    def remove_layer(self, lid):
        """A Stars and background block taken out of Post (its layer, with its settings)."""
        s = self.s
        s.remove_post_layer(lid, self)
        s.log("Combine stars & background layers block removed")
        if s.sel_step == "composition" and getattr(s, "sel_layer", None) == lid:
            s.select(step="etoiles")

    def toggle_pause(self, node):
        """Pause here for review: set on every workflow going through the block, or taken off when they all have it."""
        s = self.s
        paths = list(node.paths)
        if not paths or node.key in ("sources", "final"):
            return
        on = not all(s.has_breakpoint(q, node.key) for q in paths)
        s.set_breakpoint(paths, node.key, on)
        title = block_title(paths[0], node.key)
        if on:
            s.log("Pause set after {block}", block=title)
            self.toast(_("The run will stop after {block} for your review").format(block=title) if len(paths) == 1
                       else plural(len(paths), "The run will stop after {block} for your review in {n} workflow",
                                   "The run will stop after {block} for your review in {n} workflows", block=title))
        else:
            s.log("Pause removed after {block}", block=title)
            self.toast(_("No more pause after {block}").format(block=title))

    def mark_reviewed(self, node):
        """Mark as reviewed: the results are checked, the run will go through this block."""
        s = self.s
        pids = [q.pid for q in node.paths if s.review_state(q, node.key) == "paused"]
        if not pids:
            return
        from .model import Pause
        s.mark_reviewed(Pause(node.key, pids))
        title = block_title(node.paths[0], node.key)
        s.log("{block} marked as reviewed", block=title)
        self.toast(_("{block} marked as reviewed").format(block=title))

    def variant_made(self, pid, orig):
        """The variant is shown at the same block, side by side with the workflow it comes from."""
        self.select(index=self.s.index_of(pid))
        q = self.s.view(pid)
        self.toast(_('Variant "{name}": it shares everything up to this block').format(name=q.name))
        self.variantMade.emit(pid, orig)
