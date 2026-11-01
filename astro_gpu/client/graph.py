"""The graph of the project: Graph mode's centre, and the page that holds it (``GraphPage``).

Columns follow the processing order; a block sits in its column and the wires draw themselves. A block shared by
several workflows is drawn once, and the branch splits where they start to differ. Optional blocks are added by
dragging a card of the toolbar onto a workflow: while one is dragged, the places where it can go appear. The output of
Stacking has a port that is pulled onto Post.

``GraphPage`` is what the window mounts: the toolbar of optional blocks, the ``GraphView`` and the block ``Panel``, over
the ``ProjectModel``. It makes the edits (add or remove a block, new workflow, rename, duplicate, delete, an output sent
to Post) and tells the window, by signals, what it must do (open Inspect or Post, import frames).

Wiring (app.py, P13)::

    READY["workflows"] = True
    self.s = ProjectModel(definition, self.worker, history=self.project.history, gpu=..., device=...)   # once per project
    page = GraphPage(self.s)                                    # graph_page(); built again with the project screen
    page.openBlock.connect(lambda: self.set_mode(2 if self.s.sel_step == "final" else 1))
    page.openPost.connect(lambda: self.set_mode(2));  page.addSources.connect(self.add_sources)
    page.addSession.connect(self.add_sources)                   # "Add session…" of the Frames block
    page.variantMade.connect(lambda pid, orig: ...)             # Inspect opens the variant compared with its original
    page.changed.connect(self.refresh)                          # Run button: hm(self.s.total_seconds()); status line
    self.s.readyChanged.connect(self.refresh)                   # Run enabled once the keys are read (self.s.ready)
    undo / redo: ``self.s.undo()`` / ``self.s.redo()`` (False when there is nothing to take back); ``can_undo()``
    after a run or a stage end: ``self.s.results_changed()``; after an import: ``self.s.reload_sources()``
    Spec mode toggled: ``page.request_refresh()``; machine changed: ``self.s.set_machine(gpu, device)``
    closing the project: ``self.s.close()`` instead of ``flush()`` (it flushes, and lets go of the shared worker)
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
    QLinearGradient,
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
    COLUMNS,
    OPTIONAL,
    PAUSE_COL,
    PHASES,
    REVIEWED_COL,
    STATE_WORD,
    TOOL_COLUMN,
    TOOLBOX,
    TOOLS,
    Graph,
    Node,
    block_title,
)
from .dialogs import NewWorkflowDialog, ask_name
from .i18n import _, hm, plural
from .model import TEMPLATES, NotInEngine
from .panel import Panel
from .widgets import ACCENT, BAD, NEW, OK, WARN, clear, label

BW, BH = 120, 54            # block size (a column is wider when its title needs it: TITLE_PAD)
GAP = 12                    # between two columns: where the branches split
COLW = BW + GAP             # column pitch
LANE = 80                   # lane pitch
TOP = 74                    # room for the phase labels
MARGIN = 8                  # around the graph, left and right
MIN_SCALE = 0.6             # the smallest the default view goes to show the whole chain; below, the text cannot be read:
#                             the view keeps this scale, shows the left end (Frames) and scrolls right. Show all goes lower
TITLE_PAD = 22              # a block's title, its state stripe and its padding
OUT_MAX = BW + 30           # the widest Output column: a longer workflow name fades out (whole in the tooltip)
FADE = 18                   # the width over which a name too long for its block fades out
STATE_COL = {"done": OK, "pending": "#5d6470", "stale": WARN, "running": ACCENT, "failed": BAD,
             "interrupted": WARN, "skipped": "#5d6470"}
# the blocks whose output can go straight to the post: only where it gives an image no workflow already gives
# (before the background step, an output still carries the sky gradient; astrometry does not change the image)
EXITS = {"empilement"}
OPTIONAL_BLOCKS = ("debruitage", "etoiles", "deconv")


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
        self.port = node.key in EXITS
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        tip = f"{title}\n{summary}\n" + (_(STATE_WORD[state]) if node.key not in ("sources", "final") else "")
        if self.port:
            tip += "\n" + _("Output ●: drag it onto Post to make a layer")
        if bp is not None:
            tip += "\n" + review_tip(bp)
        self.setToolTip(tip.strip())

    def boundingRect(self):                                           # noqa: N802 - Qt's name
        left = 12 if self.bp is not None else 2         # the review dot sits on the corner
        return QRectF(-left, -left, self.w + left + (10 if self.port else 2), BH + left + 2)

    def _x_rect(self):
        return QRectF(self.w - 20, 4, 16, 16)

    def _on_port(self, pos):
        return self.port and (pos - QPointF(self.w, BH / 2)).manhattanLength() < 14

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
        p.setOpacity(0.55 if self.dim else 1.0)
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
        if self.node.key not in ("sources", "final") or self.state != "pending":
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(STATE_COL[self.state]))
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
        if self.node.key == "post" and QFontMetricsF(f).horizontalAdvance(self.title) > tw:
            # a workflow's name: it fades out at the edge of its block (the whole name is in the tooltip); "…" is
            # kept for the engine's words, whose end matters less
            p.save()
            p.setClipRect(QRectF(11, 6, tw, 20))
            p.drawText(QRectF(11, 6, tw + 400, 20), Qt.AlignLeft | Qt.AlignVCenter, self.title)
            p.restore()
            grad = QLinearGradient(11 + tw - FADE, 0, 11 + tw, 0)
            c0 = QColor(fill)
            c0.setAlpha(0)
            grad.setColorAt(0.0, c0)
            grad.setColorAt(1.0, QColor(fill))
            p.fillRect(QRectF(11 + tw - FADE, 6, FADE, 20), QBrush(grad))
        else:
            p.drawText(QRectF(11, 6, tw, 20), Qt.AlignLeft | Qt.AlignVCenter, elide(self.title, f, tw))
        f2 = QFont("Segoe UI", 8.5)
        p.setFont(f2)
        p.setPen(QColor("#8b93a1"))
        p.drawText(QRectF(11, 28, w - 15, 18), Qt.AlignLeft | Qt.AlignVCenter, elide(self.summary, f2, w - 15))
        if self.removable and self.hover:
            p.setPen(QPen(QColor("#e06c6c"), 1.6))
            xr = self._x_rect().adjusted(4, 4, -4, -4)
            p.drawLine(xr.topLeft(), xr.bottomRight())
            p.drawLine(xr.topRight(), xr.bottomLeft())
        if self.port:
            big = self.hover or self.selected or self._wire
            p.setPen(QPen(QColor(ACCENT if big else "#4a5263"), 1.5))
            p.setBrush(QColor("#15171c" if not big else ACCENT))
            p.drawEllipse(QPointF(w, BH / 2), 5.5 if big else 3.5, 5.5 if big else 3.5)
        if self.bp is not None:
            self.paint_review(p)

    def hoverEnterEvent(self, e):                                     # noqa: N802
        self.hover = True
        self.update()

    def hoverMoveEvent(self, e):                                      # noqa: N802
        self.setCursor(Qt.CrossCursor if self._on_port(e.pos()) else Qt.PointingHandCursor)

    def hoverLeaveEvent(self, e):                                     # noqa: N802
        self.hover = False
        self.update()

    def mousePressEvent(self, e):                                     # noqa: N802
        if e.button() == Qt.LeftButton:
            if self._on_port(e.pos()):
                self._wire = True
                self.view.start_wire(self)
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
        self.view.blockOpened.emit(self.node)
        e.accept()

    def contextMenuEvent(self, e):                                    # noqa: N802
        self.view.block_menu(self.node, e.screenPos())
        e.accept()


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

    def __init__(self, project, steps=None, parent=None):
        super().__init__(parent)
        self.s, self.steps = project, steps
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
        if self.drag_tool:
            used.add(TOOL_COLUMN[self.drag_tool])
        return [c for c in COLUMNS if c not in OPTIONAL or c in used]

    @staticmethod
    def column_width(key: str, names=()) -> float:
        """A column is as wide as a block, or wider when its title needs it in the language shown ("Post-traitement").
        The outputs carry the workflows' names: their column widens to the longest name, up to OUT_MAX; a longer name
        fades out at the edge of its block and is whole in the tooltip."""
        f = QFont("Segoe UI", 10)
        f.setBold(True)
        fm = QFontMetricsF(f)
        if key == "post":
            need = max((fm.horizontalAdvance(n) for n in names), default=0.0) + TITLE_PAD
            return max(BW, min(OUT_MAX, need))
        return max(BW, min(BW + 40, fm.horizontalAdvance(block_title(None, key)) + TITLE_PAD))

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
        sc.clear()
        self.ghosts = []
        self._sel_item = None
        self.final_item = None
        self.b_all.setText(_("Show all"))
        self.b_all.setToolTip(_("The whole graph in the window, smaller; off: readable size, scrolling"))
        self.b_all.adjustSize()
        s = self.s
        g = Graph(s, self.steps)
        self.graph = g
        cols = self.columns(g)
        cw = {c: self.column_width(c, [p.name for p in s.paths]) for c in cols}
        cx, x = {}, 0.0
        for c in cols:
            cx[c] = x
            x += cw[c] + GAP
        self._next_x = {c: cx[c] + cw[c] + GAP for c in cols}       # where the column after it starts
        lanes = {pid: TOP + i * LANE for i, pid in enumerate(self.lane_order(g))}
        n_lanes = max(1, len(s.paths))
        mid = TOP + (n_lanes - 1) * LANE / 2 if s.paths else TOP + LANE / 2
        for n in g.nodes:
            if n.key in ("sources", "final") or not n.paths:
                n.y = mid
            else:
                ys = [lanes[p.pid] for p in n.paths]
                n.y = (min(ys) + max(ys)) / 2
            n.col = cx[n.key]
            n.w = cw[n.key]
        bottom = TOP + n_lanes * LANE + LANE
        # phase bands
        for title, a, b in PHASES:
            span = [c for c in COLUMNS[COLUMNS.index(a): COLUMNS.index(b) + 1] if c in cx]
            pad = GAP / 2 - 1                                         # two bands never touch
            x0, x1 = cx[span[0]] - pad, cx[span[-1]] + cw[span[-1]] + pad
            sc.addItem(BandItem(QRectF(x0, 0, x1 - x0, bottom), _(title)))
        cur = s.path
        sel = g.node_of(cur, s.sel_step) if s.sel_step not in ("sources", "final") else (
            g.root if s.sel_step == "sources" else g.final)
        # wires along the chains
        for n in g.nodes:
            if n.key == "sources" or n is g.final:
                continue
            par = n.parent if n.parent is not None else g.root
            self._wire(par, n, cur is not None and cur in n.paths)
        for p in s.paths:
            self._wire(g.leaves[p.pid], g.final, cur is not None and p is cur)
        # side inputs of the selected block
        if sel is not None and sel.key not in ("sources", "final"):
            for src, lab in g.side_inputs(sel):
                self._side_wire(src, sel, lab)
        # blocks
        for n in g.nodes:
            if n.key not in ("sources", "final") and not n.paths:
                continue
            title = block_title(n.paths[0] if n.paths else None, n.key)
            removable = n.key in OPTIONAL_BLOCKS
            dim = cur is not None and n.key not in ("sources", "final") and cur not in n.paths
            it = BlockItem(self, n, title, g.summary(n), g.state(n), n is sel, dim, removable, n.w, g.breakpoint(n))
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
        x1 = cx["post"] + cw["post"]
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
    def start_wire(self, item):
        self._pull = item
        self._pull_path = self.scene().addPath(QPainterPath(), QPen(QColor(ACCENT), 2, Qt.DashLine))
        self._pull_path.setZValue(9)

    def _on_post(self, pos):
        fi = self.final_item
        return fi is not None and fi.sceneBoundingRect().adjusted(-16, -16, 16, 16).contains(pos)

    def move_wire(self, pos):
        a = self._pull.mapToScene(QPointF(self._pull.w, BH / 2))
        path = QPainterPath(a)
        dx = max(30.0, (pos.x() - a.x()) * 0.5)
        path.cubicTo(QPointF(a.x() + dx, a.y()), QPointF(pos.x() - dx, pos.y()), pos)
        self._pull_path.setPath(path)
        hot = self._on_post(pos)
        if self.final_item.target != hot:
            self.final_item.target = hot
            self.final_item.update()
        x, y = self.mapFromScene(pos).x(), self.mapFromScene(pos).y()
        if x > self.viewport().width() - 20 or y > self.viewport().height() - 20:
            self.ensureVisible(QRectF(pos, pos), 20, 20)

    def end_wire(self, pos):
        node = self._pull.node
        self.scene().removeItem(self._pull_path)
        self.final_item.target = False
        self.final_item.update()
        if self._on_post(pos):
            QTimer.singleShot(0, lambda: self.sendToPost.emit(node))

    def _wire(self, a: Node, b: Node, strong: bool):
        """A wire from a block to the next block of its workflows. A branch bends in the gap right after the block it
        leaves (the last one its workflows share), then runs straight on its own lane past the columns it skips, as
        the mockup draws a workflow that leaves the trunk early."""
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
        if node.key == "post":
            p = node.paths[0]
            m.addAction(_("Rename")).triggered.connect(lambda: self.workflowMenu.emit("rename", p.pid))
            m.addAction(_("Duplicate")).triggered.connect(lambda: self.workflowMenu.emit("duplicate", p.pid))
            m.addSeparator()
            m.addAction(_("Delete workflow")).triggered.connect(lambda: self.workflowMenu.emit("delete", p.pid))
        elif node.key in OPTIONAL_BLOCKS:
            if node.shared:
                m.addAction(_("Remove from the {n} workflows").format(n=len(node.paths))).triggered.connect(
                    lambda: self.remove_block(node, None))
                for p in node.paths:
                    m.addAction(_('Remove from "{name}" only').format(name=p.name)).triggered.connect(
                        lambda _x=False, q=p: self.remove_block(node, [q.pid]))
            else:
                m.addAction(_("Remove")).triggered.connect(lambda: self.remove_block(node, None))
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


class GraphPage(QWidget):
    """Graph mode: the toolbar of optional blocks, the graph and the panel of the selected block."""
    openBlock = Signal()                   # double-click or Enter on a block: Inspect (Post when sel_step is "final")
    openPost = Signal()                    # "Open Post" of the Post block
    addSources = Signal()                  # "Import folder…" of the Frames block
    addSession = Signal()
    variantMade = Signal(int, int)         # a variant (id) was made from this workflow (id): Inspect compares them
    changed = Signal()                     # the page was built again: the window updates its Run button and status
    toasted = Signal(str)

    def __init__(self, project, steps=None, parent=None):
        super().__init__(parent)
        self.s, self.steps = project, steps
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
        self.graph = GraphView(project, steps)
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
        pn = self.panel
        pn.changed.connect(self.request_refresh)
        pn.openTune.connect(self.openBlock.emit)
        pn.openPost.connect(self.openPost.emit)
        pn.variantMade.connect(self.variant_made)
        pn.addSources.connect(self.addSources.emit)
        pn.addSession.connect(self.addSession.emit)
        pn.renamed.connect(self.rename)
        pn.toast.connect(self.toast)
        pn.removeTool.connect(lambda: self.remove_tool(self.current_node(), None))
        for sig in (project.pathsChanged, project.currentChanged, project.costChanged, project.sourcesChanged,
                    project.readyChanged):
            sig.connect(self.request_refresh)
        for key in ("Return", "Enter"):
            ret = QShortcut(QKeySequence(key), self.graph)
            ret.setContext(Qt.WidgetWithChildrenShortcut)
            ret.activated.connect(self.openBlock.emit)
        self.refresh()

    # ------------------------------------------------------------ refresh
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
        g = g or Graph(self.s, self.steps)
        k = self.s.sel_step
        if k == "sources":
            return g.root
        if k == "final":
            return g.final
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
            elif p.end and COLUMNS.index(TOOL_COLUMN[k]) > COLUMNS.index(p.end):
                state, text = "in", _("after the output")
            elif s.has_tool(p.opts, k):
                state, text = "in", _("already there")
            else:
                dt = s.tool_cost(p.opts, k)
                sure = s.opts_measured(s.with_tool(p.opts, k)[0]) if hasattr(s, "opts_measured") else True
                state, text = "add", ((f"+{hm(dt)}" if sure else f"≈ +{hm(dt)}") if dt >= 30 else _("same duration"))
            card = ToolCard(self, k, state, text, new)
            card.setEnabled(p is not None and not new)
            self.tl.addWidget(card)
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
        s.select(step=node.key)

    def select(self, index=None, step=None):
        s = self.s
        s.pending_change = None
        s.select(index, step)
        node = self.current_node()
        s.scope = [q.pid for q in node.paths] if node is not None and node.key not in ("sources", "final") else None

    # ------------------------------------------------------------ actions
    def new_workflow(self, start=None):
        s = self.s
        d = NewWorkflowDialog(s, start, self)
        if not d.exec():
            return
        name = d.name.text().strip() or _("New workflow")
        copy_of = d.cur - len(TEMPLATES)
        if copy_of >= 0:                                   # a copy keeps its options, settings and exit
            p = s.duplicate(s.paths[copy_of], name)
        else:
            p = s.add_path(name, d.opts, d.params)
        s.log('Workflow "{name}" created', name=p.name)
        self.select(index=s.paths.index(p), step="post")

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
        node = Graph(s, self.steps).node_of(s.view(pid), s.sel_step)
        s.scope = [q.pid for q in node.paths] if node is not None else [pid]
        if notes:
            self.toast(" · ".join(_(n) for n in notes))

    def remove_tool(self, node, pids):
        s = self.s
        if node is None:
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
