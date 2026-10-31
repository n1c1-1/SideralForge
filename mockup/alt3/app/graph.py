"""The graph of the project: the Conception mode's centre.

Columns follow the processing order; a block sits in its column and the wires draw themselves. A block shared by
several workflows is drawn once, and the branch splits where they start to differ. Tools are dropped onto a workflow:
while one is dragged, the places where it can go appear.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsScene, QGraphicsView, QLabel, QMenu

from blocks import COLUMNS, OPTIONAL, PHASES, TOOL_COLUMN, Graph, Node, block_title
from i18n import _
from widgets import ACCENT, OK, WARN

BW, BH = 124, 54            # block size
COLW = 142                  # column pitch
LANE = 80                   # lane pitch
TOP = 74                    # room for the phase labels
STATE_COL = {"done": OK, "pending": "#5d6470", "stale": WARN}
STATE_WORD = {"done": "Up to date", "pending": "Pending", "stale": "Outdated"}
# the blocks whose output can go straight to the post: only where it gives an image no workflow already gives
# (before the background step, an output still carries the sky gradient; astrometry does not change the image)
EXITS = {"empilement"}


def elide(text, font, width):
    return QFontMetrics(font).elidedText(text, Qt.ElideRight, int(width))


class BlockItem(QGraphicsItem):
    def __init__(self, view, node: Node, title, summary, state, selected, dim, removable):
        super().__init__()
        self.view, self.node = view, node
        self.title, self.summary, self.state = title, summary, state
        self.selected, self.dim, self.removable = selected, dim, removable
        self.hover = False
        self.target = False                    # the Post block, while an output is pulled over it
        self._wire = False
        self.port = node.key in EXITS
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        tip = f"{title}\n{summary}\n" + (_(STATE_WORD[state]) if node.key not in ("sources", "final") else "")
        if self.port:
            tip += "\n" + _("Output ●: drag it onto Post to make a layer")
        self.setToolTip(tip.strip())

    def boundingRect(self):
        return QRectF(-2, -2, BW + 14, BH + 4)

    def _x_rect(self):
        return QRectF(BW - 20, 4, 16, 16)

    def _on_port(self, pos):
        return self.port and (pos - QPointF(BW, BH / 2)).manhattanLength() < 14

    def paint(self, p: QPainter, opt, widget=None):
        p.setRenderHint(QPainter.Antialiasing)
        p.setOpacity(0.55 if self.dim else 1.0)
        r = QRectF(0, 0, BW, BH)
        ghost = self.node.key == "sources" and self.state == "pending"
        fill = "#1d2129" if ghost else ("#2b3446" if self.selected or self.target else
                                        ("#272c36" if self.hover else "#22262e"))
        pen = QPen(QColor(ACCENT if self.selected or self.target else "#3a4150"),
                   2 if self.selected or self.target else 1, Qt.DashLine if ghost else Qt.SolidLine)
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
        tw = BW - 18 - (18 if self.removable and self.hover else 0)
        p.drawText(QRectF(11, 6, tw, 20), Qt.AlignLeft | Qt.AlignVCenter, elide(self.title, f, tw))
        f2 = QFont("Segoe UI", 8.5)
        p.setFont(f2)
        p.setPen(QColor("#8b93a1"))
        p.drawText(QRectF(11, 28, BW - 15, 18), Qt.AlignLeft | Qt.AlignVCenter, elide(self.summary, f2, BW - 15))
        if self.removable and self.hover:
            p.setPen(QPen(QColor("#e06c6c"), 1.6))
            xr = self._x_rect().adjusted(4, 4, -4, -4)
            p.drawLine(xr.topLeft(), xr.bottomRight())
            p.drawLine(xr.topRight(), xr.bottomLeft())
        if self.port:
            big = self.hover or self.selected or self._wire
            p.setPen(QPen(QColor(ACCENT if big else "#4a5263"), 1.5))
            p.setBrush(QColor("#15171c" if not big else ACCENT))
            p.drawEllipse(QPointF(BW, BH / 2), 5.5 if big else 3.5, 5.5 if big else 3.5)

    def hoverEnterEvent(self, e):
        self.hover = True
        self.update()

    def hoverMoveEvent(self, e):
        self.setCursor(Qt.CrossCursor if self._on_port(e.pos()) else Qt.PointingHandCursor)

    def hoverLeaveEvent(self, e):
        self.hover = False
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            if self._on_port(e.pos()):
                self._wire = True
                self.view.start_wire(self)
            elif self.removable and self._x_rect().contains(e.pos()):
                self.view.remove_block(self.node, None)
            else:
                self.view.blockClicked.emit(self.node)
        e.accept()

    def mouseMoveEvent(self, e):
        if self._wire:
            self.view.move_wire(e.scenePos())

    def mouseReleaseEvent(self, e):
        if self._wire:
            self._wire = False
            self.view.end_wire(e.scenePos())
        e.accept()

    def mouseDoubleClickEvent(self, e):
        self.view.blockOpened.emit(self.node)
        e.accept()

    def contextMenuEvent(self, e):
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

    def boundingRect(self):
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

    def hoverEnterEvent(self, e):
        if self.on_click:
            self.hot = True
            self.update()

    def hoverLeaveEvent(self, e):
        if self.on_click:
            self.hot = False
            self.update()

    def mousePressEvent(self, e):
        if self.on_click:
            self.on_click()
        e.accept()


class BandItem(QGraphicsItem):
    def __init__(self, rect, text):
        super().__init__()
        self.rect, self.text = rect, text
        self.setZValue(-10)

    def boundingRect(self):
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
    blockOpened = Signal(object)           # Node: double-click, to the Mise au point
    newWorkflow = Signal()
    toolDropped = Signal(str, int)         # tool key, workflow pid
    removeTool = Signal(object, object)    # Node, list of pids or None (all the workflows of the block)
    workflowMenu = Signal(str, int)        # action, pid
    sendToPost = Signal(object)            # Node whose output was pulled to the Post block

    def __init__(self, project, steps, parent=None):
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
        self.toast = QLabel(self)
        self.toast.setStyleSheet("background: #262a33; border: 1px solid #3a4150; border-radius: 6px; "
                                 "padding: 6px 10px; color: #e8ebf0;")
        self.toast.hide()
        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self.toast.hide)
        from PySide6.QtWidgets import QPushButton
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

    def rebuild(self):
        sc = self.scene()
        sc.clear()
        self.ghosts = []
        self._sel_item = None
        s = self.s
        g = Graph(s, self.steps)
        self.graph = g
        cols = self.columns(g)
        cx = {c: i * COLW for i, c in enumerate(cols)}
        lanes = {p.pid: TOP + i * LANE for i, p in enumerate(s.paths)}
        n_lanes = max(1, len(s.paths))
        mid = TOP + (n_lanes - 1) * LANE / 2 if s.paths else TOP + LANE / 2
        for n in g.nodes:
            if n.key in ("sources", "final") or not n.paths:
                n.y = mid
            else:
                ys = [lanes[p.pid] for p in n.paths]
                n.y = (min(ys) + max(ys)) / 2
            n.col = cx[n.key]
        bottom = TOP + n_lanes * LANE + (LANE if s.sources_loaded or s.paths else LANE)
        # phase bands
        for title, a, b in PHASES:
            span = [c for c in COLUMNS[COLUMNS.index(a): COLUMNS.index(b) + 1] if c in cx]
            x0, x1 = cx[span[0]] - 8, cx[span[-1]] + BW + 8
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
            title = block_title(n.paths[0] if n.paths and n.key == "post" else (n.paths[0] if n.paths else None),
                                n.key)
            removable = n.key in ("debruitage", "etoiles", "deconv")
            dim = cur is not None and n.key not in ("sources", "final") and cur not in n.paths
            it = BlockItem(self, n, title, g.summary(n), g.state(n), n is sel, dim, removable)
            it.setPos(n.col, n.y - BH / 2)
            it.setZValue(5)
            sc.addItem(it)
            if n is g.final:
                self.final_item = it
            if n is sel:
                self._sel_item = it
        # the next workflow
        if True:
            y = TOP + len(s.paths) * LANE - BH / 2 if s.paths else mid - BH / 2
            x0 = cx["calibration"]
            x1 = cx["post"] + BW
            gh = GhostItem(QRectF(x0, y, x1 - x0, BH), "+  " + _("New workflow"), self.newWorkflow.emit)
            sc.addItem(gh)
        # where the dragged tool can go
        if self.drag_tool:
            key = self.drag_tool
            for p in s.paths:
                end = getattr(p, "end", None)
                if s.has_tool(p.opts, key) or (end and COLUMNS.index(TOOL_COLUMN[key]) > COLUMNS.index(end)):
                    continue
                x = cx[TOOL_COLUMN[key]]
                short = block_title(None, TOOL_COLUMN[key])
                gh = GhostItem(QRectF(x + 4, lanes[p.pid] - BH / 2 + 4, BW - 8, BH - 8), short, None, p.pid)
                gh.setZValue(8)
                sc.addItem(gh)
                self.ghosts.append(gh)
        r = sc.itemsBoundingRect().adjusted(-24, -10, 24, 24)
        sc.setSceneRect(r)
        self.fit()

    # ------------------------------------------------------------ pulling an output to the post
    def start_wire(self, item):
        self._pull = item
        self._pull_path = self.scene().addPath(QPainterPath(), QPen(QColor(ACCENT), 2, Qt.DashLine))
        self._pull_path.setZValue(9)

    def _on_post(self, pos):
        fi = getattr(self, "final_item", None)
        return fi is not None and fi.sceneBoundingRect().adjusted(-16, -16, 16, 16).contains(pos)

    def move_wire(self, pos):
        a = self._pull.mapToScene(QPointF(BW, BH / 2))
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
        x1, y1 = a.col + BW, a.y
        x2, y2 = b.col, b.y
        path = QPainterPath(QPointF(x1, y1))
        dx = max(30.0, (x2 - x1) * 0.5) if abs(y2 - y1) > 1 else 0
        if dx:
            path.cubicTo(QPointF(x1 + dx, y1), QPointF(x2 - dx, y2), QPointF(x2, y2))
        else:
            path.lineTo(QPointF(x2, y2))
        it = self.scene().addPath(path, QPen(QColor("#9aa3b2" if strong else "#3d4452"), 2.2 if strong else 1.6))
        it.setZValue(1)

    def _side_wire(self, a: Node, b: Node, lab: str):
        x1, y1 = a.col + BW / 2, a.y + BH / 2
        x2, y2 = b.col + BW / 2, b.y + BH / 2
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
        r = self.scene().sceneRect()
        if r.isEmpty():
            return
        k = min(1.3, (self.viewport().width() - 10) / r.width(), (self.viewport().height() - 10) / r.height())
        if not self.b_all.isChecked():
            k = max(k, 0.8)                            # below, the text cannot be read: scroll instead
        self.resetTransform()
        self.scale(k, k)
        self.centerOn(r.center())
        it = getattr(self, "_sel_item", None)
        if it is not None and it.scene() is self.scene():
            self.ensureVisible(it, 80, 40)               # the chosen block stays in view when the graph scrolls

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.fit()
        self.toast.move(16, self.height() - self.toast.height() - 16)
        self.b_all.adjustSize()
        self.b_all.move(self.width() - self.b_all.width() - 14, 10)

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
        elif node.key in ("debruitage", "etoiles", "deconv"):
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
        if not m.isEmpty():
            m.exec(pos)

    def remove_block(self, node: Node, pids):
        self.removeTool.emit(node, pids)

    # ------------------------------------------------------------ drag and drop of tools
    def _tool(self, e):
        t = e.mimeData().text() if e.mimeData().hasText() else ""
        return t[5:] if t.startswith("tool:") else None

    def dragEnterEvent(self, e):
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

    def dragMoveEvent(self, e):
        hit = self._ghost_at(e.position().toPoint())
        for g in self.ghosts:
            if g.hot != (g is hit):
                g.hot = g is hit
                g.update()
        if hit:
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragLeaveEvent(self, e):
        self.drag_tool = None
        self.rebuild()

    def dropEvent(self, e):
        hit = self._ghost_at(e.position().toPoint())
        k = self.drag_tool
        self.drag_tool = None
        if hit and k:
            e.acceptProposedAction()
            pid = hit.pid
            QTimer.singleShot(0, lambda: self.toolDropped.emit(k, pid))
        else:
            self.rebuild()
