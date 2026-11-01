"""Inspect: the image of one block, large.

At the top, the current workflow as a strip of blocks; in the middle, the result of the chosen block with the inspection
tools (auto-stretch, histogram, 1:1, pixel readout in ADU and sky coordinates); on the right, the same panel as in
Graph, with the measurements. The Frames block opens the sessions and the frame browser.

The pictures come from the done results of the engine (the arrays of ``api.stage_array``, the previews of
``api.stage_outputs``), loaded and stretched by the worker (``services/inspect``): nothing here reads a file. An Outdated
block shows the result it had ("previous result"); a Pending one says so, with Run to here.

Mounting (the window's ``inspect_page``): ``steps = Steps(model)``; ``browser = SourcesView(model)``;
``frames = FramesView(model, browser)``; ``tune = TuneView(model, steps, frames, post_page)``. The view refreshes itself
from the model's signals while it is visible. Its signals: ``back`` (the Graph button, Esc), ``runHere`` (Run to the
selected block of the current workflow), ``openPost``, ``removeTool`` (Remove on the panel of an optional block: the
graph's ``remove_tool(node, None)``), ``toasted(text)``. Importing is the views' own: ``sources.import_folder(model, parent,
folder=None, new_session=False)`` serves File > Import folder… and Import folder as new project.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPainterPath, QPen, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedLayout,
    QStackedWidget,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from . import prefs
from .blocks import ENGINE_BLOCK, Graph, block_state, block_title, chain, stages_of
from .charts import FramesChart, SpectralChart
from .i18n import _, hm, number
from .panel import STATE_COLOR, STATE_WORD, Panel
from .services.inspect import ShownImage, VisualSpec
from .steps import Steps, fnum, kv
from .viewer import StaticSource, Viewer
from .widgets import ACCENT, WARN, clear, label

STATE_COL = dict(STATE_COLOR, pending="#5d6470")      # the strip: a block not computed yet is a dark grey


class StretchSource:
    """A picture shown with the automatic display stretch, or linear (as stored). The data is never changed."""

    def __init__(self, img: ShownImage, stretched: bool = True):
        self.img, self.stretched = img, stretched
        self.version = 0

    def _src(self):
        im = self.img
        return im.stretched if (self.stretched or im.linear is None) else im.linear

    def size(self):
        return self._src().size()

    def region(self, level, x0, y0, x1, y1):
        return self._src().region(level, x0, y0, x1, y1)


class Histogram(QWidget):
    """The histogram of what is shown: three channels, logarithmic counts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(90)
        self.h = None

    def set_counts(self, counts):
        """``counts``: (3, 256) counts of the values of the picture shown."""
        self.h = None if counts is None else [np.log1p(np.asarray(c, float)) for c in counts]
        self.update()

    def paintEvent(self, e):                                         # noqa: N802 - Qt's name
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#181b21"))
        if self.h is not None:
            w, hgt = self.width(), self.height() - 4
            top = max(float(x.max()) for x in self.h) or 1
            for c, col in enumerate(("#e06c6c", "#5fbf7f", "#5ea1ff")):
                path = QPainterPath(QPointF(0, hgt))
                for i, v in enumerate(self.h[c]):
                    path.lineTo(QPointF(i * w / 255, hgt - v / top * (hgt - 4)))
                path.lineTo(QPointF(w, hgt))
                p.setPen(QPen(QColor(col), 1))
                c_ = QColor(col)
                c_.setAlpha(50)
                p.setBrush(c_)
                p.drawPath(path)
        p.end()


class Strip(QFrame):
    """The current workflow, block after block."""
    picked = Signal(str)
    back = Signal()
    workflow = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("strip")
        self.setStyleSheet("QFrame#strip { background: #1c1f26; border-bottom: 1px solid #2a2f38; }")
        self.h = QHBoxLayout(self)
        self.h.setContentsMargins(10, 8, 10, 8)
        self.h.setSpacing(4)

    def build(self, s, g: Graph):
        clear(self.h)
        bk = QPushButton("◂  " + _("Graph"))
        bk.setToolTip(_("Back to the graph (Esc)"))
        bk.clicked.connect(self.back.emit)
        self.h.addWidget(bk)
        self.h.addSpacing(10)
        p = s.path
        if s.sel_step == "final":
            self.h.addWidget(label(_("Post") + "  ·  " + _("the workflows' layers, combined"), "h2"))
            self.h.addStretch(1)
            return
        if p is not None:
            cb = QComboBox()
            for q in s.paths:
                cb.addItem(q.name, q.pid)
            cb.setCurrentIndex(s.current)
            cb.setMinimumWidth(200)
            cb.currentIndexChanged.connect(lambda i: self.workflow.emit(i))
            self.h.addWidget(cb)
            self.h.addSpacing(10)
        keys = ["sources"] + (chain(g.steps, p) if p is not None else [])
        for i, k in enumerate(keys):
            if i:
                a = QLabel("›")
                a.setStyleSheet("color: #6b7280; background: transparent;")
                self.h.addWidget(a)
            node = g.node_of(p, k)
            st = g.state(node) if node is not None else "pending"
            b = QPushButton(block_title(None, k) if k == "post" else block_title(p, k))
            sel = k == s.sel_step
            b.setStyleSheet(f"QPushButton {{ border-left: 3px solid {STATE_COL.get(st, '#5d6470')}; padding: 4px 8px; "
                            f"background: {'#2b3446' if sel else '#22262e'}; "
                            f"border-top: 1px solid {ACCENT if sel else '#353b47'}; "
                            f"border-right: 1px solid {ACCENT if sel else '#353b47'}; "
                            f"border-bottom: 1px solid {ACCENT if sel else '#353b47'}; "
                            f"{'font-weight: 600; color: #ffffff;' if sel else ''} }}")
            b.setToolTip(_(STATE_WORD[st]))
            b.clicked.connect(lambda _x=False, kk=k: self.picked.emit(kk))
            self.h.addWidget(b)
        if p is not None:
            a = QLabel("›")
            a.setStyleSheet("color: #6b7280; background: transparent;")
            self.h.addWidget(a)
            b = QPushButton(_("Post"))
            b.setStyleSheet("QPushButton { padding: 4px 8px; }")
            b.clicked.connect(lambda: self.picked.emit("final"))
            self.h.addWidget(b)
        self.h.addStretch(1)


class ImageSlot(QWidget):
    """One place for a picture: a note while it loads (or when it cannot be shown), then the viewer or the chart."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stack = QStackedLayout(self)
        self.note = label("", "dim")
        self.note.setAlignment(Qt.AlignCenter)
        self.note.setWordWrap(True)
        self.stack.addWidget(self.note)
        self.content: QWidget | None = None
        self.image: ShownImage | None = None
        self.viewer: Viewer | None = None

    def message(self, text: str, warn: bool = False):
        self.note.setObjectName("warn" if warn else "dim")
        self.note.setStyleSheet("")
        self.note.style().unpolish(self.note)
        self.note.style().polish(self.note)
        self.note.setText(text)
        self.stack.setCurrentWidget(self.note)

    def put(self, w: QWidget):
        if self.content is not None:
            self.stack.removeWidget(self.content)
            self.content.deleteLater()
        self.content = w
        self.stack.addWidget(w)
        self.stack.setCurrentWidget(w)


class TuneView(QWidget):
    back = Signal()
    pick = Signal(str)                    # a block of the strip was chosen
    workflow = Signal(int)                # a workflow of the strip's list was chosen
    runHere = Signal()                    # Run to here: the selected block of the current workflow
    openPost = Signal()
    removeTool = Signal()                 # Remove, on the panel of an optional block
    toasted = Signal(str)                 # a word to say (what came with a change, why it was refused)

    def __init__(self, project, steps, sources_view, post_page, parent=None, service=None):
        super().__init__(parent)
        steps = steps if steps is not None else Steps(project, service)
        self.s, self.steps = project, steps
        self.service = service or steps.service
        self.img_tab = 0
        self.cmp_pid = None
        self.picked = None                # the frame picked on the chart (an index of the frame table)
        self.stretched = bool(prefs.get("auto_stretch"))
        self.show_hist = False
        self._pending = False
        self._dirty = True
        self._want: dict[int, int] = {}
        self._slots: list[ImageSlot] = []
        self._views: dict[tuple, tuple] = {}              # picture key -> (scale, cx, cy) the user left it at
        self._syncing = False
        self.viewers: list[Viewer] = []
        self.readout = None
        self.hist = None
        self.table = None                                 # the frame table of the Selection chart
        self.chart: FramesChart | None = None
        self._frame_token = 0
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.strip = Strip()
        self.strip.back.connect(self.back.emit)
        self.strip.picked.connect(self._strip_pick)
        self.strip.workflow.connect(self._strip_workflow)
        v.addWidget(self.strip)
        self.pages = QStackedWidget()
        v.addWidget(self.pages, 1)
        body = QWidget()
        h = QHBoxLayout(body)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(14, 8, 14, 6)
        mv.setSpacing(6)
        self.head = QHBoxLayout()
        mv.addLayout(self.head)
        self.center = QVBoxLayout()
        mv.addLayout(self.center, 1)
        self.foot = QHBoxLayout()
        mv.addLayout(self.foot)
        h.addWidget(mid, 1)
        self.panel = Panel(project, steps, tune=True)
        h.addWidget(self.panel)
        self.pages.addWidget(body)
        self.sources = sources_view
        self.pages.addWidget(sources_view)
        self.post = post_page
        self.pages.addWidget(post_page)
        pn = self.panel
        pn.changed.connect(self.request_refresh)
        pn.openPost.connect(self.openPost.emit)
        pn.variantMade.connect(self._variant_made)
        pn.addSources.connect(self._import)
        pn.addSession.connect(self._add_session)
        pn.renamed.connect(lambda pid, name: self.s.rename(pid, name))
        pn.toast.connect(self.toasted.emit)
        pn.removeTool.connect(self.removeTool.emit)
        for sig in (project.pathsChanged, project.currentChanged, project.costChanged, project.sourcesChanged,
                    project.readyChanged):
            sig.connect(self.request_refresh)
        steps.changed.connect(self._facts_arrived)
        self.service.imageReady.connect(self._image_ready)
        self.service.imageFailed.connect(self._image_failed)
        self.service.tableReady.connect(self._table_ready)
        self.service.frameReady.connect(self._frame_ready)
        self.service.previewReady.connect(self._preview_ready)
        esc = QShortcut(QKeySequence("Escape"), self)
        esc.setContext(Qt.WidgetWithChildrenShortcut)
        esc.activated.connect(self.back.emit)

    # ------------------------------------------------------------ selection (what the strip does)
    def select(self, index=None, step=None):
        s = self.s
        s.pending_change = None
        s.select(index, step)
        g = Graph(s, self.steps)
        node = g.node_of(s.path, s.sel_step) if s.sel_step not in ("sources", "final") else None
        s.scope = [q.pid for q in node.paths] if node is not None else None

    def _strip_pick(self, k: str):
        self.select(step=k)
        self.pick.emit(k)

    def _strip_workflow(self, i: int):
        self.select(index=i)
        self.workflow.emit(i)

    def _import(self):
        from .sources import import_folder
        import_folder(self.s, self.window(), service=self.service)

    def _add_session(self):
        from .sources import import_folder
        import_folder(self.s, self.window(), new_session=True, service=self.service)

    def _variant_made(self, pid, orig):
        """The variant is shown at the same block, side by side with the workflow it comes from."""
        self.cmp_pid = orig
        self.select(index=self.s.index_of(pid))
        q = self.s.view(pid)
        if q is not None:
            self.toasted.emit(_('Variant "{name}": it shares everything up to this block').format(name=q.name))

    # ------------------------------------------------------------ refresh
    def request_refresh(self, *_a):
        self._dirty = True
        if self.isVisible() and not self._pending:
            self._pending = True
            QTimer.singleShot(0, self.refresh)

    def showEvent(self, e):                                          # noqa: N802 - Qt's name
        super().showEvent(e)
        if self._dirty:
            self.request_refresh()

    def _facts_arrived(self):
        if self.pages.currentIndex() == 0 and self.panel.node is not None:
            self.panel.refresh(self.panel.node)

    def _remember_view(self):
        """Where the user left each picture (zoom and position), kept while the view is built again."""
        for sl in self._slots:
            if sl.viewer is not None and sl.image is not None and sl.viewer._fitted:
                self._views[sl.image.key] = (sl.viewer.scale, sl.viewer.cx, sl.viewer.cy)

    def refresh(self, g: Graph | None = None):
        self._pending = False
        self._dirty = False
        s = self.s
        g = g or Graph(s, self.steps)
        self.strip.build(s, g)
        if s.sel_step == "sources":
            self.pages.setCurrentIndex(1)
            if hasattr(self.sources, "refresh"):
                self.sources.refresh()
            return
        if s.sel_step == "final":
            self.pages.setCurrentIndex(2)
            return
        self.pages.setCurrentIndex(0)
        p = s.path
        node = g.node_of(p, s.sel_step)
        self.panel.refresh(node)
        self._remember_view()
        clear(self.head)
        clear(self.center)
        clear(self.foot)
        self._slots, self.viewers, self.readout, self.hist, self.chart = [], [], None, None, None
        self._want.clear()
        if node is None or p is None:
            return
        k = s.sel_step
        state = block_state(s, p, k)
        self.head.addWidget(label(f"{block_title(p, k)}  ·  {p.name}" if k != "post" else p.name, "h1"))
        if state in ("pending", "running", "failed", "interrupted", "skipped") or \
                (state == "stale" and not self._has_previous(p, k)):
            self.head.addStretch(1)
            self.pending(p, k, state)
            return
        if state == "stale":
            self._previous_tag(p, k)
        self.head.addStretch(1)
        if k == "selection":
            self.selection_center(p)
            return
        vis = self.visuals(p, k)
        if not vis:
            self.center.addWidget(label(_("No image for this block."), "dim"), 0, Qt.AlignCenter)
            return
        i = min(self.img_tab, len(vis) - 1)
        while i > 0 and vis[i].spec:
            i -= 1
        if len(vis) > 1:
            tb = QTabBar()
            tb.setDrawBase(False)
            tb.setExpanding(False)
            for sp in vis:
                n = tb.addTab(_(sp.caption) + ("  ◇" if sp.spec else ""))
                if sp.spec:
                    tb.setTabEnabled(n, False)
                    tb.setTabToolTip(n, "◇ " + _("Not in the engine yet."))
            tb.setCurrentIndex(i)
            tb.currentChanged.connect(self._tab)
            self.head.addWidget(tb)
        self.head.addSpacing(12)
        self.head.addWidget(label(_("Compare with"), "dim"))
        cmp_ = QComboBox()
        cmp_.setMinimumWidth(180)
        cmp_.addItem("—", None)
        for q in s.paths:
            if q.pid != p.pid and k in chain(self.steps, q):
                cmp_.addItem(q.name, q.pid)
        j = cmp_.findData(self.cmp_pid)
        cmp_.setCurrentIndex(max(j, 0))
        if j < 0:
            self.cmp_pid = None
        cmp_.setEnabled(cmp_.count() > 1)
        cmp_.currentIndexChanged.connect(lambda jj: self._cmp(cmp_.itemData(jj)))
        self.head.addWidget(cmp_)
        q = next((x for x in s.paths if x.pid == self.cmp_pid), None)
        spec = vis[i]
        if q is None:
            sl = self._slot(0)
            self.center.addWidget(sl, 1)
            self._load(0, p, k, spec)
        else:
            row = QHBoxLayout()
            for slot, w in enumerate((p, q)):
                col = QVBoxLayout()
                same = g.node_of(w, k) is node
                col.addWidget(label(w.name + ("  ·  " + _("same block") if same and w is q else ""), "h2"))
                if block_state(s, w, k) == "pending":
                    col.addWidget(label(_("Not computed yet."), "dim"), 1, Qt.AlignCenter)
                else:
                    wv = self.visuals(w, k)
                    sl = self._slot(slot)
                    col.addWidget(sl, 1)
                    self._load(slot, w, k, wv[min(i, len(wv) - 1)])
                row.addLayout(col, 1)
            self.center.addLayout(row, 1)
        if spec.kind != "chart":
            self.tools(spec)

    def visuals(self, p, k) -> list[VisualSpec]:
        """The pictures of a block; the ◇ ones (not in the engine yet) only in Spec mode."""
        return [v for v in self.steps.visuals(p, k) if not v.spec or prefs.spec()]

    def _has_previous(self, p, k) -> bool:
        """Whether an Outdated block has a result to show: one of its pictures' stages is done or has its previous one."""
        stages = ["register"] if k == "selection" else [v.stage for v in self.steps.visuals(p, k) if not v.spec]
        return any(self.steps.key_of(p, x, self.steps.which_for(p, x)) for x in stages)

    def _previous_tag(self, p, k):
        tag = QLabel(_("previous result"))
        tag.setStyleSheet(f"color: {WARN}; border: 1px solid {WARN}; border-radius: 9px; padding: 1px 8px;")
        self.head.addWidget(tag)
        self.head.addWidget(self._run_button(p, k))

    # ------------------------------------------------------------ the pictures
    def _slot(self, n: int) -> ImageSlot:
        sl = ImageSlot()
        sl.setMinimumHeight(300)
        while len(self._slots) <= n:
            self._slots.append(None)
        self._slots[n] = sl
        return sl

    def _load(self, slot: int, p, k, spec: VisualSpec):
        """Show a picture in a slot: from the cache at once, else after the worker has loaded it."""
        which = self.steps.which_for(p, spec.stage)
        sl = self._slots[slot]
        key = self.steps.key_of(p, spec.stage, which)
        hit = self.service.cache.get((spec.stage, key, spec.name, which)) if key else None
        if hit is not None:
            self._show(slot, hit, spec, k)
            return
        sl.message(_("Loading the image…"))
        self._want[slot] = self.service.request_image(slot, p.pid, spec, which)
        sl._spec, sl._k = spec, k

    def _image_ready(self, slot: int, token: int, im: ShownImage):
        if self._want.get(slot) != token or slot >= len(self._slots) or self._slots[slot] is None:
            return
        sl = self._slots[slot]
        self._show(slot, im, getattr(sl, "_spec", im.spec), getattr(sl, "_k", self.s.sel_step))

    def _image_failed(self, slot: int, token: int, why: str):
        if self._want.get(slot) != token or slot >= len(self._slots) or self._slots[slot] is None:
            return
        self._slots[slot].message(_("This image is not available: {why}").format(why=why), warn=True)

    def _show(self, slot: int, im: ShownImage, spec: VisualSpec, k: str):
        sl = self._slots[slot]
        sl.image = im
        if im.curves is not None:
            ch = SpectralChart(im.curves.lam, im.curves.S)
            sl.put(ch)
            return
        vw = Viewer()
        vw.setMinimumHeight(300)
        vw.set_source(StretchSource(im, self.stretched or not spec.linear))
        left = self._views.get(im.key)
        if left is not None:
            vw.zoom_to(*left)
        ov = self.steps.overlay(k, spec)
        if ov:
            vw.overlays.append(ov)
        vw.hovered.connect(lambda x, y, n=slot: self._hover(n, x, y))
        sl.viewer = vw
        sl.put(vw)
        self.viewers = [x.viewer for x in self._slots if x is not None and x.viewer is not None]
        if len(self.viewers) == 2 and len([x for x in self._slots if x is not None]) == 2:
            a, b = self.viewers
            a.viewChanged.connect(lambda: self._sync(a, b))
            b.viewChanged.connect(lambda: self._sync(b, a))
        if slot == 0:
            self._set_histogram(im)

    def _set_histogram(self, im: ShownImage):
        if self.hist is not None:
            self.hist.set_counts(im.hist.get("stretched" if (self.stretched or im.linear is None) else "linear"))

    # ------------------------------------------------------------ inspection tools
    def tools(self, spec: VisualSpec):
        """Under the image: auto-stretch (linear data only), histogram, fit and 1:1, and the pixel under the cursor."""
        if spec.linear:
            st = QPushButton(_("Auto-stretch"))
            st.setCheckable(True)
            st.setObjectName("toggle")
            st.setChecked(self.stretched)
            st.setToolTip(_("Display only: the data stays linear"))
            st.toggled.connect(self._stretch)
            self.foot.addWidget(st)
        hb = QPushButton(_("Histogram"))
        hb.setCheckable(True)
        hb.setObjectName("toggle")
        hb.setChecked(self.show_hist)
        hb.toggled.connect(self._hist)
        self.foot.addWidget(hb)
        fit = QPushButton(_("Fit"))
        fit.clicked.connect(lambda: [v.fit() for v in self.viewers])
        one = QPushButton("1:1")
        one.clicked.connect(lambda: [v.zoom_to(1.0) for v in self.viewers])
        self.foot.addWidget(fit)
        self.foot.addWidget(one)
        self.foot.addSpacing(12)
        self.readout = label("", "small")
        self.readout.setTextFormat(Qt.RichText)
        self.foot.addWidget(self.readout, 1)
        if self.show_hist:
            self.hist = Histogram()
            self.center.addWidget(self.hist)
            im = self._slots[0].image if self._slots and self._slots[0] is not None else None
            if im is not None:
                self._set_histogram(im)

    def _stretch(self, on):
        self.stretched = on
        prefs.set("auto_stretch", on)
        for sl in self._slots:
            if sl is not None and sl.viewer is not None and isinstance(sl.viewer.source, StretchSource):
                sl.viewer.source.stretched = on
                sl.viewer.source.version += 1
                sl.viewer.refresh()
        if self._slots and self._slots[0] is not None and self._slots[0].image is not None:
            self._set_histogram(self._slots[0].image)

    def _hist(self, on):
        self.show_hist = on
        self.refresh()

    def _hover(self, slot: int, x: float, y: float):
        sl = self._slots[slot] if slot < len(self._slots) else None
        im = sl.image if sl is not None else None
        if self.readout is None or im is None:
            return
        w, h = im.size
        if not (0 <= x < w and 0 <= y < h):
            self.readout.setText("")
            return
        xi, yi = int(x), int(y)
        if im.lin is not None:
            r, g, b = (int(round(float(c))) for c in im.lin[:, yi, xi])
            vals = f"R {r}  G {g}  B {b} ADU"
        else:
            r, g, b = (int(c) for c in im.stretched.arr[yi, xi])
            vals = f"R {r}  G {g}  B {b}"
        sky = ""
        if im.wcs is not None:
            gw, gh = im.grid or self.s.image_size() or im.size
            f = gw / w
            try:
                ra, dec = im.wcs.pixel_to_world_values((xi + 0.5) * f, (yi + 0.5) * f)
                sky = "   " + sky_text(float(ra), float(dec))
            except (ValueError, ArithmeticError):
                sky = ""
        self.readout.setText(f"x {xi}  y {yi}   {vals}{sky}")

    # ------------------------------------------------------------ helpers
    def _sync(self, a, b):
        if self._syncing:
            return
        self._syncing = True
        try:
            if a.source and b.source and a.source.size() == b.source.size():
                b.zoom_to(a.scale, a.cx, a.cy)
        finally:
            self._syncing = False

    def _tab(self, i):
        self.img_tab = i
        self.refresh()

    def _cmp(self, pid):
        self.cmp_pid = pid
        self.refresh()

    # ------------------------------------------------------------ a block not computed
    def _run_button(self, p, k) -> QPushButton:
        s = self.s
        plan = s.plan_to_block(p, k)
        rb = QPushButton(_("Run to here") + "  ·  " + approx(plan.seconds, plan.measured))
        titles = []
        for j in plan.jobs:
            key = ENGINE_BLOCK.get(j.block)
            t = block_title(p, key) if key else j.block
            if t not in titles:
                titles.append(t)
        rb.setToolTip(", ".join(titles))
        rb.setEnabled(s.sources_loaded and bool(plan.jobs))
        rb.clicked.connect(self.runHere.emit)
        return rb

    def pending(self, p, k, state="pending"):
        s = self.s
        plan = s.plan_to_block(p, k)
        box = QVBoxLayout()
        box.setAlignment(Qt.AlignCenter)
        if state in ("pending", "stale"):
            box.addWidget(label(_("Not computed yet"), "h2"), 0, Qt.AlignHCenter)
        else:
            box.addWidget(label(_(STATE_WORD[state]), "h2"), 0, Qt.AlignHCenter)
            st = next((s.state_info(p, x) for x in stages_of(s, p, k) if s.state(p, x) == state), None)
            if st is not None and st.reason:
                box.addWidget(label(_(st.reason), "dim"), 0, Qt.AlignHCenter)
        box.addSpacing(8)
        rb = self._run_button(p, k)
        rb.setObjectName("primary")
        box.addWidget(rb, 0, Qt.AlignHCenter)
        if prefs.spec():
            pv = QPushButton(_("Preview on a region") + "  ·  " + approx(max(plan.seconds / 12, 30), False) + "  ◇")
            pv.setToolTip("◇ " + _("Not in the engine yet."))
            pv.setEnabled(False)
            box.addWidget(pv, 0, Qt.AlignHCenter)
        if not s.sources_loaded:
            box.addWidget(label(_("No frames yet."), "warn"), 0, Qt.AlignHCenter)
        self.center.addStretch(1)
        self.center.addLayout(box)
        self.center.addStretch(2)

    # ------------------------------------------------------------ the frames chart, and the frame under the cursor
    def _thresholds(self) -> tuple[float, float]:
        """The thresholds the chart shows: the settings', or the ones just dragged while Replace / Try as variant is asked."""
        s = self.s
        t, f = s.thresholds_now()
        ch = s.pending_change
        if ch and ch.get("kind") == "params" and ch.get("step") == "selection":
            for _stage, name, v in ch["items"]:
                if name == "min_transparency":
                    t = 100 * float(v)
                elif name == "max_fwhm_ratio":
                    f = float(v) * (s.fwhm_median() or 1.0)
        return t, f

    def selection_center(self, p):
        self.center.addWidget(label(_("Loading the image…"), "dim"), 0, Qt.AlignCenter)
        self._table_token = self.service.request_table(p.pid)
        self._table_wid = p.pid

    def _table_ready(self, wid: int, token: int, table):
        if self.pages.currentIndex() != 0 or self.s.sel_step != "selection" or wid != getattr(self, "_table_wid", None) \
                or token != getattr(self, "_table_token", None):
            return
        clear(self.center)
        if isinstance(table, str):
            self.center.addStretch(1)
            self.center.addWidget(label(_("The frames chart appears once registration is computed."), "dim"), 0,
                                  Qt.AlignHCenter)
            self.center.addStretch(2)
            return
        s = self.s
        self.table = table
        t, f = self._thresholds()
        ch = FramesChart(table.rows, t, f)
        ch.sugg = s.suggested_thresholds()
        ch.current = self.picked if self.picked is not None and self.picked < ch.n else None
        ch.setMinimumHeight(300)
        ch.changed.connect(lambda c=ch: self._chart_changed(c))
        ch.picked.connect(self._pick)
        ch.setToolTip(_("Thresholds: drag · frame: click"))
        self.chart = ch
        self.center.addWidget(ch, 3)
        self.kept_label = label("", "dim")
        self.center.addWidget(self.kept_label)
        self._kept = (table.kept, table.total)
        self._kept_text()
        if abs(t - 100 * table.min_transparency) > 1e-6 or abs(f - table.max_fwhm_ratio * (s.fwhm_median() or 1.0)) > 1e-6:
            self._ask_preview(ch)                  # the thresholds shown are not the stack's: what do they keep?
        bottom = QHBoxLayout()
        self.pick_view = Viewer()
        self.pick_view.setMinimumHeight(200)
        bottom.addWidget(self.pick_view, 1)
        self.pick_side = QVBoxLayout()
        side = QWidget()
        side.setFixedWidth(260)
        side.setLayout(self.pick_side)
        bottom.addWidget(side)
        self.center.addLayout(bottom, 2)
        self._pick(self.picked if self.picked is not None and self.picked < ch.n else 0)

    def _ask_preview(self, c):
        """The engine counts the frames the dragged thresholds keep (selection_preview, in the worker)."""
        m = self.s.fwhm_median() or 1.0
        self._preview_token = self.service.request_preview(self.s.path.pid, c.tmin / 100, c.fmax / m)

    def _preview_ready(self, token: int, kept: int, total: int):
        if token == getattr(self, "_preview_token", None):
            self._kept = (kept, total)
            self._kept_text()

    def _kept_text(self):
        if self.chart is None or getattr(self, "kept_label", None) is None:
            return
        try:
            s = self.s
            kept, n = self._kept
            exp = s.info().get("exposure_s") or 0
            self.kept_label.setText(_("{kept} of {n} frames kept · {h} h").format(
                kept=kept, n=n, h=fnum(kept * exp / 3600)))
        except RuntimeError:                                  # the label went with a rebuild
            pass

    def _chart_changed(self, c):
        """Frames excluded by hand count for the whole project; thresholds go through the panel (Replace or Try as
        variant, when computed results would be lost)."""
        if self.table is not None:
            now = {(r.session, r.name) for r in self.table.rows if r.excluded}
            want = {(self.table.rows[i].session, self.table.rows[i].name) for i in c.hand}
            if now != want:
                self._set_hand(want - now, True)
                self._set_hand(now - want, False)
        t, f = self._thresholds()
        if abs(c.tmin - t) > 1e-6 or abs(c.fmax - f) > 1e-6:
            self._ask_preview(c)
            self.panel.propose_selection(c.tmin, c.fmax)

    def _set_hand(self, pairs, excluded: bool):
        pairs = [x for x in pairs if x[0] is not None]
        if pairs:
            self.s.set_excluded(pairs, excluded)
            touch = getattr(self.s, "_touch", None)
            if touch is not None:
                touch()                                       # the model does not mark an exclusion as an edit

    def _pick(self, i):
        self.picked = i
        if self.table is None or not (0 <= i < len(self.table.rows)):
            return
        row = self.table.rows[i]
        clear(self.pick_side)
        self.pick_side.addWidget(label(row.name, "h2"))
        ex = row.excluded or bool(row.status)
        tr = 100 * row.transparency / self.chart.p90 if self.chart is not None else 0.0
        self.pick_side.addWidget(kv([(_("Time"), (row.time or "")[11:16]), ("FWHM", f"{fnum(row.fwhm, 2)} px"),
                                     (_("Transparency"), f"{tr:.0f} %"),
                                     (_("State"), _("excluded") if ex else _("kept"))], pad=10))
        b1 = QPushButton(_("Keep") if row.excluded else _("Exclude"))
        b1.setEnabled(row.session is not None)
        b1.clicked.connect(lambda: (self._set_hand([(row.session, row.name)], not row.excluded), self.request_refresh()))
        self.pick_side.addWidget(b1, 0, Qt.AlignLeft)
        self.pick_side.addStretch(1)
        self._frame_token = self.service.request_frame(row.path, 800, True)

    def _frame_ready(self, token: int, path: str, arr):
        if token != self._frame_token or self.table is None or not hasattr(self, "pick_view"):
            return
        try:
            self.pick_view.set_source(StaticSource(arr))
        except RuntimeError:
            pass


def approx(seconds: float, measured: bool) -> str:
    """A duration, with ≈ when it is an estimate ("≤ 1 min" is one already)."""
    t = hm(seconds)
    return t if measured or t.startswith("≤") else "≈ " + t


def sky_text(ra: float, dec: float) -> str:
    """RA and Dec as the readout writes them: 02h42m06.6s +61°18′01″ (a decimal comma in French)."""
    rh = (ra % 360.0) / 15.0
    h = int(rh)
    m = int(rh % 1 * 60)
    sec = (rh * 60 % 1) * 60
    d = abs(dec)
    sign = "+" if dec >= 0 else "−"
    dd, dm = int(d), int(d % 1 * 60)
    ds = (d * 60 % 1) * 60
    sec_txt = number(sec, 1).rjust(4, "0")
    return f"RA {h:02d}h{m:02d}m{sec_txt}s   Dec {sign}{dd:02d}°{dm:02d}′{ds:02.0f}″"


__all__ = ["TuneView", "Strip", "Histogram", "StretchSource", "ImageSlot", "sky_text"]
