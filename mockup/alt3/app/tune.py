"""Inspect: the image of one block, large.

At the top, the current workflow as a strip of blocks; in the middle, the result of the chosen block with the
inspection tools (auto-stretch, histogram, 1:1, pixel readout in ADU and sky coordinates); on the right, the same panel
as in Graph, with the measurements. The Frames block opens the sessions and the frame browser.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QStackedWidget, QTabBar,
                               QVBoxLayout, QWidget)

import prefs
from blocks import Graph, block_title, chain
from charts import FramesChart, SpectralChart
from i18n import _
from model import DATA, hm
from panel import STATE_WORD, Panel
from runs import run_blocks
from steps import BY_KEY, fnum, kv
from viewer import StaticSource, Viewer
from widgets import ACCENT, OK, WARN, clear, label

STATE_COL = {"done": OK, "pending": "#5d6470", "stale": WARN}
LINEAR = {"calibration", "alignement", "selection", "empilement", "astrometrie", "fond", "couleur", "debruitage",
          "deconv", "etoiles"}                         # blocks whose data is linear: shown auto-stretched or not


def run_stages(s, steps, p, key):
    b = BY_KEY[key]
    st = list(steps.run_stages(p, b))
    if key == "etoiles" and "starless" in s.all_stages(p):
        st.append("starless")
    return st


class StretchSource:
    """A block's image, shown with the automatic display stretch (as prepared) or linear (darker, as stored)."""

    def __init__(self, src, stretched=True):
        self.src, self.stretched = src, stretched
        self.version = 0

    def size(self):
        return self.src.size()

    def region(self, level, x0, y0, x1, y1):
        a = self.src.region(level, x0, y0, x1, y1)
        if self.stretched:
            return a
        return (np.power(a.astype(np.float32) / 255, 2.4) * 255).astype(np.uint8)


class Histogram(QWidget):
    """The histogram of what is shown: three channels, logarithmic counts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(90)
        self.h = None

    def set_image(self, a: np.ndarray):
        a = a[::4, ::4].reshape(-1, 3)
        self.h = [np.log1p(np.bincount(a[:, c], minlength=256)) for c in range(3)]
        self.update()

    def paintEvent(self, e):
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
        keys = ["sources"] + (chain_keys(s, g) if p is not None else [])
        for i, k in enumerate(keys):
            if i:
                a = QLabel("›")
                a.setStyleSheet("color: #6b7280; background: transparent;")
                self.h.addWidget(a)
            node = g.node_of(p, k)
            st = g.state(node) if node is not None else "pending"
            b = QPushButton(block_title(None, k) if k == "post" else block_title(p, k))
            sel = k == s.sel_step
            b.setStyleSheet(f"QPushButton {{ border-left: 3px solid {STATE_COL[st]}; padding: 4px 8px; "
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


def chain_keys(s, g):
    return chain(g.steps, s.path) if s.path is not None else []


class TuneView(QWidget):
    back = Signal()
    pick = Signal(str)                    # a block of the strip
    workflow = Signal(int)

    def __init__(self, project, steps, sources_view, post_page, parent=None):
        super().__init__(parent)
        self.s, self.steps = project, steps
        self.img_tab = 0
        self.cmp_pid = None
        self.picked = None
        self.stretched = bool(prefs.get("auto_stretch"))
        self.show_hist = False
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.strip = Strip()
        self.strip.back.connect(self.back.emit)
        self.strip.picked.connect(self.pick.emit)
        self.strip.workflow.connect(self.workflow.emit)
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

    # ------------------------------------------------------------ refresh
    def refresh(self, g: Graph):
        s = self.s
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
        clear(self.head)
        clear(self.center)
        clear(self.foot)
        self.viewers = []
        if node is None:
            return
        k = s.sel_step
        b = BY_KEY[k]
        state = self.steps.state(p, b)
        if k == "etoiles" and "starless" in s.all_stages(p) and state == "done":
            state = s.state(p, "starless") if s.state(p, "starless") != "done" else "done"
        self.head.addWidget(label(f"{block_title(p, k)}  ·  {p.name}" if k != "post" else p.name, "h1"))
        if state == "pending":
            self.head.addStretch(1)
            self.pending(p, k)
            return
        if state == "stale":
            tag = QLabel(_("previous result"))
            tag.setStyleSheet(f"color: {WARN}; border: 1px solid {WARN}; border-radius: 9px; padding: 1px 8px;")
            self.head.addWidget(tag)
            items = s.pending_upto(p, run_stages(s, self.steps, p, k))
            rb = QPushButton(_("Run to here") + f"  ·  {hm(sum(s.seconds(x) for _x, x in items))}")
            rb.clicked.connect(lambda: run_blocks(s, self.steps, items, _('Run "{name}" to {block}').format(
                name=p.name, block=block_title(p, k)), self.window()))
            self.head.addWidget(rb)
        self.head.addStretch(1)
        if k == "selection":
            self.selection_center()
            return
        vis = self.visuals(p, k)
        if not vis:
            self.center.addWidget(label(_("No image for this block."), "dim"), 0, Qt.AlignCenter)
            return
        i = min(self.img_tab, len(vis) - 1)
        if len(vis) > 1:
            tb = QTabBar()
            tb.setDrawBase(False)
            tb.setExpanding(False)
            for _k, cap in vis:
                tb.addTab(cap)
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
        if q is None:
            self.center.addWidget(self.visual(p, k, vis[i]), 1)
        else:
            row = QHBoxLayout()
            for w in (p, q):
                col = QVBoxLayout()
                same = g.node_of(w, k) is node
                col.addWidget(label(w.name + ("  ·  " + _("same block") if same and w is q else ""), "h2"))
                if self.steps.state(w, b) == "pending":
                    col.addWidget(label(_("Not computed yet."), "dim"), 1, Qt.AlignCenter)
                else:
                    wv = self.visuals(w, k)
                    col.addWidget(self.visual(w, k, wv[min(i, len(wv) - 1)]), 1)
                row.addLayout(col, 1)
            self.center.addLayout(row, 1)
            vws = [x for x in self.viewers if isinstance(x, Viewer)]
            if len(vws) == 2:
                vws[0].viewChanged.connect(lambda: self._sync(vws[0], vws[1]))
                vws[1].viewChanged.connect(lambda: self._sync(vws[1], vws[0]))
        self.tools(k)

    # ------------------------------------------------------------ inspection tools
    def tools(self, k):
        """Under the image: auto-stretch, histogram, fit and 1:1, and the pixel under the cursor."""
        if not self.viewers:
            return
        if k in LINEAR:
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
        for v in self.viewers:
            v.hovered.connect(lambda x, y, vv=v: self._hover(vv, x, y))
        if self.show_hist:
            self.hist = Histogram()
            src = self.viewers[0].source
            w, h = src.size()
            self.hist.set_image(src.region(1, 0, 0, -(-w // 2), -(-h // 2)))
            self.center.addWidget(self.hist)

    def _stretch(self, on):
        self.stretched = on
        prefs.set("auto_stretch", on)
        for v in self.viewers:
            if isinstance(v.source, StretchSource):
                v.source.stretched = on
                v.source.version += 1
                v.refresh()

    def _hist(self, on):
        self.show_hist = on
        self.window().request_refresh()

    def _hover(self, v, x, y):
        src = v.source
        w, h = src.size()
        if not (0 <= x < w and 0 <= y < h):
            self.readout.setText("")
            return
        px = src.region(0, int(x), int(y), int(x) + 1, int(y) + 1)[0, 0].astype(float)
        adu = [int((c / 255) ** 2.4 * 16383) for c in px] if self.stretched else [int(c / 255 * 16383) for c in px]
        m = self.s.data["stages"].get("astrometry", {}).get("metrics", {})
        sky = ""
        if m.get("ra") is not None:
            W, H = self.s.data["size"]
            scale = m.get("scale_arcsec", 2.0) * (self.s.data["size"][0] / w)
            dec = m["dec"] - (y - h / 2) * scale / 3600
            ra = m["ra"] + (x - w / 2) * scale / 3600 / max(math.cos(math.radians(dec)), 0.1)
            rh = ra / 15
            sky = (f"   RA {int(rh):02d}h{int(rh % 1 * 60):02d}m{(rh * 60 % 1) * 60:04.1f}s   "
                   f"Dec {'+' if dec >= 0 else '−'}{int(abs(dec)):02d}°{int(abs(dec) % 1 * 60):02d}′"
                   f"{(abs(dec) * 60 % 1) * 60:02.0f}″")
        self.readout.setText(f"x {int(x)}  y {int(y)}   R {adu[0]}  G {adu[1]}  B {adu[2]} ADU{sky}")

    # ------------------------------------------------------------ helpers
    def _sync(self, a, b):
        if getattr(self, "_syncing", False):
            return
        self._syncing = True
        try:
            if a.source and b.source and a.source.size() == b.source.size():
                b.zoom_to(a.scale, a.cx, a.cy)
        finally:
            self._syncing = False

    def _tab(self, i):
        self.img_tab = i
        self.window().request_refresh()

    def _cmp(self, pid):
        self.cmp_pid = pid
        self.window().request_refresh()

    def visuals(self, p, k):
        vis = [(kind, _(cap)) for kind, cap in self.steps.visuals(p, k)]
        if k == "etoiles":
            vis = [("img:starless", _("Starless image"))] + vis
        if k == "post":
            vis = [(kind, _("Output, suggested settings")) for kind, _c in vis]
        return vis

    def visual(self, p, k, vis):
        kind, _c = vis
        if kind == "chart:spectral":
            ch = SpectralChart(self.s.data["spectral"]["lam"], self.s.data["spectral"]["S"])
            return ch
        vw = Viewer()
        vw.setMinimumHeight(300)
        src = self.steps.source(kind)
        vw.set_source(StretchSource(src, self.stretched or k not in LINEAR))
        ov = self.steps.overlay(k, kind)
        if ov:
            vw.overlays.append(ov)
        self.viewers.append(vw)
        return vw

    def pending(self, p, k):
        s = self.s
        items = s.pending_upto(p, run_stages(s, self.steps, p, k))
        sec = sum(s.seconds(x) for _x, x in items)
        box = QVBoxLayout()
        box.setAlignment(Qt.AlignCenter)
        box.addWidget(label(_("Not computed yet"), "h2"), 0, Qt.AlignHCenter)
        box.addSpacing(8)
        rb = QPushButton(_("Run to here") + f"  ·  {hm(sec)}")
        rb.setObjectName("primary")
        rb.setEnabled(s.sources_loaded and bool(items))
        rb.setToolTip(", ".join(dict.fromkeys(self.steps.step_titles(p, [x for _x, x in items]))))
        rb.clicked.connect(lambda: run_blocks(s, self.steps, items, _('Run "{name}" to {block}').format(
            name=p.name, block=block_title(p, k)), self.window()))
        box.addWidget(rb, 0, Qt.AlignHCenter)
        pv = QPushButton(_("Preview on a region") + f"  ·  ≈ {hm(max(sec / 12, 30))}")
        pv.setToolTip(_("Runs the blocks on a region drawn on a frame, at full resolution: a close preview before "
                        "a long run"))
        pv.setEnabled(s.sources_loaded and bool(items))
        pv.clicked.connect(lambda: self.window().flash(_("Demo: draw the region on a frame, then the preview runs")))
        box.addWidget(pv, 0, Qt.AlignHCenter)
        if not s.sources_loaded:
            box.addWidget(label(_("No frames yet."), "warn"), 0, Qt.AlignHCenter)
        self.center.addStretch(1)
        self.center.addLayout(box)
        self.center.addStretch(2)

    # ---- the frames chart, and the frame under the cursor
    def selection_center(self):
        s = self.s
        ch = FramesChart(s.data["frames"], 0.6, 2.0)
        ch.tmin, ch.fmax = s.thresholds_now()
        ch.sugg = s.suggested_thresholds()
        ch.hand = set(s.frames_excluded)
        ch.current = self.picked
        ch.setMinimumHeight(300)
        ch.changed.connect(lambda c=ch: self._chart_changed(c))
        ch.picked.connect(self._pick)
        ch.setToolTip(_("Thresholds: drag · frame: click"))
        self.center.addWidget(ch, 3)
        n = len(s.data["frames"]["fwhm"])
        kept = s.kept_frames()
        self.center.addWidget(label(_("{kept} of {n} frames kept · {h} h").format(
            kept=kept, n=n, h=fnum(kept * s.data["frames"]["exposure_s"] / 3600)), "dim"))
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
        self._pick(self.picked if self.picked is not None else
                   next((i for i, nm in enumerate(s.data["frames"]["name"]) if "DSC02333" in nm), 0))

    def _chart_changed(self, c):
        """Frames excluded by hand count for the whole project; thresholds go through the panel (Replace or
        Try as variant, when computed results would be lost)."""
        s = self.s
        if set(c.hand) != set(s.frames_excluded):
            s.set_selection(hand=c.hand)
        t, f = s.thresholds_now()
        if abs(c.tmin - t) > 1e-6 or abs(c.fmax - f) > 1e-6:
            self.panel.propose_selection(c.tmin, c.fmax)
        self.window().request_refresh()

    def _pick(self, i):
        self.picked = i
        clear(self.pick_side)
        f = self.s.data["frames"]
        name = f["name"][i].rsplit(".", 1)[0]
        tr = 100 * f["transp"][i] / float(np.percentile(f["transp"], 90))
        ex = bool(self.s.excluded_mask()[i])
        self.pick_side.addWidget(label(f["name"][i], "h2"))
        self.pick_side.addWidget(kv([(_("Time"), f["time"][i]), ("FWHM", f"{fnum(f['fwhm'][i], 2)} px"),
                                     (_("Transparency"), f"{tr:.0f} %"),
                                     (_("State"), _("excluded") if ex else _("kept"))], pad=10))
        b1 = QPushButton(_("Keep") if i in self.s.frames_excluded else _("Exclude"))
        b1.clicked.connect(lambda: (self.s.set_selection(hand=set(self.s.frames_excluded) ^ {i}),
                                    self.window().request_refresh()))
        self.pick_side.addWidget(b1, 0, Qt.AlignLeft)
        self.pick_side.addStretch(1)
        path = DATA / "raws" / "lights" / f"{name}.jpg"
        if path.exists():
            self.pick_view.set_source(StaticSource(np.asarray(Image.open(path).convert("RGB"))))
