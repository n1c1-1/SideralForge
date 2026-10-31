"""Post view: the image at the centre, the computed workflows as layers, simplified sliders, masks from the data.

The layer images are the ones prepared by build/prepare.py. A slider jumps between real renderings prepared at 5
positions; several sliders moved at once add up their differences (an approximation of the mockup). The blending of
the layers through their masks is computed here, on those prepared images.
"""
from __future__ import annotations

import copy
from collections import OrderedDict

import numpy as np
from PIL import Image
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap, QShortcut, QKeySequence
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QScrollArea, QSlider, QStackedWidget, QVBoxLayout, QWidget)
from scipy import ndimage

from model import DATA, Session
from viewer import Viewer
from widgets import NotchSlider, clear, fit_height, hline, label, para

MODES = ["Normal", "Éclaircir", "Écran", "Multiplier"]


def half(a: np.ndarray) -> np.ndarray:
    """Half resolution (mean of 2 x 2), odd sizes rounded up like the viewer expects; images and masks alike."""
    h, w = a.shape[:2]
    pad = [(0, h % 2), (0, w % 2)] + [(0, 0)] * (a.ndim - 2)
    b = np.pad(a.astype(np.float32), pad, mode="edge")
    out = b.reshape(b.shape[0] // 2, 2, b.shape[1] // 2, 2, *b.shape[2:]).mean(axis=(1, 3))
    return out.astype(a.dtype) if a.dtype == np.uint8 else out


class ImageStore:
    """Prepared images, decoded on demand (full and half resolution), most recent kept in memory."""

    def __init__(self, cap=40):
        self.cache: OrderedDict = OrderedDict()
        self.cap = cap

    def get(self, rel: str, level: int) -> np.ndarray:
        key = (rel, level)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if level == 0:
            a = np.asarray(Image.open(DATA / rel).convert("RGB"))
        else:
            a = half(self.get(rel, 0))
        self.cache[key] = a
        while len(self.cache) > self.cap:
            self.cache.popitem(last=False)
        return a

    def mask(self, rel: str, level: int) -> np.ndarray:
        key = ("mask", rel, level)
        if key in self.cache:
            return self.cache[key]
        m = np.asarray(Image.open(DATA / rel).convert("L"), np.float32) / 255
        if level:
            m = half(m)
        self.cache[key] = m
        return m


def new_layer(pid: int, base=False) -> dict:
    return {"pid": pid, "visible": base, "opacity": 100, "mode": "Normal", "sliders": {},
            "mask": {"parts": [], "feather": 0, "invert": False}}


class Compositor:
    """Blends the visible layers (bottom to top) through their masks, then the final adjustment."""

    def __init__(self, session: Session, store: ImageStore):
        self.s = session
        self.store = store
        self.state = {"layers": [], "final": {"noir": 0, "tons": 0, "sat": 0}}
        self.version = 0
        self._mask_cache = {}
        self._brush = {}                                   # id(part) -> (raster at full res)
        self._half = None

    # ------------------------------------------------------------ source protocol for the viewer
    def size(self):
        return tuple(self.s.data["size"])

    def region(self, level, x0, y0, x1, y1):
        if level == 1:
            if self._half is None:
                h, w = -(-self.s.data["size"][1] // 2), -(-self.s.data["size"][0] // 2)
                self._half = self.compose(1, 0, 0, w, h)
            return self._half[y0:y1, x0:x1]
        return self.compose(0, x0, y0, x1, y1)

    def changed(self, masks=False):
        self.version += 1
        self._half = None
        if masks:
            self._mask_cache.clear()

    # ------------------------------------------------------------ one layer
    def layer_rgb(self, lay: dict, level, sl) -> np.ndarray:
        p = self.path(lay["pid"])
        info = self.s.layers[p.layer]
        base = self.store.get(info["base"], level)[sl].astype(np.float32)
        out = base.copy()
        for k, pos in lay["sliders"].items():
            files = info["sliders"].get(k)
            if pos == 2 or not files or not self.s.slider_state(p, k)[0]:
                continue
            out += self.store.get(files[pos], level)[sl].astype(np.float32) - base
        return out

    def path(self, pid):
        return next(q for q in self.s.paths if q.pid == pid)

    # ------------------------------------------------------------ masks
    def part_raster(self, part: dict, lay: dict, level: int) -> np.ndarray:
        W, H = self.s.data["size"]
        if part["src"] in ("etoiles", "halpha"):
            return self.store.mask(self.s.data["masks"][part["src"]]["file"], level)
        if part["src"] == "lum":
            p = self.path(lay["pid"])
            img = self.store.get(self.s.layers[p.layer]["base"], level).astype(np.float32) / 255
            L = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
            lo, hi = part["lo"] / 100, part["hi"] / 100
            soft = 0.06
            a = np.clip((L - (lo - soft)) / (2 * soft), 0, 1)
            b = np.clip(((hi + soft) - L) / (2 * soft), 0, 1)
            return a * b
        if part["src"] == "pinceau":
            r = self._brush.get(part["id"])
            if r is None:
                r = np.zeros((H, W), np.float32)
                for x, y, rad in part["strokes"]:
                    self.stamp(r, x, y, rad)
                self._brush[part["id"]] = r
            return half(r) if level else r
        raise ValueError(part["src"])

    @staticmethod
    def stamp(r: np.ndarray, x, y, rad):
        H, W = r.shape
        xa, xb = int(max(0, x - rad - 1)), int(min(W, x + rad + 2))
        ya, yb = int(max(0, y - rad - 1)), int(min(H, y + rad + 2))
        if xa >= xb or ya >= yb:
            return
        yy, xx = np.mgrid[ya:yb, xa:xb]
        d = np.hypot(xx - x, yy - y)
        r[ya:yb, xa:xb] = np.maximum(r[ya:yb, xa:xb], np.clip((rad - d) / (0.35 * rad), 0, 1))

    def mask(self, lay: dict, level: int) -> np.ndarray | None:
        m = lay["mask"]
        if not m["parts"]:
            return None
        key = (id(lay), level, repr(m), self.version_mask(lay))
        hit = self._mask_cache.get(key)
        if hit is not None:
            return hit
        out = None
        for part in m["parts"]:
            r = self.part_raster(part, lay, level)
            if part.get("invert"):
                r = 1 - r
            if out is None:
                out = r.copy() if part["op"] != "sub" else 1 - r
            elif part["op"] == "add":
                out = np.maximum(out, r)
            elif part["op"] == "sub":
                out = out * (1 - r)
            else:
                out = out * r
        if m["feather"]:
            out = ndimage.gaussian_filter(out, m["feather"] / (2 if level else 1))
        if m["invert"]:
            out = 1 - out
        self._mask_cache = {k: v for k, v in self._mask_cache.items() if k[0] != id(lay) or k[1] != level}
        self._mask_cache[key] = out
        return out

    def version_mask(self, lay):
        return tuple(len(p.get("strokes", ())) for p in lay["mask"]["parts"])

    # ------------------------------------------------------------ composition
    def compose(self, level, x0, y0, x1, y1, layers=None, final=True) -> np.ndarray:
        sl = (slice(y0, y1), slice(x0, x1))
        layers = self.state["layers"] if layers is None else layers
        vis = [lay for lay in layers if lay["visible"]]
        if not vis:
            return np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        comp = None
        for lay in reversed(vis):                          # the list is top first
            src = self.layer_rgb(lay, level, sl)
            if comp is None:
                comp = src
                continue
            m = self.mask(lay, level)
            a = (lay["opacity"] / 100) * (np.ones(src.shape[:2], np.float32) if m is None else m[sl])
            a = a[..., None]
            mode = lay["mode"]
            if mode == "Éclaircir":
                src = np.maximum(comp, src)
            elif mode == "Écran":
                src = 255 - (255 - comp) * (255 - src) / 255
            elif mode == "Multiplier":
                src = comp * src / 255
            comp = comp * (1 - a) + src * a
        comp = np.clip(comp, 0, 255)
        if final:
            comp = self.final_adjust(comp)
        return comp.astype(np.uint8)

    def final_adjust(self, c: np.ndarray) -> np.ndarray:
        f = self.state["final"]
        if not any(f.values()):
            return c
        x = c / 255
        black = 0.04 * f["noir"]
        x = np.clip((x - black) / (1 - black), 0, 1)
        g = 1.25 ** (-f["tons"])
        x = x ** g
        if f["sat"]:
            L = (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])[..., None]
            x = np.clip(L + (1 + 0.25 * f["sat"]) * (x - L), 0, 1)
        return x * 255


class StaticPair:
    """Viewer source for the 'before' side of the curtain: the composite with every slider at its suggestion."""

    def __init__(self, comp: Compositor):
        self.c = comp
        self.version = 0

    def size(self):
        return self.c.size()

    def region(self, level, x0, y0, x1, y1):
        base = copy.deepcopy(self.c.state["layers"])
        for lay in base:
            lay["sliders"] = {}
        return self.c.compose(level, x0, y0, x1, y1, layers=base, final=False)


class LayerRow(QFrame):
    clicked = Signal(int)
    toggled = Signal(int, bool)

    def __init__(self, idx, name, sub, thumb: QPixmap, visible, selected, has_mask, parent=None):
        super().__init__(parent)
        self.idx = idx
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(("QFrame { background: #2b3446; border-radius: 4px; }" if selected else
                            "QFrame { background: transparent; border-radius: 4px; } QFrame:hover "
                            "{ background: #232831; }") + " QLabel { background: transparent; }")
        h = QHBoxLayout(self)
        h.setContentsMargins(6, 4, 6, 4)
        self.eye = QCheckBox()
        self.eye.setChecked(visible)
        self.eye.setToolTip("Afficher le calque")
        self.eye.toggled.connect(lambda v: self.toggled.emit(self.idx, v))
        h.addWidget(self.eye)
        th = QLabel()
        th.setPixmap(thumb)
        h.addWidget(th)
        if has_mask is not None:
            mk = QLabel()
            mk.setPixmap(has_mask)
            mk.setToolTip("Masque du calque")
            h.addWidget(mk)
        v = QVBoxLayout()
        v.setSpacing(0)
        a = QLabel(name)
        a.setStyleSheet("font-weight: 600;")
        b = QLabel(sub)
        b.setObjectName("small")
        v.addWidget(a)
        v.addWidget(b)
        h.addLayout(v, 1)

    def mousePressEvent(self, e):
        self.clicked.emit(self.idx)


class PostView(QWidget):
    goProcessing = Signal(int, str)                        # workflow id, step to show
    goResults = Signal(int)                                # workflow id

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.s = session
        self.store = ImageStore()
        self.comp = Compositor(session, self.store)
        self.before = StaticPair(self.comp)
        self.undo, self.redo = [], []
        self.sel = 0
        self.mask_view = False
        self.expert_open = False
        self._painting = None
        self._thumbs = {}
        # one layer per workflow computed to the end; bottom layer: Complet when there is one; above: hidden
        lays = [new_layer(p.pid) for p in self.s.paths if p.rendered]
        order = sorted(lays, key=lambda l: self.comp.path(l["pid"]).layer == "complet")
        if order:
            order[-1]["visible"] = True
        self.comp.state["layers"] = order
        self.sel = len(order) - 1

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        # ---- left: masks
        self.left = QWidget()
        self.left.setObjectName("panel")
        self.left.setFixedWidth(290)
        lv = QVBoxLayout(self.left)
        lv.setContentsMargins(12, 12, 12, 12)
        self.mask_box = QVBoxLayout()
        lv.addLayout(self.mask_box)
        lv.addStretch(1)
        root.addWidget(self.left)
        # ---- centre: viewer and its bar
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(10, 6, 10, 6)
        self.zoom_lab = label("", "dim")
        self.b_fit = QPushButton("Ajuster")
        self.b_100 = QPushButton("100 %")
        self.b_cmp = QPushButton("Avant / après")
        self.b_cmp.setCheckable(True)
        self.b_cmp.setObjectName("toggle")
        self.b_cmp.setToolTip("Avant : tous les curseurs à la suggestion. Après : tes réglages. (touche C)")
        self.b_mask = QPushButton("Voir le masque")
        self.b_mask.setCheckable(True)
        self.b_mask.setObjectName("toggle")
        self.b_mask.setToolTip("Le masque du calque choisi en rouge sur l'image (touche M)")
        self.b_undo, self.b_redo = QPushButton("↶"), QPushButton("↷")
        self.b_undo.setToolTip("Annuler (Ctrl+Z)")
        self.b_redo.setToolTip("Rétablir (Ctrl+Y)")
        self.b_export = QPushButton("Exporter…")
        for w in (self.b_fit, self.b_100, self.zoom_lab):
            bar.addWidget(w)
        bar.addStretch(1)
        for w in (self.b_cmp, self.b_mask, self.b_undo, self.b_redo, self.b_export):
            bar.addWidget(w)
        mv.addLayout(bar)
        self.viewer = Viewer()
        self.viewer.set_source(self.comp)
        self.viewer.overlays.append(self._overlay)
        mv.addWidget(self.viewer, 1)
        self.hint = label("", "small")
        self.hint.setContentsMargins(10, 4, 10, 6)
        mv.addWidget(self.hint)
        root.addWidget(mid, 1)
        # ---- right: layers and sliders
        right = QWidget()
        right.setObjectName("panel")
        right.setFixedWidth(360)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(10, 12, 10, 10)
        rv.addWidget(label("Calques", "h2"))
        rv.addWidget(label("Un calque par workflow calculé. Le plus haut recouvre les autres là où son masque le "
                           "laisse passer.", "small", wrap=True))
        self.layers_box = QVBoxLayout()
        self.layers_box.setSpacing(2)
        rv.addLayout(self.layers_box)
        rv.addWidget(hline())
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sl_host = QWidget()
        self.sl_box = QVBoxLayout(self.sl_host)
        self.sl_box.setContentsMargins(0, 0, 6, 0)
        sc.setWidget(self.sl_host)
        rv.addWidget(sc, 1)
        root.addWidget(right)

        self.b_fit.clicked.connect(self.viewer.fit)
        self.b_100.clicked.connect(lambda: self.viewer.zoom_to(1.0))
        self.b_cmp.toggled.connect(self._compare)
        self.b_mask.toggled.connect(self._mask_toggle)
        self.b_undo.clicked.connect(self.do_undo)
        self.b_redo.clicked.connect(self.do_redo)
        self.b_export.clicked.connect(self.export)
        self.viewer.viewChanged.connect(self._zoom_label)
        self.viewer.stroke.connect(self._stroke)
        self.viewer.strokeEnd.connect(self._stroke_end)
        for seq, fn in (("Ctrl+Z", self.do_undo), ("Ctrl+Y", self.do_redo), ("C", self.b_cmp.toggle),
                        ("M", self.b_mask.toggle), ("F", self.viewer.fit), ("1", lambda: self.viewer.zoom_to(1.0))):
            sc_ = QShortcut(QKeySequence(seq), self)
            sc_.setContext(Qt.WidgetWithChildrenShortcut)
            sc_.activated.connect(fn)
        self.s.pathsChanged.connect(self.sync_paths)
        self.s.testChanged.connect(self.refresh)
        self.rebuild()

    # ------------------------------------------------------------ state, undo
    @property
    def layers(self):
        return self.comp.state["layers"]

    def push(self):
        self.undo.append(copy.deepcopy(self.comp.state))
        self.redo.clear()
        self.b_undo.setEnabled(True)
        self.b_redo.setEnabled(False)

    def do_undo(self):
        if not self.undo:
            return
        self.redo.append(copy.deepcopy(self.comp.state))
        self.comp.state = self.undo.pop()
        self._after_restore()

    def do_redo(self):
        if not self.redo:
            return
        self.undo.append(copy.deepcopy(self.comp.state))
        self.comp.state = self.redo.pop()
        self._after_restore()

    def _after_restore(self):
        self.comp._brush.clear()
        self.sel = min(self.sel, len(self.layers) - 1)
        self.comp.changed(masks=True)
        self.rebuild()

    def sync_paths(self):
        had = bool(self.layers)
        known = {lay["pid"] for lay in self.layers}
        for p in self.s.paths:
            if p.rendered and p.pid not in known:
                self.layers.insert(0, new_layer(p.pid))
        alive = {p.pid for p in self.s.paths}
        self.comp.state["layers"] = [lay for lay in self.layers if lay["pid"] in alive]
        if self.layers and not any(lay["visible"] for lay in self.layers):
            self.layers[-1]["visible"] = True
        self.sel = len(self.layers) - 1 if not had else min(self.sel, len(self.layers) - 1)
        self.comp.changed(masks=True)
        self.rebuild()

    def refresh(self):
        self.comp.changed()
        self.before.version += 1
        self._apply_test()
        self.viewer.refresh()

    def _apply_test(self):
        if self.s.test_on:
            gx0, gx1, gy0, gy1 = self.s.data["crop"]
            F = self.s.data["F"]
            zx0, zy0, zx1, zy1 = self.s.test_zone
            self.viewer.dim_outside = ((zx0 - gx0) / F, (zy0 - gy0) / F, (zx1 - gx0) / F, (zy1 - gy0) / F)
            self.viewer.banner = "APERÇU · zone de test : proche du résultat final, pas identique"
        else:
            self.viewer.dim_outside = None
            self.viewer.banner = ""

    def showEvent(self, e):
        self._apply_test()
        if self.s.test_on and self.viewer.dim_outside:
            QTimer.singleShot(0, lambda: self.viewer.show_rect(*self.viewer.dim_outside))
        super().showEvent(e)

    # ------------------------------------------------------------ building the panels
    def thumb(self, lay) -> QPixmap:
        p = self.comp.path(lay["pid"])
        key = p.layer
        if key not in self._thumbs:
            a = self.store.get(self.s.layers[key]["base"], 1)
            im = Image.fromarray(a).resize((64, 43))
            a2 = np.ascontiguousarray(np.asarray(im))
            self._thumbs[key] = QPixmap.fromImage(QImage(a2.data, 64, 43, 192, QImage.Format_RGB888).copy())
        return self._thumbs[key]

    def mask_thumb(self, lay) -> QPixmap | None:
        m = self.comp.mask(lay, 1)
        if m is None:
            return None
        im = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).resize((64, 43))
        a = np.ascontiguousarray(np.asarray(im))
        return QPixmap.fromImage(QImage(a.data, 64, 43, 64, QImage.Format_Grayscale8).copy())

    def rebuild(self):
        clear(self.layers_box)
        top = QFrame()
        top.setStyleSheet("QFrame { background: #20242c; border-radius: 4px; } QLabel { background: transparent; }")
        th = QHBoxLayout(top)
        th.setContentsMargins(8, 4, 8, 4)
        tl = QLabel("Réglage final")
        tl.setStyleSheet("font-weight: 600;")
        th.addWidget(tl)
        th.addWidget(label("courbe et couleur sur le mélange", "small"), 1)
        fb = QPushButton("Régler")
        fb.clicked.connect(lambda: self.select(-1))
        th.addWidget(fb)
        self.layers_box.addWidget(top)
        n = len(self.layers)
        for i, lay in enumerate(self.layers):
            p = self.comp.path(lay["pid"])
            sub = "base" if i == n - 1 else f"{lay['mode']} · {lay['opacity']} %"
            if p.provisional:
                sub += " · aperçu"
            if not p.computed:
                sub += " · à recalculer"
            row = LayerRow(i, p.name, sub, self.thumb(lay), lay["visible"], i == self.sel,
                           self.mask_thumb(lay) if i < n - 1 else None)
            row.clicked.connect(self.select)
            row.toggled.connect(self._toggle_visible)
            self.layers_box.addWidget(row)
        waiting = [p for p in self.s.paths if not p.rendered]
        if waiting:
            self.layers_box.addSpacing(4)
            self.layers_box.addWidget(label("Pas encore calculés, donc sans calque :", "small"))
            for p in waiting:
                b = QPushButton(f"« {p.name} » : voir où il en est  →")
                b.setObjectName("link")
                b.clicked.connect(lambda _=False, pid=p.pid: self.goResults.emit(pid))
                self.layers_box.addWidget(b)
        self.b_undo.setEnabled(bool(self.undo))
        self.b_redo.setEnabled(bool(self.redo))
        self.build_sliders()
        self.build_masks()
        self.hint.setText(self._hint_text())
        self.viewer.refresh()

    def _hint_text(self):
        if self.sel < 0:
            return "Réglage final : il s'applique au mélange de tous les calques."
        lay = self.layers[self.sel]
        p = self.comp.path(lay["pid"])
        t = f"Calque « {p.name} »."
        if self.sel == len(self.layers) - 1:
            t += " C'est la base : les calques au-dessus la recouvrent là où leur masque le laisse passer."
        elif not lay["visible"]:
            t += " Caché : coche la case pour l'afficher."
        elif not lay["mask"]["parts"]:
            t += " Sans masque, il recouvre toute l'image : ajoute un masque à gauche."
        return t

    def select(self, i):
        self.sel = i
        self.rebuild()

    def _toggle_visible(self, i, v):
        self.push()
        self.layers[i]["visible"] = v
        self.comp.changed()
        self.rebuild()

    # ------------------------------------------------------------ sliders
    def build_sliders(self):
        clear(self.sl_box)
        if self.sel < 0:
            self.sl_box.addWidget(label("Réglage final", "h2"))
            f = self.comp.state["final"]
            for key, name, ends in (("noir", "Point noir", ("plus clair", "plus sombre")),
                                    ("tons", "Tons moyens", ("plus sombres", "plus clairs")),
                                    ("sat", "Saturation globale", ("plus pâle", "plus vive"))):
                ns = NotchSlider(key, name, ends)
                ns.set_pos(f[key] + 2)
                ns.moved.connect(self._final_moved)
                self.sl_box.addWidget(ns)
            self.sl_box.addWidget(label("Calculé dans la maquette sur l'image affichée (courbe simple).", "small",
                                        wrap=True))
            self.sl_box.addStretch(1)
            return
        lay = self.layers[self.sel]
        p = self.comp.path(lay["pid"])
        head = QHBoxLayout()
        head.addWidget(label(f"Curseurs · {p.name}", "h2"), 1)
        self.sl_box.addLayout(head)
        if self.sel < len(self.layers) - 1:
            row = QHBoxLayout()
            row.addWidget(label("Mélange", "dim"))
            cb = QComboBox()
            cb.addItems(MODES)
            cb.setCurrentText(lay["mode"])
            cb.currentTextChanged.connect(self._mode)
            row.addWidget(cb)
            row.addWidget(label("Opacité", "dim"))
            op = QSlider(Qt.Horizontal)
            op.setRange(0, 100)
            op.setValue(lay["opacity"])
            op.sliderReleased.connect(lambda o=op: self._opacity(o.value()))
            row.addWidget(op, 1)
            self.sl_box.addLayout(row)
        group = None
        for sd in self.s.data["sliders"]:
            if sd["group"] != group:
                group = sd["group"]
                self.sl_box.addSpacing(6)
                self.sl_box.addWidget(label(group, "dim"))
            ns = NotchSlider(sd["key"], sd["name"], sd["ends"])
            ok, why, need = self.s.slider_state(p, sd["key"])
            has_imgs = sd["key"] in self.s.layers[p.layer]["sliders"]
            if ok and not has_imgs:
                ok, why, need = False, "pas préparé dans la maquette pour ce calque", None
            ns.set_available(ok, why, self.s.unlock_text(p, need) if need else "")
            ns.set_pos(lay["sliders"].get(sd["key"], 2) if ok else 2)
            ns.moved.connect(self._slider_moved)
            ns.unlock.connect(lambda k, pid=p.pid: self.goProcessing.emit(pid, "etoiles" if k != "bruit"
                                                                         else "debruitage"))
            ns.setToolTip(self._expert_tip(sd, p))
            self.sl_box.addWidget(ns)
        self.sl_box.addSpacing(6)
        ex = QCheckBox("Expert : réglages du moteur")
        ex.setToolTip("Les réglages du moteur que chaque curseur simplifié déplace")
        self.expert = QLabel()
        self.expert.setObjectName("small")
        self.expert.setWordWrap(True)
        self.expert.setVisible(self.expert_open)
        ex.setChecked(self.expert_open)
        self._fill_expert(lay, p)
        ex.toggled.connect(lambda v: (setattr(self, "expert_open", v), self.expert.setVisible(v)))
        self.sl_box.addWidget(ex)
        self.sl_box.addWidget(self.expert)
        self.sl_box.addStretch(1)

    def _expert_tip(self, sd, p):
        if self.s.layers[p.layer]["kind"] != "engine":
            return "Calque classique : étirement automatique simple (approximation de la maquette)."
        return "\n".join(f"{sd['labels'][k]} : {' · '.join(str(x) for x in v)}" for k, v in sd["engine"].items())

    def _fill_expert(self, lay, p):
        if self.s.layers[p.layer]["kind"] != "engine":
            self.expert.setText("Ce calque n'utilise pas le moteur de rendu : il ne montre que la pile avec un "
                                "étirement automatique. Le vrai client aurait ici les réglages de cet étirement.")
            return
        rows = []
        for sd in self.s.data["sliders"]:
            pos = lay["sliders"].get(sd["key"], 2)
            for k, v in sd["engine"].items():
                mark = "" if pos == 2 else f"  (suggéré {v[2]})"
                rows.append(f"<b>{sd['labels'][k]}</b> = {v[pos]}{mark}")
        self.expert.setText("Chaque curseur simplifié déplace ensemble ces réglages du moteur (73 au total dans le "
                            "client actuel) :<br>" + "<br>".join(rows))

    def _slider_moved(self, key, pos):
        self.push()
        lay = self.layers[self.sel]
        if pos == 2:
            lay["sliders"].pop(key, None)
        else:
            lay["sliders"][key] = pos
        if not lay["visible"]:
            lay["visible"] = True
        self.comp.changed()
        QTimer.singleShot(0, self.rebuild_layers_only)

    def _final_moved(self, key, pos):
        self.push()
        self.comp.state["final"][key] = pos - 2
        self.comp.changed()
        self.viewer.refresh()

    def rebuild_layers_only(self):
        self.rebuild()

    def _mode(self, m):
        self.push()
        self.layers[self.sel]["mode"] = m
        self.comp.changed()
        self.rebuild()

    def _opacity(self, v):
        self.push()
        self.layers[self.sel]["opacity"] = v
        self.comp.changed()
        self.rebuild()

    # ------------------------------------------------------------ masks
    def build_masks(self):
        clear(self.mask_box)
        self.viewer.mode = "pan"
        self.mask_box.addWidget(label("Masque du calque", "h2"))
        if self.sel < 0 or self.sel == len(self.layers) - 1:
            self.mask_box.addWidget(label("La base n'a pas de masque : choisis un calque au-dessus." if self.sel >= 0
                                          else "Le réglage final s'applique partout.", "small", wrap=True))
            return
        lay = self.layers[self.sel]
        m = lay["mask"]
        p = self.comp.path(lay["pid"])
        self.mask_box.addWidget(label(f"Où « {p.name} » recouvre les calques en dessous.", "small", wrap=True))
        if not m["parts"]:
            self.mask_box.addWidget(label("Pas de masque : le calque recouvre toute l'image.", "dim", wrap=True))
        for i, part in enumerate(m["parts"]):
            self.mask_box.addWidget(self._part_row(i, part, lay))
        self.mask_box.addSpacing(6)
        self.mask_box.addWidget(label("Ajouter au masque", "dim"))
        grid = QVBoxLayout()
        for src, name, tip in (("etoiles", "★  Étoiles du catalogue", self.s.data["masks"]["etoiles"]["note"]),
                               ("halpha", "◐  Émission H-alpha", self.s.data["masks"]["halpha"]["note"]),
                               ("lum", "◑  Plage de luminosité", "Les zones dont la luminosité est dans une plage."),
                               ("pinceau", "✎  Pinceau", "Peindre le masque à la main sur l'image.")):
            b = QPushButton(name)
            b.setToolTip(tip)
            b.setStyleSheet("text-align: left; padding-left: 10px;")
            b.clicked.connect(lambda _=False, s_=src: self.add_part(s_))
            grid.addWidget(b)
        self.mask_box.addLayout(grid)
        self.mask_box.addSpacing(6)
        if m["parts"]:
            fr_ = QHBoxLayout()
            fr_.addWidget(label("Adoucir le bord", "dim"))
            fs = QSlider(Qt.Horizontal)
            fs.setRange(0, 40)
            fs.setValue(m["feather"])
            fs.sliderReleased.connect(lambda f=fs: self._feather(f.value()))
            fr_.addWidget(fs, 1)
            self.mask_box.addLayout(fr_)
            inv = QCheckBox("Inverser le masque")
            inv.setChecked(m["invert"])
            inv.toggled.connect(self._invert)
            self.mask_box.addWidget(inv)
            clr = QPushButton("Effacer le masque")
            clr.clicked.connect(self._clear_mask)
            self.mask_box.addWidget(clr)
        brush = [pt for pt in m["parts"] if pt["src"] == "pinceau"]
        if brush:
            self.viewer.mode = "brush"
            row = QHBoxLayout()
            row.addWidget(label("Taille du pinceau", "dim"))
            bs = QSlider(Qt.Horizontal)
            bs.setRange(5, 200)
            bs.setValue(int(self.viewer.brush_r))
            bs.valueChanged.connect(lambda v: setattr(self.viewer, "brush_r", float(v)))
            row.addWidget(bs, 1)
            self.mask_box.addLayout(row)
            self.mask_box.addWidget(label("Peins sur l'image. Clic droit ou molette pour te déplacer.", "small",
                                          wrap=True))

    def _part_row(self, i, part, lay):
        fr_ = QFrame()
        fr_.setStyleSheet("QFrame { background: #23272f; border-radius: 4px; } QLabel { background: transparent; }")
        v = QVBoxLayout(fr_)
        v.setContentsMargins(8, 6, 8, 6)
        h = QHBoxLayout()
        names = {"etoiles": "Étoiles du catalogue", "halpha": "Émission H-alpha", "lum": "Plage de luminosité",
                 "pinceau": "Pinceau"}
        h.addWidget(QLabel(names[part["src"]]), 1)
        if i > 0:
            op = QComboBox()
            op.addItems(["ajouter", "retirer", "croiser"])
            op.setCurrentIndex({"add": 0, "sub": 1, "inter": 2}[part["op"]])
            op.setToolTip("Ajouter : réunir · Retirer : enlever du masque · Croiser : garder seulement l'intersection")
            op.currentIndexChanged.connect(lambda k, j=i: self._part_op(j, ["add", "sub", "inter"][k]))
            h.addWidget(op)
        rm = QPushButton("×")
        rm.setFixedSize(26, 24)
        rm.setStyleSheet("padding: 0;")
        rm.setToolTip("Retirer cet élément du masque")
        rm.clicked.connect(lambda _=False, j=i: self._part_remove(j))
        h.addWidget(rm)
        v.addLayout(h)
        if part["src"] == "lum":
            for k, name in (("lo", "de"), ("hi", "à")):
                r = QHBoxLayout()
                r.addWidget(label(name, "small"))
                s = QSlider(Qt.Horizontal)
                s.setRange(0, 100)
                s.setValue(part[k])
                s.sliderReleased.connect(lambda s_=s, j=i, kk=k: self._part_val(j, kk, s_.value()))
                r.addWidget(s, 1)
                r.addWidget(label(f"{part[k]} %", "small"))
                v.addLayout(r)
        if part["src"] == "pinceau":
            v.addWidget(label(f"{len(part['strokes'])} touches de pinceau", "small"))
        return fr_

    def add_part(self, src):
        self.push()
        lay = self.layers[self.sel]
        part = {"src": src, "op": "add" if lay["mask"]["parts"] else "add"}
        if src == "lum":
            part.update(lo=55, hi=100)
        if src == "pinceau":
            part.update(id=f"b{len(self.undo)}_{id(part)}", strokes=[])
        lay["mask"]["parts"].append(part)
        lay["visible"] = True
        self.comp.changed(masks=True)
        if not self.b_mask.isChecked():
            self.b_mask.setChecked(True)
        self.rebuild()

    def _part_op(self, j, op):
        self.push()
        self.layers[self.sel]["mask"]["parts"][j]["op"] = op
        self.comp.changed(masks=True)
        self.rebuild()

    def _part_val(self, j, k, v):
        self.push()
        self.layers[self.sel]["mask"]["parts"][j][k] = v
        self.comp.changed(masks=True)
        self.rebuild()

    def _part_remove(self, j):
        self.push()
        self.layers[self.sel]["mask"]["parts"].pop(j)
        self.comp.changed(masks=True)
        self.rebuild()

    def _feather(self, v):
        self.push()
        self.layers[self.sel]["mask"]["feather"] = v
        self.comp.changed(masks=True)
        self.rebuild()

    def _invert(self, v):
        self.push()
        self.layers[self.sel]["mask"]["invert"] = v
        self.comp.changed(masks=True)
        self.rebuild()

    def _clear_mask(self):
        self.push()
        self.layers[self.sel]["mask"] = {"parts": [], "feather": 0, "invert": False}
        self.comp.changed(masks=True)
        self.rebuild()

    def _stroke(self, x, y, first):
        if self.sel < 0:
            return
        lay = self.layers[self.sel]
        parts = [pt for pt in lay["mask"]["parts"] if pt["src"] == "pinceau"]
        if not parts:
            return
        part = parts[-1]
        if first:
            self.push()
            lay = self.layers[self.sel]
            part = [pt for pt in lay["mask"]["parts"] if pt["src"] == "pinceau"][-1]
        r = self.viewer.brush_r
        last = self._painting if not first else None
        pts = [(x, y)]
        if last is not None:
            d = np.hypot(x - last[0], y - last[1])
            n = int(d / (r * 0.3)) + 1
            pts = [(last[0] + (x - last[0]) * k / n, last[1] + (y - last[1]) * k / n) for k in range(1, n + 1)]
        raster = self.comp._brush.get(part["id"])
        if raster is None:
            W, H = self.s.data["size"]
            raster = np.zeros((H, W), np.float32)
            self.comp._brush[part["id"]] = raster
        for px, py in pts:
            part["strokes"].append((px, py, r))
            Compositor.stamp(raster, px, py, r)
        self._painting = (x, y)
        self.comp.changed(masks=True)
        self.viewer.refresh()

    def _stroke_end(self):
        self._painting = None
        self.rebuild()

    # ------------------------------------------------------------ viewer helpers
    def _compare(self, on):
        self.viewer.compare = self.before if on else None
        self.viewer.update()

    def _mask_toggle(self, on):
        self.mask_view = on
        self.viewer.update()

    def _zoom_label(self):
        self.zoom_lab.setText(f"zoom {self.viewer.scale * 100:.0f} %")

    def _overlay(self, p: QPainter, v: Viewer):
        if not self.mask_view or self.sel < 0 or self.sel >= len(self.layers) - 1:
            return
        m = self.comp.mask(self.layers[self.sel], 1)
        if m is None:
            return
        key = (id(m), m.shape)
        if getattr(self, "_ov_key", None) != key:
            a = np.zeros(m.shape + (4,), np.uint8)
            a[..., 0] = 230
            a[..., 1] = 60
            a[..., 2] = 60
            a[..., 3] = (np.clip(m, 0, 1) * 120).astype(np.uint8)
            a = np.ascontiguousarray(a)
            self._ov_img = QImage(a.data, a.shape[1], a.shape[0], 4 * a.shape[1], QImage.Format_RGBA8888).copy()
            self._ov_key = key
        W, H = self.s.data["size"]
        p.drawImage(QRectF(v.img_to_w(0, 0), v.img_to_w(W, H)), self._ov_img)

    def export(self):
        d = QDialog(self)
        d.setWindowTitle("Exporter")
        v = QVBoxLayout(d)
        W, H = self.s.data["crop"][1] - self.s.data["crop"][0], self.s.data["crop"][3] - self.s.data["crop"][2]
        v.addWidget(label("Exporter l'image finale", "h2"))
        v.addWidget(label(f"Pleine résolution : {W} × {H} px. Chaque calque est recalculé par le moteur à pleine "
                          "résolution (environ 2 s chacun), puis mélangé avec ses masques.", "dim", wrap=True))
        fmt = QComboBox()
        fmt.addItems(["TIFF 16 bits", "PNG 16 bits", "JPEG (partage)"])
        v.addWidget(fmt)
        bar = QProgressBar()
        bar.setRange(0, 100)
        v.addWidget(bar)
        note = label("Maquette : l'export est simulé, aucun fichier n'est écrit.", "warn", wrap=True)
        v.addWidget(note)
        go = QPushButton("Exporter")
        go.setObjectName("primary")
        v.addWidget(go)
        t = QTimer(d)

        def tick():
            bar.setValue(bar.value() + 4)
            if bar.value() >= 100:
                t.stop()
                go.setText("Terminé (simulé)")

        t.timeout.connect(tick)
        go.clicked.connect(lambda: (bar.setValue(0), t.start(60)))
        d.resize(420, 220)
        d.exec()


class PostPage(QWidget):
    """The Post view, or what is missing for it when no workflow has been computed yet."""
    goWorkflow = Signal()

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.s = session
        self.pages = QStackedWidget()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.pages)
        self.empty = QWidget()
        ev = QVBoxLayout(self.empty)
        ev.setAlignment(Qt.AlignCenter)
        ev.addWidget(label("Pas encore de calque", "h1"), 0, Qt.AlignHCenter)
        self.why = para("")
        ev.addWidget(self.why, 0, Qt.AlignHCenter)
        b = QPushButton("Aller aux workflows  →")
        b.setObjectName("link")
        b.clicked.connect(self.goWorkflow.emit)
        ev.addWidget(b, 0, Qt.AlignHCenter)
        self.pages.addWidget(self.empty)
        self.view = PostView(session)
        self.pages.addWidget(self.view)
        session.pathsChanged.connect(self.update_page)
        self.update_page()

    def update_page(self):
        n = sum(p.rendered for p in self.s.paths)
        if not self.s.paths:
            self.why.setText("Chaque workflow calculé donne ici un calque, avec ses curseurs simplifiés. Ce projet "
                             "n'a pas encore de workflow.")
        else:
            who = ("Le workflow du projet n'a" if len(self.s.paths) == 1 else
                   f"Aucun des {len(self.s.paths)} workflows du projet n'a")
            self.why.setText(f"Chaque workflow calculé donne ici un calque. {who} encore tourné jusqu'au bout : "
                             "bouton Lancer en haut à droite, ou étape par étape avec « Lancer jusqu'ici ».")
        fit_height(self.why)
        self.pages.setCurrentIndex(1 if n else 0)

    def refresh(self):
        self.update_page()
        if self.pages.currentIndex() == 1:
            self.view.refresh()
