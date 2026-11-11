"""The template gallery (issue #43): how a workflow starts, chosen on a picture of its graph.

Six templates (colour or filters, minimal or full; one image; several images), and the user's own. On a blank page they
are shown in place of the graph, behind a link (``TemplatePage``): a click makes the workflow. A project that has
workflows adds one in a window (``TemplateGallery``): a click shows the graph larger with what the template is for, its
duration and its name; Create makes the workflow. A template asks for nothing (issue #114): it only places its blocks,
an Image block's file and a Session's frames are chosen in the graph afterwards.

A template (``GALLERY``) is a starting point only: the workflow it makes is an ordinary one, every block of which can
be changed, added, removed or cut later. Its graph is drawn here from the template itself (``draw_graph``): the blocks
in the graph's order, a block cut hatched in grey, one greyed by a cut outlined in amber, two sessions joined before
Registration, an image file in place of the frames.

The Filters templates (issue #43): a monochrome camera's filters, each its own workflow (its sessions: those of that
filter), and for the full one a Combine channels block in the Composition section that combines their outputs with the
palette chosen on the card. A template says which of the project's sessions it uses.

Texts are English and go through _() where shown.
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from ..core import workflow as wfl
from . import prefs
from .blocks import KEY_ENGINE_BLOCK, TITLES, TRUNK
from .i18n import _, hm
from .icons import set_icon
from .widgets import ACCENT, cap_dialog, label, size_within

FULL = dict(debruitage=True, etoiles="retirer", fond="simple")
CLASSIC_DN = dict(debruitage=True, etoiles="garder", fond="simple")
CLASSIC = dict(debruitage=False, etoiles="garder", fond="simple")

#: the tabs, in order: key and title
TABS = (("frames", "Templates"), ("mine", "My templates"), ("copy", "Workflow copy"))    # compact names (decision 12)
#: a user's template file (Export / Import)
TEMPLATE_SUFFIX = ".sftemplate"

#: the palettes the Filters templates offer, and their filters
PALETTES = {"SHO": ("SII", "Ha", "OIII"), "HOO": ("Ha", "OIII"), "RGB": ("R", "G", "B"), "LRGB": ("L", "R", "G", "B")}
#: the templates: key, tab, name, what it is for, the workflow's options (the views'), and: kind and palette (Filters),
#: source ("image"), plain (each image straight to Post), images (several files), recommended
GALLERY: list[dict] = [
    dict(key="colour_min", tab="frames", name="Colour minimal", opts=CLASSIC,
         text="Calibration, registration, selection and stacking, then astrometry, the sky gradient and the colours. "
              "The quickest complete workflow, without a neural network."),
    dict(key="colour_full", tab="frames", name="Colour full", opts=FULL, recommended=True,
         text="The same, then denoise on the two half-stacks and the stars taken off: the starless image and the stars "
              "come to Post as two layers."),
    dict(key="image", tab="frames", name="Image", opts=CLASSIC, source="image", plain=True, blocks=[],
         text="One image to Post as a layer, a FITS, TIFF, PNG or JPEG file: blocks are added from the palette."),
    dict(key="filters_min", tab="frames", name="Filters minimal", opts=CLASSIC, kind="filters", palette="HOO",
         filters=PALETTES["HOO"],
         text="A monochrome camera: each filter gets its own workflow, and each result is a layer of Post."),
    dict(key="filters_full", tab="frames", name="Filters full", opts=CLASSIC_DN, kind="fusion", palette="SHO",
         filters=PALETTES["SHO"],
         text="Each filter's workflow with denoise and its stars taken off, then a Combine channels block makes the "
              "colour image with the palette chosen here; the stars come back by a Combine layers block."),
    dict(key="images", tab="frames", name="Several images", opts=CLASSIC, source="image", plain=True, blocks=[], images=3,
         text="Several images to Post, one layer each: each Image block takes its file in the graph."),
]

#: the template chosen first in the frames' tab
RECOMMENDED = "colour_full"
#: the engine's cut blocks → the views' block keys
_CUT_KEY = {"calibration": "calibration", "registration": "alignement", "astrometry": "astrometrie",
            "gradient": "fond", "color": "couleur"}
#: a block's name in the pictures: the graph's own, whole (control round: "Calib.", "Astrom." were abbreviations); the
#: last block of a template is where its image goes, Post
_PICTURE_NAMES = {"post": "Post", "combine": "Combine layers"}


def picture_name(key: str) -> str:
    """The name a template's picture writes in a block: the graph's, never shortened."""
    return _(_PICTURE_NAMES.get(key) or TITLES.get(key, key))


def template_source(t: dict) -> str:
    """The source of a template's workflow: frames, an image, a stretched image."""
    if t.get("stretched"):                       # "Already stretched image" is an image too: its kind first (1.7)
        return "stretched"
    if t.get("source") in wfl.SOURCES:
        return t["source"]
    return "stretched" if t.get("stretched") else ("image" if t.get("source") == "image" else "frames")


def template_blocks(t: dict, mono: bool = False) -> list[dict]:
    """The graph a template makes (issue #47): only the blocks it uses (no block cut or greyed); a monochrome
    workflow has no Color block. A user's template keeps its own graph."""
    if t.get("blocks") is not None:
        return [dict(n) for n in t["blocks"]]
    from .model import to_engine
    o = dict(t.get("opts") or {})
    end = KEY_ENGINE_BLOCK.get(t.get("end"), t.get("end")) if t.get("end") else None
    g = wfl.from_options(to_engine(o), end, template_source(t))
    drop = set(t.get("drop") or ())
    if mono or t.get("kind") in ("filters", "fusion"):
        drop.add("color")
    g = [n for n in g if n["block"] not in drop and not n["off"]]
    if t.get("kind") == "fusion" and not any(n["block"] == "stars" for n in g):
        g.append(wfl.node("stars"))               # option A: each channel's stars taken off
    return g


def chain_of(t: dict) -> list[tuple[str, str]]:
    """The blocks a template's graph shows, in order: (views' key, "")."""
    o = t.get("opts") or {}
    if t.get("blocks") is not None or not t.get("kind"):
        from .model import node_key
        keys = [node_key(n) for n in template_blocks(t) if n["block"] != "calibration"]
        if "etoiles" in keys and "post" in keys:
            keys.remove("post")
            keys.insert(keys.index("etoiles") + 1, "post")
        if (t.get("stretched") or t.get("plain")) and "post" not in keys:
            keys.append("post")
        return [(k, "") for k in keys]
    cut = {_CUT_KEY[b] for b in t.get("cut") or ()}
    if t.get("kind") in ("filters", "fusion"):                 # each filter's own workflow (draw_filters)
        keys = ["calibration", "empilement", "astrometrie", "fond"]
        keys += ["debruitage"] if o.get("debruitage") else []
        keys += ["etoiles"] if t["kind"] == "fusion" else []       # option A: the stars off in each
        return [(k, "") for k in keys + ["post"]]
    if t.get("stretched"):
        return [("post", "")]
    if t.get("source") == "image":
        keys = ["astrometrie", "fond", "couleur"]
    else:
        keys = ["alignement", "selection", "empilement"]
        if t.get("end") != "empilement":
            keys += ["astrometrie", "fond", "couleur"]
            keys += ["debruitage"] if o.get("debruitage") else []
            keys += ["etoiles"] if o.get("etoiles") == "retirer" else []
    keys.append("post")
    out = []
    for k in keys:
        grey = k == "couleur" and "astrometrie" in cut and k not in cut
        out.append((k, "cut" if k in cut else ("grey" if grey else "")))
    return out


#: the width of the chosen template's graph, beside the cards (GUI exploration 2: 492 cut the blocks' names)
PREVIEW_W = 612
#: the smallest letter a block's name is written in, in a picture; a name wider than its block at this size makes the
#: chain go on, on a second row (the names are never cut, GUI exploration 2)
NAME_PT = 6


def _name_width(text: str, pt: int = NAME_PT) -> int:
    f = QFont("Segoe UI", pt)
    f.setBold(True)
    return QFontMetrics(f).horizontalAdvance(text) + 7


def _draw_name(p: QPainter, r: QRectF, text: str, font: QFont):
    """A block's name in its box, whole (control round: never shortened): in a smaller letter if need be, down to
    NAME_PT, then on two lines (at its "/" or anywhere)."""
    f = QFont(font)
    room = r.width() - 7
    while f.pointSize() > NAME_PT and QFontMetrics(f).horizontalAdvance(text) > room:
        f.setPointSize(f.pointSize() - 1)
    p.setFont(f)
    box = r.adjusted(4, 0, -1, 0)
    if QFontMetrics(f).horizontalAdvance(text) <= room:
        p.drawText(box, Qt.AlignVCenter | Qt.AlignLeft, text)
    else:
        p.drawText(box, Qt.AlignVCenter | Qt.AlignLeft | Qt.TextWrapAnywhere, text.replace("/", "/\u200b"))


def draw_graph(t: dict, w: int, h: int, names: bool = False) -> QPixmap:
    """A picture of a template's graph: its sources (a night with its Calibration, or image files), then its blocks; a
    cut block hatched, a greyed one outlined in amber."""
    if t.get("kind") in ("filters", "fusion"):
        return draw_filters(t, w, h, names)
    pm = QPixmap(w, h)
    pm.fill(QColor("#15171c"))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    chain = chain_of(t)
    image = t.get("source") == "image"
    nights = int(t.get("images") or 1)
    mono = t.get("filters") or ()
    rows = len(mono) if mono else nights
    lead = 1 if image else (3 if mono else 2)     # the image; a night and its Calibration; (mono) and its stack
    n = lead + len(chain)
    gap = max(4.0, w * 0.012)

    def name_of(key):
        kb, _t, num = key.partition("~")                    # a block placed twice (issue #47): its number
        return key if key in mono else picture_name(kb) + (f" {num}" if num else "")

    def width_of(n):
        return (w - 2 * gap - (n - 1) * gap - (gap * 2 if rows > 1 else 0)) / n
    bw = width_of(n)
    wrap = len(chain)                                   # the chain's blocks on the first row, the others on a second
    if names and len(chain) > 2 and bw < max(_name_width(name_of(k)) for k, _h in chain):
        wrap = max(1, (n + 1) // 2 - lead)
        bw = width_of(max(lead + wrap, len(chain) - wrap))
    hb = h if wrap == len(chain) else h / 2             # the height of a row of the picture
    bh = min(hb * 0.22 if rows > 1 else hb * 0.34, max(bw * 0.62, hb * (0.18 if rows > 2 else 0.24)))
    if hb < h and rows == 1:
        bh = min(hb * 0.6, max(bh, hb * 0.42))
    mid = hb / 2
    font = QFont("Segoe UI", min(10, max(6, int(bh * 0.28))))
    font.setBold(True)
    p.setFont(font)

    def block(x, y, key, how=""):
        r = QRectF(x, y - bh / 2, bw, bh)
        p.setPen(QPen(QColor("#3a4150"), 1))
        p.setBrush(QColor("#22262e"))
        p.drawRoundedRect(r, 4, 4)
        if how == "cut":
            p.save()
            path = QPainterPath()
            path.addRoundedRect(r, 4, 4)
            p.setClipPath(path)
            p.setPen(QPen(QColor("#3a4150"), 1))
            for i in range(-int(bh), int(bw) + 1, 6):
                p.drawLine(QPointF(x + i, y + bh / 2), QPointF(x + i + bh, y - bh / 2))
            p.restore()
            p.setPen(QPen(QColor("#9aa3b2"), 1.4))
            p.drawLine(QPointF(x, y), QPointF(x + bw, y))
        if how == "grey":
            p.setPen(QPen(QColor("#c9a227"), 1.2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r.adjusted(1, 1, -1, -1), 4, 4)
        stripe = {"cut": "#6b7280", "grey": "#c9a227"}.get(how, "#5fbf7f" if key != "session" else "#7aa2d6")
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(stripe))
        p.drawRect(QRectF(x, y - bh / 2 + 2, 3, bh - 4))
        if names:
            p.setPen(QColor("#e8ebf0" if how != "cut" else "#8b93a1"))
            _draw_name(p, r, name_of(key), font)
            p.setFont(font)
        return r

    def wire(x1, y1, x2, y2):
        p.setPen(QPen(QColor("#6f7888"), 1.4))
        path = QPainterPath(QPointF(x1, y1))
        if abs(y1 - y2) > 0.5:
            mx = (x1 + x2) / 2
            path.cubicTo(QPointF(mx, y1), QPointF(mx, y2), QPointF(x2, y2))
        else:
            path.lineTo(QPointF(x2, y2))
        p.drawPath(path)

    x = gap
    if image:
        ends = []
        for i in range(rows):
            y = mid + (i - (rows - 1) / 2) * (bh + max(4.0, h * 0.06))
            r = block(x, y, "image")
            p.setPen(QPen(QColor("#9aa3b2"), 1))             # a file's corner
            p.drawLine(QPointF(r.right() - 7, r.top()), QPointF(r.right(), r.top() + 7))
            ends.append((r.right(), y))
        x += bw + gap
    else:
        pitch = bh + max(4.0, h * 0.06)
        ys = [mid + (i - (rows - 1) / 2) * pitch for i in range(rows)]
        ends = []
        calib = t.get("kind") or any(n["block"] == "calibration" for n in template_blocks(t))
        for i, y in enumerate(ys):
            r = block(x, y, mono[i] if mono else "session")
            if not calib:                                 # no Calibration block (frames calibrated elsewhere)
                ends.append((r.right(), y))
                continue
            wire(x + bw, y, x + bw + gap, y)
            r = block(x + bw + gap, y, "calibration")
            if mono:                                      # each filter registered and stacked on its own
                wire(r.right(), y, r.right() + gap, y)
                r = block(r.right() + gap, y, "empilement")
            ends.append((r.right(), y))
        x += lead * (bw + gap)
        if rows > 1:                                          # the join: one bar before Registration
            xb = x + gap * 0.5
            for ex, ey in ends:
                wire(ex, ey, xb, ey)
            p.setPen(QPen(QColor("#9aa3b2"), 2.2))
            p.drawLine(QPointF(xb, min(ys)), QPointF(xb, max(ys)))
            ends = [(xb, mid)]
            x += gap * 2
    for i, (k, how) in enumerate(chain):
        if i == wrap:                                     # the chain goes on, on the second row, from the left
            (ex, ey), y2 = ends[0], h * 0.75
            p.setPen(QPen(QColor("#6f7888"), 1.4))
            p.setBrush(Qt.NoBrush)
            turn = QPainterPath(QPointF(ex, ey))
            for pt in ((ex + gap * 0.5, ey), (ex + gap * 0.5, (ey + y2) / 2), (gap * 0.5, (ey + y2) / 2),
                       (gap * 0.5, y2), (gap, y2)):
                turn.lineTo(QPointF(*pt))
            p.drawPath(turn)
            x, mid = gap, y2
            ends = []
        for ex, ey in ends:
            wire(ex, ey, x, mid)
        r = block(x, mid, k, how)
        ends = [(r.right(), mid)]
        x += bw + gap
    p.end()
    return pm


def draw_filters(t: dict, w: int, h: int, names: bool = False) -> QPixmap:
    """A filters template's graph: a row per filter (its session, then its own workflow up to its Output), the rows
    joined into a Combine channels block (a fusion's palette)."""
    pm = QPixmap(w, h)
    pm.fill(QColor("#15171c"))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    chain = [k for k, _h in chain_of(t)]
    rows = list(t["filters"])
    fusion = t["kind"] == "fusion"
    n = 1 + len(chain) + (2 if fusion else 0)
    gap = max(4.0, w * (0.008 if names else 0.012))      # with the names: narrower gaps, wider blocks
    bw = (w - 2 * gap - (n - 1) * gap - (gap * 2 if fusion else 0)) / n
    pitch_max = (h - 8) / max(len(rows), 1)
    bh = min(h * 0.34, pitch_max * 0.78, max(bw * 0.62, h * 0.12))
    font = QFont("Segoe UI", min(10, max(6, int(bh * 0.3))))
    font.setBold(True)

    def block(x, y, text, colour):
        r = QRectF(x, y - bh / 2, bw, bh)
        p.setPen(QPen(QColor("#3a4150"), 1))
        p.setBrush(QColor("#22262e"))
        p.drawRoundedRect(r, 4, 4)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(colour))
        p.drawRect(QRectF(x, y - bh / 2 + 2, 3, bh - 4))
        if names:
            p.setPen(QColor("#e8ebf0"))
            _draw_name(p, r, text, font)
        return r

    def wire(x1, y1, x2, y2):
        p.setPen(QPen(QColor("#6f7888"), 1.4))
        path = QPainterPath(QPointF(x1, y1))
        mx = (x1 + x2) / 2
        path.cubicTo(QPointF(mx, y1), QPointF(mx, y2), QPointF(x2, y2))
        p.drawPath(path)

    pitch = min(pitch_max, bh + max(4.0, h * 0.06))
    ys = [h / 2 + (i - (len(rows) - 1) / 2) * pitch for i in range(len(rows))]
    ends = []
    for row, y in zip(rows, ys):
        x = gap
        r = block(x, y, row, "#7aa2d6")
        for k in chain:
            x = r.right() + gap
            wire(r.right(), y, x, y)
            r = block(x, y, picture_name(k), "#5fbf7f")
        ends.append((r.right(), y))
    if fusion:
        yf = h / 2
        x = ends[0][0] + 3 * gap
        r = block(x, yf, _("Combine channels") + " " + t.get("palette", ""), "#c9a227")
        r2 = block(r.right() + gap, yf, _("Combine layers"), "#c9a227")     # the stars put back (option A)
        wire(r.right(), yf, r2.left(), yf)
        for ex, ey in ends:
            wire(ex, ey, r.left(), h / 2)
    p.end()
    return pm


#: a template card's width (its picture 214 px, and its margins)
CARD_W = 232


def with_palette(t: dict, palette: str) -> dict:
    """A Filters template with this palette, and its filters."""
    return {**t, "palette": palette, "filters": PALETTES[palette]}


class TemplateCard(QFrame):
    """A template in the gallery: its graph's thumbnail and its name."""
    clicked = Signal(object)
    chosen = Signal(object)

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self.t = t
        self.setObjectName("tcard")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedWidth(CARD_W)
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(4)
        self.pic = QLabel()
        self.pic.setPixmap(draw_graph(t, 214, 76))
        v.addWidget(self.pic)
        name = _(t["name"]) + ("  ·  " + _("Recommended") if t.get("recommended") else "")
        lb = QLabel(name)
        lb.setStyleSheet("font-weight: 600; background: transparent;")
        v.addWidget(lb)
        self.set_selected(False)

    def set_selected(self, on: bool):
        self.setStyleSheet(f"QFrame#tcard {{ border: 1px solid {ACCENT if on else '#2e333d'}; border-radius: 8px; "
                           f"background: {'#232a36' if on else '#1c1f26'}; }} QLabel {{ background: transparent; }}")

    def set_palette(self, palette: str):
        self.t = with_palette(self.t, palette)
        self.pic.setPixmap(draw_graph(self.t, 214, 76))

    def mousePressEvent(self, e):                                     # noqa: N802 - Qt's name
        self.clicked.emit(self.t)

    def mouseDoubleClickEvent(self, e):                               # noqa: N802
        self.chosen.emit(self.t)


def palette_box(current: str) -> QComboBox:
    """The palette of a Filters template."""
    cb = QComboBox()
    cb.addItems(list(PALETTES))
    cb.setCurrentText(current)
    cb.setToolTip(_("The filters shot: each gets its Session block and its workflow"))
    return cb


def ask_files(parent, t: dict) -> list[str]:
    """The files Open images… asks for (several: as many as chosen); [] when cancelled. Never a template's: a template
    asks for nothing (issue #114)."""
    if t.get("images"):
        files, _x = QFileDialog.getOpenFileNames(parent, _("Images"), prefs.last_folder(), image_filter())
    else:
        f, _x = QFileDialog.getOpenFileName(parent, _("Image"), prefs.last_folder(), image_filter())
        files = [f] if f else []
    if files:
        prefs.remember_folder(files[0])
    return files


class TemplatePage(QWidget):
    """The templates in place of the blank graph: a click on a card makes its workflow, asking for nothing (issue #114:
    its Image blocks wait for their file, chosen in their panel); ``back``: the blank page again."""
    chosen = Signal(object)
    back = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cards: list[TemplateCard] = []
        out = QVBoxLayout(self)
        out.setContentsMargins(24, 18, 24, 18)
        out.setSpacing(12)
        top = QHBoxLayout()
        top.addWidget(label(_("Start from a template"), "h1"), 1)
        self.back_b = QPushButton(_("Blank page"))
        self.back_b.setToolTip(_("Build the workflow from the palette on the left"))
        self.back_b.clicked.connect(self.back.emit)
        top.addWidget(self.back_b)
        out.addLayout(top)
        self.scroll = sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.NoFrame)
        host = QWidget()
        self.grid = QGridLayout(host)
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.grid.setSpacing(12)
        sc.setWidget(host)
        out.addWidget(sc, 1)
        self._cols = 0
        sc.viewport().installEventFilter(self)          # its width decides the columns (a scroll bar coming takes some)

    def eventFilter(self, obj, e):                                    # noqa: N802 - Qt's name
        if e.type() == QEvent.Resize and obj is self.scroll.viewport():
            self._reflow()
        return super().eventFilter(obj, e)

    def _reflow(self):
        """The cards in as many columns as the page's width holds, row after row, as a picture browser's grid (issue
        #133); each row as tall as its tallest card, a tall page's room left empty below them."""
        sp = self.grid.spacing()
        room = self.scroll.viewport().width() - self.grid.contentsMargins().left() - self.grid.contentsMargins().right()
        cols = max(1, min(len(self.cards), (room + sp) // (CARD_W + sp))) if self.cards else 1
        if cols == self._cols:                          # build() sets it to 0: its new cards are placed
            return
        self._cols = cols
        for c in self.cards:
            self.grid.removeWidget(c)
        for n, c in enumerate(self.cards):
            row = self.cards[n - n % cols:n - n % cols + cols]
            c.setFixedHeight(max(x.own_h for x in row))
            self.grid.addWidget(c, n // cols, n % cols)

    def build(self):
        """The cards, the user's templates after the six."""
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self.cards = []
        for t in GALLERY + user_templates():
            c = TemplateCard(t)
            what = label(_(t["text"]), "small", wrap=True)
            c.layout().addWidget(what)
            if t.get("kind"):
                box = palette_box(t["palette"])
                box.currentTextChanged.connect(c.set_palette)
                c.layout().addWidget(box)
            # issue #133: its stretch made the cards, and the grid, take the page's whole height. A card is as tall as
            # the tallest of its row (_reflow), never taller: the room left goes below its content, which keeps its
            # top. Its own height is that of its text wrapped in its width (a label's size hint guesses a wider one)
            c.layout().addStretch(1)
            c.ensurePolished()
            c.own_h = c.heightForWidth(CARD_W)
            c.clicked.connect(self.chosen.emit)
            self.cards.append(c)
        self._cols = 0
        self._reflow()


class TemplateGallery(QDialog):
    """New workflow: the template gallery. After ``exec()``: ``choice`` (the template, or {"copy": workflow} for a
    copy), ``name`` (a QLineEdit). No file is asked here (issue #114): an image template's Image blocks wait for theirs."""

    def __init__(self, session, start=None, parent=None, templates: list[dict] | None = None):
        super().__init__(parent)
        self.s = session
        self.templates = list(GALLERY if templates is None else templates)
        self.setWindowTitle(_("New workflow"))
        # never taller than a 1366 x 768 screen's room: 768 less the task bar and this window's title bar (control
        # round: 700 was taller than that)
        size_within(self, 1180, 660)
        cap_dialog(self)
        self.choice: dict | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)
        root.addWidget(label(_("New workflow"), "h1"))
        root.addWidget(label(_("Choose how the workflow starts: every block can then be changed, added from the "
                               "palette, removed or switched off."), "small", wrap=True))
        self.tabs = QTabBar()
        self.tabs.setStyleSheet(f"QTabBar::tab {{ color: #c9cfd8; background: #232831; border: 1px solid #353b47; "
                                f"padding: 6px 14px; margin-right: 2px; border-top-left-radius: 6px; "
                                f"border-top-right-radius: 6px; }} QTabBar::tab:selected {{ color: #ffffff; "
                                f"background: #2b3446; border-color: {ACCENT}; font-weight: 600; }} "
                                f"QTabBar::tab:hover {{ color: #ffffff; }}")
        self.tab_keys = []
        self.templates += user_templates()
        for key, title in TABS:
            if key == "copy" and not getattr(session, "paths", None):
                continue
            if key == "mine" and not any(t["tab"] == "mine" for t in self.templates):
                continue
            self.tabs.addTab(_(title))
            self.tab_keys.append(key)
        self.tabs.currentChanged.connect(self._tab)
        root.addWidget(self.tabs)
        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, 1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.NoFrame)
        host = QWidget()
        self.grid = QGridLayout(host)
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.grid.setSpacing(10)
        sc.setWidget(host)
        body.addWidget(sc, 1)
        side = QFrame()
        side.setObjectName("side")
        side.setStyleSheet("QFrame#side { background: #1c1f26; border-radius: 8px; } QLabel { background: transparent; }")
        side.setFixedWidth(PREVIEW_W + 28)              # the room left by two columns of cards: a wide preview
        sv = QVBoxLayout(side)
        sv.setContentsMargins(14, 14, 14, 14)
        sv.setSpacing(8)
        self.preview = QLabel()
        sv.addWidget(self.preview)
        self.title = label("", "h2")
        sv.addWidget(self.title)
        self.text = label("", None, wrap=True)
        sv.addWidget(self.text)
        self.duration = label("", "small", wrap=True)
        sv.addWidget(self.duration)
        sv.addStretch(1)
        # the filters shot, by their palette: each gets its Session block and its workflow (GUI exploration 7.2)
        self.palette_row = QWidget()
        flr = QHBoxLayout(self.palette_row)
        flr.setContentsMargins(0, 0, 0, 0)
        flr.addWidget(label(_("Palette"), "dim"))
        self.palette = palette_box("SHO")
        self.palette.currentTextChanged.connect(self._palette)
        flr.addWidget(self.palette)
        flr.addStretch(1)
        sv.addWidget(self.palette_row)
        nr = QHBoxLayout()
        nr.addWidget(label(_("Name"), "dim"))
        self.name = QLineEdit()
        nr.addWidget(self.name, 1)
        sv.addLayout(nr)
        bb = QHBoxLayout()
        imp = set_icon(QPushButton(_("Import…")), "import", 16)
        imp.setToolTip(_("A template saved as a file (Export… of a template of yours)"))
        imp.clicked.connect(self._import)
        bb.addWidget(imp)
        self.export_b = set_icon(QPushButton(_("Export…")), "export", 16)
        self.export_b.setToolTip(_("Save this template of yours as a file, to share it"))
        self.export_b.clicked.connect(self._export)
        bb.addWidget(self.export_b)
        self.delete_b = set_icon(QPushButton(_("Delete")), "delete", 16)
        self.delete_b.clicked.connect(self._delete)
        bb.addWidget(self.delete_b)
        bb.addStretch(1)
        cancel = QPushButton(_("Cancel"))
        cancel.clicked.connect(self.reject)
        self.ok = QPushButton(_("Create"))
        self.ok.setObjectName("primary")
        self.ok.setDefault(True)
        self.ok.clicked.connect(self._create)
        bb.addWidget(cancel)
        bb.addWidget(self.ok)
        sv.addLayout(bb)
        body.addWidget(side)
        self.cards: list[TemplateCard] = []
        if start is not None and "copy" in self.tab_keys:
            self.tabs.setCurrentIndex(self.tab_keys.index("copy"))
            self._tab(self.tab_keys.index("copy"))
            self.pick(next(c.t for c in self.cards if c.t.get("copy") is start))
        else:
            self._tab(self.tabs.currentIndex())

    # ------------------------------------------------------------ the cards
    def _copies(self) -> list[dict]:
        return [dict(key=f"copy:{p.pid}", tab="copy", name=_("{name} (copy)").format(name=p.name), copy=p,
                     opts=p.opts, cut=p.opts.get("cut") or [], end=p.end, source="image" if p.source else None,
                     text=_("The workflow “{name}” with all its settings: to try one change against it.").format(
                         name=p.name)) for p in self.s.paths]

    def _tab(self, i: int):
        key = self.tab_keys[i] if 0 <= i < len(self.tab_keys) else "frames"
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self.cards = []
        items = self._copies() if key == "copy" else [t for t in self.templates if t["tab"] == key]
        for n, t in enumerate(items):
            c = TemplateCard(t)
            c.clicked.connect(self.pick)
            c.chosen.connect(lambda t: (self.pick(t), self._create()))
            self.grid.addWidget(c, n // 2, n % 2)
            self.cards.append(c)
        first = next((c.t for c in self.cards if c.t.get("key") == RECOMMENDED), self.cards[0].t if self.cards else None)
        if first is not None:
            self.pick(first)

    def pick(self, t: dict):
        self.choice = t
        for c in self.cards:
            c.set_selected(c.t is t)
        self.preview.setPixmap(draw_graph(t, PREVIEW_W, 190, names=True))
        self.title.setText(_(t["name"]) if not t.get("copy") else t["name"])
        self.text.setText(_(t["text"]) if not t.get("copy") else t["text"])
        self.palette_row.setVisible(self._asks_filters(t))
        self.palette.blockSignals(True)
        self.palette.setCurrentText(t.get("palette", "SHO"))
        self.palette.blockSignals(False)
        mine = bool(t.get("user"))
        self.export_b.setVisible(mine)
        self.delete_b.setVisible(mine)
        dur = ""
        if t.get("source") != "image" and not t.get("kind") and not t.get("user") and hasattr(self.s, "opts_seconds"):
            trunk = {k: v for k, v in (getattr(self.s.path, "params", {}) or {}).items() if k in TRUNK} \
                if getattr(self.s, "path", None) is not None else {}
            try:
                dur = _("About {t} on this machine").format(t=hm(self.s.opts_seconds(_view_opts(t), trunk)))
            except Exception:  # noqa: BLE001 - an estimate is a hint
                dur = ""
        self.duration.setText(dur)
        base = t["name"] if t.get("copy") else t["palette"] if t.get("kind") == "fusion" else _(t["name"])
        if t.get("copy"):
            base = _("{name} (copy)").format(name=t["copy"].name)
        self.name.setText(self.s.unique_name(base) if hasattr(self.s, "unique_name") else base)
        self._update()

    def _palette(self, palette: str):
        """Another palette: its filters are the template's."""
        t = with_palette(self.choice or {}, palette)
        self.choice = t
        self.preview.setPixmap(draw_graph(t, PREVIEW_W, 190, names=True))
        if t.get("kind") == "fusion":
            self.name.setText(self.s.unique_name(palette) if hasattr(self.s, "unique_name") else palette)
        self._update()

    def needs(self, t: dict) -> list[str]:
        """What a filters template uses of the project's sessions."""
        kind = t.get("kind")
        have = self.s.filters() if hasattr(self.s, "filters") else {}
        if kind == "filters":                             # none yet: those of the palette (GUI exploration 7.2)
            return sorted(have)
        return [f for f in t["filters"] if f in have] if kind == "fusion" else []    # a missing filter: an empty Session

    def to_create(self, t: dict) -> list[str]:
        """The Session blocks a template makes, empty, for the filters the project does not have yet."""
        have = self.s.filters() if hasattr(self.s, "filters") else {}
        if t.get("kind") == "fusion" or (t.get("kind") == "filters" and not have):
            return [f for f in t["filters"] if f not in have]
        return []

    def _asks_filters(self, t: dict) -> bool:
        """A palette is chosen for a Combine channels block, or for filter workflows of a project without filters yet."""
        return t.get("kind") == "fusion" or (t.get("kind") == "filters" and
                                             not (self.s.filters() if hasattr(self.s, "filters") else {}))

    def _update(self):
        t = self.choice or {}
        if t.get("kind"):
            used = self.needs(t)
            new = self.to_create(t)
            self.duration.setText((_("Uses: {what}").format(what=", ".join(used)) if used else "") +
                                  ("\n" + _("Empty Session blocks are made for {what}: add their frames there.").format(
                                      what=", ".join(new)) if new else ""))
        self.ok.setEnabled(bool(t))                        # nothing to fill first (issue #114): a template is placed

    def _create(self):
        if not self.ok.isEnabled():
            return
        self.accept()

    # ------------------------------------------------------------ the user's templates (issue #47)
    def _reload(self):
        self.templates = [t for t in self.templates if not t.get("user")] + user_templates()

    def _import(self):
        f, _x = QFileDialog.getOpenFileName(self, _("Import a template"), prefs.last_folder(),
                                            _("Templates (*{suffix})").format(suffix=TEMPLATE_SUFFIX))
        if not f:
            return
        prefs.remember_folder(f)
        try:
            t = import_template(f)
        except (OSError, ValueError):                      # said once, plainly (GUI exploration 7.5)
            self.duration.setText(_("“{name}” is not a template file of this program.").format(name=Path(f).name))
            return
        self._reload()
        self._show_tab("mine")
        self.pick(next(c.t for c in self.cards if c.t.get("key") == t["key"]))

    def _export(self):
        t = self.choice or {}
        if not t.get("user"):
            return
        f, _x = QFileDialog.getSaveFileName(self, _("Export the template"),
                                            str(Path(prefs.last_folder()) / (t["name"] + TEMPLATE_SUFFIX)),
                                            _("Templates (*{suffix})").format(suffix=TEMPLATE_SUFFIX))
        if f:
            prefs.remember_folder(f)
            export_template(t, f)

    def _delete(self):
        t = self.choice or {}
        from PySide6.QtWidgets import QMessageBox
        if t.get("user") and QMessageBox.question(
                self, _("Delete"), _("Delete the template “{name}”? It cannot be undone.").format(name=t["name"])) \
                == QMessageBox.Yes:
            delete_user_template(t["key"])
            self._reload()
            self._show_tab(t["tab"] if any(x["tab"] == t["tab"] for x in self.templates) else "frames")

    def _show_tab(self, key: str):
        if key not in self.tab_keys:                      # a tab that had no template yet
            i = [k for k, _t in TABS if k in self.tab_keys or k == key].index(key)
            self.tabs.insertTab(i, _(dict(TABS)[key]))
            self.tab_keys.insert(i, key)
        self.tabs.setCurrentIndex(self.tab_keys.index(key))
        self._tab(self.tab_keys.index(key))

    # ------------------------------------------------------------ what the page reads
    @property
    def opts(self) -> dict:
        return _view_opts(self.choice or {})


def image_filter() -> str:
    """The file dialog's filter of an Image block (GUI exploration 7.3): every format it reads, then any file."""
    return (_("Images (*.fits *.fit *.fts *.tif *.tiff *.png *.jpg *.jpeg)") + ";;" + _("All files (*)"))


def template_dialog(parent, name: str) -> tuple[str, str] | None:
    """Save as template: its name and its gallery tab, in one dialog (issue #47). None when cancelled."""
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QFormLayout
    d = QDialog(parent)
    d.setWindowTitle(_("Save template"))
    f = QFormLayout(d)
    ed = QLineEdit(name)
    ed.selectAll()
    f.addRow(_("Name"), ed)
    cb = QComboBox()
    for k, t in TABS:
        if k != "copy":
            cb.addItem(_(t), k)
    cb.setCurrentIndex(cb.findData("mine"))
    f.addRow(_("Gallery tab"), cb)
    bb = QDialogButtonBox()
    bb.addButton(_("Save"), QDialogButtonBox.AcceptRole)
    bb.addButton(_("Cancel"), QDialogButtonBox.RejectRole)
    bb.accepted.connect(d.accept)
    bb.rejected.connect(d.reject)
    f.addRow(bb)
    if d.exec() != QDialog.Accepted or not ed.text().strip():
        return None
    return ed.text().strip(), cb.currentData()


def user_templates() -> list[dict]:
    """The user's own templates (issue #47), from the preferences."""
    out = []
    for t in prefs.get("templates") or []:
        if isinstance(t, dict) and t.get("key") and isinstance(t.get("blocks"), list):
            out.append({**t, "user": True, "tab": t["tab"] if t.get("tab") in dict(TABS) else "mine", "opts": CLASSIC})
    return out


def make_template(name: str, tab: str, text: str, source: str, blocks: list[dict]) -> dict:
    """A user's template of a workflow's graph (checked)."""
    blocks = wfl.check_blocks(wfl.with_output(blocks, source), source)       # its Output where it belongs (#49)
    key = "user:" + "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
    return {"key": key, "tab": tab if tab in dict(TABS) and tab != "copy" else "mine", "name": name.strip(),
            "text": text.strip(), "source": source, "blocks": blocks}


def save_user_template(t: dict) -> dict:
    """Keep a template in the preferences (one of the same name is replaced)."""
    keep = [x for x in prefs.get("templates") or [] if isinstance(x, dict) and x.get("key") != t["key"]]
    prefs.set("templates", keep + [{k: t[k] for k in ("key", "tab", "name", "text", "source", "blocks")}])
    return t


def delete_user_template(key: str) -> None:
    prefs.set("templates", [x for x in prefs.get("templates") or [] if isinstance(x, dict) and x.get("key") != key])


def export_template(t: dict, path: str) -> None:
    Path(path).write_text(json.dumps({"format": "float-template", "version": 1,
                                      **{k: t[k] for k in ("tab", "name", "text", "source", "blocks")}},
                                     indent=1, ensure_ascii=False), encoding="utf-8")


def import_template(path: str) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(d, dict) or d.get("format") != "float-template":
        raise ValueError(_("not a template file"))
    return save_user_template(make_template(str(d.get("name") or Path(path).stem), str(d.get("tab") or "mine"),
                                            str(d.get("text") or ""), str(d.get("source") or "frames"),
                                            list(d.get("blocks") or [])))


def _view_opts(t: dict) -> dict:
    """The views' options of a template's workflow, its cut blocks with them."""
    o = dict(t.get("opts") or {})
    if t.get("cut"):
        o["cut"] = list(t["cut"])
    return o
