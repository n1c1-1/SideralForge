"""The template gallery (issue #43): how a workflow starts, chosen on a picture of its graph.

Tabs by kind of task (from frames, from an image, monochrome, a copy of a workflow); each template a card with a
thumbnail of its graph. A click shows the graph larger with what the template is for, its duration, its name, and
for an image the file; Create makes the workflow. The gallery opens after a new project is made and from "+ New
workflow".

A template (``GALLERY``) is a starting point only: the workflow it makes is an ordinary one, every block of which can
be changed, added, removed or cut later. Its graph is drawn here from the template itself (``draw_graph``): the blocks
in the graph's order, a block cut hatched in grey, one greyed by a cut outlined in amber, two sessions joined before
Registration, an image file in place of the frames.

The Filters tab (issue #43): a monochrome camera's filters, each its own workflow (its sessions: those of that filter),
then a Fusion block in the Composition section combines their outputs with a palette; a colour camera's image shot
through a dual-band filter, its red and its green-blue split into H-alpha and O III, then the same fusion. A template
says which of the project's sessions it uses and which filter is missing.

Texts are English and go through _() where shown.
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
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
from .widgets import ACCENT, label, size_within

FULL = dict(debruitage=True, etoiles="retirer", fond="simple")
CLASSIC_DN = dict(debruitage=True, etoiles="garder", fond="simple")
CLASSIC = dict(debruitage=False, etoiles="garder", fond="simple")

#: the tabs, in order: key and title
TABS = (("frames", "From frames"), ("mono", "Filters"), ("image", "From image"), ("mine", "My templates"),
        ("copy", "Workflow copy"))                    # compact names (decision 12): two words, no articles
#: a user's template file (Export / Import)
TEMPLATE_SUFFIX = ".sftemplate"

#: the templates: key, tab, name, what it is for, the workflow's options (the views'), the blocks it cuts (the
#: engine's names), and: end (its output leaves from Stacking), sessions (drawn with two nights calibrated each on its
#: own: the project calibrates its sessions separately), source ("image"), planned (not offered yet), recommended
GALLERY: list[dict] = [
    dict(key="blank", tab="frames", name="Blank", opts=CLASSIC, blank=True,
         text="An empty page: build the workflow yourself from the palette on the left of the graph. Start with a "
              "Session (a night of frames) or an Image; each block then goes where it can, and the blocks that cannot "
              "go yet say why."),
    dict(key="full", tab="frames", name="Full", opts=FULL, recommended=True,
         text="Calibration, registration and stacking, then astrometry, the sky gradient taken off, the colours from "
              "the Gaia stars, denoise on the two half-stacks and the stars taken off: the starless image and the stars "
              "come to Post as two layers. The usual choice for a deep-sky object."),
    dict(key="classic_dn", tab="frames", name="Classic + denoise", opts=CLASSIC_DN,
         text="The same without taking the stars off: one image to Post, denoised. For a star field, a cluster or a "
              "galaxy whose stars stay part of the picture."),
    dict(key="classic", tab="frames", name="Classic stack", opts=CLASSIC,
         text="Calibration, registration, stacking, astrometry, the sky gradient and the colours: the image as the "
              "stack gives it, without denoise. The quickest complete workflow."),
    dict(key="nights", tab="frames", name="Several nights", opts=FULL, sessions=2,
         text="Each night is a Session block with its own lights, darks and flats, calibrated on its own; all the "
              "nights are then registered and stacked together. Add the other nights with Add session…, then "
              "choose their calibration frames in their blocks."),
    dict(key="calibrated", tab="frames", name="Frames already calibrated", opts=FULL, drop=["calibration"],
         text="For frames another program has already calibrated (darks and flats applied): no Calibration block, "
              "the frames are taken as they are. Everything else is the Full workflow."),
    dict(key="aligned", tab="frames", name="Frames already aligned", opts=FULL, drop=["registration"],
         text="For frames already aligned on each other: no Registration block, each frame stays where it is; the "
              "stars are still measured to weigh and select the frames."),
    dict(key="raw", tab="frames", name="Raw stack", opts=CLASSIC, end="empilement",
         text="Calibration, registration and stacking only: the stack goes straight to Post, its sky gradient and its "
              "colours as they are. To check a night quickly, or to finish the image in another program."),
    dict(key="image", tab="image", name="Process a stacked image", opts=CLASSIC, source="image",
         text="Starts from a linear image (a FITS or TIFF file straight out of a stack, not stretched): astrometry, "
              "the sky gradient and the colours from the Gaia stars, then Post. Denoise needs the two half-stacks of a "
              "stack: it is not offered here."),
    dict(key="image_colours", tab="image", name="Stacked image, its colours kept", opts=CLASSIC, source="image",
         drop=["color"],
         text="The same without the Color block: the image keeps its own colours (it was colour calibrated already). "
              "Astrometry stays for the sky gradient's survey."),
    dict(key="image_plain", tab="image", name="Stacked image, gradient only", opts=CLASSIC, source="image",
         drop=["astrometry", "color"],
         text="Only the sky gradient is taken off, then Post: no astrometry, no colour calibration. For a field with "
              "too few stars to be solved, or without the network."),
    dict(key="image_stretched", tab="image", name="Already stretched image", opts=CLASSIC, source="image",
         stretched=True,
         text="An image already stretched (a TIFF, PNG or JPEG finished elsewhere, not linear) goes to Post as it is. "
              "Add Stars/background to take its stars off: the starless image and the stars come to Post as "
              "two layers."),
    dict(key="mono_each", tab="mono", name="One workflow per filter", opts=CLASSIC_DN, kind="filters",
         filters=("any",),
         text="A monochrome camera: each filter of the project's sessions gets its own complete workflow (its "
              "sessions calibrated, registered, stacked, astrometry, sky gradient, denoise); each output is a layer "
              "of Post. Colour calibration is not offered: the colours come from combining the filters."),
    dict(key="sho", tab="mono", name="SHO (Hubble palette)", opts=CLASSIC_DN, kind="fusion", palette="SHO",
         filters=("SII", "Ha", "OIII"),
         text="Narrowband: one workflow per filter, its stars taken off; a Fusion block combines the starless images: "
              "S II red, H-alpha green, O III blue (aligned through their plate solutions, brought to the same scale). "
              "The stars come back from the channels' stars, by the Combine block; the Fusion's panel offers other "
              "star sources (one channel's, a colour session's)."),
    dict(key="hso", tab="mono", name="HSO", opts=CLASSIC_DN, kind="fusion", palette="HSO",
         filters=("Ha", "SII", "OIII"),
         text="Narrowband with H-alpha red, S II green, O III blue: the red nebulae stay red."),
    dict(key="hoo", tab="mono", name="HOO", opts=CLASSIC_DN, kind="fusion", palette="HOO", filters=("Ha", "OIII"),
         text="Two filters: H-alpha red, O III green and blue."),
    dict(key="rgb", tab="mono", name="RGB with filters", opts=CLASSIC_DN, kind="fusion", palette="RGB",
         filters=("R", "G", "B"),
         text="A monochrome camera with red, green and blue filters: one workflow per filter, combined into a colour "
              "image."),
    dict(key="lrgb", tab="mono", name="LRGB", opts=CLASSIC_DN, kind="fusion", palette="LRGB",
         filters=("L", "R", "G", "B"),
         text="RGB with a luminance filter: the colour from R, G and B, the detail and the depth from L (each pixel's "
              "luminance is L's)."),
    dict(key="dualband", tab="mono", name="Dual band (colour camera)", opts=CLASSIC_DN, kind="dualband",
         palette="HOO",
         text="A colour camera behind a dual-band filter (H-alpha and O III): one workflow stacks it and takes its "
              "stars off, then a Fusion block takes its starless red as H-alpha and its green and blue as O III, "
              "combined as HOO; its own stars, in their colours, come back by the Combine block."),
]

#: the template chosen first in the frames' tab
RECOMMENDED = "full"
#: the engine's cut blocks → the views' block keys
_CUT_KEY = {"calibration": "calibration", "registration": "alignement", "astrometry": "astrometrie",
            "gradient": "fond", "color": "couleur"}
_SHORT = {"calibration": "Calib.", "alignement": "Regist.", "selection": "Select.", "empilement": "Stack",
          "astrometrie": "Astrom.", "fond": "Gradient", "couleur": "Color", "debruitage": "Denoise",
          "etoiles": "Stars/bg", "post": "Post", "image": "Image", "session": "Session", "combine": "Recombine"}


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
    if t.get("blank"):
        return []
    from .model import to_engine
    o = dict(t.get("opts") or {})
    end = KEY_ENGINE_BLOCK.get(t.get("end"), t.get("end")) if t.get("end") else None
    g = wfl.from_options(to_engine(o), end, template_source(t))
    drop = set(t.get("drop") or ())
    if mono or t.get("kind") in ("filters", "fusion"):
        drop.add("color")
    g = [n for n in g if n["block"] not in drop and not n["off"]]
    if t.get("kind") in ("fusion", "dualband") and not any(n["block"] == "stars" for n in g):
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
        if t.get("stretched") and "post" not in keys:
            keys.append("post")
        return [(k, "") for k in keys]
    cut = {_CUT_KEY[b] for b in t.get("cut") or ()}
    if t.get("kind") in ("filters", "fusion", "dualband"):     # each filter's own workflow (draw_filters)
        keys = ["calibration", "empilement", "astrometrie", "fond"]
        keys += ["couleur"] if t["kind"] == "dualband" else []
        keys += ["debruitage"] if o.get("debruitage") else []
        keys += ["etoiles"] if t["kind"] in ("fusion", "dualband") else []    # option A: the stars off in each
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


def draw_graph(t: dict, w: int, h: int, names: bool = False) -> QPixmap:
    """A picture of a template's graph: its sources (one or two nights with their Calibration, or an image file),
    then its blocks; a cut block hatched, a greyed one outlined in amber."""
    if t.get("kind") in ("filters", "fusion", "dualband"):
        return draw_filters(t, w, h, names)
    pm = QPixmap(w, h)
    pm.fill(QColor("#15171c"))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    if t.get("blank"):                                   # an empty page: a dashed place to start from
        r = QRectF(w * 0.3, h * 0.3, w * 0.4, h * 0.4)
        p.setPen(QPen(QColor("#4a5263"), 1.4, Qt.DashLine))
        p.drawRoundedRect(r, 6, 6)
        p.setPen(QColor("#8b93a1"))
        p.setFont(QFont("Segoe UI", max(8, int(h * 0.12))))
        p.drawText(r, Qt.AlignCenter, "+")
        p.end()
        return pm
    chain = chain_of(t)
    image = t.get("source") == "image"
    nights = 1 if image else int(t.get("sessions") or 1)
    mono = t.get("filters") or ()
    rows = len(mono) if mono else nights
    lead = 1 if image else (3 if mono else 2)     # the image; a night and its Calibration; (mono) and its stack
    n = lead + len(chain)
    gap = max(4.0, w * 0.012)

    def name_of(key):
        kb, _t, num = key.partition("~")                    # a block placed twice (issue #47): its number
        return key if key in mono else _(_SHORT.get(kb, TITLES.get(kb, kb))) + (f" {num}" if num else "")

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
        if t.get("planned"):
            stripe = "#5d6470"
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(stripe))
        p.drawRect(QRectF(x, y - bh / 2 + 2, 3, bh - 4))
        if names:
            p.setPen(QColor("#e8ebf0" if how != "cut" else "#8b93a1"))
            text = name_of(key)
            f = QFont(font)
            while f.pointSize() > 6 and QFontMetrics(f).horizontalAdvance(text) > bw - 7:
                f.setPointSize(f.pointSize() - 1)          # the name whole, in a smaller letter if need be
            p.setFont(f)
            p.drawText(r.adjusted(4, 0, -1, 0), Qt.AlignVCenter | Qt.AlignLeft,
                       QFontMetrics(f).elidedText(text, Qt.ElideRight, int(bw - 5)))
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
        r = block(x, mid, "image")
        p.setPen(QPen(QColor("#9aa3b2"), 1))                 # a file's corner
        p.drawLine(QPointF(r.right() - 7, r.top()), QPointF(r.right(), r.top() + 7))
        ends = [(r.right(), mid)]
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
    if t.get("planned"):
        p.setPen(QColor("#c9a227"))
        f = QFont("Segoe UI", max(7, int(h * 0.07)))
        f.setBold(True)
        p.setFont(f)
        p.drawText(QRectF(0, 0, w - 6, h * 0.22), Qt.AlignRight | Qt.AlignTop, _("planned"))
    p.end()
    return pm


def draw_filters(t: dict, w: int, h: int, names: bool = False) -> QPixmap:
    """A filters template's graph: a row per filter (its session, then its own workflow up to its Output), the rows
    joined into a Fusion block (a fusion's palette); a dual-band one: one colour row split into H-alpha and O III."""
    pm = QPixmap(w, h)
    pm.fill(QColor("#15171c"))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    chain = [k for k, _h in chain_of(t)]
    kind = t["kind"]
    rows = (["Colour"] if kind == "dualband" else (["Ha", "OIII"] if kind == "filters" else list(t["filters"])))
    fusion = kind in ("fusion", "dualband")
    n = 1 + len(chain) + (2 if fusion else 0)
    gap = max(4.0, w * (0.008 if names else 0.012))      # with the names: narrower gaps, wider blocks
    bw = (w - 2 * gap - (n - 1) * gap - (gap * 2 if fusion else 0)) / n
    # one row (dual band) whose names do not fit: the Fusion and the Recombine go on a second row (never cut names)
    below = names and fusion and len(rows) == 1 and \
        bw < max(_name_width(x) for x in [_(_SHORT.get(k, TITLES.get(k, k))) for k in chain] +
                 [_("Fusion") + " " + t.get("palette", ""), _("Recombine")])
    if below:
        n = 1 + len(chain)
        bw = (w - 2 * gap - (n - 1) * gap) / n
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
            f = QFont(font)
            while f.pointSize() > 6 and QFontMetrics(f).horizontalAdvance(text) > bw - 7:
                f.setPointSize(f.pointSize() - 1)
            p.setFont(f)
            p.setPen(QColor("#e8ebf0"))
            p.drawText(r.adjusted(4, 0, -1, 0), Qt.AlignVCenter | Qt.AlignLeft,
                       QFontMetrics(f).elidedText(text, Qt.ElideRight, int(bw - 5)))
        return r

    def wire(x1, y1, x2, y2):
        p.setPen(QPen(QColor("#6f7888"), 1.4))
        path = QPainterPath(QPointF(x1, y1))
        mx = (x1 + x2) / 2
        path.cubicTo(QPointF(mx, y1), QPointF(mx, y2), QPointF(x2, y2))
        p.drawPath(path)

    pitch = min(pitch_max, bh + max(4.0, h * 0.06))
    ys = [h / 2 + (i - (len(rows) - 1) / 2) * pitch for i in range(len(rows))] if not below else [h * 0.3]
    ends = []
    for row, y in zip(rows, ys):
        x = gap
        r = block(x, y, _(row) if row == "Colour" else row, "#7aa2d6")
        for k in chain:
            x = r.right() + gap
            wire(r.right(), y, x, y)
            r = block(x, y, _(_SHORT.get(k, TITLES.get(k, k))), "#5fbf7f")
        ends.append((r.right(), y))
    if fusion:
        yf = h * 0.72 if below else h / 2
        if below:                                       # the row's end turns down to the second row, on the left
            (ex, ey) = ends[0]
            p.setPen(QPen(QColor("#6f7888"), 1.4))
            p.setBrush(Qt.NoBrush)
            turn = QPainterPath(QPointF(ex, ey))
            for pt in ((ex + gap * 0.5, ey), (ex + gap * 0.5, (ey + yf) / 2), (gap * 0.5, (ey + yf) / 2),
                       (gap * 0.5, yf)):
                turn.lineTo(QPointF(*pt))
            p.drawPath(turn)
            ends = [(gap * 0.5, yf)]
        x = ends[0][0] + 3 * gap
        r = block(x, yf, _("Fusion") + " " + t.get("palette", ""), "#c9a227")
        r2 = block(r.right() + gap, yf, _("Recombine"), "#c9a227")     # the stars put back (option A)
        wire(r.right(), yf, r2.left(), yf)
        if kind == "dualband":                          # one colour image, two channels
            for _lab, dy in (("Ha", -bh * 0.25), ("OIII", bh * 0.25)):
                wire(ends[0][0], ends[0][1], r.left(), yf + dy)
        else:
            for ex, ey in ends:
                wire(ex, ey, r.left(), h / 2)
    p.end()
    return pm


class TemplateCard(QFrame):
    """A template in the gallery: its graph's thumbnail and its name."""
    clicked = Signal(object)
    chosen = Signal(object)

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self.t = t
        self.setObjectName("tcard")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedWidth(232)
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(4)
        pic = QLabel()
        pic.setPixmap(draw_graph(t, 214, 76))
        v.addWidget(pic)
        name = _(t["name"]) + ("  ·  " + _("Recommended") if t.get("recommended") else "")
        lb = QLabel(name)
        lb.setStyleSheet("font-weight: 600; background: transparent;" + (" color: #8b93a1;" if t.get("planned") else ""))
        v.addWidget(lb)
        self.set_selected(False)

    def set_selected(self, on: bool):
        self.setStyleSheet(f"QFrame#tcard {{ border: 1px solid {ACCENT if on else '#2e333d'}; border-radius: 8px; "
                           f"background: {'#232a36' if on else '#1c1f26'}; }} QLabel {{ background: transparent; }}")

    def mousePressEvent(self, e):                                     # noqa: N802 - Qt's name
        self.clicked.emit(self.t)

    def mouseDoubleClickEvent(self, e):                               # noqa: N802
        self.chosen.emit(self.t)


class TemplateGallery(QDialog):
    """New workflow: the template gallery. After ``exec()``: ``choice`` (the template, or {"copy": workflow} for a
    copy), ``name`` (a QLineEdit), ``image`` (the file of an image template)."""

    def __init__(self, session, start=None, parent=None, templates: list[dict] | None = None):
        super().__init__(parent)
        self.s = session
        self.templates = list(GALLERY if templates is None else templates)
        self.setWindowTitle(_("New workflow"))
        size_within(self, 1180, 700)                      # never taller than a 1366 x 768 screen's room
        self.choice: dict | None = None
        self.image = ""
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
        self.file_row = QWidget()
        fr = QHBoxLayout(self.file_row)
        fr.setContentsMargins(0, 0, 0, 0)
        fr.addWidget(label(_("Image"), "dim"))
        self.file = QLineEdit()
        self.file.setPlaceholderText(_("a FITS or TIFF file, straight out of a stack"))
        self.file.textChanged.connect(self._update)
        fr.addWidget(self.file, 1)
        b = QPushButton(_("Browse…"))
        b.clicked.connect(self._browse)
        fr.addWidget(b)
        sv.addWidget(self.file_row)
        # One workflow per filter on a project without filters yet (GUI exploration 7.2): the filters shot, chosen
        # here, each get an empty Session block and its workflow
        self.filter_row = QWidget()
        flr = QHBoxLayout(self.filter_row)
        flr.setContentsMargins(0, 0, 0, 0)
        flr.addWidget(label(_("Filters"), "dim"))
        self.filter_b: dict[str, QPushButton] = {}
        from ..core.project2 import FILTERS
        for f in [x for x in FILTERS if x]:
            fb = QPushButton(f)
            fb.setCheckable(True)
            fb.setObjectName("face")
            fb.setMinimumWidth(40)
            fb.toggled.connect(lambda _on: self._update())
            flr.addWidget(fb)
            self.filter_b[f] = fb
        flr.addStretch(1)
        sv.addWidget(self.filter_row)
        self.why = label("", "warn", wrap=True)             # why Create is greyed (GUI exploration 7.5)
        sv.addWidget(self.why)
        nr = QHBoxLayout()
        nr.addWidget(label(_("Name"), "dim"))
        self.name = QLineEdit()
        nr.addWidget(self.name, 1)
        sv.addLayout(nr)
        bb = QHBoxLayout()
        imp = QPushButton(_("Import…"))
        imp.setToolTip(_("A template saved as a file (Export… of a template of yours)"))
        imp.clicked.connect(self._import)
        bb.addWidget(imp)
        self.export_b = QPushButton(_("Export…"))
        self.export_b.setToolTip(_("Save this template of yours as a file, to share it"))
        self.export_b.clicked.connect(self._export)
        bb.addWidget(self.export_b)
        self.delete_b = QPushButton(_("Delete"))
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
        if key not in ("copy", "frames", "mine"):          # the blank page, in every tab of templates (issue #47)
            items = [t for t in self.templates if t.get("blank")] + items
        for n, t in enumerate(items):
            c = TemplateCard(t)
            c.clicked.connect(self.pick)
            c.chosen.connect(lambda t: (self.pick(t), self._create()))
            self.grid.addWidget(c, n // 2, n % 2)
            self.cards.append(c)
        first = next((c.t for c in self.cards if c.t.get("key") == RECOMMENDED),
                     next((c.t for c in self.cards if not c.t.get("blank")), self.cards[0].t if self.cards else None))
        if first is not None:
            self.pick(first)

    def pick(self, t: dict):
        self.choice = t
        for c in self.cards:
            c.set_selected(c.t is t)
        self.preview.setPixmap(draw_graph(t, PREVIEW_W, 190, names=True))
        self.title.setText(_(t["name"]) if not t.get("copy") else t["name"])
        self.text.setText(_(t["text"]) if not t.get("copy") else t["text"])
        self.file_row.setVisible(template_source(t) in ("image", "stretched") and not t.get("copy"))
        self.file.setPlaceholderText(_("a TIFF, PNG or JPEG file, already stretched") if template_source(t) ==
                                     "stretched" else _("a FITS or TIFF file, straight out of a stack"))
        self.filter_row.setVisible(self._asks_filters(t))
        mine = bool(t.get("user"))
        self.export_b.setVisible(mine)
        self.delete_b.setVisible(mine)
        dur = ""
        if t.get("planned"):
            dur = _("Not in this version yet: planned.")
        elif t.get("source") != "image" and not t.get("kind") and not t.get("blank") and not t.get("user") \
                and hasattr(self.s, "opts_seconds"):
            trunk = {k: v for k, v in (getattr(self.s.path, "params", {}) or {}).items() if k in TRUNK} \
                if getattr(self.s, "path", None) is not None else {}
            try:
                dur = _("About {t} on this machine").format(t=hm(self.s.opts_seconds(_view_opts(t), trunk)))
            except Exception:  # noqa: BLE001 - an estimate is a hint
                dur = ""
        self.duration.setText(dur)
        base = t["name"] if t.get("copy") else _(t["name"])
        if t.get("copy"):
            base = _("{name} (copy)").format(name=t["copy"].name)
        self.name.setText(self.s.unique_name(base) if hasattr(self.s, "unique_name") else base)
        self._update()

    def _browse(self):
        f, _x = QFileDialog.getOpenFileName(self, _("Image"), prefs.last_folder(), image_filter())
        if f:
            prefs.remember_folder(f)
            self.file.setText(f)

    def needs(self, t: dict) -> tuple[list[str], list[str]]:
        """(what the template uses of the project's sessions, what is missing) for a filters template."""
        kind = t.get("kind")
        have = self.s.filters() if hasattr(self.s, "filters") else {}
        if kind == "filters":                             # none yet: the filters chosen here (GUI exploration 7.2)
            return sorted(have), []
        if kind == "fusion":                              # a missing filter gets an empty Session block (#47)
            return ([f for f in t["filters"] if f in have], [])
        if kind == "dualband":
            colour = [s for s in getattr(getattr(self.s, "p", None), "sessions", []) if not s.filter]
            return ([_("colour sessions")] if colour else [], [])
        return [], []

    def to_create(self, t: dict) -> list[str]:
        """The Session blocks a template makes, empty, for the filters the project does not have yet."""
        have = self.s.filters() if hasattr(self.s, "filters") else {}
        if t.get("kind") == "fusion":
            return [f for f in t["filters"] if f not in have]
        if t.get("kind") == "filters" and not have:
            return self.chosen_filters()
        if t.get("kind") == "dualband":
            colour = [s for s in getattr(getattr(self.s, "p", None), "sessions", []) if not s.filter]
            return [] if colour else [_("colour")]
        return []

    def _asks_filters(self, t: dict) -> bool:
        """One workflow per filter, on a project whose sessions name no filter yet: its filters are chosen here."""
        return t.get("kind") == "filters" and not (self.s.filters() if hasattr(self.s, "filters") else {})

    def chosen_filters(self) -> list[str]:
        return [f for f, b in self.filter_b.items() if b.isChecked()]

    def _update(self):
        t = self.choice or {}
        ok = bool(t) and not t.get("planned")
        why = ""
        if t.get("kind"):
            used, missing = self.needs(t)
            new = self.to_create(t)
            ok = ok and not missing
            if self._asks_filters(t) and not new:
                ok, why = False, _("Choose the filters you shot: each gets its Session block and its workflow.")
            self.duration.setText((_("Uses: {what}").format(what=", ".join(used)) if used else "") +
                                  ("\n" + _("Missing: {what}. Add a session with that filter (its Session block), then "
                                             "come back.").format(what=", ".join(missing)) if missing else "") +
                                  ("\n" + _("Empty Session blocks are made for {what}: add their frames there.").format(
                                      what=", ".join(new)) if new else ""))
        if template_source(t) in ("image", "stretched") and not t.get("copy") and not t.get("blank"):
            from ..stages.image import image_problem
            f = self.file.text().strip()                 # a greyed Create says why (GUI exploration 7.5)
            why = _(image_problem(f)).format(name=Path(f).name)
            ok = ok and not why
        self.why.setText(why)
        self.why.setVisible(bool(why))
        self.ok.setEnabled(ok)
        self.ok.setToolTip(why)

    def _create(self):
        if not self.ok.isEnabled():
            return
        self.image = self.file.text().strip()
        t = self.choice or {}
        if self._asks_filters(t):                          # the filters chosen go with the choice (7.2)
            self.choice = {**t, "filters": tuple(self.chosen_filters())}
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
            out.append({**t, "user": True, "tab": t.get("tab") or "mine", "opts": CLASSIC})
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
    Path(path).write_text(json.dumps({"format": "sideralforge-template", "version": 1,
                                      **{k: t[k] for k in ("tab", "name", "text", "source", "blocks")}},
                                     indent=1, ensure_ascii=False), encoding="utf-8")


def import_template(path: str) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(d, dict) or d.get("format") != "sideralforge-template":
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
