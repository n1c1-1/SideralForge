"""The application's icons: one SVG per role in ``icons/``, the only way to get an icon.

The folder holds the 13 block icons (drawn for this program on a 32 px grid, stroke 2, butt caps, miter joins) and the
generic icons from IBM Carbon (Apache-2.0, ``icons/LICENSE-carbon.txt``; ``CARBON`` says which Carbon icon each role is,
Help > About lists them). A file has no colour of its own: ``currentColor`` is replaced in its text before QSvgRenderer
reads it (Qt renders SVG Tiny, where ``currentColor`` is not reliable), with the colour of a ``state`` taken from the
theme's constants. Two sizes only: 16 px (rows, small buttons), 20 px (bars, the palette, the strip of icons).

    set_icon(button, "undo", 16, tip=_("Undo (Ctrl+Z)"))
    IconLabel("warning", 16, "warning")
    draw(painter, QRectF(x, y, 16, 16), "remove", state="error")       # a view that paints itself

Renderings are cached per (role, size, colour, device pixel ratio). Colour means a state only: ``selected`` (the
accent), ``warning``, ``error``, ``ok``; the rest is ``normal``, ``dim`` or ``disabled``.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from .widgets import ACCENT, BAD, DISABLED, ICON, ICON_DIM, INK, OK, WARN

FOLDER = Path(__file__).parent / "icons"
#: the program's own icon: not a role, it has its colours
APP_ICON = Path(__file__).parent / "float.svg"
SIZES = (16, 20)
DEFAULT_SIZE = 16

#: the state of an icon, and the theme's colour it is drawn in
COLOURS = {"normal": ICON, "dim": ICON_DIM, "disabled": DISABLED, "selected": ACCENT, "error": BAD, "warning": WARN,
           "ok": OK, "on_accent": INK}

#: role -> the Carbon icon its file is (for the credits): the block icons are not there, they are the program's own
CARBON = {
    "add": "add", "remove": "close", "delete": "trash-can", "duplicate": "copy", "undo": "undo", "redo": "redo",
    "reset": "reset", "more": "overflow-menu--horizontal", "chevron_down": "chevron--down",
    "chevron_right": "chevron--right", "chevron_left": "chevron--left", "drag": "draggable", "back": "arrow--left",
    "forward": "arrow--right", "previous": "skip--back", "next": "skip--forward", "fold_left": "side-panel--close",
    "fold_right": "side-panel--open", "pin": "pin", "warning": "warning--alt", "not_computable": "error",
    "failed": "misuse--outline", "off": "power", "info": "information", "ok": "checkmark", "eye": "view",
    "eye_off": "view--off", "mask": "contrast", "fit": "fit-to-screen", "actual_size": "zoom--reset", "play": "play",
    "pause": "pause", "run": "play--filled--alt", "stop": "stop--filled--alt", "folder": "folder",
    "open": "folder--open", "folder_missing": "folder--off", "file": "document", "new": "document--add",
    "import": "document--import", "export": "document--export", "save": "save", "quit": "exit",
    "close_project": "close--outline", "home": "home", "settings": "settings", "adjust": "settings--adjust",
    "graph": "flow", "inspect": "image--search", "layers": "layers", "log": "terminal", "about": "help",
    "branch": "branch", "link": "link", "reference": "star--filled", "radio_on": "radio-button--checked",
    "radio_off": "radio-button", "missing": "circle-dash", "section": "list--boxes", "running": "in-progress",
    "waiting": "pending", "skipped": "subtract", "compute": "chip", "catalogue": "data--base", "display": "screen",
    "measure": "ruler", "samples": "grid", "thresholds": "filter", "palette": "color-palette",
    "deconvolve": "center--circle", "filter_layer": "function-math",
}
#: the block icons, drawn for the program: the palette's block -> role
BLOCK_ROLES = {"session": "block_session", "image": "block_image", "calibration": "block_calibration",
               "registration": "block_alignment", "selection": "block_selection", "stacking": "block_stacking",
               "astrometry": "block_astrometry", "gradient": "block_gradient", "denoise": "block_denoise",
               "color": "block_colour", "stars": "block_stars", "fusion": "block_fusion",
               "composition": "block_recombine", "deconv": "deconvolve", "align_images": "block_alignment"}
#: every role: one file each
ROLES = sorted(set(CARBON) | {r for r in BLOCK_ROLES.values() if r.startswith("block_")})

_text: dict[str, str] = {}                              # role -> its SVG text
_cache: dict[tuple, QPixmap] = {}                       # (role, size, colour, device pixel ratio) -> the rendering


def svg_text(role: str, colour: str) -> str:
    """The role's SVG, in the colour given (what QSvgRenderer reads)."""
    if role not in ROLES:
        raise KeyError(f"no icon role {role!r}")
    src = _text.get(role)
    if src is None:
        src = _text[role] = (FOLDER / f"{role}.svg").read_text(encoding="utf-8")
    return src.replace("currentColor", colour)


def _colour(state: str) -> str:
    try:
        return COLOURS[state]
    except KeyError:
        raise KeyError(f"no icon state {state!r}") from None


def _check_size(size: int) -> int:
    if size not in SIZES:
        raise ValueError(f"icon size {size}: only {SIZES}")
    return size


def _device_ratio(dpr: float | None) -> float:
    if dpr is not None:
        return float(dpr)
    screen = QGuiApplication.primaryScreen()
    return float(screen.devicePixelRatio()) if screen is not None else 1.0


def pixmap(role: str, size: int = DEFAULT_SIZE, state: str = "normal", dpr: float | None = None) -> QPixmap:
    """The icon drawn at ``size`` logical pixels, sharp at the device pixel ratio ``dpr`` (default: the main screen)."""
    _check_size(size)
    ratio = _device_ratio(dpr)
    colour = _colour(state)
    key = (role, size, colour, round(ratio, 3))
    hit = _cache.get(key)
    if hit is not None:
        return hit
    px = int(round(size * ratio))
    img = QImage(px, px, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    rend = QSvgRenderer(QByteArray(svg_text(role, colour).encode("utf-8")))
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    rend.render(p, QRectF(0, 0, px, px))
    p.end()
    pm = QPixmap.fromImage(img)
    pm.setDevicePixelRatio(ratio)
    _cache[key] = pm
    return pm


def app_icon() -> QIcon:
    """The program's icon (the window, the taskbar), drawn for each size the system may ask."""
    rend = QSvgRenderer(str(APP_ICON))
    ic = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        rend.render(p, QRectF(0, 0, size, size))
        p.end()
        ic.addPixmap(QPixmap.fromImage(img))
    return ic


def _ratios() -> set[float]:
    """The device pixel ratios a QIcon is made for: 1 and every screen's."""
    out = {1.0}
    for screen in QGuiApplication.screens():
        out.add(round(float(screen.devicePixelRatio()), 3))
    return out


def icon(role: str, size: int = DEFAULT_SIZE, state: str = "normal") -> QIcon:
    """A QIcon of the role: its state's colour; disabled, the disabled colour; on (a checked action), the accent."""
    ic = QIcon()
    for ratio in sorted(_ratios()):
        ic.addPixmap(pixmap(role, size, state, ratio), QIcon.Normal, QIcon.Off)
        ic.addPixmap(pixmap(role, size, "disabled", ratio), QIcon.Disabled, QIcon.Off)
        ic.addPixmap(pixmap(role, size, "selected" if state == "normal" else state, ratio), QIcon.Normal, QIcon.On)
    return ic


def set_icon(button, role: str, size: int = DEFAULT_SIZE, state: str = "normal", tip: str | None = None):
    """Puts the icon on a button or a tool button (``tip``: its full name, for an icon-only control)."""
    button.setIcon(icon(role, size, state))
    button.setIconSize(QSize(size, size))
    button.setProperty("iconRole", role)
    if tip is not None:
        button.setToolTip(tip)
    return button


def icon_only(button, role: str, tip: str, size: int = DEFAULT_SIZE, state: str = "normal", box: int | None = None):
    """An icon-only button: no text, its full name in the tooltip (``box``: a square of that side)."""
    button.setText("")
    set_icon(button, role, size, state, tip)
    if box is not None:
        button.setFixedSize(box, box)
        button.setStyleSheet("QPushButton { padding: 0; }")
    return button


def menu_button(button):
    """A button that opens a menu: its text, then a chevron on its right."""
    set_icon(button, "chevron_down", 16, "dim")
    button.setLayoutDirection(Qt.RightToLeft)
    if button.menu() is not None:
        button.menu().setLayoutDirection(Qt.LeftToRight)
    return button


def stamped(base: QPixmap, role: str, state: str = "error") -> QPixmap:
    """A copy of a thumbnail with the icon on its top right corner, on a dark square."""
    out = QPixmap(base)
    p = QPainter(out)
    r = out.deviceIndependentSize()
    box = QRectF(r.width() - 22, 2, 20, 20)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(15, 17, 21, 200))
    p.drawRoundedRect(box, 3, 3)
    draw(p, box.adjusted(2, 2, -2, -2), role, state)
    p.end()
    return out


def draw(p: QPainter, rect: QRectF, role: str, state: str = "normal", size: int = DEFAULT_SIZE):
    """The icon in a widget or a graphics item that paints itself, sharp at the painter's device ratio and scale."""
    t = p.worldTransform()
    scale = max(abs(t.m11()), abs(t.m12()), 1e-3)
    dev = p.device()
    ratio = (dev.devicePixelRatioF() if dev is not None else 1.0) * scale
    pm = pixmap(role, size, state, max(0.25, round(min(ratio, 4.0) * 4) / 4))  # a few ratios only, on a zoomed graph
    p.save()
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    p.drawPixmap(rect, pm, QRectF(0, 0, pm.width(), pm.height()))
    p.restore()


class IconLabel(QLabel):
    """The icon as a small label (a mark in a row); it draws itself at its own screen's ratio."""

    def __init__(self, role: str, size: int = DEFAULT_SIZE, state: str = "normal", tip: str = "", parent=None):
        super().__init__(parent)
        self.role, self.icon_size, self.state = role, size, state
        self.setFixedSize(size, size)
        self.setProperty("iconRole", role)
        if tip:
            self.setToolTip(tip)

    def set_role(self, role: str, state: str | None = None):
        self.role = role
        self.setProperty("iconRole", role)
        if state is not None:
            self.state = state
        self.update()

    def paintEvent(self, e):                             # noqa: N802 - Qt's name
        p = QPainter(self)
        p.drawPixmap(0, 0, pixmap(self.role, self.icon_size, self.state if self.isEnabled() else "disabled",
                                  self.devicePixelRatioF()))
        p.end()


class IconText(QWidget):
    """A line of text after its icon (a warning, a note): the text wraps, the icon stays at its first line."""

    def __init__(self, role: str, text: str = "", state: str = "normal", name: str | None = None, size: int = DEFAULT_SIZE,
                 parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(5)
        self.label = QLabel(text, self)                  # the first child: the text is what a search for labels finds
        self.label.setWordWrap(True)
        if name:
            self.label.setObjectName(name)
        self.icon = IconLabel(role, size, state)
        h.addWidget(self.icon, 0, Qt.AlignTop)
        h.addWidget(self.label, 1)

    def text(self) -> str:
        return self.label.text()

    def setText(self, text: str):                        # noqa: N802 - Qt's name
        self.label.setText(text)
