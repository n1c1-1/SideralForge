"""The window on small laptop screens (issue #44): the size ranges, the compact look and the folding right panel.

The window is measured in logical pixels, never the screen: Qt already applies the system's 125 % or 150 %, and a
window that does not fill the screen has less room than the screen. Three ranges, with a dead band so that dragging the
window's edge does not flicker:

    large    width >= 1600 and height >= 880     exactly today's layout
    medium   width 1280-1599 or height 740-879   compact look
    small    width < 1280 or height < 740        also: the right panel as icons, the blocks' palette folded

The automatic layout only applies until the user acts: a panel pinned, folded, resized or toggled is his choice, kept
per size range (the setting ``layout``); View > Reset layout forgets them. The setting ``layout_density`` (Settings)
is "auto", "compact" (always compact) or "comfortable" (never: today's layout on any screen).

``Dock`` is a page's right panel: docked (today's), folded to a strip of icons (a click opens it over the main area, Esc
or a click elsewhere closes it, the pin docks it again) or hidden (a small arrow at the window's edge brings it back).
``wheel_gesture`` tells a mouse wheel (zoom) from two fingers on a touchpad (pan) for the graph and the viewers.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QFrame, QToolButton, QVBoxLayout, QWidget

from . import prefs
from .i18n import _

LARGE, MEDIUM, SMALL = "large", "medium", "small"
LARGE_W, LARGE_H = 1600, 880
SMALL_W, SMALL_H = 1280, 740
DEAD_BAND = 20                  # a range is left this far past its threshold: the window's edge dragged does not flicker
MIN_WINDOW = (1024, 640)        # the smallest window that still shows every page (Windows' own design floor is 1024 x 640)
STRIP_W = 40                    # the panel folded to icons
ARROW_W, ARROW_H = 12, 46       # the arrow left by a hidden side
FLOAT_W = 400                   # a panel opened over the main area, when it has no width of its own

#: the compact look: margins, spacing and buttons tighter, the text unchanged and never below 9 pt (set on the window:
#: a large screen has no style sheet here at all)
COMPACT_QSS = """
QPushButton { padding: 2px 8px; }
QPushButton#seg { padding: 4px 14px; font-size: 10pt; }
QPushButton#face { padding: 2px 10px; }
QTabBar::tab { padding: 3px 9px; }
QTabWidget#faces > QTabBar::tab { padding: 3px 18px; font-size: 10pt; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { padding: 1px 4px; }
QTreeWidget::item { padding: 2px 2px; }
"""


OVERLAY_QSS = "QWidget#panel { background: #1c1f26; border-left: 1px solid #353b47; }"


def size_class(w: int, h: int, was: str | None = None) -> str:
    """The size range of a window ``w`` x ``h``; ``was``: the range it is in now, which it keeps inside the dead band."""
    margin_large = DEAD_BAND if was == LARGE else -DEAD_BAND if was is not None else 0
    margin_small = DEAD_BAND if was == SMALL else -DEAD_BAND if was is not None else 0
    if w >= LARGE_W - margin_large and h >= LARGE_H - margin_large:
        return LARGE
    if w < SMALL_W + margin_small or h < SMALL_H + margin_small:
        return SMALL
    return MEDIUM


def density() -> str:
    """The setting: "auto", "compact" (always) or "comfortable" (never: today's layout on any screen)."""
    d = prefs.get("layout_density")
    return d if d in ("auto", "compact", "comfortable") else "auto"


def effective(cls: str) -> str:
    """The range whose automatic layout applies: the setting can turn it off, or make every screen at least compact."""
    d = density()
    if d == "comfortable":
        return LARGE
    if d == "compact" and cls == LARGE:
        return MEDIUM
    return cls


def compact(cls: str) -> bool:
    return effective(cls) != LARGE


def choices(cls: str) -> dict:
    """What the user chose in this size range: his panels' state (only what he acted on)."""
    all_ = prefs.get("layout") or {}
    return dict(all_.get(cls) or {})


def remember(cls: str, **kw):
    """Keep a choice of the user for this size range; the automatic layout no longer applies to it."""
    all_ = dict(prefs.get("layout") or {})
    cur = dict(all_.get(cls) or {})
    cur.update(kw)
    all_[cls] = cur
    prefs.set("layout", all_)


def reset():
    """View > Reset layout: back to the automatic layout of every size range, Post's panel width included."""
    prefs.set("layout", {})
    prefs.set("post_panel_width", 0)
    prefs.set("post_panel_shown", True)


def state(cls: str) -> dict:
    """The panels of this size range: the user's choices, else the automatic layout.

    right: docked | icons, left: docked | folded; hidden (Tab), right_hidden (Shift+Tab), max (Ctrl+Space)."""
    small = effective(cls) == SMALL
    s = {"right": "icons" if small else "docked", "left": "folded" if small else "docked",
         "hidden": False, "right_hidden": False, "max": False}
    s.update(choices(cls))
    return s


# ---------------------------------------------------------------------------------------------- wheel and pinch
def wheel_gesture(e) -> tuple[str, float, float]:
    """What a wheel event asks: ("zoom", steps, 0), a step being one notch of a mouse wheel, or ("pan", dx, dy) in pixels.

    Windows sends a touchpad as wheel events: a pinch has Ctrl held, a two-finger scroll finer deltas than a notch's 120
    (a high-resolution mouse sends those too: it pans, and Ctrl+wheel zooms with it). macOS and Wayland give the phase or
    pixel deltas of a touchpad, and their pinch comes as a native gesture, which ``pinch_factor`` reads."""
    a, px = e.angleDelta(), e.pixelDelta()
    if e.modifiers() & Qt.ControlModifier:
        return "zoom", (a.y() or px.y() * 4) / 120.0, 0.0
    touch = not px.isNull() or e.phase() != Qt.NoScrollPhase or (a.y() % 120 != 0 or a.x() % 120 != 0)
    if not touch:
        return "zoom", a.y() / 120.0, 0.0
    if px.isNull():                                    # about what a notch of the wheel moves a list by
        return "pan", a.x() / 3.0, a.y() / 3.0
    return "pan", float(px.x()), float(px.y())


def pinch_factor(e) -> float | None:
    """The zoom factor of a native pinch (macOS, Wayland), or None for any other event."""
    if e.type() == QEvent.NativeGesture and e.gestureType() == Qt.ZoomNativeGesture:
        return 1.0 + e.value()
    return None


# ---------------------------------------------------------------------------------------------- the folding panel
def pin_icon() -> QIcon:
    """The pin that docks a panel again: drawn, as the app has no icon files."""
    pm = QPixmap(16, 16)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor("#aab2bf"), 1.6))
    p.setBrush(QColor("#aab2bf"))
    p.drawEllipse(QPointF(5.5, 5.5), 3.2, 3.2)
    p.drawLine(QPointF(8, 8), QPointF(13.5, 13.5))
    p.end()
    return QIcon(pm)


class Dock(QObject):
    """A page's right panel: docked, folded to a strip of icons, or hidden (see the module's text).

    ``put`` and ``take`` place the panel in, and out of, its page's layout (a splitter or a box). ``sections``, when
    given, says the panel's sections: [(title, shows it)] for the strip's icons; the strip then has a first icon that
    opens the panel at its top."""
    chosen = Signal(str)                               # the user's gesture: "docked" or "icons", to remember

    def __init__(self, page: QWidget, panel: QWidget, put, take, sections=None, title: str = ""):
        super().__init__(page)
        self.page, self.panel, self.put, self.take, self.sections = page, panel, put, take, sections or (lambda: [])
        self.title = title
        self.state = "docked"
        self._open = False
        self.strip = QFrame(page)
        self.strip.setObjectName("panel")
        self.strip.setFixedWidth(STRIP_W)
        self.strip.setStyleSheet("QFrame#panel { background: #1c1f26; border-left: 1px solid #2a2f38; }")
        v = QVBoxLayout(self.strip)
        v.setContentsMargins(4, 8, 4, 8)
        v.setSpacing(4)
        self.pin = QToolButton()
        self.pin.setIcon(pin_icon())
        self.pin.setAutoRaise(True)
        self.pin.setToolTip(_("Dock the panel"))
        self.pin.clicked.connect(lambda: self._choose("docked"))
        v.addWidget(self.pin, 0, Qt.AlignHCenter)
        self.icons = QVBoxLayout()
        self.icons.setSpacing(4)
        v.addLayout(self.icons)
        v.addStretch(1)
        self.strip.hide()
        self.arrow = QToolButton(page)
        self.arrow.setText("‹")
        self.arrow.setFixedSize(ARROW_W, ARROW_H)
        self.arrow.setToolTip(_("Show the panel again"))
        self.arrow.setStyleSheet("QToolButton { background: #1c1f26; border: 1px solid #2a2f38; color: #aab2bf; "
                                 "border-top-left-radius: 5px; border-bottom-left-radius: 5px; font-weight: 600; }")
        self.arrow.hide()
        page.installEventFilter(self)
        self.refresh_icons()

    # ------------------------------------------------------------ states
    def set_state(self, st: str):
        """docked | icons | hidden."""
        if st == self.state:
            return
        self.close_overlay()
        was = self.state
        self.state = st
        if st == "docked":
            self.strip.hide()
            self.arrow.hide()
            self.put()
            self.panel.show()
            return
        if was == "docked":
            self.take()
        self.panel.hide()
        self.strip.setVisible(st == "icons")
        self.arrow.setVisible(st == "hidden")
        self.place()

    def _choose(self, st: str):
        self.set_state(st)
        self.chosen.emit(st)

    def fold(self):
        """The panel's » button."""
        self._choose("icons")

    def retranslate(self):
        """A change of language on a page that is not built again."""
        self.pin.setToolTip(_("Dock the panel"))
        self.arrow.setToolTip(_("Show the panel again"))
        self.refresh_icons()

    # ------------------------------------------------------------ the icons
    def refresh_icons(self):
        """The strip's icons: one for the panel, one per section (their first letter, the name in the tooltip)."""
        while self.icons.count():
            it = self.icons.takeAt(0)
            if it.widget() is not None:
                it.widget().deleteLater()
        rows = [(self.title or _("Panel"), None)] + list(self.sections())
        for i, (title, show) in enumerate(rows):
            b = QToolButton()
            b.setText("≡" if i == 0 else next((c.upper() for c in title if c.isalpha()), "•"))
            b.setToolTip(title)
            b.setAutoRaise(True)
            b.setFixedSize(STRIP_W - 8, STRIP_W - 12)
            b.setStyleSheet("QToolButton { color: #c9cfd8; font-weight: 600; border: none; border-radius: 4px; }"
                            "QToolButton:hover { background: #2b3446; }")
            b.clicked.connect(lambda _x=False, f=show: self.open_overlay(f))
            self.icons.addWidget(b)

    # ------------------------------------------------------------ over the main area
    def is_open(self) -> bool:
        return self._open

    def _width(self) -> int:
        w = self.panel.width()
        return w if self.panel.minimumWidth() or w > 100 else FLOAT_W

    def open_overlay(self, show_section=None):
        """The folded panel, over the main area, at ``show_section``."""
        if self.state != "icons":
            return
        p = self.panel
        if not self._open:
            self._open = True
            QApplication.instance().installEventFilter(self)
            # over the graph or the image: its own background and an edge (docked, it sits on the page's)
            p.setAttribute(Qt.WA_StyledBackground, True)
            p.setStyleSheet(OVERLAY_QSS)
        p.setParent(self.page)
        self._place_overlay()
        p.raise_()
        p.show()
        self.strip.raise_()
        if show_section is not None:
            QTimer.singleShot(0, self.panel, show_section)

    def close_overlay(self):
        if not self._open:
            return
        self._open = False
        QApplication.instance().removeEventFilter(self)
        self.panel.setAttribute(Qt.WA_StyledBackground, False)
        self.panel.setStyleSheet("")
        self.panel.hide()

    def _place_overlay(self):
        w = self._width()
        s = self.strip.geometry()
        self.panel.setGeometry(max(0, s.x() - w), s.y(), w, s.height())

    def place(self):
        """The arrow of a hidden panel, at the middle of the page's right edge."""
        pg = self.page
        self.arrow.move(pg.width() - ARROW_W, (pg.height() - ARROW_H) // 2)
        self.arrow.raise_()

    def eventFilter(self, o, e):                                     # noqa: N802 - Qt's name
        t = e.type()
        if o is self.page and t == QEvent.Resize:
            self.place()
            if self._open:
                self._place_overlay()
            return False
        if self._open:
            if t == QEvent.Hide and o is self.page:
                self.close_overlay()
            elif t == QEvent.ShortcutOverride and e.key() == Qt.Key_Escape:
                e.accept()                                    # Escape closes the panel, before any shortcut of the page
                return True
            elif t == QEvent.KeyPress and e.key() == Qt.Key_Escape:
                self.close_overlay()
                return True
            elif t == QEvent.MouseButtonPress and isinstance(o, QWidget) and self._outside(e.globalPosition().toPoint()):
                self.close_overlay()                           # the click goes on to what is under it, as a flyout's
        return False

    def _outside(self, gp: QPoint) -> bool:
        if QApplication.activePopupWidget() is not None or QApplication.activeModalWidget() is not None:
            return False                                       # a combo's list, a menu or a dialog of the panel
        for w in (self.panel, self.strip):
            if w.isVisible() and w.rect().contains(w.mapFromGlobal(gp)):
                return False
        return True


def fold_button(owner: QWidget, on_click, tip: str, text: str = "»") -> QToolButton:
    """The panel's » at its top right corner, over the head's empty margin (a child placed by ``place_corner``)."""
    b = QToolButton(owner)
    b.setText(text)
    b.setAutoRaise(True)
    b.setToolTip(tip)
    b.setFixedSize(20, 14)
    b.setStyleSheet("QToolButton { border: none; color: #8b93a1; font-size: 11pt; padding: 0; }"
                    "QToolButton:hover { color: #e8ebf0; }")
    b.clicked.connect(on_click)
    return b


def place_corner(b: QToolButton, owner: QWidget):
    b.move(owner.width() - b.width() - 2, 1)
    b.raise_()

