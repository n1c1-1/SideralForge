"""The window on small laptop screens (issue #44): the size ranges, the compact look and the folding right panel.

The window is measured in logical pixels, never the screen: Qt already applies the system's 125 % or 150 %, and a
window that does not fill the screen has less room than the screen. Three ranges, with a dead band so that dragging the
window's edge does not flicker:

    large    width >= 1600 and height >= 880     exactly today's layout
    medium   width 1280-1599 or height 740-879   compact look
    small    width < 1280 or height < 740        also: the right panel as icons, the blocks' palette automatic

The automatic layout only applies until the user acts: a panel pinned, folded, resized or toggled is his choice, kept
per size range (the setting ``layout``); View > Reset layout forgets them. The setting ``layout_density`` (Settings)
is "auto", "compact" (always compact) or "comfortable" (never: today's layout on any screen).

``Dock`` is a page's side panel: docked (today's), folded to a strip of icons (a click opens it over the main area, Esc
or a click elsewhere closes it, the pin docks it again), automatic (the same strip, opened by the mouse resting on it and
closed when the mouse leaves) or hidden (a small arrow at the window's edge brings it back).
``wheel_gesture`` tells a mouse wheel (zoom) from two fingers on a touchpad (pan) for the graph and the viewers.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QFrame,
    QLabel,
    QLayout,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import prefs
from .i18n import _
from .icons import draw, icon_only, set_icon

LARGE, MEDIUM, SMALL = "large", "medium", "small"
LARGE_W, LARGE_H = 1600, 880
SMALL_W, SMALL_H = 1280, 740
DEAD_BAND = 20                  # a range is left this far past its threshold: the window's edge dragged does not flicker
MIN_WINDOW = (1024, 640)        # the smallest window that still shows every page (Windows' own design floor is 1024 x 640)
STRIP_W = 40                    # the panel folded to icons
ARROW_W, ARROW_H = 16, 46       # the arrow left by a hidden side
FLOAT_W = 400                   # a panel opened over the main area, when it has no width of its own
UNFOLD_MS, FOLD_MS = 200, 300   # automatic: the mouse only crossing it opens nothing, a slip closes nothing (as the palette)

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


OVERLAY_QSS = {"right": "QWidget#panel { background: #1c1f26; border-left: 1px solid #353b47; }",
               "left": "QWidget#panel { background: #1c1f26; border-right: 1px solid #353b47; }"}
#: a panel's three ways of showing, as the blocks' palette (its menu's words)
SHOWN = (("auto", "Automatic"), ("docked", "Kept open"), ("icons", "Kept folded"))


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
    """View > Reset layout: back to the automatic layout of every size range, the widths of Post's sides included."""
    prefs.set("layout", {})
    prefs.set("post_panel_width", 0)
    prefs.set("post_layers_width", 0)


def state(cls: str) -> dict:
    """The panels of this size range: the user's choices, else the automatic layout.

    right: docked | icons, left: docked | folded | auto (its icons, unfolded under the mouse); hidden (Tab), right_hidden
    (Shift+Tab), max (Ctrl+Space)."""
    small = effective(cls) == SMALL
    s = {"right": "icons" if small else "docked", "left": "auto" if small else "docked",
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
class Dock(QObject):
    """A page's side panel: docked, folded to a strip of icons, automatic or hidden (see the module's text).

    ``put`` and ``take`` place the panel in, and out of, its page's layout (a splitter or a box). ``sections``, when
    given, says the panel's sections: [(title, shows it, icon role)] for the strip's icons (the role may be left out);
    the strip then has a first icon, ``role``, that opens the panel at its top. ``side``: "right" or "left"."""
    chosen = Signal(str)                               # the user's gesture: "docked", "icons" or "auto", to remember

    def __init__(self, page: QWidget, panel: QWidget, put, take, sections=None, title: str = "", role: str = "adjust",
                 side: str = "right"):
        super().__init__(page)
        self.page, self.panel, self.put, self.take, self.sections = page, panel, put, take, sections or (lambda: [])
        self.title, self.role, self.side = title, role, side
        self.state = "docked"
        self._open = False
        self._unfold_t = QTimer(self, singleShot=True, interval=UNFOLD_MS)
        self._unfold_t.timeout.connect(self._hover_open)
        self._fold_t = QTimer(self, singleShot=True, interval=FOLD_MS)
        self._fold_t.timeout.connect(self._hover_close)
        self.strip = QFrame(page)
        self.strip.setObjectName("panel")
        self.strip.setFixedWidth(STRIP_W)
        edge = "border-left" if side == "right" else "border-right"
        self.strip.setStyleSheet(f"QFrame#panel {{ background: #1c1f26; {edge}: 1px solid #2a2f38; }}")
        v = QVBoxLayout(self.strip)
        v.setContentsMargins(4, 8, 4, 8)
        v.setSpacing(4)
        self.pin = QToolButton()
        set_icon(self.pin, "pin", 16, "dim")
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
        set_icon(self.arrow, "chevron_left" if side == "right" else "chevron_right", 16, "dim")
        self.arrow.setFixedSize(ARROW_W, ARROW_H)
        self.arrow.setToolTip(_("Show the panel again"))
        inner = "left" if side == "right" else "right"
        self.arrow.setStyleSheet("QToolButton { background: #1c1f26; border: 1px solid #2a2f38; padding: 0; "
                                 f"border-top-{inner}-radius: 5px; border-bottom-{inner}-radius: 5px; }}")
        self.arrow.hide()
        page.installEventFilter(self)
        self.strip.installEventFilter(self)
        panel.installEventFilter(self)
        self.refresh_icons()

    # ------------------------------------------------------------ states
    def set_state(self, st: str):
        """docked | icons | auto | hidden."""
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
        self.strip.setVisible(st in ("icons", "auto"))
        self.arrow.setVisible(st == "hidden")
        self.place()

    def _choose(self, st: str):
        self.set_state(st)
        self.chosen.emit(st)

    def fold(self):
        """The panel's fold button."""
        self._choose("icons")

    def menu(self, button: QWidget):
        """The panel's three ways of showing (``SHOWN``), as the blocks' palette's menu."""
        m = QMenu(button)
        for st, word in SHOWN:
            a = m.addAction(_(word))
            a.setCheckable(True)
            a.setChecked(st == self.state)
            a.setData(st)
        a = m.exec(button.mapToGlobal(button.rect().bottomLeft()))
        st = a.data() if a is not None else None
        m.deleteLater()
        if st and st != self.state:
            self._choose(st)

    def retranslate(self):
        """A change of language on a page that is not built again."""
        self.pin.setToolTip(_("Dock the panel"))
        self.arrow.setToolTip(_("Show the panel again"))
        self.refresh_icons()

    # ------------------------------------------------------------ the icons
    def refresh_icons(self):
        """The strip's icons: one for the panel, one per section (the name in the tooltip)."""
        while self.icons.count():
            it = self.icons.takeAt(0)
            if it.widget() is not None:
                it.widget().hide()
                it.widget().deleteLater()
        rows = [(self.title or _("Panel"), None, self.role)] + [tuple(x) + ("section",) * (3 - len(x)) for x in self.sections()]
        for title, show, role in rows:
            b = QToolButton()
            set_icon(b, role, 20, tip=title)
            b.setAutoRaise(True)
            b.setFixedSize(STRIP_W - 8, STRIP_W - 12)
            b.setStyleSheet("QToolButton { border: none; border-radius: 4px; padding: 0; }"
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
        if self.state not in ("icons", "auto"):
            return
        self._unfold_t.stop()
        self._fold_t.stop()
        p = self.panel
        if not self._open:
            self._open = True
            QApplication.instance().installEventFilter(self)
            # over the graph or the image: its own background and an edge (docked, it sits on the page's)
            p.setAttribute(Qt.WA_StyledBackground, True)
            p.setStyleSheet(OVERLAY_QSS[self.side])
        p.setParent(self.page)
        self._place_overlay()
        p.raise_()
        p.show()
        self.strip.raise_()
        if show_section is not None:
            QTimer.singleShot(0, self.panel, show_section)

    def close_overlay(self):
        self._unfold_t.stop()
        self._fold_t.stop()
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
        x = max(0, s.x() - w) if self.side == "right" else s.right() + 1
        self.panel.setGeometry(x, s.y(), w, s.height())

    def place(self):
        """The arrow of a hidden panel, at the middle of the page's edge."""
        pg = self.page
        self.arrow.move(pg.width() - ARROW_W if self.side == "right" else 0, (pg.height() - ARROW_H) // 2)
        self.arrow.raise_()

    # ------------------------------------------------------------ automatic: opened by the mouse resting on the strip
    def _hover_open(self):
        if self.state == "auto" and not self._open and self.strip.underMouse():
            self.open_overlay()

    def _hover_close(self):
        """The mouse left: closed, unless the user is still at work in it (a button held, a list or a menu open, a text
        being typed), then looked at again later."""
        if self.state != "auto" or not self._open or self.strip.underMouse() or self.panel.underMouse():
            return
        fw = QApplication.focusWidget()
        typing = isinstance(fw, (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit)) and self.panel.isAncestorOf(fw)
        if QApplication.mouseButtons() != Qt.NoButton or QApplication.activePopupWidget() is not None or typing:
            self._fold_t.start()
            return
        self.close_overlay()

    def eventFilter(self, o, e):                                     # noqa: N802 - Qt's name
        t = e.type()
        if o is self.page and t == QEvent.Resize:
            self.place()
            if self._open:
                self._place_overlay()
            return False
        if self.state == "auto" and (o is self.strip or o is self.panel):
            if t == QEvent.Enter:
                self._fold_t.stop()
                if not self._open and o is self.strip:
                    self._unfold_t.start()
            elif t == QEvent.Leave:
                self._unfold_t.stop()
                if self._open:
                    self._fold_t.start()
        if self._open:
            if t == QEvent.Hide and o is self.page:
                self.close_overlay()
            elif t == QEvent.ShortcutOverride and e.key() == Qt.Key_Escape and self._in_window(o):
                e.accept()                                    # Escape closes the panel, before any shortcut of the page
                return True
            elif t == QEvent.KeyPress and e.key() == Qt.Key_Escape and self._in_window(o):
                self.close_overlay()
                return True
            elif t == QEvent.MouseButtonPress and self._in_window(o) and self._outside(e.globalPosition().toPoint()):
                self.close_overlay()                           # the click goes on to what is under it, as a flyout's
        return False

    def _in_window(self, o) -> bool:
        """Whether an event goes to a widget of the page's window: a combo's list, a menu or a dialog is a window of
        its own, and keeps its click and its Escape."""
        return isinstance(o, QWidget) and o.window() is self.page.window()

    def _outside(self, gp: QPoint) -> bool:
        for w in (self.panel, self.strip):
            if w.isVisible() and w.rect().contains(w.mapFromGlobal(gp)):
                return False
        return True


# ---------------------------------------------------------------------------------------------- folding sections
def _widgets_of(layout: QLayout) -> list[QWidget]:
    out: list[QWidget] = []
    for i in range(layout.count()):
        it = layout.itemAt(i)
        if it.widget() is not None:
            out.append(it.widget())
        elif it.layout() is not None:
            out += _widgets_of(it.layout())
    return out


class SectionTitle(QLabel):
    """A section's title: a click folds it, Alt-click opens only that one, as the reference editors' panels do. Its
    arrow shows when folded or under the mouse, so that a large window looks as it did."""

    def __init__(self, text, sections):
        super().__init__(text)
        # a title stands out from the settings under it (#119: Starless and Stars alone were not told from the rest)
        self.setObjectName("sectiontitle")
        self.setStyleSheet("QLabel#sectiontitle { font-weight: 600; color: #e6edf7; padding: 6px 0 3px 0; "
                           "margin-top: 8px; border-top: 1px solid #353b47; }")
        self.sections, self.folded, self.rows, self._hover, self._shown = sections, False, [], False, []
        self.setToolTip(_("Click: fold or unfold · Alt-click: only this section"))
        self.setCursor(Qt.PointingHandCursor)

    def set_folded(self, on: bool):
        """Only what it hid comes back: a row its panel keeps hidden for its own reasons stays hidden."""
        if on == self.folded:
            return
        self.folded = on
        if on:
            self._shown = [w for w in self.rows if not (w.isHidden() and w.testAttribute(Qt.WA_WState_ExplicitShowHide))]
            for w in self._shown:
                w.setVisible(False)
        else:
            for w in self._shown:                      # setVisible: a slider's ``show`` is its value's text
                w.setVisible(True)
            self._shown = []
        self.update()

    def enterEvent(self, e):                                         # noqa: N802 - Qt's name
        self._hover = True
        self.update()

    def leaveEvent(self, e):                                         # noqa: N802
        self._hover = False
        self.update()

    def mouseReleaseEvent(self, e):                                  # noqa: N802
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.sections.clicked(self, bool(e.modifiers() & Qt.AltModifier))

    def paintEvent(self, e):                                         # noqa: N802
        super().paintEvent(e)
        if self.folded or self._hover:
            p = QPainter(self)
            r = self.contentsRect()
            draw(p, QRectF(r.right() - 17, r.center().y() - 7, 16, 16), "chevron_right" if self.folded else "chevron_down",
                 "normal" if self.folded else "dim")
            p.end()


#: the sections drawn as icons on a folded strip: their English titles (a title's count stays out), the rest "section"
SECTION_ROLES = {"Sessions": "block_session", "Lights": "block_image", "Layers": "layers", "Stars": "block_stars",
                 "Palette": "palette", "Channels": "block_colour", "Measurements": "measure", "Samples · exclusions": "samples",
                 "Outputs": "export", "Rule": "thresholds", "Thresholds": "thresholds", "Speed · memory": "compute"}


def section_role(title: str) -> str:
    """The icon of a section, from its title as shown (in the interface's language)."""
    base = title.split("  ·  ")[0]
    return next((role for english, role in SECTION_ROLES.items() if _(english) == base), "section")


class Sections:
    """The folding sections of a settings column (a box of titles and rows, built again at each change): a section is
    what lies between its title and the next one. What is folded is kept per kind of block or layer (``key``), and only
    the user folds: on a small window too, several stay open and the column scrolls (#119; the small window used to
    open one at a time, closing the others)."""

    def __init__(self, box, scroll):
        self.box, self.scroll = box, scroll
        self.titles: list[SectionTitle] = []
        self.key = ""
        self.folded: dict[str, set[str]] = {}          # kind -> the titles of its folded sections

    def reset(self):
        self.titles = []

    def title(self, text: str) -> SectionTitle:
        t = SectionTitle(text, self)
        self.titles.append(t)
        return t

    def finish(self, key: str):
        """The box is built: each title takes the rows under it, and the folds of this kind apply."""
        self.key = key
        cur = None
        for i in range(self.box.count()):
            it = self.box.itemAt(i)
            w = it.widget()
            if isinstance(w, SectionTitle):
                cur = w
            elif cur is not None and w is not None:
                if w.objectName() != "disclosure":     # the Advanced settings' line stays
                    cur.rows.append(w)
            elif cur is not None and it.layout() is not None:
                cur.rows += _widgets_of(it.layout())
        self.apply()

    def apply(self):
        off = self.folded.get(self.key, set())
        for t in self.titles:
            t.set_folded(t.text() in off)

    def clicked(self, t: SectionTitle, alt: bool):
        off = self.folded.setdefault(self.key, set())
        if alt:
            off.clear()
            off.update(x.text() for x in self.titles if x is not t)
        elif t.text() in off:
            off.discard(t.text())
        else:
            off.add(t.text())
        self.apply()

    def items(self) -> list[tuple[str, object, str]]:
        """For the strip of icons: (title, what opens that section and brings it in view, its icon)."""
        return [(t.text(), (lambda x=t: self.show(x)), section_role(t.text())) for t in self.titles]

    def show(self, t: SectionTitle):
        if t.folded:
            self.clicked(t, False)
        self.scroll.ensureWidgetVisible(t, 0, 10)


def fold_button(owner: QWidget, on_click, tip: str, role: str = "fold_right") -> QToolButton:
    """The panel's fold icon at its top right corner, over the head's empty margin (a child placed by ``place_corner``)."""
    b = QToolButton(owner)
    icon_only(b, role, tip, 16, "dim")
    b.setAutoRaise(True)
    b.setIconSize(QSize(16, 16))
    b.setFixedSize(20, 18)
    b.setStyleSheet("QToolButton { border: none; padding: 0; } QToolButton:hover { background: #2b3446; border-radius: 3px; }")
    b.clicked.connect(on_click)
    return b


def place_corner(b: QToolButton, owner: QWidget):
    b.move(owner.width() - b.width() - 2, 1)
    b.raise_()

