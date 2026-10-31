"""Style and small shared widgets.

Frozen once the shell is done (DESIGN §6.1): a view that needs a new widget keeps it in its own module.
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

ACCENT, OK, WARN, SHARED, BAD = "#5ea1ff", "#5fbf7f", "#e6a23c", "#a395ff", "#e06c6c"
NEW = "#c9b37e"                       # the ◇ mark: what the engine does not have yet (Spec mode)
DISABLED = "#5d6470"

QSS = f"""
* {{ font-family: 'Segoe UI'; font-size: 10pt; color: #d8dce4; }}
QMainWindow, QWidget#root {{ background: #15171c; }}
QWidget#panel, QFrame#panel {{ background: #1c1f26; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QLabel#h1 {{ font-size: 13pt; font-weight: 600; }}
QLabel#h2 {{ font-size: 10.5pt; font-weight: 600; color: #e8ebf0; }}
QLabel#dim {{ color: #8b93a1; }}
QLabel#small {{ color: #8b93a1; font-size: 9pt; }}
QLabel#reason {{ color: #9aa3b2; font-size: 9pt; font-style: italic; }}
QLabel#warn {{ color: {WARN}; font-size: 9pt; }}
QLabel:disabled {{ color: {DISABLED}; }}
QPushButton {{ background: #2a2f39; border: 1px solid #353b47; border-radius: 4px; padding: 4px 10px; }}
QPushButton:hover {{ background: #323846; }}
QPushButton:pressed {{ background: #3a4252; }}
QPushButton:disabled {{ color: {DISABLED}; border-color: #2a2f39; }}
QPushButton#primary {{ background: {ACCENT}; color: #0f1115; border: none; font-weight: 600; }}
QPushButton#primary:hover {{ background: #7ab3ff; }}
QPushButton#primary:disabled {{ background: #2a2f39; color: {DISABLED}; border: 1px solid #353b47; }}
QPushButton::menu-indicator {{ image: none; width: 0; }}
QPushButton#seg {{ background: transparent; border: none; border-radius: 0; padding: 8px 22px;
                  font-size: 11pt; color: #8b93a1; border-bottom: 2px solid transparent; }}
QPushButton#seg:checked {{ color: #ffffff; border-bottom: 2px solid {ACCENT}; }}
QPushButton#seg:disabled {{ color: {DISABLED}; }}
QPushButton#face {{ padding: 4px 16px; }}
QPushButton#face:checked {{ background: #2b3446; border-color: {ACCENT}; color: #ffffff; font-weight: 600; }}
QPushButton#toggle:checked {{ background: #4a3a1d; border-color: {WARN}; color: #ffd28a; }}
QPushButton#link {{ background: transparent; border: none; color: {ACCENT}; padding: 0; text-align: left; }}
QPushButton#link:hover {{ text-decoration: underline; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: #12141a; border: 1px solid #353b47;
    border-radius: 3px; padding: 2px 6px; selection-background-color: {ACCENT}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: {DISABLED};
    border-color: #2a2f39; }}
QComboBox QAbstractItemView {{ background: #1c1f26; border: 1px solid #353b47; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px; }}
QRadioButton::indicator {{ width: 12px; height: 12px; border: 2px solid #6b7280; border-radius: 8px; background: #12141a; }}
QRadioButton::indicator:checked {{ border: 2px solid {ACCENT}; background: qradialgradient(cx:0.5, cy:0.5,
    radius:0.5, fx:0.5, fy:0.5, stop:0 {ACCENT}, stop:0.55 {ACCENT}, stop:0.62 #12141a, stop:1 #12141a); }}
QRadioButton::indicator:disabled {{ border-color: #3a4150; }}
QCheckBox:disabled, QRadioButton:disabled {{ color: {DISABLED}; }}
QTreeWidget, QListWidget {{ background: #1c1f26; border: none; outline: none; }}
QTreeWidget::item {{ padding: 4px 2px; }}
QTreeWidget::item:selected, QListWidget::item:selected {{ background: #2b3446; }}
QTabBar::tab {{ background: transparent; padding: 5px 12px; color: #8b93a1; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: #ffffff; border-bottom: 2px solid {ACCENT}; }}
QTabWidget#faces::pane {{ border: 1px solid #3a4150; border-top-right-radius: 6px; border-bottom-left-radius: 6px;
    border-bottom-right-radius: 6px; top: -1px; background: #15171c; }}
QTabWidget#faces > QTabBar::tab {{ background: #1c1f26; color: #8b93a1; border: 1px solid #3a4150; border-bottom: none;
    border-top-left-radius: 6px; border-top-right-radius: 6px; padding: 7px 28px; margin-right: 3px;
    font-size: 10.5pt; }}
QTabWidget#faces > QTabBar::tab:selected {{ background: #15171c; color: #ffffff; font-weight: 600;
    border-color: #556079; border-bottom: 1px solid #15171c; }}
QTabWidget#faces > QTabBar::tab:!selected:hover {{ color: #d8dce4; background: #232831; }}
QSlider::groove:horizontal {{ height: 4px; background: #353b47; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: #4b6fa8; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #e8ebf0; width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; }}
QSlider::handle:horizontal:disabled {{ background: #4a505c; }}
QSlider::sub-page:horizontal:disabled {{ background: #353b47; }}
QToolTip {{ background: #262a33; color: #e8ebf0; border: 1px solid #3a4150; padding: 4px; }}
QMenu {{ background: #1c1f26; border: 1px solid #353b47; }}
QMenu::item:selected {{ background: #2b3446; }}
QMenu::item:disabled {{ color: {DISABLED}; }}
QProgressBar {{ background: #12141a; border: 1px solid #353b47; border-radius: 3px; height: 10px; text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: #353b47; border-radius: 4px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: #353b47; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QDialog {{ background: #1c1f26; }}
"""


def label(text="", name=None, wrap=False) -> QLabel:
    lb = QLabel(text)
    if name:
        lb.setObjectName(name)
    lb.setWordWrap(wrap)
    return lb


def para(text, name="dim", width=560) -> QLabel:
    """A centred paragraph of fixed width whose height follows its text (empty states)."""
    lb = QLabel(text)
    lb.setObjectName(name)
    lb.setWordWrap(True)
    lb.setAlignment(Qt.AlignCenter)
    lb.setFixedWidth(width)
    fit_height(lb)
    return lb


def fit_height(lb: QLabel):
    """Fixes the height of a wrapped label to its text, with the style sheet's font (layouts with an alignment
    flag give it only one line otherwise)."""
    lb.ensurePolished()
    r = lb.fontMetrics().boundingRect(0, 0, lb.width(), 10000, int(Qt.TextWordWrap | lb.alignment()), lb.text())
    lb.setFixedHeight(r.height() + 6)


def hline() -> QFrame:
    f = QFrame()
    f.setFixedHeight(1)
    f.setStyleSheet("background: #2a2f38;")
    return f


def clear(layout):
    while layout.count():
        it = layout.takeAt(0)
        w = it.widget()
        if w is not None:
            w.hide()                                       # not drawn again before it is deleted
            w.deleteLater()
        elif it.layout() is not None:
            clear(it.layout())


def spec_mark(text: str) -> str:
    """A name followed by the ◇ mark of what the engine does not have yet (rich text, for a QLabel)."""
    return f"{text} <span style='color:{NEW}'>◇</span>"


class ElidedLabel(QLabel):
    """A one-line label for a name or a path: without room, its text is shortened with "…" (in the middle for a
    path, "F:/astro…/Rosette") and the whole text goes to its tooltip."""

    def __init__(self, text="", name=None, mode=Qt.ElideMiddle, parent=None):
        super().__init__(parent)
        if name:
            self.setObjectName(name)
        self._mode = mode
        self._full = ""
        self.setText(text)

    def setText(self, text):                         # noqa: N802 - Qt's name
        self._full = text or ""
        self._elide()

    def full_text(self) -> str:
        return self._full

    def _elide(self):
        self.ensurePolished()
        shown = self.fontMetrics().elidedText(self._full, self._mode, max(self.width(), 1))
        QLabel.setText(self, shown)
        self.setToolTip(self._full if shown != self._full else "")

    def resizeEvent(self, e):                        # noqa: N802
        super().resizeEvent(e)
        self._elide()

    def sizeHint(self):                              # noqa: N802
        self.ensurePolished()
        m = self.contentsMargins()
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + m.left() + m.right() + 2,
                     super().sizeHint().height())

    def minimumSizeHint(self):                       # noqa: N802
        return QSize(24, super().minimumSizeHint().height())


class Box(QFrame):
    """A framed note: impact of a change, cost, warning."""

    def __init__(self, color=WARN, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"QFrame#box {{ background: #23211c; border: 1px solid {color}; border-radius: 6px; }}"
                           "QFrame#box QLabel { background: transparent; }")
        self.setObjectName("box")
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(10, 8, 10, 8)
        self.v.setSpacing(6)
