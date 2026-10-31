"""Style and small shared widgets."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

ACCENT, OK, WARN, SHARED, BAD = "#5ea1ff", "#5fbf7f", "#e6a23c", "#a395ff", "#e06c6c"

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
QPushButton {{ background: #2a2f39; border: 1px solid #353b47; border-radius: 4px; padding: 4px 10px; }}
QPushButton:hover {{ background: #323846; }}
QPushButton:pressed {{ background: #3a4252; }}
QPushButton:disabled {{ color: #5d6470; border-color: #2a2f39; }}
QPushButton#primary {{ background: {ACCENT}; color: #0f1115; border: none; font-weight: 600; }}
QPushButton#primary:hover {{ background: #7ab3ff; }}
QPushButton#primary:disabled {{ background: #2a2f39; color: #5d6470; border: 1px solid #353b47; }}
QPushButton::menu-indicator {{ image: none; width: 0; }}
QPushButton#seg {{ background: transparent; border: none; border-radius: 0; padding: 8px 22px;
                  font-size: 11pt; color: #8b93a1; border-bottom: 2px solid transparent; }}
QPushButton#seg:checked {{ color: #ffffff; border-bottom: 2px solid {ACCENT}; }}
QPushButton#face {{ padding: 4px 16px; }}
QPushButton#face:checked {{ background: #2b3446; border-color: {ACCENT}; color: #ffffff; font-weight: 600; }}
QPushButton#toggle:checked {{ background: #4a3a1d; border-color: {WARN}; color: #ffd28a; }}
QPushButton#link {{ background: transparent; border: none; color: {ACCENT}; padding: 0; text-align: left; }}
QPushButton#link:hover {{ text-decoration: underline; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: #12141a; border: 1px solid #353b47;
    border-radius: 3px; padding: 2px 6px; selection-background-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: #1c1f26; border: 1px solid #353b47; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px; }}
QRadioButton::indicator {{ width: 12px; height: 12px; border: 2px solid #6b7280; border-radius: 8px; background: #12141a; }}
QRadioButton::indicator:checked {{ border: 2px solid {ACCENT}; background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 {ACCENT}, stop:0.55 {ACCENT}, stop:0.62 #12141a, stop:1 #12141a); }}
QRadioButton::indicator:disabled {{ border-color: #3a4150; }}
QCheckBox:disabled, QRadioButton:disabled {{ color: #5d6470; }}
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
    lb.setTextInteractionFlags(Qt.TextSelectableByMouse) if False else None
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


class Card(QFrame):
    """A measurement: a big value and what it means."""

    def __init__(self, value: str, text: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QFrame { background: #23272f; border-radius: 6px; }")
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 8, 12, 8)
        v.setSpacing(2)
        a = QLabel(value)
        a.setStyleSheet("font-size: 14pt; font-weight: 600; background: transparent;")
        b = QLabel(text)
        b.setWordWrap(True)
        b.setStyleSheet("color: #9aa3b2; font-size: 9pt; background: transparent;")
        v.addWidget(a)
        v.addWidget(b)


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


class NotchSlider(QWidget):
    """A simplified slider: 5 positions, the middle one is the suggestion; greyed with its reason when unavailable."""
    moved = Signal(str, int)
    unlock = Signal(str)

    def __init__(self, key, name, ends, parent=None):
        super().__init__(parent)
        self.key = key
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 2, 0, 6)
        v.setSpacing(1)
        top = QHBoxLayout()
        self.name = QLabel(name)
        self.val = QLabel("suggéré")
        self.val.setObjectName("small")
        self.reset = QPushButton("↺")
        self.reset.setFixedSize(22, 20)
        self.reset.setToolTip("Revenir à la suggestion")
        self.reset.setStyleSheet("padding: 0;")
        self.reset.clicked.connect(lambda: self.s.setValue(2))
        top.addWidget(self.name)
        top.addStretch(1)
        top.addWidget(self.val)
        top.addWidget(self.reset)
        v.addLayout(top)
        row = QHBoxLayout()
        self.lo, self.hi = QLabel(ends[0]), QLabel(ends[1])
        for lb in (self.lo, self.hi):
            lb.setObjectName("small")
        self.s = QSlider(Qt.Horizontal)
        self.s.setRange(0, 4)
        self.s.setPageStep(1)
        self.s.setValue(2)
        self.s.setTickPosition(QSlider.TicksBelow)
        self.s.valueChanged.connect(self._changed)
        row.addWidget(self.lo)
        row.addWidget(self.s, 1)
        row.addWidget(self.hi)
        v.addLayout(row)
        self.why = QLabel()
        self.why.setObjectName("warn")
        self.why.setWordWrap(True)
        self.btn = QLabel()
        self.btn.setWordWrap(True)
        self.btn.setTextFormat(Qt.RichText)
        self.btn.linkActivated.connect(lambda _: self.unlock.emit(self.key))
        v.addWidget(self.why)
        v.addWidget(self.btn)
        self.set_available(True)
        self._changed(2, emit=False)

    def _changed(self, pos, emit=True):
        txt = {0: "−−", 1: "−", 2: "suggéré", 3: "+", 4: "++"}[pos]
        self.val.setText(txt)
        self.reset.setVisible(pos != 2 and self.s.isEnabled())
        if emit:
            self.moved.emit(self.key, pos)

    def set_pos(self, pos):
        self.s.blockSignals(True)
        self.s.setValue(pos)
        self.s.blockSignals(False)
        self._changed(pos, emit=False)

    def set_available(self, ok, why="", action=""):
        self.s.setEnabled(ok)
        self.name.setStyleSheet("" if ok else "color: #6b7280;")
        self.why.setVisible(not ok)
        self.btn.setVisible(not ok and bool(action))
        self.why.setText(f"Indisponible : {why}." if why else "")
        self.btn.setText(f"<a href='#' style='color:{ACCENT}; text-decoration:none'>{action}  →</a>" if action else "")
        self.reset.setVisible(ok and self.s.value() != 2)
