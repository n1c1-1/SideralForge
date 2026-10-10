"""Visual style: a calm dark theme suited to judging astronomical images (neutral greys, one warm accent)."""
from __future__ import annotations

from PySide6.QtGui import QColor, QPalette

ACCENT = "#d9a441"        # warm amber, readable on dark grey, evokes H-beta / sodium rather than UI blue
BG = "#1b1d21"
BG2 = "#23262b"
BG3 = "#2c3036"
FG = "#e4e2dd"
FG2 = "#a7abb3"
OK = "#6fbf73"
WARN = "#e0a43c"
ERR = "#e06c5f"
RUN = "#5aa0d8"

STATE_COLOURS = {"done": OK, "ready": FG, "stale": WARN, "pending": FG2, "failed": ERR, "interrupted": WARN, "running": RUN}
STATE_SYMBOLS = {"done": "✔", "ready": "○", "stale": "↻", "pending": "○", "failed": "✖",
                 "interrupted": "⏸", "running": "▶"}

QSS = f"""
* {{ font-size: 10pt; }}
QMainWindow, QDialog, QWidget {{ background: {BG}; color: {FG}; }}
QDockWidget::title {{ background: {BG2}; padding: 6px 8px; font-weight: 600; color: {FG2};
                      text-transform: uppercase; letter-spacing: 1px; }}
QToolBar {{ background: {BG2}; border: none; spacing: 6px; padding: 4px; }}
QToolButton, QPushButton {{ background: {BG3}; border: 1px solid #3a3f47; border-radius: 4px; padding: 5px 10px; }}
QToolButton:hover, QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:default, QPushButton#primary {{ background: {ACCENT}; color: #1b1d21; border: none; font-weight: 600; }}
QPushButton:disabled, QToolButton:disabled {{ color: #6b7079; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextBrowser {{
    background: {BG2}; border: 1px solid #3a3f47; border-radius: 3px; padding: 3px; selection-background-color: {ACCENT};
    selection-color: #1b1d21; }}
QListWidget, QTreeWidget, QTableWidget {{ background: {BG2}; border: none; }}
QListWidget::item {{ padding: 6px 4px; }}
QListWidget::item:selected, QTreeWidget::item:selected {{ background: {BG3}; border-left: 3px solid {ACCENT}; color: {FG}; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: {BG2}; padding: 7px 16px; color: {FG2}; border: none; }}
QTabBar::tab:selected {{ color: {FG}; border-bottom: 2px solid {ACCENT}; }}
QGroupBox {{ border: 1px solid #343840; border-radius: 4px; margin-top: 14px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {FG2}; }}
QSlider::groove:horizontal {{ height: 4px; background: #3a3f47; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 12px; margin: -5px 0; border-radius: 6px; }}
QSlider::sub-page:horizontal {{ background: #8a6a2c; border-radius: 2px; }}
QProgressBar {{ background: {BG2}; border: none; border-radius: 3px; text-align: center; height: 16px; color: {FG}; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QScrollArea {{ border: none; }}
QLabel#muted {{ color: {FG2}; }}
QLabel#badge_auto {{ color: {OK}; font-size: 8pt; }}
QLabel#badge_user {{ color: {ACCENT}; font-size: 8pt; }}
QLabel#h1 {{ font-size: 13pt; font-weight: 600; }}
QToolTip {{ background: {BG3}; color: {FG}; border: 1px solid {ACCENT}; padding: 6px; }}
"""


def palette() -> QPalette:
    p = QPalette()
    for role, c in ((QPalette.Window, BG), (QPalette.WindowText, FG), (QPalette.Base, BG2), (QPalette.AlternateBase, BG3),
                    (QPalette.Text, FG), (QPalette.Button, BG3), (QPalette.ButtonText, FG),
                    (QPalette.Highlight, ACCENT), (QPalette.HighlightedText, BG), (QPalette.ToolTipBase, BG3),
                    (QPalette.ToolTipText, FG)):
        p.setColor(role, QColor(c))
    return p
