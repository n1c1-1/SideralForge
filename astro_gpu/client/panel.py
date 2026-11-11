"""The panel of the selected block, on the right in Graph and Inspect.

One line per setting. What there is to know about a setting (its suggestion and why, its range, its use, a tip)
shows under the list when the setting is clicked or changed. When the block is shared by several workflows, the panel
says which ones a change goes to; a change that would throw away computed results offers Replace or Try as variant.
Blocks with several methods show them as cards: what each does, where it runs, how long on this machine. A Stars and
background block shows its layer's settings in Post (how the stars go back, the sliders): a change here is a change
there, and the other way round (``ProjectModel.set_post_layer``, ``postChanged``).

The settings come from the model's cache of the engine's ``describe`` (the worker fills it): the panel shows what it has
and builds itself again when the rest arrives. The ◇ methods and settings (not in the engine yet) show in Spec mode
only, disabled.

Issue #44: the section titles fold (``layout.Sections``); Alt-click keeps one open and closes the others; on a small window
the panel opens with one section at a time, and the ``layout.Dock`` of its page folds the whole panel to icons.

Classes: ``Panel``, its rows (``CompactRow``), the method cards (``MethodCard``, ``StackMethodCard`` with its
``InfoDot``), ``Details`` (what is known about the setting clicked) and ``ProjectForm`` (the Project block's form, in the
centre of Inspect, issue #89).
"""
from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import QLocale, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .. import texts
from ..core import workflow as wfl
from . import prefs
from .blocks import BY_KEY, STATE_WORD, block_state, block_title
from .i18n import _, all_workflows, hm, language, number
from .icons import IconLabel, IconText, icon_only, set_icon
from .layout import Sections
from .model import NotInEngine, method_of, method_value, methods_of
from .settings import RangeBar, SettingRow, fmt, needed_stages, selection_rule_text, step_settings
from .settings import loading as settings_loading
from .widgets import ACCENT, BAD, NEW, OK, WARN, clear, hline, label

STATE_COLOR = {"done": "#5fbf7f", "pending": "#8b93a1", "stale": WARN, "running": ACCENT, "failed": BAD,
               "interrupted": WARN, "skipped": "#8b93a1"}
OPTIONAL_BLOCKS = ("debruitage", "etoiles", "deconv")
EDITOR_W = 104                      # every editor of a row has this width...
BACK_W = 22                         # ...then the unit column, then the ↺ button (its room kept when hidden)
UNIT_MAX = 76                       # a longer unit wraps on two lines


def kv(rows, pad=14) -> QLabel:
    """Two columns of text: a name and its value."""
    html = "<table cellspacing='0' cellpadding='3'>" + "".join(
        f"<tr><td style='color:#8b93a1; padding-right:{pad}px' valign='top'>{k}</td><td>{v}</td></tr>"
        for k, v in rows) + "</table>"
    lb = QLabel(html)
    lb.setTextFormat(Qt.RichText)
    lb.setWordWrap(True)
    return lb


def unit_column(units) -> int:
    """The width of the unit column of a panel: its longest unit (a longer one than UNIT_MAX wraps), 0 without any.
    The same width on every row puts every editor at the same x."""
    units = [u for u in units if u]
    if not units:
        return 0
    probe = label("", "small")
    probe.ensurePolished()
    return min(UNIT_MAX, max(probe.fontMetrics().horizontalAdvance(u) for u in units) + 6)


def unit_label(text: str, width: int) -> QLabel:
    u = label(text or "", "small")
    u.setFixedWidth(width)
    u.ensurePolished()
    u.setWordWrap(bool(text) and u.fontMetrics().horizontalAdvance(text) > width)
    u.setToolTip(text or "")
    return u


def name_label(text: str) -> QLabel:
    """The name of a setting: on two lines when the row has no room for it (long French names), never cut."""
    n = QLabel(text)
    n.setWordWrap(True)
    return n


def retain(w: QWidget, shown: bool) -> QWidget:
    """Hidden, the widget keeps its room, so that the columns after it do not move."""
    sp = w.sizePolicy()
    sp.setRetainSizeWhenHidden(True)
    w.setSizePolicy(sp)
    if not shown or w.parentWidget() is not None:  # shown without a parent yet, it would open as a window of its
        w.setVisible(shown)                         # own for a moment and close the menu open (Stacking's, 2nd round)
    return w


def spec_mark(row_layout, new):
    """The ◇ mark of what the engine does not have yet: only in Spec mode."""
    if new and prefs.spec():
        mark = QLabel("◇")
        mark.setStyleSheet(f"color: {NEW};")
        mark.setToolTip(_("Not in the engine yet"))
        row_layout.addWidget(mark)


class CommentEdit(QPlainTextEdit):
    """Several lines typed: applied when the focus leaves, as a field's Enter (the panel is not built again under it)."""
    finished = Signal(str)

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setMaximumHeight(120)
        self.setPlaceholderText(_("Notes on this project"))
        self.setStyleSheet("QPlainTextEdit { background: #15171c; border: 1px solid #353b47; border-radius: 4px; }")

    def focusOutEvent(self, e):                                       # noqa: N802
        self.finished.emit(self.toPlainText())
        super().focusOutEvent(e)


# ---------------------------------------------------------------------------------------------- the Project block
#: what the target field's dot says (``ProjectModel.target_state``): green known, orange unknown here, grey empty
TARGET_DOTS = {"known": OK, "unknown": WARN, "empty": "#6b7382"}
#: how complete the block is (``ProjectModel.project_state``, issue #89): the colour of the block and of its line
PROJECT_COLOURS = {"none": BAD, "partial": WARN, "complete": OK}
#: what is known of the capture, kept with the target: (key, name, unit, tooltip)
CAPTURE = (("ra", "RA", "°", "The target's right ascension, when the catalogues here do not know it"),
           ("dec", "Dec", "°", "The target's declination, when the catalogues here do not know it"),
           ("focal_mm", "Focal length", "mm", "The focal length of the instrument: Astrometry starts from it"),
           ("pixel_um", "Pixel size", "µm", "The size of the camera's pixels: Astrometry starts from it"))
#: the shooting details the files do not say, in ``ACQUISITION``'s order: (key, name, tooltip)
SHOOTING = (("instrument", "Instrument", "The telescope or lens, as the Report and the text to paste give it"),
            ("mount", "Mount", "The mount, as the Report gives it"),
            ("filter", "Filter", "The filter in front of the camera, as the Report gives it"),
            ("location", "Location", "Where the frames were taken, as the Report gives it"),
            ("date", "Date", "The night of the shooting: the Report takes it from here, never from the camera's clock"))


def capture_text(t, key: str) -> str:
    return "" if getattr(t, key) is None else f"{getattr(t, key):g}"


def capture_typed(s, key: str, ed: QLineEdit):
    """A capture value typed (a decimal comma taken as a point); what is not a number is left as typed, not applied."""
    text = ed.text().replace(",", ".").strip()
    try:
        value = float(text) if text else None
    except ValueError:
        return
    if value != getattr(s.p.target, key):
        s.set_target_values(**{key: value})


def project_state_line(ps) -> QLabel:
    """What the Project block misses to be complete, in its colour (issue #89)."""
    lb = label("", wrap=True)
    set_project_state(lb, ps)
    lb.setToolTip(_("Red: no target; orange: shooting details missing; green: complete"))
    return lb


def set_project_state(lb: QLabel, ps):
    lb.setText("● " + ps.line)
    lb.setStyleSheet(f"color: {PROJECT_COLOURS[ps.level]}; font-weight: 600;")


class ProjectForm(QScrollArea):
    """The Project block's form, in the centre of Inspect (issue #89): its name, its target (its state as in its panel),
    the capture data, the shooting details the Report reads (the same values: one place in the project, the Report's
    fields and these stay in step) and the comments; at its top, in one line, what is missing for the block to be
    complete. A field applies at Enter or when it is left, as the panel's, through the model's setters (Undo, Save).
    Built once while it is shown: ``load`` puts the project's values back in the fields not being typed in, never
    building them again under the user's fingers. It scrolls on a small window."""

    WIDTH = 640                                          # the form's column, at most
    NAME_W = 140                                         # the column of the fields' names

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.s = s = project
        self.setObjectName("projectForm")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("QScrollArea#projectForm { background: transparent; } "
                           "QScrollArea#projectForm > QWidget > QWidget { background: transparent; }")
        host = QWidget()
        outer = QHBoxLayout(host)
        outer.setContentsMargins(16, 12, 16, 16)
        col = QWidget()                                  # centred, as wide as the room up to WIDTH
        col.setMaximumWidth(self.WIDTH)
        outer.addStretch(1)
        outer.addWidget(col, 100)
        outer.addStretch(1)
        self.v = v = QVBoxLayout(col)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        self.state = project_state_line(s.project_state())
        v.addWidget(self.state)
        form = self._form(_("Project"))
        self.name = self._field(form, _("Name"), _("The project's name: the top bar, the window's title and Home show it; "
                                                   "its folder keeps its own"))
        self.name.editingFinished.connect(self._name_typed)
        row = QHBoxLayout()
        self.dot = QLabel("●")
        row.addWidget(self.dot)
        self.target = QLineEdit()
        self.target.setToolTip(_("The object photographed: Astrometry starts from it, the Report takes it as its title"))
        self.target.editingFinished.connect(lambda: s.set_target_name(self.target.text()))
        row.addWidget(self.target, 1)
        form.addRow(self._name(_("Target")), row)
        under = QVBoxLayout()                            # what the target's state says, under its field
        under.setSpacing(4)
        self.note = label("", "small", wrap=True)
        under.addWidget(self.note)
        self.replace = QPushButton()
        self.replace.setToolTip(_("The object the solved field shows, in place of the target typed"))
        self.replace.clicked.connect(lambda: s.set_target_name(s.target_state().replace))
        under.addWidget(self.replace, 0, Qt.AlignLeft)
        form.addRow("", under)
        # what Astrometry starts from (the panel's Capture): named by its use, not "Prise de vue" next to the "Fiche de
        # prise de vue" below it in French
        form = self._form(_("Astrometry"))
        self.capture: dict[str, QLineEdit] = {}
        for key, word, unit, tip in CAPTURE:
            row = QHBoxLayout()
            e = QLineEdit()
            e.setPlaceholderText(_("optional"))
            e.setToolTip(_(tip))
            e.setFixedWidth(140)
            e.editingFinished.connect(lambda k=key, x=e: capture_typed(s, k, x))
            row.addWidget(e)
            row.addWidget(label(unit, "dim"))
            row.addStretch(1)
            form.addRow(self._name(_(word)), row)
            self.capture[key] = e
        form = self._form(_("Shooting details"))
        form.addRow(label(_("What the files do not say. Kept in the project."), "small", wrap=True))
        self.shooting: dict[str, QLineEdit] = {}
        for key, word, tip in SHOOTING:
            self.shooting[key] = e = self._field(form, _(word), _(tip))
            e.editingFinished.connect(lambda k=key, x=e: s.set_shooting_detail(k, x.text()))
        self._form(_("Comments"))
        self.comment = CommentEdit("")
        self.comment.setToolTip(_("Notes on this project"))
        self.comment.finished.connect(s.set_description)
        v.addWidget(self.comment)
        v.addStretch(1)
        self.setWidget(host)
        self.load()

    def _form(self, title: str) -> QFormLayout:
        """A section: its title as the panel's, then its labelled rows."""
        t = label(title)
        t.setStyleSheet("color: #aab2bf; font-weight: 600; margin-top: 10px; margin-bottom: 2px;")
        self.v.addWidget(t)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(6)
        self.v.addLayout(form)
        return form

    def _name(self, text: str) -> QLabel:
        """A field's name, all of one width: the fields of every section start at the same x."""
        n = label(text, "dim")
        n.setFixedWidth(self.NAME_W)
        return n

    def _field(self, form: QFormLayout, name: str, tip: str) -> QLineEdit:
        e = QLineEdit()
        e.setToolTip(tip)
        form.addRow(self._name(name), e)
        return e

    def _name_typed(self):
        if not self.s.set_project_name(self.name.text()):    # emptied: the name it had comes back
            self.name.setText(self.s.name)

    def load(self):
        """The project's values in the fields (not in the one being typed in), the target's state and what is missing."""
        s = self.s
        t, st = s.p.target, s.target_state()
        values = [(self.name, s.name), (self.target, t.name or "")] + \
            [(self.capture[k], capture_text(t, k)) for k in self.capture] + \
            [(self.shooting[k], s.p.acquisition.get(k, "")) for k in self.shooting]
        for ed, text in values:
            if not ed.hasFocus() and ed.text() != text:
                ed.setText(text)
        if not self.comment.hasFocus() and self.comment.toPlainText() != (s.p.description or ""):
            self.comment.setPlainText(s.p.description or "")
        self.target.setPlaceholderText(st.name or "NGC 7000")
        self.dot.setStyleSheet(f"color: {TARGET_DOTS[st.dot]}; font-size: 13pt;")
        self.note.setText(st.note)
        self.replace.setText(_("Replace by {name}").format(name=st.replace))
        self.replace.setVisible(bool(st.replace))
        set_project_state(self.state, s.project_state())


class CompactRow(SettingRow):
    """A setting on one line: name, value, back to the suggestion."""
    focused = Signal(dict)

    def __init__(self, r: dict, unit_w: int | None = None, parent=None):
        QFrame.__init__(self, parent)
        self.r = r
        if unit_w is None:
            unit_w = unit_column([r.get("unit")])
        self.setObjectName("row")
        self.setStyleSheet("QFrame#row { background: transparent; border-radius: 4px; }"
                           "QFrame#row:hover { background: #20242c; } QLabel { background: transparent; }")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        h = QHBoxLayout()
        h.setContentsMargins(4, 2, 2, 2)
        h.setSpacing(6)
        outer.addLayout(h)
        name = name_label(r["label"])
        name.setStyleSheet(f"color: {WARN};" if r["changed"] else "")
        h.addWidget(name, 1)
        spec_mark(h, r["kind"] == "new")
        ed = self._editor()
        self.editor = ed                                # the panel gives it the focus back after a rebuild (#24)
        back = r["changed"] and not r.get("locked")
        if isinstance(ed, QComboBox):                 # a choice: its own line, full width, never truncated
            ed.setToolTip(ed.currentText() if ed.isEnabled() else ed.toolTip())
            line = QHBoxLayout()
            line.setContentsMargins(4, 0, 2, 4)
            line.addWidget(ed, 1)
            line.addWidget(self._back(r, back))
            outer.addLayout(line)
            self.setToolTip(r.get("help") or "")
            return
        ed.setFixedWidth(EDITOR_W)
        h.addWidget(ed)
        h.addWidget(unit_label(r.get("unit") or "", unit_w))
        h.addWidget(self._back(r, back))
        self.setToolTip(r.get("help") or "")

    def _back(self, r, shown: bool) -> QPushButton:
        rb = icon_only(QPushButton(), "reset", _("Back to the suggested value: {v}").format(
            v=fmt(r["suggested"], r["type"], r.get("choices"))), 16, box=BACK_W)
        rb.clicked.connect(lambda: self.changed.emit(self.r, None))
        return retain(rb, shown)

    def mousePressEvent(self, e):                       # noqa: N802 - Qt's name
        self.focused.emit(self.r)
        super().mousePressEvent(e)


def durations_text(times, here: str, estimated: bool = False) -> str:
    """The duration of a method on each device it runs on, as rich text: "GPU ≈ 8 min · CPU ≈ 23 h", the device this
    machine uses in bold. ``times``: {"gpu"|"cpu": (seconds, measured)}, or None while the worker has not read them.
    ``estimated``: every figure is an estimate (the spec's methods the engine does not have yet)."""
    if times is None:
        return "…"
    parts = []
    here = here if here in times else "cpu"                  # a method that only runs on the CPU runs there
    for dev in ("gpu", "cpu"):
        if dev not in times:
            continue
        sec, measured = times[dev]
        d = hm(sec)
        t = html.escape(f"{_(dev.upper())} {'' if (measured and not estimated) or d.startswith('≤') else '≈ '}{d}")
        parts.append(f"<b style='color:#e6edf7'>{t}</b>" if dev == here else t)
    return " · ".join(parts)


class MethodCard(QFrame):
    """One way of computing a block: its name, what it does, how long it takes on each device it runs on (the device
    this machine uses in bold). ``times``: see ``durations_text``; ``here``: "gpu" or "cpu"."""
    clicked = Signal()

    def __init__(self, m, on, ok, times, here, why="", parent=None):
        super().__init__(parent)
        self.ok = ok
        self.setObjectName("meth")
        self.setCursor(Qt.PointingHandCursor if ok and not on else Qt.ArrowCursor)
        border = ACCENT if on else "#353b47"
        self.setStyleSheet(f"QFrame#meth {{ background: {'#2b3446' if on else '#20242c'}; border: 1px solid "
                           f"{border}; border-radius: 6px; }} QLabel {{ background: transparent; }}")
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 6, 10, 6)
        v.setSpacing(1)
        n = IconText("radio_on" if on else "radio_off", _(m["name"]), "selected" if on else "dim")
        n.label.setStyleSheet("font-weight: 600;" + ("" if ok else " color: #6b7280;"))
        v.addWidget(n)
        top = QHBoxLayout()
        top.setSpacing(6)
        spec_mark(top, m.get("new"))
        d = label(durations_text(times, here, bool(m.get("new"))), "small", wrap=True)
        d.setTextFormat(Qt.RichText)
        d.setToolTip(_("Duration on each device; in bold, the one this machine uses · ≈ an estimate"))
        top.addWidget(d, 1)
        v.addLayout(top)
        v.addWidget(label(_(m["what"]), "small", wrap=True))
        if why:
            v.addWidget(label(why, "warn", wrap=True))

    def mousePressEvent(self, e):                       # noqa: N802 - Qt's name
        if self.ok:
            self.clicked.emit()


class InfoDot(IconLabel):
    """A small "i" in a circle: what it is about shows in its tooltip."""

    def __init__(self, tip: str, parent=None):
        super().__init__("info", 16, "dim", parent=parent)
        self.setObjectName("infodot")
        self.setToolTip(tip)
        self.setCursor(Qt.WhatsThisCursor)


class StackMethodCard(QFrame):
    """A stacking method: its name, what it does; the engine's advice as a discreet "Recommended" outline in the accent
    colour, with an "i" whose tooltip says why. ``clicked`` asks for this method (the advice never chooses it)."""
    clicked = Signal()

    def __init__(self, key: str, on: bool, recommended: bool = False, why: str = "", enabled: bool = True,
                 parent=None):
        super().__init__(parent)
        from .settings import method_text
        self.key, self.on, self.recommended, self.ok = key, on, recommended, enabled and not on
        self.setObjectName("meth")
        self.setCursor(Qt.PointingHandCursor if self.ok else Qt.ArrowCursor)
        border = ACCENT if on else "#353b47"
        self.setStyleSheet(f"QFrame#meth {{ background: {'#2b3446' if on else '#20242c'}; border: 1px solid "
                           f"{border}; border-radius: 6px; }} QLabel {{ background: transparent; }}")
        name, what = method_text(key)
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 6, 10, 6)
        v.setSpacing(2)
        top = QHBoxLayout()
        top.setSpacing(6)
        n = IconText("radio_on" if on else "radio_off", name, "selected" if on else "dim")
        n.label.setWordWrap(False)
        n.label.setStyleSheet("font-weight: 600;" + ("" if enabled else " color: #6b7280;"))
        top.addWidget(n)
        top.addStretch(1)
        self.tag = self.info = None
        if recommended:
            self.tag = QLabel(_("Recommended"))
            self.tag.setObjectName("rec")
            self.tag.setStyleSheet(f"QLabel#rec {{ color: {ACCENT}; border: 1px solid {ACCENT}; border-radius: 8px; "
                                   "padding: 0px 7px; font-size: 8.5pt; }")
            top.addWidget(self.tag)
            if why:
                self.info = InfoDot(why)
                top.addWidget(self.info)
        v.addLayout(top)
        if what:
            v.addWidget(label(what, "small", wrap=True))

    def mousePressEvent(self, e):                       # noqa: N802 - Qt's name
        if self.ok:
            self.clicked.emit()


class Details(QFrame):
    """Everything about one setting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("det")
        self.setStyleSheet("QFrame#det { background: #191c22; border-top: 1px solid #2a2f38; } "
                           "QLabel { background: transparent; }")
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(12, 10, 12, 10)
        self.v.setSpacing(4)
        self.hide()

    def show_setting(self, r):
        clear(self.v)
        self.v.addWidget(label(r["label"], "h2"))
        if r["type"] in ("int", "float") and r.get("min") is not None and r.get("max") is not None:
            self.v.addWidget(RangeBar(r["min"], r["max"], r["suggested"], r["value"]))
        sug = fmt(r["suggested"], r["type"], r.get("choices"))
        self.v.addWidget(label(_("Suggested: {v}").format(v=sug) + (f" · {r['reason']}" if r.get("reason") else ""),
                               "reason", wrap=True))
        if r.get("help"):
            self.v.addWidget(label(r["help"], "small", wrap=True))
        if r.get("tip"):
            t = label(r["tip"], "small", wrap=True)
            t.setStyleSheet("color: #b9c2cf; font-size: 9pt;")
            self.v.addWidget(t)
        if r.get("drift"):
            self.v.addWidget(label(_("The current result used {v}: the suggested value changed since.").format(
                v=fmt(r.get("result_value"), r["type"], r.get("choices"))), "warn", wrap=True))
        if r.get("locked"):
            self.v.addWidget(label(_("Set by the options of the workflow."), "small", wrap=True))
        if r["kind"] == "new" and prefs.spec():
            self.v.addWidget(label("◇ " + _("Not in the engine yet."), "small"))
        self.show()


class Panel(QWidget):
    changed = Signal()                  # a setting changed (the window rebuilds the graph)
    openTune = Signal()                 # "Browse frames" and the like
    openPost = Signal()
    addSession = Signal()
    renamed = Signal(int, str)
    openLayer = Signal(int)             # Open in Post (a Post layer's id)
    removeTool = Signal()
    saveTemplate = Signal(int)              # issue #47: Save as template… of a workflow
    variantMade = Signal(int, int)      # new workflow, the one it is a variant of
    toast = Signal(str)                 # a word to say (what came with a change, why it was refused)
    cutToggled = Signal()               # the block's switch: cut it, or restore it (issue #43)
    selectBlock = Signal(str)           # a greyed block's reason: select the block that was cut
    restoreBlock = Signal(str)          # a greyed block's panel: restore the block that was cut
    sectionsChanged = Signal()          # its sections were built (issue #44: the strip of icons follows)

    def __init__(self, project, steps=None, tune=False, parent=None):
        super().__init__(parent)
        self.s, self.steps, self.tune = project, steps, tune
        self.corner = None                         # the » button that folds the panel (its page's, issue #44)
        self.expert = False
        self.focus_key = None
        self.node = None
        self._unit_w = 0                    # the unit column of the rows shown
        self.setObjectName("panel")
        self.setFixedWidth(390)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        head = QWidget()
        self.hv = QVBoxLayout(head)
        self.hv.setContentsMargins(14, 12, 14, 8)
        self.hv.setSpacing(6)
        v.addWidget(head)
        v.addWidget(hline())
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._sc = sc
        host = QWidget()
        self.bv = QVBoxLayout(host)
        self.bv.setContentsMargins(10, 8, 10, 8)
        self.bv.setSpacing(1)
        sc.setWidget(host)
        self.folds = Sections(self.bv, sc)              # its section titles fold (issue #44)
        v.addWidget(sc, 1)
        self.details = Details()
        v.addWidget(self.details)
        self._later = QTimer(self)
        self._later.setSingleShot(True)
        self._later.setInterval(0)
        self._later.timeout.connect(self._rebuild)
        project.paramsChanged.connect(self._params_arrived)
        if hasattr(project, "postChanged"):
            project.postChanged.connect(self._post_changed)
        for sig in (project.selectionChanged, project.readyChanged):
            sig.connect(self._again)

    def _again(self, *_a):
        """What the worker was asked for arrived: build the panel again (once, whatever the count of answers)."""
        if self.node is not None:
            self._later.start()

    def _post_changed(self, source):
        """Post changed the composition: a Stars and background block shown here shows it (not while one of its
        sliders is held, nor for a change made here)."""
        if source is not self and self.node is not None and self.node.key == "composition":
            self._later.start()

    def _params_arrived(self, wid: int, stage: str):
        """Settings arrived: only the panel that shows them builds itself again."""
        node = self.node
        if node is None or node.key in ("sources", "final", "composition") or not node.paths:
            return
        if any(q.pid == wid for q in node.paths) and stage in needed_stages(self.s, node.paths[0], node.ref):
            self._later.start()

    def _rebuild(self):
        """Not while a setting is being typed: the editor would go away under the user's fingers."""
        fw = QApplication.focusWidget()
        if fw is not None and self.isAncestorOf(fw) and isinstance(fw, QLineEdit | QAbstractSpinBox | QPlainTextEdit):
            self._later.start(400)
            return
        self.refresh(self.node)

    # ------------------------------------------------------------ build
    def refresh(self, node):
        # issue #24: built again at each change; the same block keeps its scroll and the setting being edited keeps
        # the focus (it was lost after Enter or Tab in a number field)
        same = (node is not None and self.node is not None and node.key == self.node.key
                and [q.pid for q in node.paths] == [q.pid for q in self.node.paths])
        fw = QApplication.focusWidget()
        had_focus = fw is not None and self.isAncestorOf(fw)
        scroll = self._sc.verticalScrollBar().value() if same else 0
        self._refresh(node)
        self._finish_sections()
        if same:
            # the scroll area as context, never the panel in the lambda: a panel closed meanwhile is not kept with its
            # model (GUI exploration 4: every closed project stayed in memory)
            QTimer.singleShot(0, self._sc, lambda v=scroll, sb=self._sc.verticalScrollBar(): sb.setValue(v))  # noqa: B008
            if had_focus and self.focus_key is not None:
                shown = [self.bv.itemAt(i).widget() for i in range(self.bv.count())]   # not the rows going away
                row = next((r for r in shown if isinstance(r, CompactRow) and r.r.get("key") == self.focus_key), None)
                ed = getattr(row, "editor", None)
                if ed is not None and ed.isEnabled():
                    QTimer.singleShot(0, ed.setFocus)

    def _refresh(self, node):
        clear(self.hv)
        clear(self.bv)
        self.folds.reset()
        self.details.hide()
        if node is not None and not self._alive(node):     # its block went away (Ctrl+Z, a removal): the page gives
            node = None                                     # the panel the block selected now
        self.node = node
        if node is None:
            return
        k = node.key
        if k == "project":
            self.project_panel()
        elif k == "align_images":
            self.align_panel(node)
        elif k == "session" or (k == "sources" and len(self.s.sessions) == 1):      # #48: even empty, its lists
            self.session_panel(node.sid if node.sid is not None else self.s.sessions[0]["id"])
        elif k == "sources" and self.s.sources_loaded and self.s.stack_mode == "together":
            self.all_nights()
        elif k == "sources":
            self.sources()
        elif k == "image":
            self.image_panel(node)
        elif k == "fusion":
            self.fusion_panel(node)
        elif node.cut or node.grey:
            self.cut_panel(node)
        elif k == "final":
            self.hv.addWidget(label(_("Post"), "h1"))
            from .blocks import Graph
            from .post3 import tr_name
            rows = []
            for name, pid in Graph(self.s).post_layers:
                q = self.s.view(pid) if pid is not None else None
                if q is None:
                    rows.append((html.escape(tr_name(name, True)), _("image")))
                    continue
                src = _("from {block}").format(block=block_title(q, q.end)) if q.end else ""
                st = _("Up to date") if q.rendered and q.computed else (_("Outdated") if q.rendered else _("Not computed yet"))
                rows.append((html.escape(tr_name(name)), " · ".join(x for x in (st, src) if x)))
            if rows:
                self.bv.addWidget(self.group_title(_("Layers")))
                self.bv.addWidget(kv(rows, pad=10))
                self.bv.addSpacing(8)
                b = QPushButton(_("Open Post"))
                b.setObjectName("primary")
                if not any(q.rendered for q in self.s.paths):   # opened as its double-click does (2nd round): Post
                    b.setToolTip(_("Nothing computed yet: Post lists the layers to come"))   # lists them

                b.clicked.connect(self.openPost.emit)
                self.bv.addWidget(b, 0, Qt.AlignLeft)
            else:                                         # no layer left: what makes one (control round)
                self.bv.addWidget(label(_("No layer: a block's image becomes one by “To Post” in its menu, or by "
                                          "pulling its ● onto Post."), "dim", wrap=True))
            self.bv.addStretch(1)
        elif k == "composition":
            self.composition(node)
        else:
            self.block(node)

    def _alive(self, node) -> bool:
        """Whether a block shown is still in the project: its session and its workflows (GUI exploration 1.6)."""
        if node.key in ("sources", "final", "project"):
            return True
        if node.sid is not None and node.sid not in {x["id"] for x in self.s.sessions}:
            return False
        if node.key == "session":
            return True
        alive = {q.pid for q in self.s.paths}
        return bool(node.paths) and all(q.pid in alive for q in node.paths)

    def align_panel(self, node):
        """Align images: the reference among its images, the images it takes."""
        s = self.s
        group = s.align_group(s.view(node.paths[0].pid))
        self.hv.addWidget(label(_("Align images"), "h1"))
        self.bv.addWidget(label(_(BY_KEY["align_images"]["what"]), "small", wrap=True))
        self.bv.addWidget(self.group_title(_("Reference")))
        cb = QComboBox()
        for q in group:
            cb.addItem(q.name, q.pid)
        cb.setCurrentIndex(cb.findData(group[0].source["align"]) if group else -1)
        cb.setToolTip(_("The image the others are brought onto: they take its size and framing"))
        cb.activated.connect(lambda _i: s.set_align_reference(group[0], cb.currentData()))
        self.bv.addWidget(cb)
        self.bv.addWidget(self.group_title(_("Images")))
        for q in group:
            self.bv.addWidget(label(q.name, "dim"))
        self.bv.addStretch(1)

    def project_panel(self):
        """The Project block: what it misses to be complete, the target and what is known of the capture, the comments;
        in Graph, the way to its whole form in Inspect (issue #89)."""
        s = self.s
        t, st = s.p.target, s.target_state()
        self.hv.addWidget(label(s.name, "h1"))
        self.hv.addWidget(project_state_line(s.project_state()))
        if not self.tune:
            b = QPushButton(_("Open form"))
            b.setToolTip(_("Opens the block in Inspect: its name, its target and the shooting details the Report reads"))
            b.clicked.connect(self.openTune.emit)
            self.hv.addWidget(b, 0, Qt.AlignLeft)
        self.bv.addWidget(self.group_title(_("Target")))
        row = QHBoxLayout()
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {TARGET_DOTS[st.dot]}; font-size: 13pt;")
        row.addWidget(dot)
        ed = QLineEdit(t.name or "")
        ed.setPlaceholderText(st.name or "NGC 7000")
        ed.editingFinished.connect(lambda: s.set_target_name(ed.text()))
        row.addWidget(ed, 1)
        self.bv.addLayout(row)
        self.bv.addWidget(label(st.note, "small", wrap=True))
        if st.replace:
            b = QPushButton(_("Replace by {name}").format(name=st.replace))
            b.clicked.connect(lambda: s.set_target_name(st.replace))
            self.bv.addWidget(b, 0, Qt.AlignLeft)
        self.bv.addWidget(self.group_title(_("Capture")))
        for key, word, unit, tip in CAPTURE:
            row = QHBoxLayout()
            row.addWidget(label(_(word), "dim"), 1)
            e = QLineEdit(capture_text(t, key))
            e.setPlaceholderText(_("optional"))
            e.setToolTip(_(tip))
            e.setFixedWidth(110)
            e.editingFinished.connect(lambda k=key, x=e: capture_typed(s, k, x))
            row.addWidget(e)
            u = label(unit, "dim")
            u.setFixedWidth(22)
            row.addWidget(u)
            self.bv.addLayout(row)
        self.bv.addWidget(self.group_title(_("Comments")))
        note = CommentEdit(s.p.description or "")
        note.finished.connect(s.set_description)
        self.bv.addWidget(note)
        self.bv.addStretch(1)

    def sources(self):
        s = self.s
        self.hv.addWidget(label(_("Frames"), "h1"))
        if not s.sources_loaded:
            b = QPushButton(_("Open frames"))
            b.setObjectName("primary")
            b.clicked.connect(self.openTune.emit)
            self.bv.addWidget(b, 0, Qt.AlignLeft)
            self.bv.addStretch(1)
            return
        self.bv.addWidget(self.group_title(_("Sessions")))
        for ses in s.sessions:
            darks = str(ses["darks"]) if ses["darks"] else _("none")
            self.bv.addWidget(kv([(html.escape(ses["night"]), _("{lights} lights · {flats} flats · darks: {darks}").format(
                lights=ses["lights"], flats=ses["flats"], darks=darks))], pad=10))
            if ses.get("rotation"):
                self.bv.addWidget(label(_("Camera rotated {deg}°: own flats").format(
                    deg=number(ses["rotation"])), "warn", wrap=True))
        self.bv.addSpacing(6)
        self.bv.addWidget(kv([(_("Excluded by hand"), str(len(s.frames_excluded))),
                              (_("Stacking"), _("all sessions together") if s.stack_mode == "together" else
                               _("one stack per session"))]))
        self.bv.addSpacing(8)
        row = QHBoxLayout()
        if not self.tune:
            b = QPushButton(_("Open frames"))
            b.clicked.connect(self.openTune.emit)
            row.addWidget(b)
        a = QPushButton(_("Add session…"))
        a.clicked.connect(self.addSession.emit)
        row.addWidget(a)
        row.addStretch(1)
        self.bv.addLayout(row)
        self.bv.addStretch(1)

    # ------------------------------------------------------------ issue #43: sessions, cut blocks, image source
    KIND_TITLES = (("lights", "Lights"), ("darks", "Darks"), ("flats", "Flats"), ("bias", "Bias"),
                   ("flat_darks", "Dark flats"))       # one name everywhere, as the import's (#49)

    def name_edited(self, ed, old: str, pid: int | None = None, sid: int | None = None):
        """A workflow's or a session's name typed in the panel (GUI exploration 2, decision 8): an empty name, or
        another's, is refused with the reason, the old one back in the field."""
        t = ed.text().strip()
        gone = (sid is not None and sid not in [x.id for x in self.s.p.sessions]) or \
            (pid is not None and self.s.view(pid) is None)
        if t == old or gone:                              # its session or workflow went (Ctrl+Z): nothing to name
            return
        why = self.s.name_problem(t, pid=pid, sid=sid)
        if why:
            ed.setText(old)
            self.toast.emit(_("Name refused: {why}").format(why=why))
        elif sid is not None:
            self.s.rename_session(sid, t)
        else:
            self.renamed.emit(pid, t)

    def session_panel(self, sid: int):
        """A Session block: its name, its filter, and what it holds kind by kind; its files are chosen in its Inspect
        (the one place for them)."""
        from ..core.project2 import FILTERS
        s = self.s
        ses = s.session(sid)
        counts = next((x for x in s.sessions if x["id"] == sid), {})
        top = QHBoxLayout()
        ed = QLineEdit(ses.name)
        ed.setStyleSheet("QLineEdit { font-size: 13pt; font-weight: 600; background: transparent; border: none; "
                         "border-bottom: 1px dashed #3a4150; border-radius: 0; } "
                         f"QLineEdit:focus {{ border-bottom: 1px solid {ACCENT}; }}")
        ed.setToolTip(_("Session name"))
        ed.editingFinished.connect(lambda: self.name_edited(ed, ses.name, sid=sid))
        top.addWidget(ed, 1)
        self.hv.addLayout(top)
        if not self.tune:                                 # as every block: its Delete under its name (expl. 2)
            acts = QHBoxLayout()
            x = QPushButton(_("Delete"))
            why = s.session_removal_problem(sid)
            x.setEnabled(not why)                         # greyed, with why (its menu says the same)
            x.setToolTip(_("Cannot be deleted: {why}").format(why=why) if why else
                         _("Take this session out of the project, its files kept (Del)"))
            x.clicked.connect(self.removeTool.emit)
            acts.addWidget(x)
            acts.addStretch(1)
            self.hv.addLayout(acts)
        row = QHBoxLayout()
        row.addWidget(label(_("Filter"), "dim"))
        cb = QComboBox()
        for f in FILTERS:
            cb.addItem(f or _("none (colour camera)"), f)
        cb.setCurrentIndex(max(0, cb.findData(ses.filter)))
        cb.setToolTip(_("The filter of these lights (a monochrome camera): the workflows by filter and their Combine channels block "
                        "use it; none for a colour camera."))
        cb.currentIndexChanged.connect(lambda _i: s.set_session_filter(sid, cb.currentData()))
        row.addWidget(cb)
        row.addStretch(1)
        self.hv.addLayout(row)
        n = {"lights": counts.get("lights", 0), "darks": counts.get("darks", 0), "flats": counts.get("flats", 0),
             "bias": counts.get("bias", 0), "flat_darks": counts.get("dark_flats", 0)}
        from .blocks import gone_kinds
        from .frames import type_warning_text
        gone = gone_kinds(s, {sid})
        warns = s.file_warnings(sid)                       # issue #58: files that may be mistakes, shown on their kind
        note = getattr(s, "colour_note", lambda _i: None)(sid)
        if note:                                           # issue #57: the limit of frames already in colour, said
            from .frames import colour_note_text
            lb = IconText("info", colour_note_text(note), "dim", "colour_note")
            self.bv.addWidget(lb)
        for kind, title in self.KIND_TITLES:
            if kind == "lights":
                kept = counts.get("kept", n["lights"])
                # a file the engine does not read (a JPEG added by mistake) is not used (GUI exploration 6.10)
                kept = max(kept - sum(w["n"] for w in warns if w["key"] == "unread" and w["kind"] == "lights"), 0)
                # nor a light that cannot be read (damaged, or not a FITS file: round 3, "5 / 5 used")
                kept = max(kept - getattr(s, "unreadable_lights", lambda _i: 0)(sid), 0)
                count = _("{kept} / {n} used").format(kept=kept, n=n["lights"])
            else:
                count = str(n[kind]) if n[kind] else _("none")
            if not n[kind] and kind in gone:              # not an empty folder: one not found (GUI exploration 4)
                count = _("folder not found")
            self.bv.addWidget(self.group_title(_(title) + "  ·  " + count))
            shared = s.shared_with(sid, kind) if kind != "lights" else []
            if shared:
                names = ", ".join(s.session(o).name for o in shared)
                self.bv.addWidget(IconText("link", _("shared with {names}: one master").format(names=names), "dim", "small"))
            for w in [w for w in warns if w["kind"] == kind]:      # issue #58: a file that may be a mistake
                self.bv.addWidget(IconText("warning", type_warning_text(w), "warning", "warn"))
        self.bv.addSpacing(8)
        if not self.tune:
            b = QPushButton(_("Open frames") if n["lights"] else _("Add lights"))
            b.setObjectName("primary")
            b.setToolTip(_("Opens its files in Inspect: add them there, kind by kind, and review its frames"))
            b.clicked.connect(self.openTune.emit)
            self.bv.addWidget(b, 0, Qt.AlignLeft)
        self.bv.addStretch(1)

    def all_nights(self):
        """A project made before issue #43: its nights are calibrated together. They can be calibrated each on its
        own (Calibration and what follows are computed again)."""
        s = self.s
        self.hv.addWidget(label(_("All nights"), "h1"))
        self.hv.addWidget(label(_("Calibrated together: one master dark and one master flat for every night."),
                                "small", wrap=True))
        self.bv.addWidget(self.group_title(_("Sessions")))
        for ses in s.sessions:
            self.bv.addWidget(kv([(html.escape(ses["night"]), _("{lights} lights · {flats} flats · darks: {darks}").format(
                lights=ses["lights"], flats=ses["flats"], darks=ses["darks"] or _("none")))], pad=10))
        self.bv.addSpacing(8)
        b = QPushButton(_("Calibrate per night"))
        b.setObjectName("primary")
        b.setToolTip(_("Each night with its own calibration frames, then all stacked together. Calibration and "
                       "what follows are computed again."))
        b.clicked.connect(self._separate)
        self.bv.addWidget(b, 0, Qt.AlignLeft)
        a = QPushButton(_("Add session…"))
        a.clicked.connect(self.addSession.emit)
        self.bv.addWidget(a, 0, Qt.AlignLeft)
        self.bv.addStretch(1)

    def _separate(self):
        self.s.set_stack_mode("per_session")
        self.changed.emit()
        self.toast.emit(_("Per night · to recompute · Ctrl+Z"))

    def image_panel(self, node):
        """The Image block of a workflow that starts from an image (issue #43)."""
        s = self.s
        p = node.paths[0]
        src = p.source or {}
        self.hv.addWidget(label(_("Image"), "h1"))
        acts = QHBoxLayout()                             # as every block: its Delete under its name
        x = QPushButton(_("Delete"))
        x.setToolTip(_("Take this block out, with the blocks after it (Del)"))
        x.clicked.connect(self.removeTool.emit)
        acts.addWidget(x)
        acts.addStretch(1)
        self.hv.addLayout(acts)
        path = str(src.get("path") or "")
        stretched = bool(src.get("stretched"))
        if path:
            f = Path(path)
            self.bv.addWidget(kv([(_("File"), html.escape(f.name)), (_("Folder"), html.escape(f.parent.as_posix()))],
                                 pad=10))
            from ..stages.image import image_problem
            why = image_problem(path)                     # moved, or its disk unplugged (GUI exploration 7.3)
            if why:
                self.bv.addWidget(label(_(why).format(name=f.name), "warn", wrap=True))
        else:
            self.bv.addWidget(label(_("No file yet: choose the image this block reads."), "warn", wrap=True))
        self.bv.addSpacing(6)
        row = QHBoxLayout()
        b = QPushButton(_("Change image…") if path else _("Choose image…"))
        b.setObjectName("primary" if not path else "")
        b.clicked.connect(lambda: self._choose_image(p))
        row.addWidget(b)
        row.addStretch(1)
        self.bv.addLayout(row)
        # its kind decides what can follow it: while nothing follows, either; then the one the blocks need
        alone = not s.graph(p)
        self.bv.addWidget(self.group_title(_("Kind")))
        row = QHBoxLayout()
        for value, text, tip in ((False, _("Linear"), _("Straight out of a stack: the blocks after it read it as a "
                                                         "stack")),
                                 (True, _("Stretched"), _("Already stretched: it goes to Post as it is"))):
            k = QPushButton(text)
            k.setCheckable(True)
            k.setObjectName("face")
            k.setChecked(value == stretched)
            k.setEnabled(alone or value == stretched)
            k.setToolTip(tip if alone else _("Fixed by the blocks that follow it"))
            k.clicked.connect(lambda _c=False, v=value: s.set_image_file(p, stretched=v))
            row.addWidget(k)
        row.addStretch(1)
        self.bv.addLayout(row)
        self.bv.addSpacing(6)
        if not stretched:
            self.bv.addWidget(label(_("The blocks after it read it as a stack. Denoise needs the two half-stacks of a "
                                      "stack: it is not offered here."), "small", wrap=True))
        self.bv.addStretch(1)

    def _choose_image(self, p):
        from . import prefs as _prefs
        from .gallery import image_filter
        f, _x = QFileDialog.getOpenFileName(self, _("Image"), _prefs.last_folder(), image_filter())
        if f:
            _prefs.remember_folder(f)
            self.set_image(p, f)

    def set_image(self, p, f: str) -> bool:
        """An Image block's file, refused with its reason when it is not an image read here (GUI exploration 7.3)."""
        from ..stages.image import image_problem
        why = image_problem(f)
        if why:
            self.toast.emit(_(why).format(name=Path(f).name))
            return False
        self.s.set_image_file(p, f)
        return True

    def fusion_panel(self, node):
        """A Combine channels block (issue #43): its workflow's name, its palette, its channels and the workflows they come
        from; its output is a layer of Post."""
        from ..stages.fusion import PALETTES
        s = self.s
        p = node.paths[0]
        f = (p.source or {}).get("fusion") or {}
        top = QHBoxLayout()
        ed = QLineEdit(p.name)
        ed.setStyleSheet("QLineEdit { font-size: 13pt; font-weight: 600; background: transparent; border: none; "
                         "border-bottom: 1px dashed #3a4150; border-radius: 0; } "
                         f"QLineEdit:focus {{ border-bottom: 1px solid {ACCENT}; }}")
        ed.editingFinished.connect(lambda: self.name_edited(ed, p.name, pid=p.pid))
        top.addWidget(ed, 1)
        st = self.graph_state(node)
        chip = QLabel(_(STATE_WORD[st]))
        chip.setStyleSheet(f"color: {STATE_COLOR[st]}; font-size: 9pt;")
        top.addWidget(chip)
        self.hv.addLayout(top)
        self.hv.addWidget(label(_("Combine channels"), "small"))
        acts = QHBoxLayout()                             # as the Image block: its Delete under its name (#34)
        x = QPushButton(_("Delete"))
        x.setToolTip(_("Take this block out, with the blocks after it (Del)"))
        x.clicked.connect(self.removeTool.emit)
        acts.addWidget(x)
        acts.addStretch(1)
        self.hv.addLayout(acts)
        if node.broken:                                  # why Run is refused, as the graph's red circle says (round 3)
            why = label(_("Run refused: {why}").format(why=node.broken.rstrip(".")), "warn", wrap=True)
            self.hv.addWidget(why)
        self.bv.addWidget(self.group_title(_("Palette")))
        roles = [i["role"] for i in f.get("inputs") or []]
        read = PALETTES.get(f.get("palette"), (tuple(roles), None))[0]     # the channels its palette reads (7.6)
        row = QHBoxLayout()
        for pal, (need, _m) in PALETTES.items():
            b = QPushButton(_(pal) if pal in ("RGB", "LRGB") else pal)    # RVB, LRVB in French; SHO... as they are
            b.setCheckable(True)
            b.setObjectName("face")
            b.setMinimumWidth(48)                          # the five fit the panel: they shrink below the style's 80 px
            b.setChecked(pal == f.get("palette"))
            ok = set(need) <= set(roles)
            b.setEnabled(ok)
            b.setToolTip(_("Needs the {roles} channels").format(roles=", ".join(need)) if not ok else "")
            b.clicked.connect(lambda _c=False, x=pal: s.set_fusion_palette(p, x))
            row.addWidget(b)
        row.addStretch(1)
        self.bv.addLayout(row)
        self.bv.addWidget(self.group_title(_("Channels")))
        words = {"L": _("grey"), "R": _("red"), "G": _("green"), "B": _("blue"),
                 "GB": _("green + blue")}
        weights = f.get("weights") or {}
        for i in [i for i in f.get("inputs") or [] if i["role"] in read]:
            q = s.view(i["pid"])
            row = QHBoxLayout()
            row.setSpacing(10)                             # the channel's name apart from its text (#27)
            role = label(f"<b>{html.escape(i['role'])}</b>", None)
            role.setMinimumWidth(36)
            row.addWidget(role, 0)
            row.addWidget(self.channel_source(p, i, q, words.get(i.get("channel", "L"), "")), 1)
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 5.0)
            sp.setSingleStep(0.05)
            sp.setDecimals(2)
            sp.setValue(float(weights.get(i["role"], 1.0)))
            sp.setToolTip(_("The channel's weight in the palette, after the balance (1: as balanced). Lower a "
                            "channel too faint, whose noise the balance raised."))
            sp.editingFinished.connect(lambda r=i["role"], b=sp: s.set_fusion_weight(p, r, b.value()))
            row.addWidget(sp)
            self.bv.addLayout(row)
        unread = [r for r in roles if r not in read]
        if unread:
            self.bv.addWidget(label(_("Unused by {palette}: {roles}").format(
                palette=f.get("palette", ""), roles=", ".join(unread)), "small", wrap=True))
        self.fusion_stars(p, f, [r for r in roles if r in read])
        self.bv.addSpacing(6)
        self.bv.addWidget(label(_("Each channel is its workflow's output, aligned on the first channel through the "
                                  "plate solutions (each channel's workflow needs its Astrometry) and brought to the "
                                  "same scale; the palette makes the colours. Its output is a layer of Post."),
                                "small", wrap=True))
        self.bv.addStretch(1)

    def channel_source(self, p, i: dict, q, word: str):
        """What a channel reads (issue #55): its workflow, chosen on its row among the stacks of its filter and the
        linear images; said only, when there is no other."""
        s = self.s
        choices = s.channel_choices(p, i["role"])
        if len(choices) < 2:
            return label(_("{workflow} · {channel}").format(workflow=html.escape(q.name if q is not None else "?"),
                                                            channel=word), "small")
        cb = QComboBox()
        cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)     # as the Stars list below
        cb.setMinimumContentsLength(10)
        for x, why in choices:
            cb.addItem(_("{workflow} · {channel}").format(workflow=x.name, channel=word), x.pid)
            if why:
                item = cb.model().item(cb.count() - 1)
                item.setEnabled(False)
                item.setToolTip(why)
        cb.setCurrentIndex(cb.findData(i["pid"]))
        cb.setToolTip(_("The workflow this channel reads: a stack of its filter, or a linear image's output"))
        cb.activated.connect(lambda k, r=i["role"]: s.set_fusion_channel(p, r, cb.itemData(k)))
        return cb

    def fusion_stars(self, p, f: dict, roles: list[str]):
        """Option A: the palette on the channels' starless images, then the stars from a source, put back by the
        Combine layers block."""
        s = self.s
        cur = f.get("stars") or {}
        self.bv.addWidget(self.group_title(_("Stars")))
        choices = [(_("With channels (no split)"), None, ""),
                   (_("Channel stars, same palette"), {"mode": "palette"}, "")]
        choices += [(_("{channel} stars (neutral)").format(channel=r), {"mode": "channel", "role": r}, "")
                    for r in roles]
        choices += [(_("“{workflow}” stars (own colours)").format(workflow=q.name),
                     {"mode": "workflow", "pid": q.pid}, why) for q, why in s.star_choices(p)]
        cb = QComboBox()
        cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)     # a long workflow name: elided,
        cb.setMinimumContentsLength(20)                                              # never wider than the panel
        for text, val, why in choices:
            cb.addItem(text, val)
            if why:                                      # issue #55: an image already stretched, greyed with its reason
                item = cb.model().item(cb.count() - 1)
                item.setEnabled(False)
                item.setToolTip(why)
        cb.setCurrentIndex(next((i for i, (_t, v, _w) in enumerate(choices) if (v or {}) == cur), 0))
        cb.setToolTip(_("The palette is made on the channels' starless images; these stars are put back by the "
                        "Combine layers block. Each channel's workflow gets Stars/background."))
        cb.activated.connect(lambda i: s.set_fusion_stars(p, cb.itemData(i)))
        self.bv.addWidget(cb)
        if cur:
            self.bv.addWidget(label(_("The stars are taken off in each channel first: stars of different sizes from "
                                      "one filter to another leave no coloured rings."), "small", wrap=True))

    def cut_panel(self, node):
        """A block the user cut, or greyed by a cut: what it means, and the way back."""
        s = self.s
        p = s.path if s.path in node.paths else node.paths[0]
        top = QHBoxLayout()
        top.addWidget(label(block_title(p, node.key), "h1", wrap=True), 1)
        self.hv.addLayout(top)
        self.hv.addWidget(label(p.name if len(node.paths) == 1 else all_workflows(len(node.paths)), "small"))
        if node.cut:
            self.bv.addWidget(label(_("Switched off: this block does not change the data, which passes through to "
                                      "the next block. Its settings are kept."), None, wrap=True))
            b = QPushButton(_("Switch on") + "  (D)")
            b.setObjectName("primary")
            b.clicked.connect(self.cutToggled.emit)
            self.bv.addSpacing(6)
            self.bv.addWidget(b, 0, Qt.AlignLeft)
        else:
            self.bv.addWidget(label(_("Blocked: {why} · image passes through").format(
                why=node.grey), "warn", wrap=True))
            cause = {_("needs Astrometry"): "astrometrie", _("needs Denoise"): "debruitage"}.get(node.grey)
            if cause and cause in s.display_keys(p):
                self.bv.addSpacing(6)
                if s.is_cut(p, cause):
                    r = QPushButton(_("Switch on {block}").format(block=block_title(None, cause)))
                    r.setObjectName("primary")
                    r.clicked.connect(lambda: self.restoreBlock.emit(cause))
                    self.bv.addWidget(r, 0, Qt.AlignLeft)
                b = QPushButton(_("Go to {block}").format(block=block_title(None, cause)))
                b.clicked.connect(lambda: self.selectBlock.emit(cause))
                self.bv.addWidget(b, 0, Qt.AlignLeft)
        self.bv.addStretch(1)

    def block(self, node):
        s = self.s
        p = s.path if s.path in node.paths else node.paths[0]
        k = node.key
        top = QHBoxLayout()
        if k == "post":
            ed = QLineEdit(p.name)
            ed.setStyleSheet("QLineEdit { font-size: 13pt; font-weight: 600; background: transparent; border: none; "
                             "border-bottom: 1px dashed #3a4150; border-radius: 0; } "
                             f"QLineEdit:focus {{ border-bottom: 1px solid {ACCENT}; }}")
            ed.setToolTip(_("Workflow name"))
            ed.editingFinished.connect(lambda: self.name_edited(ed, p.name, pid=p.pid))
            top.addWidget(ed, 1)
        else:
            from .blocks import node_title
            title = node_title(s, node) if node.paths else block_title(p, k)   # a block placed twice: its number
            if node.sid is not None:
                title += " · " + s.session(node.sid).name
            top.addWidget(label(title, "h1", wrap=True), 1)     # a long name (#36) wraps, never cut
        st = self.graph_state(node)
        chip = QLabel(_(STATE_WORD[st]))
        chip.setStyleSheet(f"color: {STATE_COLOR[st]}; font-size: 9pt;")
        top.addWidget(chip)
        kb = k.partition("~")[0]
        graph_block = kb not in ("sources", "final", "session", "composition", "fusion", "image", "post") and \
            (node.sid is None or kb == "calibration") and not self.tune      # a session's Calibration too (expl. 2)
        acts = QHBoxLayout()                             # the block's actions, under its name (issue #47)
        acts.setSpacing(6)
        if graph_block:                                  # any block can go when what follows allows it
            x = QPushButton(_("Delete"))
            x.setToolTip(_("Take this block out of the workflow (Del)") if node.sid is None else
                         _("Take the Calibration out of every session of the workflow (Del)"))
            x.clicked.connect(self.removeTool.emit)
            acts.addWidget(x)
        if graph_block:                                   # question 7: greyed with its reason, as in the menu
            why_off = "" if node.cut else s.can_switch_off(p, k)
            cut = QPushButton(_("Switch on") if node.cut else _("Switch off"))
            cut.setToolTip(_("Cannot be switched off: {why}").format(why=why_off) if why_off else
                           _("Switch this block off (D): the data passes through it; its settings are kept"))
            cut.setEnabled(not why_off)
            cut.clicked.connect(self.cutToggled.emit)
            acts.addWidget(cut)
        if k == "post" and not self.tune:                # the workflow as a template of the gallery
            tpl = QPushButton(_("Save template…"))
            tpl.setToolTip(_("Keep this workflow's graph as a template of the gallery"))
            tpl.clicked.connect(lambda: self.saveTemplate.emit(p.pid))
            acts.addWidget(tpl)
        self.hv.addLayout(top)
        if acts.count():
            acts.addStretch(1)
            self.hv.addLayout(acts)
        # which workflows a change goes to (the Stars block has no setting: nothing to apply)
        if node.shared and k != "etoiles":
            self.hv.addWidget(label(_("Apply to:"), "dim"))
            box = QVBoxLayout()
            box.setSpacing(2)
            allp = [q.pid for q in node.paths]
            cur = s.scope if s.scope and set(s.scope) <= set(allp) else allp
            choices = [(all_workflows(len(node.paths)), allp)] + \
                      [(_('"{name}" only').format(name=q.name), [q.pid]) for q in node.paths]
            if len(choices) > 4:                       # many workflows: a list instead of a column of buttons
                cb = QComboBox()
                for text, pids in choices:
                    cb.addItem(text, pids)
                cb.setCurrentIndex(next((i for i, (_t, ps) in enumerate(choices) if sorted(ps) == sorted(cur)), 0))
                cb.currentIndexChanged.connect(lambda i: self._scope(cb.itemData(i)))
                box.addWidget(cb)
                choices = []
            for text, pids in choices:
                b = QPushButton(text)
                b.setObjectName("scope")
                b.setCheckable(True)
                b.setChecked(sorted(cur) == sorted(pids))
                b.setStyleSheet("QPushButton#scope { text-align: left; padding: 4px 10px; }"
                                f"QPushButton#scope:checked {{ background: #2b3446; border-color: {ACCENT}; "
                                "color: #ffffff; font-weight: 600; }")
                b.clicked.connect(lambda _x=False, ps=pids: self._scope(ps))
                box.addWidget(b)
            self.hv.addLayout(box)
        elif k != "post":
            self.hv.addWidget(label(p.name, "small"))
        self.variant_row(p)
        self.choice_box(node, p)
        # measurements, in Inspect
        if self.tune and self.steps is not None and st != "pending":
            facts = self.steps.facts(p, k)
            if facts:
                self.bv.addWidget(self.group_title(_("Measurements")))
                self.bv.addWidget(kv(facts, pad=10))
                self.bv.addSpacing(6)
        if k == "calibration":              # a Calibration placed in parallel: its own frames
            from .ownframes import own_frames
            own_frames(self, node, p)
        if k == "etoiles":                  # what the block gives first: its two images, and where they go
            self.outputs(node)
        if k.partition("~")[0] in ("fond", "debruitage", "etoiles", "deconv", "synth"):     # Gradient 2 too (round 3)
            self.methods(p, k)
        if k == "fond" and self.tune and getattr(self, "grad_editor", None) is not None:
            self.bv.addWidget(self.group_title(_("Samples · exclusions")))
            self.bv.addWidget(self.grad_editor.controls(p))
        elif k == "fond" and not self.tune:            # issue #15: the samples are edited on the image, in Inspect
            b = QPushButton(_("Edit samples…"))
            b.setToolTip(_("Opens the block in Inspect: place, move or remove the samples and the excluded areas on "
                           "the image"))
            b.clicked.connect(self.openTune.emit)
            self.bv.addWidget(b, 0, Qt.AlignLeft)
            self.bv.addSpacing(6)
        sk = node.ref                                  # a session's block: that session's settings (issue #43)
        groups = self._groups(p, sk)
        units = [r.get("unit") for _t, rs in groups for r in rs]          # hidden rows too: the column stays put
        if k == "selection":
            units += [_("% of the best"), "px", _("× median")]
        self._unit_w = unit_column(units)
        extra = 0
        if k == "selection":
            groups, extra = self.selection_rule(p, groups)
            self.selection(p)
        if k == "empilement" and len(s.sessions) > 1:
            self.stack_sessions()
        if k == "empilement":
            self.stack_methods(p)
        if k == "post" and p.end:
            self.bv.addWidget(kv([(_("Output of"), block_title(p, p.end))]))
        if k == "etoiles":
            self.bv.addWidget(label(_("No setting here: the network is applied as it was trained. Its two images are "
                                      "adjusted in a Combine layers block, or each as a layer of Post."),
                                    "small", wrap=True))
        else:
            self.settings(p, sk, groups, extra)
        self.bv.addStretch(1)

    def stack_sessions(self):
        """Several nights: one stack of all (one per night is ◇: the engine stacks all the sessions together)."""
        self.bv.addWidget(self.group_title(_("Sessions")))
        row = QHBoxLayout()
        for text, mode in ((_("All together"), "together"), (_("One per session"), "per_session")):
            if mode == "per_session" and not prefs.spec():
                continue
            b = QPushButton(text + ("  ◇" if mode == "per_session" else ""))
            b.setCheckable(True)
            b.setObjectName("face")
            b.setChecked(self.s.stack_mode == mode)
            b.setEnabled(mode == "together")
            if mode == "per_session":
                b.setToolTip("◇ " + _("Not in the engine yet."))
            row.addWidget(b)
        row.addStretch(1)
        self.bv.addLayout(row)

    # ------------------------------------------------------------ replace or try as a variant
    def variant_row(self, p):
        """A variant says what it is a variant of, and is kept or discarded from here."""
        src = self.s.view(p.variant_of) if p.variant_of is not None else None
        if src is None:
            return
        row = QHBoxLayout()
        row.addWidget(label(_('Variant of "{name}"').format(name=src.name), "dim"), 1)
        keep = QPushButton(_("Keep"))
        keep.setToolTip(_('Replaces "{name}" and takes its name').format(name=src.name))
        # what each did, said with Ctrl+Z (control round, decision 11)
        keep.clicked.connect(lambda: (self.s.keep_variant(p), self.changed.emit(), self.toast.emit(
            _("Variant “{name}” kept: it replaces “{orig}” · Ctrl+Z").format(name=p.name, orig=src.name))))
        drop = QPushButton(_("Discard"))
        drop.clicked.connect(lambda: (self.s.drop_variant(p), self.changed.emit(), self.toast.emit(
            _("Variant “{name}” discarded · Ctrl+Z").format(name=p.name))))
        row.addWidget(keep)
        row.addWidget(drop)
        self.hv.addLayout(row)

    def choice_box(self, node, p):
        ch = self.s.pending_change
        if not ch or ch["step"] != node.key:
            return
        s = self.s
        paths = [q for q in s.paths if q.pid in ch["pids"]]
        f = QFrame()
        f.setObjectName("choice")
        f.setStyleSheet(f"QFrame#choice {{ background: #2a2316; border: 1px solid {WARN}; border-radius: 6px; }}"
                        " QLabel { background: transparent; }")
        v = QVBoxLayout(f)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(6)
        v.addWidget(label(f"<b>{html.escape(ch['what'])}</b>", wrap=True))
        c = s.cost_of_change(paths, ch)
        rep = QPushButton(_("Replace") + f"  ·  {'' if c.measured else '≈ '}{hm(c.seconds)}")
        rep.setToolTip(_("The current results of {names} are computed again").format(
            names=", ".join(q.name for q in paths)))
        rep.clicked.connect(self._replace)
        v.addWidget(rep)
        cv = s.cost_of_change([p], ch)
        var = QPushButton(_("Try as variant") + f"  ·  {'' if cv.measured else '≈ '}{hm(cv.seconds)}")
        var.setObjectName("primary")
        var.setToolTip(_('"{name}" stays as it is; the variant shares everything up to this block').format(
            name=p.name))
        var.clicked.connect(lambda: self._variant(p))
        v.addWidget(var)
        cancel = QPushButton(_("Cancel"))
        cancel.clicked.connect(self._cancel)
        v.addWidget(cancel, 0, Qt.AlignLeft)
        self.hv.addWidget(f)

    def propose(self, stage, name, value, suggested, what, short):
        """A change of setting: made at once when nothing computed is lost; otherwise Replace, or Try as variant."""
        self._propose(dict(kind="param", stage=stage, name=name, value=value, suggested=suggested), what, short)

    def propose_opt(self, key, value, what, short):
        """A change of option, such as a method."""
        self._propose(dict(kind="opt", key=key, value=value), what, short)

    def _propose(self, ch, what, short):
        s = self.s
        paths = s.scope_paths()
        try:
            if s.throws_away(paths, ch):
                s.pending_change = dict(ch, what=what, short=short, step=self.node.key, pids=[q.pid for q in paths])
            else:
                s.pending_change = None
                notes = s.apply_change(paths, ch)
                s.log("{what} · {names}", what=what, names=", ".join(q.name for q in paths))
                if notes:
                    self.toast.emit(" · ".join(_(n) for n in notes))
        except NotInEngine:
            s.pending_change = None
            self.toast.emit("◇ " + _("Not in the engine yet."))
        except (ValueError, wfl.OptionRefused) as e:
            s.pending_change = None
            self.toast.emit(_(str(e)))
        self.changed.emit()

    def _replace(self):
        ch, self.s.pending_change = self.s.pending_change, None
        paths = [q for q in self.s.paths if q.pid in ch["pids"]]
        try:
            notes = self.s.apply_change(paths, ch)
        except (ValueError, wfl.OptionRefused) as e:
            self.toast.emit(_(str(e)))
            self.changed.emit()
            return
        self.s.log("{what} · replaced in {names}", what=ch["what"], names=", ".join(q.name for q in paths))
        said = _("{what} · replaced in {names}").format(what=ch["what"], names=", ".join(q.name for q in paths))
        self.toast.emit(" · ".join([said] + [_(n) for n in notes] + ["Ctrl+Z"]))     # said, as the others (decision 11)
        self.changed.emit()

    def _variant(self, p):
        ch, self.s.pending_change = self.s.pending_change, None
        try:
            q = self.s.add_variant(p, ch, f"{p.name} · {ch['short']}")
        except (ValueError, wfl.OptionRefused) as e:
            self.toast.emit(_(str(e)))
            self.changed.emit()
            return
        self.s.log('Variant "{name}" created', name=q.name)
        self.variantMade.emit(q.pid, p.pid)

    # ------------------------------------------------------------ the method of a block
    def methods(self, p, k):
        """The ways of computing this block, as cards: what it does, where it runs, how long on this machine."""
        s = self.s
        from .model import METHOD_OPT
        cur = method_of(p, k)                           # a block placed twice ("fond~2"): its own node's (round 3)
        again, k = "~" in k or k == "synth", k.partition("~")[0]          # Synthetic stars: a block of the graph only
        self.bv.addWidget(self.group_title(_("Method")))
        opt = METHOD_OPT[k]
        spec = prefs.spec()
        for m in methods_of(k):
            if m["new"] and not spec:
                continue
            on = m["key"] == cur
            ok = (s.gpu or m["runs"] != "gpu") and not m["new"]
            times = s.method_times(p, k, m["key"])          # read by the worker: None until it has
            why = ""
            if m["new"]:
                why = "◇ " + _("Not in the engine yet.")
            elif not ok:
                t = hm(times["cpu"][0]) if times and "cpu" in times else "…"
                why = _("Needs a GPU (≈ {t} on the CPU).").format(t=t)
                if on:
                    why = _("Computed earlier on a GPU; running it again here takes ≈ {t}.").format(t=t)
            if k == "synth" and m["key"] == "instrument" and not on and hasattr(s, "state_info") and \
                    s.state_info(p, "astrometry") is None:      # the night test pass: chosen, it failed in 1 s
                ok, why = False, _("Needs the image's scale: place the Astrometry block in this workflow.")
            card = MethodCard(m, on, ok, times, "gpu" if s.gpu else "cpu", why)
            if not on and ok and again:
                card.clicked.connect(lambda ek=m["engine_key"], nm=m["name"]: self._node_method(ek, _(nm)))
            elif not on and ok:
                card.clicked.connect(lambda mk=m["key"], nm=m["name"], sh=m["short"]: self.propose_opt(
                    opt, method_value(k, mk), _("Method: {name}").format(name=_(nm)), _(sh)))
            self.bv.addWidget(card)

    def _node_method(self, method: str, name: str):
        """The method of a block placed twice, chosen on its cards: stored on its node, one Undo step, said (round 3:
        its Method list was greyed "Set by the options of the workflow")."""
        from .blocks import node_title
        node = self.node
        if node is None or not node.paths:
            return
        if self.s.set_node_method(node.paths, node.key, method):
            self.toast.emit(_("{block}: {method} · Ctrl+Z").format(block=node_title(self.s, node), method=name))
        self.changed.emit()

    def stack_methods(self, p):
        """The stacking methods the engine offers, as cards; the engine's advice marks one "Recommended" (with its
        reason under an "i"), and never changes the method by itself. Below, only the settings of the method shown."""
        from .settings import STACK_METHOD, colour_frames, method_advice, method_text, stack_methods
        keys = stack_methods(self.s, p)
        if colour_frames(self.s, p):                       # issue #57: said, not silently left out
            self.bv.addWidget(self.group_title(_("Method")))
            why = label(_("Classic stack: Bayer drizzle does not apply to frames already in colour (no mosaic of raw "
                          "colour samples to drop)."), "small", wrap=True)
            why.setObjectName("colour_no_drizzle")
            self.bv.addWidget(why)
            return
        if len(keys) < 2:
            return
        info = self.s.param_info(p, *STACK_METHOD)
        advice = method_advice(self.s, p)
        self.bv.addWidget(self.group_title(_("Method")))
        for key in keys:
            rec = advice is not None and advice[0] == key
            card = StackMethodCard(key, key == info.value, rec, advice[1] if rec else "", not info.locked)
            if card.ok:
                name = method_text(key)[0]
                card.clicked.connect(lambda k_=key, nm=name: self.propose(
                    STACK_METHOD[0], STACK_METHOD[1], k_, None, _("Stacking method: {name}").format(name=nm),
                    nm.lower()))
            self.bv.addWidget(card)
        if advice is None and self.s.state(p, "register") != "done":
            # the advice reads the registration (star size, coverage of the frames kept): none without it
            why = label(_("The recommended method shows once the registration is computed."), "small", wrap=True)
            why.setObjectName("advice_pending")
            self.bv.addWidget(why)

    def _cancel(self):
        self.s.pending_change = None
        self.changed.emit()

    def outputs(self, node):
        """The Stars block's two images, in the colours of its ports, and where each goes (its Stars and background
        blocks, Post as it is), or how to send it."""
        from .blocks import STAR_OUTPUTS, Graph
        from .graph import IMAGE_COL
        g = Graph(self.s)
        mine = [n for n in g.compositions if n.parent is not None and n.parent._bk == node._bk]
        self.bv.addWidget(self.group_title(_("Outputs")))
        names = {"starless": _("Starless image"), "stars": _("Stars alone")}
        for what, _word in STAR_OUTPUTS:
            to = [block_title(n.paths[0], "composition") + (f" ({n.paths[0].name})" if len(node.paths) > 1 else "")
                  for n in mine]
            if any(src._bk == node._bk and part == what for src, part in g.direct):
                to.append(_("Post layer"))
            to = to or [_("unused: drag to Post or Composition")]
            lb = QLabel(f"<span style='color:{IMAGE_COL[what]}'>●</span> <b>{html.escape(names[what])}</b>"
                        + "".join(f"<br>&nbsp;&nbsp;&nbsp;→ {html.escape(t)}" for t in to))
            lb.setTextFormat(Qt.RichText)
            lb.setWordWrap(True)
            if len(node.paths) == 1:            # issue #8: sent to Post from here too, not only by dragging its dot
                row = QHBoxLayout()
                row.addWidget(lb, 1)
                p = node.paths[0]
                sent = any(x.get("pid") == p.pid and x.get("part") == what for x in self.s.post_layers())
                if sent:                                     # issue #29: a link to Post can be taken back
                    b = QPushButton(_("Out of Post"))
                    b.setToolTip(_("Take this image out of Post (its layers and their filters)"))
                    b.clicked.connect(lambda _c=False, w=what, pp=p: self._part_out_of_post(pp, w))
                else:
                    b = QPushButton(_("To Post"))
                    b.setToolTip(_("This image as a layer of Post, as it is (no Combine layers "
                                   "block)"))
                    b.clicked.connect(lambda _c=False, w=what, pp=p: self._part_to_post(pp, w))
                row.addWidget(b, 0, Qt.AlignTop)
                self.bv.addLayout(row)
            else:
                self.bv.addWidget(lb)
        if len(node.paths) == 1:
            nb = set_icon(QPushButton(_("Combine layers").replace("&", "&&")), "add", 16)
            nb.setToolTip(_("A layer of Post that puts the stars back on the starless image; each of the two keeps "
                            "its own settings"))
            nb.clicked.connect(lambda _c=False, pp=node.paths[0]: self._new_composition(pp))
            self.bv.addWidget(nb, 0, Qt.AlignLeft)
        self.bv.addSpacing(6)

    def _part_to_post(self, p, part):
        """One of the Stars block's images sent to Post as a layer of its own (as dragging its dot onto Post)."""
        from .blocks import STAR_OUTPUTS
        self.s.add_post_layer(p.pid, part, self)
        self.s.log('{image} of "{name}" sent to Post', image=dict(STAR_OUTPUTS)[part], name=p.name)
        self.s.select(index=self.s.paths.index(p), step="etoiles")
        self.toast.emit(_('{image} · "{name}" to Post').format(image=_(dict(STAR_OUTPUTS)[part]), name=p.name)
                        + " · Ctrl+Z")                    # said, as the graph's menu (decision 11)

    def _part_out_of_post(self, p, part):
        """One of the Stars/background block's images sent as it is, taken out of Post (#29)."""
        from .blocks import STAR_OUTPUTS
        if self.s.remove_post_part(p.pid, part, self):
            self.s.log('{image} of "{name}" taken out of Post', image=dict(STAR_OUTPUTS)[part], name=p.name)
            self.toast.emit(_('{image} of "{name}" taken out of Post').format(
                image=_(dict(STAR_OUTPUTS)[part]), name=p.name) + " · Ctrl+Z")
        self.s.select(index=self.s.paths.index(p), step="etoiles")

    def _new_composition(self, p):
        """A Stars and background block for this workflow's Stars block (as dragging a dot onto Composition)."""
        from .post3 import tr_name
        lid = self.s.add_post_layer(p.pid, "combined", self)
        self.s.log('Combine layers for "{name}"', name=p.name)
        self.s.sel_layer = lid
        self.s.select(index=self.s.paths.index(p), step="composition")
        self.s.scope = [p.pid]
        lay = self.s.post_layer(lid) or {}
        self.toast.emit(_('Layer "{name}" ({block}) to Post').format(
            name=tr_name(str(lay.get("name") or p.name)), block=block_title(None, "composition")) + " · Ctrl+Z")

    def _one_step(self, change):
        """A change of a Combine layers block whose layer is made real by it: one undo step (decision 7)."""
        with self.s.batch():
            change()

    def composition(self, node):
        """A Stars and background block: its layer of Post, recombining the Stars block's two images. How the stars go
        back and the sliders, the same as in Post; Open in Post. A workflow whose Post was never opened gets its layer
        here (the one Post would give it)."""
        from ..render.sliders import STARS as STARS_GROUP
        from ..render.sliders import group_title
        from .post3 import BLENDS, ContSlider, offset_text
        s = self.s
        p = node.paths[0]
        def lid():                                        # the layer is made at the first change, not by showing the
            if node.lid is None:                          # block (GUI exploration 3.6: a click made a layer)
                node.lid = s.add_post_layer(p.pid, "combined", self, keep_default=False)   # it is the default one
            s.sel_layer = node.lid
            return node.lid
        if node.lid is not None:
            s.sel_layer = node.lid
        lay = (s.post_layer(node.lid) if node.lid is not None else None) or {}
        top = QHBoxLayout()
        from .blocks import node_title
        top.addWidget(label(node_title(s, node), "h1", wrap=True), 1)        # "Combine layers 2": as the graph names it
        st = block_state(s, p, "etoiles")
        chip = QLabel(_(STATE_WORD[st]))
        chip.setStyleSheet(f"color: {STATE_COLOR[st]}; font-size: 9pt;")
        top.addWidget(chip)
        x = QPushButton(_("Delete"))
        x.setToolTip(_("Take this block out of Post (the Stars/background block's images stay computed)"))
        x.clicked.connect(self.removeTool.emit)
        top.addWidget(x)
        self.hv.addLayout(top)
        self.hv.addWidget(label(p.name, "small"))
        from .blocks import BY_KEY
        self.bv.addWidget(label(_(BY_KEY["composition"]["what"]), "small", wrap=True))
        if st != "done":
            self.bv.addWidget(label(_("To compute: Stars/background"),
                                    "warn", wrap=True))
        sliders = dict(lay.get("sliders") or {})

        def commit(key, v):
            if abs(v - 2) < 0.02:
                sliders.pop(key, None)
            else:
                sliders[key] = v
            with s.batch():                               # its layer made real and the value: one undo step (decision 7)
                s.set_post_layer(lid(), self, sliders=dict(sliders))
        group = None
        infos = [x for x in s.slider_infos(p, "combined") if x.available]     # the stars first: what the block is for
        for sd in sorted(infos, key=lambda x: x.group != STARS_GROUP):
            if sd.group != group:
                group = sd.group
                self.bv.addWidget(self.group_title(_(group_title(group, "combined"))))
                if group == STARS_GROUP:
                    row = QHBoxLayout()
                    row.addWidget(label(_("Star blend mode"), "dim"))
                    cb = QComboBox()
                    for b in BLENDS:
                        cb.addItem(_(b), b)
                    cb.setToolTip(_("How the stars go back on the starless image: Screen (the usual way), Linear "
                                    "dodge (add) or Lighten"))
                    cb.setCurrentIndex(max(0, BLENDS.index(lay.get("blend") or BLENDS[0])))
                    cb.currentIndexChanged.connect(lambda i, cb=cb: self._one_step(
                        lambda: s.set_post_layer(lid(), self, blend=cb.itemData(i))))
                    row.addWidget(cb, 1)
                    self.bv.addLayout(row)
            sl = ContSlider(_(sd.name), 0, 4, sliders.get(sd.key, 2.0), 2, ends=[_(e) for e in sd.ends],
                            show=offset_text)
            ok, why, _need = s.slider_state(p, sd.key, "combined")
            sl.set_available(ok, why)
            sl.done.connect(lambda v, k=sd.key: commit(k, v))
            self.bv.addWidget(sl)
        self.bv.addSpacing(8)
        b = QPushButton(_("Open Post"))
        b.setObjectName("primary")
        b.setToolTip(_("Post, with this block's layer selected: the image, the same settings, masks and filters"))
        b.clicked.connect(lambda: self.openLayer.emit(node.lid if node.lid is not None else
                                                      s.default_post_layer(p.pid)))     # never an undo step (decision 7)
        self.bv.addWidget(b, 0, Qt.AlignLeft)
        self.bv.addStretch(1)

    def show_wire(self, name: str, note: str, actions: list):
        """A wire to Post selected (control round: the panel went empty): the layer it makes, its name, what it is,
        and its actions ((text, function, tooltip))."""
        clear(self.hv)
        clear(self.bv)
        self.folds.reset()
        self.details.hide()
        self.node = None                                  # no block: nothing of a block's to build again
        self._later.stop()                                # (the block chosen before, asked to build again)
        self.hv.addWidget(label(name, "h1", wrap=True))
        self.bv.addWidget(label(note, "dim", wrap=True))
        self.bv.addSpacing(8)
        row = QHBoxLayout()
        for text, fn, tip in actions:
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _c=False, f=fn: f())
            row.addWidget(b)
        row.addStretch(1)
        self.bv.addLayout(row)
        self.bv.addStretch(1)
        self._finish_sections()

    def graph_state(self, node):
        """The block's state as the graph shows it (round 3: a session's Calibration said "Up to date" never computed,
        the last block "Up to date" where the graph said "Outdated" for the Output it holds)."""
        from .blocks import STATE_ORDER, Graph
        try:
            g = Graph(self.s)
            same = g.node_of(node.paths[0], node.ref) if node.paths else None
            if same is not None:
                return g.state(same)
        except (KeyError, ValueError, AttributeError, IndexError):
            pass
        states = {block_state(self.s, q, node.key) for q in node.paths}
        return next((w for w in STATE_ORDER if w in states), "pending")

    def _scope(self, pids):
        self.s.scope = list(pids)
        if self.s.path is None or self.s.path.pid not in pids:
            self.s.current = self.s.index_of(pids[0])
        self.changed.emit()

    def group_title(self, text):
        g = self.folds.title(text)
        g.setStyleSheet("color: #aab2bf; font-weight: 600; margin-top: 8px; margin-bottom: 2px;")
        return g

    # ------------------------------------------------------------ sections (issue #44)
    def _finish_sections(self):
        self.folds.finish((self.node.key.partition("~")[0] if self.node is not None else "") or "wire")
        self.sectionsChanged.emit()

    def resizeEvent(self, e):                                         # noqa: N802 - Qt's name
        super().resizeEvent(e)
        if self.corner is not None:
            from .layout import place_corner
            place_corner(self.corner, self)

    def selection_rule(self, p, groups):
        """The Selection block's top: the rule that decides which frames are left out, in plain words, and its setting
        rows. Returns the other groups, and how many Advanced rows were drawn here (for the Advanced settings count)."""
        title = _("Rule")
        rule = next((rs for t, rs in groups if t == title), [])
        rest = [(t, rs) for t, rs in groups if t != title]
        if not rule and settings_loading(self.s, p, "selection"):
            return groups, 0
        self.bv.addWidget(self.group_title(title))
        self.bv.addWidget(label(selection_rule_text(self.s, p), "small", wrap=True))
        for r in rule:
            if r["level"] == "normal" or (self.expert and r["level"] == "expert"):
                self.add_row(r)
        return rest, sum(r["level"] != "normal" for r in rule)

    def selection(self, p):
        s = self.s
        t, f = s.thresholds_now()
        px = s.thresholds_unit() == "px"
        live = s.thresholds_active(p)
        self.bv.addWidget(self.group_title(_("Thresholds")))
        if not live:
            why = label(_("These thresholds act only when the frame selection is “{rule}”.").format(
                rule=texts.choice_label("thresholds", language())), "small", wrap=True)
            why.setObjectName("thresholds_reason")
            self.bv.addWidget(why)
        for name, val, unit, lo, hi, step, which in (
                (_("Minimum transparency"), t, _("% of the best"), 0, 100, 1, "t"),
                (_("Maximum FWHM") if px else _("Maximum star size"), f, "px" if px else _("× median"),
                 1 if px else 1, 20 if px else 5, 0.1, "f")):
            row = QHBoxLayout()                        # the columns of a setting row (CompactRow)
            row.setContentsMargins(4, 2, 2, 2)
            row.setSpacing(6)
            row.addWidget(name_label(name), 1)
            e = QDoubleSpinBox()
            e.setLocale(QLocale())                     # a decimal comma in French
            e.setDecimals(0 if which == "t" else 1)
            e.setRange(lo, hi)
            e.setSingleStep(step)
            e.setValue(val)
            e.setFixedWidth(EDITOR_W)
            e.setEnabled(live)
            e.editingFinished.connect(lambda ed=e, wh=which: self._sel(ed.value(), wh))
            row.addWidget(e)
            row.addWidget(unit_label(unit, self._unit_w))
            room = QWidget()                               # where the ↺ of a setting row is
            room.setFixedSize(BACK_W, BACK_W)
            row.addWidget(room)
            self.bv.addLayout(row)
        s.request_selection()
        if s.state(p, "register") == "done":
            self.bv.addWidget(label(_("{kept} of {n} frames kept").format(kept=s.kept_frames(),
                                                                         n=s.selection_total()), "small"))
        else:
            self.bv.addWidget(label(_("The count appears once registration is computed."), "small", wrap=True))

    def _sel(self, value, which):
        t, f = self.s.thresholds_now()
        if which == "t" and abs(value - t) > 1e-9:
            self.propose_selection(value, f)
        elif which == "f" and abs(value - f) > 1e-9:
            self.propose_selection(t, value)

    def propose_selection(self, tmin, fmax):
        m = self.s.fwhm_median()
        ratio = round(fmax / m if m else fmax, 4)
        ch = dict(kind="params", items=[("stack", "min_transparency", round(tmin / 100, 4)),
                                        ("stack", "max_fwhm_ratio", ratio)])
        unit = "px" if m else "× median"
        self._propose(ch, _("Selection: transparency ≥ {t} %, FWHM ≤ {f} {unit}").format(
            t=f"{tmin:.0f}", f=number(fmax), unit=unit),
            _("selection {t} % / {f} {unit}").format(t=f"{tmin:.0f}", f=number(fmax), unit=unit))

    def _groups(self, p, k) -> list:
        """[(group title, [resolved settings])] of the block, with the half-stacks where they belong."""
        s = self.s
        groups = list(step_settings(s, p, k))
        if k == "empilement" and p.opts["debruitage"]:
            groups += [(_("Half-stacks") if g == _("Half-stacks") else _("Half-stacks") + f" · {g}", rs)
                       for g, rs in step_settings(s, p, "demi_piles")]
        return groups

    def settings(self, p, k, groups=None, extra_hidden=0):
        s = self.s
        groups = self._groups(p, k) if groups is None else groups
        if settings_loading(s, p, k) and not groups:
            err = next((s.params_error(p, st) for st in needed_stages(s, p, k) if s.params_error(p, st)), "")
            self.bv.addWidget(label(_("Settings unreadable: {why}").format(why=err) if err else
                                    _("Loading settings…"), "dim", wrap=True))
            return
        hidden = extra_hidden
        system = []
        for title, rs in groups:
            shown = [r for r in rs if r["level"] == "normal" or (self.expert and r["level"] == "expert")]
            hidden += sum(r["level"] != "normal" for r in rs)
            system += [r for r in rs if r["level"] == "system"]
            if not shown:
                continue
            self.bv.addWidget(self.group_title(title))
            for r in shown:
                self.add_row(r)
        if self.expert and system:
            self.bv.addWidget(self.group_title(_("Speed · memory")))
            for r in system:
                self.add_row(r)
        if hidden:
            self.bv.addSpacing(8)
            # a disclosure, as Lightroom's panels have: the chevron shows the advanced rows in their groups, or hides them
            ex = QPushButton(_("Advanced settings ({n})").format(n=hidden))
            set_icon(ex, "chevron_down" if self.expert else "chevron_right", 16, "dim")
            ex.setObjectName("disclosure")
            ex.setFlat(True)
            ex.setCursor(Qt.PointingHandCursor)
            ex.setStyleSheet("QPushButton#disclosure { text-align: left; border: none; padding: 2px 0; color: #b9c2cf; }"
                             f"QPushButton#disclosure:hover {{ color: {ACCENT}; }}")
            ex.setToolTip(_("Hide the advanced settings") if self.expert else _("Show the advanced settings"))
            ex.clicked.connect(lambda: self._expert(not self.expert))
            self.bv.addWidget(ex, 0, Qt.AlignLeft)

    def add_row(self, r):
        row = CompactRow(r, self._unit_w)
        row.changed.connect(self.set_setting)
        row.focused.connect(self._focus)
        self.bv.addWidget(row)
        if r["key"] == self.focus_key:
            self.details.show_setting(r)

    def _focus(self, r):
        self.focus_key = r["key"]
        self.details.show_setting(r)

    def _expert(self, on):
        self.expert = on
        self.refresh(self.node)

    def set_setting(self, r, value):
        """A setting was edited (value None: back to the suggestion)."""
        if r["kind"] == "new":
            return
        self.focus_key = r["key"]
        old = fmt(r["value"], r["type"], r.get("choices"))
        new = fmt(r["suggested"] if value is None else value, r["type"], r.get("choices"))
        self.propose(r["stage"], r["name"], value, r["suggested"],
                     _("{label}: {old} → {new}").format(label=r["label"], old=old, new=new),
                     f"{r['label'].lower()} {new}")
