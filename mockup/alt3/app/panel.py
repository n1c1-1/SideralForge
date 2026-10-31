"""The panel of the selected block, on the right in Graph and Inspect.

One line per setting. What there is to know about a setting (its suggestion and why, its range, its use, a tip)
shows under the list when the setting is clicked or changed. When the block is shared by several workflows, the panel
says which ones a change goes to; a change that would throw away computed results offers Replace or Try as variant.
Blocks with several methods show them as cards: what each does, where it runs, how long on this machine.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

import prefs
from blocks import block_title
from i18n import _
from model import hm
from settings import CATALOG, NEW, N, RangeBar, SettingRow, fmt, step_settings
from steps import BY_KEY, fnum, kv
from widgets import ACCENT, WARN, clear, hline, label

# ---- settings that the engine does not have yet (marked in Spec mode): the methods' own, and the pro options
CATALOG["debruitage"] += [
    ("Fine-tuning", [
        N("denoise_ft", "steps", "Fine-tuning iterations", "int", 600, "How many iterations the pretrained network "
          "is fine-tuned on the half-stacks.", lo=100, hi=3000, step=100),
        N("denoise_ft", "start", "Starting network", "choice", "Closest earlier project",
          "The network the fine-tuning starts from.", choices=["Closest earlier project", "Bundled generic network"]),
    ]),
    ("Pretrained network", [
        N("denoise_ai", "force", "Strength", "float", 0.8, "Blend between the denoised and the original image.",
          lo=0.0, hi=1.0, step=0.05),
        N("denoise_ai", "model", "Model", "choice", "Bundled", "The network applied.",
          choices=["Bundled", "GraXpert (licence to check)"]),
    ]),
    ("NL-Bayes", [
        N("denoise_cl", "force", "Strength", "float", 1.0, "How much of the measured noise is removed.", lo=0.0,
          hi=3.0, step=0.05),
        N("denoise_cl", "vst", "Stabilise photon noise", "bool", True, "Anscombe transform before denoising: the "
          "noise becomes the same in faint and bright areas."),
        N("denoise_cl", "da3d", "DA3D refinement", "bool", False, "A second pass that keeps fine detail better; "
          "slower.", level="expert"),
    ]),
]
CATALOG["calibration"] += [
    ("Debayer", [
        N("calibrate", "debayer", "Debayer method", "choice", "RCD", "How the colour of each pixel is rebuilt from "
          "the Bayer matrix. Not used with Bayer drizzle.", choices=["RCD", "VNG", "AHD", "Bilinear"]),
        N("calibrate", "cfa_pattern", "CFA pattern", "choice", "From header", "Forces the Bayer pattern when the file "
          "header is wrong.", choices=["From header", "RGGB", "BGGR", "GRBG", "GBRG"], level="expert"),
    ]),
    ("Flat calibration", [
        N("calibrate", "flat_offset", "Flat offset", "choice", "Bias from library", "What is subtracted from the "
          "flats before they divide the lights.", choices=["Bias from library", "Dark flats", "Camera black level"]),
        N("calibrate", "pedestal", "Output pedestal", "int", 0, "Added after calibration so that no pixel goes "
          "negative.", lo=0, hi=1000, step=10, unit="ADU", level="expert"),
    ]),
]
CATALOG["alignement"] += [
    ("Distortion and framing", [
        N("register", "distortion", "Distortion correction", "choice", "Automatic", "Field distortion between frames "
          "(wide fields, mosaics).", choices=["Automatic", "None", "SIP polynomial", "Thin-plate splines"]),
        N("register", "framing", "Framing", "choice", "Reference frame", "The area of the stacked image.",
          choices=["Reference frame", "Union (maximum)", "Intersection (minimum)", "Centre of gravity"]),
    ]),
]
CATALOG["empilement"] += [
    ("Advanced stacking", [
        N("stack", "local_norm", "Local normalization", "bool", False, "Matches the background of each frame locally "
          "before stacking: gradients that move from night to night are evened out."),
        N("stack", "drizzle_kernel", "Drizzle kernel", "choice", "Square", "The shape each input pixel drops on the "
          "output grid.", choices=["Square", "Point", "Turbo", "Gaussian", "Lanczos-3"], level="expert"),
    ]),
]
CATALOG["post"] = [
    ("Stretch", [
        N(st, "stretch", "Stretch", "choice", "Engine (statistical)", "How the linear output is made non-linear "
          "for Post. Post's sliders and the Stretch filter act after it.",
          choices=["Engine (statistical)", "GHS", "Arcsinh", "Histogram (STF)", "None: linear output"])
        for st in ("@render", "rendu_simple")] + [
        N(st, "linked", "Linked channels", "bool", True, "The same stretch for the three channels: the colours "
          "stay as calibrated.") for st in ("@render", "rendu_simple")]),
] + CATALOG.get("post", [])
CATALOG["deconv"] = [
    ("Deconvolution", [
        N("deconv", "iterations", "Iterations", "int", 30, "More iterations sharpen more, and raise the noise.",
          lo=5, hi=200, step=5),
        N("deconv", "stars", "Correct the stars too", "bool", True, "Rounds and tightens the stars with the local "
          "PSF."),
        N("deconv", "regularization", "Regularization", "float", 0.002, "Holds back noise and ringing.", lo=0.0,
          hi=0.05, step=0.001, level="expert"),
    ]),
]
STATE_WORD = {"done": "Up to date", "pending": "Pending", "stale": "Outdated"}


def spec_mark(row_layout, new):
    """The ◇ mark of what the engine does not have yet: only in Spec mode."""
    if new and prefs.spec():
        mark = QLabel("◇")
        mark.setStyleSheet(f"color: {NEW};")
        mark.setToolTip(_("Not in the engine yet"))
        row_layout.addWidget(mark)


class CompactRow(SettingRow):
    """A setting on one line: name, value, back to the suggestion."""
    focused = Signal(dict)

    def __init__(self, r: dict, parent=None):
        QFrame.__init__(self, parent)
        self.r = r
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
        name = QLabel(r["label"])
        name.setStyleSheet(f"color: {WARN};" if r["changed"] else "")
        h.addWidget(name, 1)
        spec_mark(h, r["kind"] == "new")
        ed = self._editor()
        if isinstance(ed, QComboBox):                 # a choice: its own line, full width, never truncated
            ed.setToolTip(ed.currentText())
            line = QHBoxLayout()
            line.setContentsMargins(4, 0, 2, 4)
            line.addWidget(ed, 1)
            rb = QPushButton("↺")
            rb.setFixedSize(22, 22)
            rb.setStyleSheet("padding: 0;")
            rb.setToolTip(_("Back to the suggested value: {v}").format(
                v=fmt(r["suggested"], r["type"], r.get("choices"))))
            rb.clicked.connect(lambda: self.changed.emit(self.r, self.r["suggested"]))
            rb.setVisible(r["changed"])
            line.addWidget(rb)
            if not r["changed"]:
                line.addSpacing(22)
            outer.addLayout(line)
            self.setToolTip(r.get("help") or "")
            return
        if ed.width() > 90:
            ed.setFixedWidth(104)
        h.addWidget(ed)
        u = label(r.get("unit") or "", "small")
        u.setFixedWidth(60 if r.get("unit") else 0)
        u.setToolTip(r.get("unit") or "")
        h.addWidget(u)
        rb = QPushButton("↺")
        rb.setFixedSize(22, 22)
        rb.setStyleSheet("padding: 0;")
        rb.setToolTip(_("Back to the suggested value: {v}").format(v=fmt(r["suggested"], r["type"], r.get("choices"))))
        rb.clicked.connect(lambda: self.changed.emit(self.r, self.r["suggested"]))
        rb.setVisible(r["changed"])
        h.addWidget(rb)
        if not r["changed"]:
            h.addSpacing(22)
        self.setToolTip(r.get("help") or "")

    def mousePressEvent(self, e):
        self.focused.emit(self.r)
        super().mousePressEvent(e)


class MethodCard(QFrame):
    """One way of computing a block: its name, what it does, where it runs and how long on this machine."""
    clicked = Signal()

    def __init__(self, m, on, ok, seconds, gpu_machine, why="", parent=None):
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
        n = QLabel(("● " if on else "○ ") + _(m["name"]))
        n.setWordWrap(True)
        n.setStyleSheet("font-weight: 600;" + ("" if ok else " color: #6b7280;"))
        v.addWidget(n)
        top = QHBoxLayout()
        top.setSpacing(6)
        spec_mark(top, m.get("new"))
        gpu = m["runs"] == "gpu"
        chip = QLabel(_({"gpu": "GPU", "cpu": "CPU", "both": "CPU or GPU"}[m["runs"]]))
        chip.setStyleSheet(f"color: {'#c9a35e' if gpu else '#8b93a1'}; border: 1px solid "
                           f"{'#c9a35e' if gpu else '#3a4150'}; border-radius: 8px; padding: 0 6px; "
                           "font-size: 8.5pt;")
        top.addWidget(chip)
        est = not gpu_machine or m.get("new")
        d = label(("≈ " if est else "") + hm(seconds), "small")
        d.setToolTip(_("On this machine") + (" · " + _("estimate") if est else ""))
        top.addWidget(d)
        top.addStretch(1)
        v.addLayout(top)
        v.addWidget(label(_(m["what"]), "small", wrap=True))
        if why:
            v.addWidget(label(why, "warn", wrap=True))

    def mousePressEvent(self, e):
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
        if r["kind"] == "new" and prefs.spec():
            self.v.addWidget(label("◇ " + _("Not in the engine yet."), "small"))
        self.show()


class Panel(QWidget):
    changed = Signal()                  # a setting changed (the window rebuilds the graph)
    openTune = Signal()                 # "Browse frames" and the like
    openPost = Signal()
    addSources = Signal()
    addSession = Signal()
    renamed = Signal(int, str)
    fondChanged = Signal(str)           # "halpha" | "simple"
    removeTool = Signal()
    variantMade = Signal(int, int)      # new workflow, the one it is a variant of

    def __init__(self, project, steps, tune=False, parent=None):
        super().__init__(parent)
        self.s, self.steps, self.tune = project, steps, tune
        self.expert = False
        self.focus_key = None
        self.node = None
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
        host = QWidget()
        self.bv = QVBoxLayout(host)
        self.bv.setContentsMargins(10, 8, 10, 8)
        self.bv.setSpacing(1)
        sc.setWidget(host)
        v.addWidget(sc, 1)
        self.details = Details()
        v.addWidget(self.details)

    # ------------------------------------------------------------ build
    def refresh(self, node):
        clear(self.hv)
        clear(self.bv)
        self.details.hide()
        self.node = node
        if node is None:
            return
        k = node.key
        if k == "sources":
            self.sources()
        elif k == "final":
            self.hv.addWidget(label(_("Post"), "h1"))
            rows = []
            for q in self.s.paths:
                end = getattr(q, "end", None)
                src = _("from {block}").format(block=block_title(q, end)) if end else ""
                st = _("Up to date") if q.rendered and q.computed else (_("Outdated") if q.rendered else
                                                                          _("Pending"))
                rows.append((q.name, " · ".join(x for x in (st, src) if x)))
            if rows:
                self.bv.addWidget(self.group_title(_("Layers")))
                self.bv.addWidget(kv(rows, pad=10))
                self.bv.addSpacing(8)
                b = QPushButton(_("Open Post"))
                b.setObjectName("primary")
                b.setEnabled(any(q.rendered for q in self.s.paths))
                b.clicked.connect(self.openPost.emit)
                self.bv.addWidget(b, 0, Qt.AlignLeft)
            self.bv.addStretch(1)
        else:
            self.block(node)

    def sources(self):
        s = self.s
        self.hv.addWidget(label(_("Frames"), "h1"))
        if not s.sources_loaded:
            b = QPushButton(_("Import folder…"))
            b.setObjectName("primary")
            b.clicked.connect(self.addSources.emit)
            self.bv.addWidget(b, 0, Qt.AlignLeft)
            self.bv.addStretch(1)
            return
        self.bv.addWidget(self.group_title(_("Sessions")))
        for ses in s.sessions:
            darks = _("library") if ses["darks"] == "library" else str(ses["darks"])
            self.bv.addWidget(kv([(ses["night"], _("{lights} lights · {flats} flats · darks: {darks}").format(
                lights=ses["lights"], flats=ses["flats"], darks=darks))], pad=10))
            if ses.get("rotation"):
                self.bv.addWidget(label(_("Camera rotated {deg}°: this session uses its own flats").format(
                    deg=fnum(ses["rotation"])), "warn", wrap=True))
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
            ed.editingFinished.connect(lambda: ed.text().strip() and ed.text().strip() != p.name
                                       and self.renamed.emit(p.pid, ed.text().strip()))
            top.addWidget(ed, 1)
        else:
            top.addWidget(label(block_title(p, k), "h1"))
            top.addStretch(1)
        st = self.graph_state(node)
        chip = QLabel(_(STATE_WORD[st]))
        col = {"done": "#5fbf7f", "pending": "#8b93a1", "stale": WARN}[st]
        chip.setStyleSheet(f"color: {col}; font-size: 9pt;")
        top.addWidget(chip)
        if k in ("debruitage", "etoiles", "deconv"):
            x = QPushButton(_("Remove"))
            x.clicked.connect(self.removeTool.emit)
            top.addWidget(x)
        self.hv.addLayout(top)
        # which workflows a change goes to
        if node.shared:
            self.hv.addWidget(label(_("Changes apply to:"), "dim"))
            box = QVBoxLayout()
            box.setSpacing(2)
            allp = [q.pid for q in node.paths]
            cur = s.scope if s.scope and set(s.scope) <= set(allp) else allp
            choices = [(_("all {n} workflows").format(n=len(node.paths)), allp)] + \
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
        if self.tune and st != "pending":
            facts = self.steps.facts(p, k)
            if facts:
                self.bv.addWidget(self.group_title(_("Measurements")))
                self.bv.addWidget(kv(facts, pad=10))
                self.bv.addSpacing(6)
        if k in ("fond", "debruitage", "etoiles", "deconv"):
            self.methods(p, k)
        if k == "selection":
            self.selection(p)
        if k == "empilement" and len(s.sessions) > 1:
            self.stack_sessions()
        if k == "post" and getattr(p, "end", None):
            self.bv.addWidget(kv([(_("Output of"), block_title(p, p.end))]))
        self.settings(p, k)
        self.bv.addStretch(1)

    def stack_sessions(self):
        """Several nights: one stack of all, or one per night as variants to compare."""
        self.bv.addWidget(self.group_title(_("Sessions")))
        row = QHBoxLayout()
        for text, mode in ((_("All together"), "together"), (_("One per session"), "per_session")):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setObjectName("face")
            b.setChecked(self.s.stack_mode == mode)
            b.clicked.connect(lambda _x=False, m=mode: self._stack_mode(m))
            row.addWidget(b)
        row.addStretch(1)
        self.bv.addLayout(row)
        if self.s.stack_mode == "per_session":
            self.bv.addWidget(label(_("Each session is stacked on its own, as a variant: compare the nights side "
                                      "by side."), "small", wrap=True))

    def _stack_mode(self, mode):
        self.s.set_stack_mode(mode)
        self.changed.emit()

    # ------------------------------------------------------------ replace or try as a variant
    def variant_row(self, p):
        """A variant says what it is a variant of, and is kept or discarded from here."""
        src = next((q for q in self.s.paths if q.pid == getattr(p, "variant_of", None)), None)
        if src is None:
            return
        row = QHBoxLayout()
        row.addWidget(label(_('Variant of "{name}"').format(name=src.name), "dim"), 1)
        keep = QPushButton(_("Keep"))
        keep.setToolTip(_('Replaces "{name}" and takes its name').format(name=src.name))
        keep.clicked.connect(lambda: (self.s.keep_variant(p), self.changed.emit()))
        drop = QPushButton(_("Discard"))
        drop.clicked.connect(lambda: (self.s.drop_variant(p), self.changed.emit()))
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
        v.addWidget(label(f"<b>{ch['what']}</b>", wrap=True))
        sec = s.cost_with(paths, ch)
        rep = QPushButton(_("Replace") + f"  ·  {hm(sec)}")
        rep.setToolTip(_("The current results of {names} are computed again").format(
            names=", ".join(q.name for q in paths)))
        rep.clicked.connect(self._replace)
        v.addWidget(rep)
        sec_v = s.cost_with([p], ch)
        var = QPushButton(_("Try as variant") + f"  ·  {hm(sec_v)}")
        var.setObjectName("primary")
        var.setToolTip(_('"{name}" stays as it is; the variant shares everything up to this block').format(
            name=p.name))
        var.clicked.connect(lambda: self._variant(p))
        v.addWidget(var)
        c = QPushButton(_("Cancel"))
        c.clicked.connect(self._cancel)
        v.addWidget(c, 0, Qt.AlignLeft)
        self.hv.addWidget(f)

    def propose(self, stage, name, value, suggested, what, short):
        """A change of setting: made at once when nothing computed is lost; otherwise Replace, or Try as variant."""
        self._propose(dict(kind="param", stage=stage, name=name, value=value, suggested=suggested), stage, what,
                      short)

    def propose_opt(self, key, value, stage, what, short):
        """A change of option, such as a method; `stage`: the first stage it changes."""
        self._propose(dict(kind="opt", key=key, value=value), stage, what, short)

    def _propose(self, ch, stage, what, short):
        s = self.s
        paths = s.scope_paths()
        if s.throws_away(paths, stage):
            s.pending_change = dict(ch, what=what, short=short, step=self.node.key, pids=[q.pid for q in paths])
        else:
            s.pending_change = None
            s.apply_change(paths, ch)
            s.log(what + " · " + ", ".join(q.name for q in paths))
        self.changed.emit()

    def _replace(self):
        ch, self.s.pending_change = self.s.pending_change, None
        paths = [q for q in self.s.paths if q.pid in ch["pids"]]
        self.s.apply_change(paths, ch)
        self.s.log(ch["what"] + " · " + _("replaced in") + " " + ", ".join(q.name for q in paths))
        self.changed.emit()

    def _variant(self, p):
        ch, self.s.pending_change = self.s.pending_change, None
        q = self.s.add_variant(p, ch, f"{p.name} · {ch['short']}")
        self.s.log(_('Variant "{name}" created').format(name=q.name))
        self.variantMade.emit(q.pid, p.pid)

    # ------------------------------------------------------------ the method of a block
    def methods(self, p, k):
        """The ways of computing this block, as cards: what it does, where it runs, how long on this machine."""
        from project import METHOD_OPT, METHODS, method_of, method_value
        s = self.s
        cur = method_of(p, k)
        self.bv.addWidget(self.group_title(_("Method")))
        stages = self.steps.stages_of(p, BY_KEY[k])
        stage = stages[0] if stages else "deconv"
        opt = METHOD_OPT[k]
        for m in METHODS[k]:
            on = m["key"] == cur
            ok = s.gpu or m["runs"] != "gpu"
            t = s._tmp(p, dict(kind="opt", key=opt, value=method_value(k, m["key"])))
            sec = sum(s.seconds(x) for x in self.steps.stages_of(t, BY_KEY[k]))
            if k == "etoiles":
                sec += s.seconds("starless")
            why = "" if ok else _("Needs an NVIDIA GPU (≈ {t} on the CPU).").format(t=hm(sec))
            if on and not ok:
                why = _("Computed earlier on a GPU; running it again here takes ≈ {t}.").format(t=hm(sec))
            card = MethodCard(m, on, ok, sec, s.gpu, why)
            if not on and ok:
                if k == "fond":
                    card.clicked.connect(lambda mk=m["key"]: self.fondChanged.emit(mk))
                else:
                    card.clicked.connect(lambda mk=m["key"], nm=m["name"], sh=m["short"]: self.propose_opt(
                        opt, method_value(k, mk), stage, _("Method: {name}").format(name=_(nm)), _(sh)))
            self.bv.addWidget(card)

    def _cancel(self):
        self.s.pending_change = None
        self.changed.emit()

    def graph_state(self, node):
        states = {self.steps.state(q, BY_KEY[node.key]) for q in node.paths}
        return "stale" if "stale" in states else ("pending" if "pending" in states else "done")

    def _scope(self, pids):
        self.s.scope = list(pids)
        if self.s.path is None or self.s.path.pid not in pids:
            i = next(i for i, q in enumerate(self.s.paths) if q.pid == pids[0])
            self.s.current = i
        self.changed.emit()

    def group_title(self, text):
        g = label(text, "dim")
        g.setStyleSheet("color: #aab2bf; font-weight: 600; margin-top: 8px; margin-bottom: 2px;")
        return g

    def selection(self, p):
        t, f = self.s.thresholds_now()
        self.bv.addWidget(self.group_title(_("Thresholds")))
        for name, val, unit, lo, hi, step, which in (
                (_("Minimum transparency"), t, _("% of the best"), 0, 100, 1, "t"),
                (_("Maximum FWHM"), f, "px", 1, 20, 0.1, "f")):
            row = QHBoxLayout()
            row.setContentsMargins(4, 2, 2, 2)
            row.addWidget(QLabel(name), 1)
            e = QDoubleSpinBox()
            e.setDecimals(0 if which == "t" else 1)
            e.setRange(lo, hi)
            e.setSingleStep(step)
            e.setValue(val)
            e.setFixedWidth(90)
            e.editingFinished.connect(lambda ed=e, wh=which: self._sel(ed.value(), wh))
            row.addWidget(e)
            u = label(unit, "small")
            u.setFixedWidth(90)
            row.addWidget(u)
            self.bv.addLayout(row)
        n = len(self.s.data["frames"]["fwhm"])
        self.bv.addWidget(label(_("{kept} of {n} frames kept").format(kept=self.s.kept_frames(), n=n), "small"))

    def _sel(self, value, which):
        t, f = self.s.thresholds_now()
        if which == "t" and abs(value - t) > 1e-9:
            self.propose_selection(value, f)
        elif which == "f" and abs(value - f) > 1e-9:
            self.propose_selection(t, value)

    def propose_selection(self, tmin, fmax):
        st, sf = self.s.suggested_thresholds()
        sugg = abs(tmin - st) < 1e-6 and abs(fmax - sf) < 1e-6
        value = None if sugg else (round(tmin, 1), round(fmax, 2))
        self.propose("stack", "selection", value, None,
                     _("Selection: transparency ≥ {t} %, FWHM ≤ {f} px").format(t=f"{tmin:.0f}", f=fnum(fmax)),
                     _("selection {t} % / {f} px").format(t=f"{tmin:.0f}", f=fnum(fmax)))

    def settings(self, p, k):
        groups = list(step_settings(self.s, p, k))
        if k == "empilement" and p.opts["debruitage"]:
            groups += [(_("Half-stacks") + f" · {g}", rs) for g, rs in step_settings(self.s, p, "demi_piles")]
        if k == "etoiles":
            groups += [(_("Neural removal") + f" · {g}", rs) for g, rs in step_settings(self.s, p, "appris")]
            groups += [(_("Starless image") + f" · {g}", rs) for g, rs in step_settings(self.s, p, "sans_etoiles")]
        hidden = 0
        system = []
        for title, rs in groups:
            shown = [r for r in rs if r["level"] == "normal" or (self.expert and r["level"] == "expert")]
            hidden += sum(r["level"] != "normal" for r in rs)
            system += [r for r in rs if r["level"] == "system"]
            if not shown:
                continue
            self.bv.addWidget(self.group_title(_(title)))
            for r in shown:
                self.add_row(r)
        if self.expert and system:
            self.bv.addWidget(self.group_title(_("System: speed and memory")))
            for r in system:
                self.add_row(r)
        if hidden:
            self.bv.addSpacing(8)
            ex = QCheckBox(_("Advanced settings ({n})").format(n=hidden))
            ex.setChecked(self.expert)
            ex.toggled.connect(self._expert)
            self.bv.addWidget(ex)

    def add_row(self, r):
        row = CompactRow(r)
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
        self.focus_key = r["key"]
        old = fmt(r["value"], r["type"], r.get("choices"))
        new = fmt(value, r["type"], r.get("choices"))
        self.propose(r["stage"], r["name"], value, r["suggested"], f"{r['label']}: {old} → {new}",
                     f"{r['label'].lower()} {new}")
