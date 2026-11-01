"""A project of the previous version (v1): the dialog that imports it, so that it opens here.

The import (``api.import_v1``) makes a new project in the workspace: the definition of the old project, and the results
that are still valid, copied under their new keys. The old folder is never written. First a dry run in the worker says
what is kept and what will be computed again, with the size to copy and the free space; then Import copies, with a
progress line, and the new project opens.
"""
from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from .. import api
from ..core import project2
from . import prefs
from .home import folder_name
from .i18n import _, language, number, plural
from .services.worker import ApiWorker
from .widgets import ElidedLabel, label

#: the stages as the views name them (the blocks they belong to)
STAGE_BLOCK = {"inventory": "Calibration", "calibrate": "Calibration", "register": "Registration", "stack": "Stacking",
               "astrometry": "Astrometry", "reference": "Astrometry", "catalogs": "Astrometry",
               "surveys": "Astrometry", "color": "Color", "denoise": "Denoise", "optics": "Stars", "stars": "Stars",
               "star_net": "Stars", "starless": "Stars", "background": "Gradient", "background_poly": "Gradient",
               "render_prep": "Output", "render_prep_learned": "Output", "render_simple": "Output"}
#: the engine's reasons for computing a stage again, as templates the client translates
REASONS = [(re.compile(r"^algorithm updated \((\w+) -> (\w+)\)$"), "algorithm updated ({0} → {1})"),
           (re.compile(r"^no result in the old project$"), "no result in the old project"),
           (re.compile(r"^'(\w+)' is recomputed$"), "{0} is computed again"),
           (re.compile(r"^'(\w+)' was run again after this result$"), "{0} was run again after this result"),
           (re.compile(r"^settings changed: (.*)$"), "settings changed: {0}"),
           (re.compile(r"^the old result is (\w+)$"), "the old result is {0}"),
           (re.compile(r"^the earlier layer belongs to another stack$"), "the earlier layer belongs to another stack")]


def size_text(n_bytes: int | None) -> str:
    """Bytes as GB or TB, with the language's decimal mark."""
    if n_bytes is None:
        return "?"
    gb = n_bytes / 1e9
    if gb >= 1000:
        return number(gb / 1000, 1) + " " + _("TB")
    return number(gb, 1 if gb >= 0.1 else 2) + " " + _("GB")


def reason_text(why: str) -> str:
    """An engine reason, translated where its template is known (the stage names it quotes too)."""
    for rx, template in REASONS:
        m = rx.match(why or "")
        if m:
            args = [_(STAGE_BLOCK.get(a, a)) if a in STAGE_BLOCK else a for a in m.groups()]
            return _(template).format(*args)
    return why


def blocks_of(stages: list[str]) -> list[str]:
    """The blocks of a list of stages, in order, each once (translated)."""
    out: list[str] = []
    for s in stages:
        b = _(STAGE_BLOCK.get(s, s))
        if b not in out:
            out.append(b)
    return out


def destination(old_root: Path) -> Path:
    """Where the imported project goes: <workspace>/<old folder name> (a free name)."""
    ws = Path(prefs.get("workspace"))
    base = folder_name(old_root.name) or "project"
    d, n = ws / base, 2
    while d.exists():
        d, n = ws / f"{base} {n}", n + 1
    return d


def _dry_run(old: str, new: str):
    return api.import_v1(old, new, dry_run=True)


def _import(old: str, new: str, say):
    return api.import_v1(old, new, progress=say)


class _Relay(QObject):
    said = Signal(str)


class PreviousVersionDialog(QDialog):
    """Import a v1 project: what is kept, what is computed again, the size; then the copy. ``result_root`` is the new
    project's folder once the import is done."""

    def __init__(self, old_root, parent=None, worker: ApiWorker | None = None):
        super().__init__(parent)
        self._own_worker = worker is None
        self.worker = worker or ApiWorker(self)          # its own thread: a copy of GBs never holds the views' jobs
        self.old = Path(old_root)
        self.new = destination(self.old)
        self.report = None
        self.result_root: Path | None = None
        self._busy = False
        self._relay = _Relay(self)
        self._relay.said.connect(self._progress)
        self.setWindowTitle(_("Import a project"))
        self.setMinimumWidth(640)
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 14)
        v.setSpacing(10)
        v.addWidget(label(_("A project of the previous version"), "h1"))
        intro = label(_("“{name}” was made with the previous version. Its workflows and the results that are still "
                        "valid are copied into a new project of the workspace; its own folder is never "
                        "changed.").format(name=self.old.name), "dim", wrap=True)
        v.addWidget(intro)
        g = QGridLayout()
        g.setHorizontalSpacing(14)
        g.setVerticalSpacing(6)
        self.rows: dict[str, QLabel] = {}
        for i, (key, title) in enumerate((("from", _("Previous project")), ("to", _("New project")),
                                          ("workflows", _("Workflows")), ("kept", _("Results kept")),
                                          ("again", _("Computed again")), ("size", _("To copy")))):
            g.addWidget(label(title, "dim"), i, 0, Qt.AlignVCenter)
            val = ElidedLabel("") if key in ("from", "to") else label("", None, wrap=True)
            g.addWidget(val, i, 1)
            self.rows[key] = val
        self.b_to = QPushButton(_("Change…"))
        self.b_to.clicked.connect(self._change)
        g.addWidget(self.b_to, 1, 2)
        g.setColumnStretch(1, 1)
        v.addLayout(g)
        self.rows["from"].setText(self.old.as_posix())
        self.rows["to"].setText(self.new.as_posix())
        self.note = label("", "warn", wrap=True)
        self.note.hide()
        v.addWidget(self.note)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.setTextVisible(False)
        v.addWidget(self.bar)
        self.status = label(_("Reading the project…"), "small")
        v.addWidget(self.status)
        bb = QHBoxLayout()
        bb.addStretch(1)
        self.b_cancel = QPushButton(_("Cancel"))
        self.b_cancel.clicked.connect(self.reject)
        self.b_ok = QPushButton(_("Import"))
        self.b_ok.setObjectName("primary")
        self.b_ok.setEnabled(False)
        self.b_ok.clicked.connect(self.start)
        bb.addWidget(self.b_cancel)
        bb.addWidget(self.b_ok)
        v.addLayout(bb)
        self._ch = f"previous:{id(self)}"
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self._check()

    # ------------------------------------------------------------ the dry run
    def _check(self):
        self.b_ok.setEnabled(False)
        self.bar.setRange(0, 0)
        self.bar.show()
        self.status.setText(_("Reading the project…"))
        self.worker.submit(self._ch + ":check", _dry_run, str(self.old), str(self.new))

    def _change(self):
        d = QFileDialog.getExistingDirectory(self, _("New project"), prefs.get("workspace"))
        if d:
            target = Path(d)
            if target.exists() and any(target.iterdir()):
                target = target / folder_name(self.old.name)
            self.new = target
            self.rows["to"].setText(self.new.as_posix())
            self._check()

    def show_report(self, rep):
        self.report = rep
        wfs = list(rep.workflows.values())
        self.rows["workflows"].setText(", ".join(wfs) if wfs else "—")
        first = next(iter(rep.workflows), None)
        kept = rep.adopted(first) if first is not None else []
        again = rep.recomputed(first) if first is not None else []
        if kept:
            b = blocks_of(kept)
            self.rows["kept"].setText(plural(len(kept), "{n} stage computed: {blocks}", "{n} stages computed: {blocks}",
                                             blocks=" → ".join([b[0], b[-1]]) if len(b) > 1 else b[0]))
        else:
            self.rows["kept"].setText(_("nothing: every stage is computed again"))
        if again:
            stage, why = again[0]
            self.rows["again"].setText(_("from {block}: {why}").format(block=_(STAGE_BLOCK.get(stage, stage)),
                                                                        why=reason_text(why)))
        else:
            self.rows["again"].setText(_("nothing"))
        free = _(", {free} free on the disk").format(free=size_text(rep.free_bytes)) \
            if rep.free_bytes is not None else ""
        self.rows["size"].setText(size_text(rep.copy_bytes) + free)
        ok = rep.enough_space
        self.note.setVisible(not ok)
        self.note.setText("" if ok else _("Not enough free space on the disk of the new project."))
        self.bar.hide()
        self.status.setText(_("Nothing is written until Import."))
        self.b_ok.setEnabled(ok)

    # ------------------------------------------------------------ the import
    def start(self):
        if self.report is None or self._busy:
            return
        self._busy = True
        self.b_ok.setEnabled(False)
        self.b_to.setEnabled(False)
        self.bar.setRange(0, 0)
        self.bar.show()
        self.status.setText(_("Copying the results…"))
        relay = self._relay
        self.worker.submit(self._ch + ":import", _import, str(self.old), str(self.new),
                           lambda msg: relay.said.emit(str(msg)))

    def _progress(self, msg: str):
        """The engine's word on what it copies (English: shown as it is in English only)."""
        if self._busy:
            self.status.setText(_("Copying the results…") + ("  " + msg if language() == "en" else ""))

    def _done(self, channel: str, _n: int, result):
        if self.worker is None:
            return
        if channel == self._ch + ":check":
            self.show_report(result)
        elif channel == self._ch + ":import":
            self._busy = False
            self.result_root = Path(self.new)
            self.accept()

    def _failed(self, channel: str, _n: int, text: str):
        if not channel.startswith(self._ch):
            return
        self._busy = False
        why = (text or "?").splitlines()[0].split(": ", 1)[-1]
        self.bar.hide()
        self.note.setText(_("The project cannot be imported: {why}").format(why=why))
        self.note.show()
        self.status.setText("")
        self.b_to.setEnabled(True)
        self.b_ok.setEnabled(False)

    def reject(self):
        if self._busy:                                   # the copy goes on to its end: it cannot be left half done
            return
        super().reject()

    def done(self, r):                                   # noqa: D401 - Qt's name
        self.release()
        super().done(r)

    def release(self):
        """Let go of the worker: a check still reading the old project's frames ends on its own (its thread is kept
        alive until then: a running thread must never be destroyed with the dialog)."""
        if self.worker is None:
            return
        for sig, slot in ((self.worker.done, self._done), (self.worker.failed, self._failed)):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        if self._own_worker:
            self.worker.stop(0)
            if self.worker._thread.isRunning():
                self.worker.setParent(None)
                _RETIRED.append(self.worker)
            _RETIRED[:] = [w for w in _RETIRED if w._thread.isRunning()]
        self.worker = None


#: the workers of closed dialogs whose last job is still running
_RETIRED: list = []


def is_previous(root) -> bool:
    """Whether the folder holds a project of the previous version (and no v2 project)."""
    root = Path(root)
    return (root / project2.V1_FILE).is_file() and not (root / project2.FILE).is_file()
