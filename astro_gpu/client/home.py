"""Home: the workspace and its projects. New project, open, import a folder as a new project, recent projects.

The workspace is read in the ApiWorker (``scan_workspace``): every folder holding a project.astroproj, except the
folders whose name starts with "_" (QC sheets, scratch), plus the recent projects kept elsewhere. Nothing is written.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
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
    QVBoxLayout,
    QWidget,
)

from ..core import project2
from . import prefs
from .i18n import _, number
from .widgets import ACCENT, ElidedLabel, label

TW, TH = 260, 150
THUMB = Path("cache") / "thumb.jpg"


# ---------------------------------------------------------------------------------------------- the scan (worker)

@dataclass
class ProjectInfo:
    """What a Home card shows of a project, read in the worker."""
    root: str
    name: str
    target: str = ""
    nights: int = 0
    integration_h: float | None = None
    date: str = ""
    state: str = "Never run"                         # English, translated when shown
    state_args: dict = field(default_factory=dict)
    thumb: QImage | None = None


def _same(a: Path, b: Path) -> bool:
    return a.as_posix().rstrip("/").lower() == b.as_posix().rstrip("/").lower()


def _manifests(root: Path, stage: str = "*"):
    """The manifests of the done results of the store (never a .tmp folder)."""
    for m in (root / "store").glob(f"{stage}/*/manifest.json"):
        if m.parent.name.endswith(".tmp") or m.parent.parent.name.startswith("_"):
            continue
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("status") == "done":
            yield d


def _running(root: Path) -> bool:
    lock = root / "logs" / "run.lock"
    if not lock.exists():
        return False
    try:
        import psutil
        return psutil.pid_exists(int(json.loads(lock.read_text(encoding="utf-8"))["pid"]))
    except Exception:  # noqa: BLE001 - an unreadable lock is not a run
        return False


def _thumb(root: Path) -> QImage | None:
    f = root / THUMB
    if not f.is_file():
        return None
    im = QImage(str(f))
    if im.isNull():
        return None
    im = im.scaled(TW, TH, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    return im.copy((im.width() - TW) // 2, (im.height() - TH) // 2, TW, TH)


def project_info(root: Path) -> ProjectInfo:
    p = project2.load(root)
    info = ProjectInfo(root.as_posix(), p.name, p.target.name or "", len(p.sessions))
    inv = max(_manifests(root, "inventory"), key=lambda d: d.get("finished") or "", default=None)
    m = (inv or {}).get("metrics") or {}
    info.integration_h = m.get("integration_h")
    when = m.get("last_time") or p.created
    if not when:
        when = datetime.fromtimestamp((root / project2.FILE).stat().st_mtime).isoformat()
    info.date = str(when)[:10]
    if _running(root):
        info.state = "Running"
    elif inv is not None or next(_manifests(root), None) is not None:
        info.state, info.state_args = "{n} workflow(s)", {"n": len(p.workflows)}
    info.thumb = _thumb(root)
    return info


def scan_workspace(workspace: str, recent: list[str]) -> tuple[list[ProjectInfo], int]:
    """The projects to show, recent ones first, then the others of the workspace from the newest; and how many
    projects the workspace holds. Folders starting with "_" are skipped."""
    ws = Path(workspace) if workspace else None
    found: list[Path] = []
    try:
        for d in ws.iterdir() if ws is not None and ws.is_dir() else ():
            if d.is_dir() and not d.name.startswith(("_", ".")) and (d / project2.FILE).is_file():
                found.append(d)
    except OSError:                                  # a drive gone or a folder refused: what was found so far
        pass
    n_ws = len(found)
    order: list[Path] = []
    for r in recent:
        d = Path(r)
        if d.name.startswith("_") or any(_same(d, x) for x in order) or not (d / project2.FILE).is_file():
            continue
        order.append(d)
    rest = [d for d in found if not any(_same(d, x) for x in order)]
    rest.sort(key=lambda d: (d / project2.FILE).stat().st_mtime, reverse=True)
    infos = []
    for d in order + rest:
        try:
            infos.append(project_info(d))
        except (OSError, ValueError, KeyError, TypeError):
            infos.append(ProjectInfo(d.as_posix(), d.name, state="Cannot be read"))
    return infos, n_ws


# ---------------------------------------------------------------------------------------------- cards

def thumb(info: ProjectInfo) -> QPixmap:
    if info.thumb is not None:
        return QPixmap.fromImage(info.thumb)
    pm = QPixmap(TW, TH)
    pm.fill(QColor("#14161b"))
    p = QPainter(pm)
    rng = np.random.default_rng(sum(map(ord, info.name)) % 2 ** 32)
    for _i in range(160):
        a = int(rng.integers(40, 220))
        p.setPen(QColor(a, a, a))
        p.drawPoint(int(rng.integers(0, TW)), int(rng.integers(0, TH)))
    p.setFont(QFont("Segoe UI", 11))
    p.setPen(QColor("#6b7280"))
    p.drawText(pm.rect(), Qt.AlignCenter, _("No preview"))
    p.end()
    return pm


def card_line(info: ProjectInfo) -> str:
    if not info.nights:
        return _("No frames yet") + " · " + info.date
    if info.integration_h:
        return _("{n} night(s) · {h} · {d}").format(n=info.nights, h=number(info.integration_h, 1) + " h",
                                                     d=info.date)
    return _("{n} night(s) · {d}").format(n=info.nights, d=info.date)


class ProjectCard(QFrame):
    clicked = Signal(str)

    def __init__(self, info: ProjectInfo, parent=None):
        super().__init__(parent)
        self.info = info
        self.setObjectName("pcard")
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("QFrame#pcard { background: #1c1f26; border: 1px solid #2e333d; border-radius: 8px; }"
                           f"QFrame#pcard:hover {{ border-color: {ACCENT}; }} QLabel {{ background: transparent; }}")
        self.setFixedWidth(TW + 20)
        self.setToolTip(info.root)
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(3)
        t = QLabel()
        t.setPixmap(thumb(info))
        v.addWidget(t)
        n = ElidedLabel(info.name, mode=Qt.ElideRight)
        n.setStyleSheet("font-weight: 600; font-size: 11pt;")
        v.addWidget(n)
        v.addWidget(ElidedLabel(info.target or " ", "dim", Qt.ElideRight))
        v.addWidget(label(card_line(info), "small"))
        v.addWidget(label(_(info.state).format(**info.state_args), "small"))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.info.root)


# ---------------------------------------------------------------------------------------------- new project

def folder_name(name: str) -> str:
    """A folder name from a project name (characters Windows refuses become "_")."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name.strip()).rstrip(". ")


class NewProjectDialog(QDialog):
    """Name, target and folder of a new project; the folder is <workspace>/<name>."""

    def __init__(self, workspace: str, name: str = "", target: str = "", parent=None):
        super().__init__(parent)
        self.workspace = Path(workspace)
        self.setWindowTitle(_("New project"))
        self.setMinimumWidth(520)
        v = QVBoxLayout(self)
        v.addWidget(label(_("New project"), "h1"))
        rows = []
        for text, ph in ((_("Name"), "NGC 7000"), (_("Target"), _("optional: filled in by astrometry"))):
            row = QHBoxLayout()
            lb = label(text, "dim")
            rows.append(lb)
            row.addWidget(lb)
            e = QLineEdit()
            e.setPlaceholderText(ph)
            row.addWidget(e, 1)
            v.addLayout(row)
            if text == _("Name"):
                self.name = e
            else:
                self.target = e
        row = QHBoxLayout()
        lb = label(_("Folder"), "dim")
        rows.append(lb)
        row.addWidget(lb)
        self.folder_lb = ElidedLabel("", "small")
        row.addWidget(self.folder_lb, 1)
        v.addLayout(row)
        width = max(80, *(r.sizeHint().width() + 8 for r in rows))
        for r in rows:
            r.setFixedWidth(width)
        self.exists = label(_("A folder with this name already exists in the workspace."), "warn")
        self.exists.setContentsMargins(width + 6, 0, 0, 0)
        v.addWidget(self.exists)
        bb = QHBoxLayout()
        bb.addStretch(1)
        c = QPushButton(_("Cancel"))
        c.clicked.connect(self.reject)
        self.ok = QPushButton(_("Create"))
        self.ok.setObjectName("primary")
        self.ok.setDefault(True)
        self.ok.clicked.connect(self.accept)
        bb.addWidget(c)
        bb.addWidget(self.ok)
        v.addLayout(bb)
        self.name.textChanged.connect(self._update)
        self.name.setText(name)
        self.target.setText(target)
        self._update()

    def folder(self) -> Path:
        return self.workspace / folder_name(self.name.text())

    def _update(self):
        f = folder_name(self.name.text())
        self.folder_lb.setText(self.workspace.as_posix() + "/" + (f or "…"))
        taken = bool(f) and self.folder().exists()
        self.exists.setVisible(taken)
        self.ok.setEnabled(bool(f) and not taken)


# ---------------------------------------------------------------------------------------------- the view

class HomeView(QWidget):
    openProject = Signal(str)                        # project folder
    createProject = Signal(str, str, str)            # folder, name, target
    importFolder = Signal()

    def __init__(self, worker, can_import: bool = False, last=None, parent=None):
        """``last``: the projects a previous Home showed (infos, count), shown until the new scan is back."""
        super().__init__(parent)
        self.worker = worker
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        left = QWidget()
        left.setObjectName("panel")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(18, 20, 18, 18)
        lv.setSpacing(8)
        lv.addWidget(label(_("Workspace"), "h2"))
        self.ws = ElidedLabel(prefs.get("workspace"), "dim")
        lv.addWidget(self.ws)
        ch = QPushButton(_("Change…"))
        ch.clicked.connect(self.change_ws)
        lv.addWidget(ch, 0, Qt.AlignLeft)
        lv.addSpacing(14)
        buttons = []
        for text, fn, primary, on in ((_("New project…"), self.new_project, True, True),
                                      (_("Import folder as new project…"), lambda: self.importFolder.emit(), False,
                                       can_import),
                                      (_("Open project…"), self.open_other, False, True)):
            b = QPushButton(text)
            if primary:
                b.setObjectName("primary")
            b.clicked.connect(lambda _x=False, f=fn: f())
            b.setEnabled(on)
            lv.addWidget(b)
            buttons.append(b)
        lv.addStretch(1)
        self.count = label("", "small")
        lv.addWidget(self.count)
        left.setFixedWidth(max([300] + [b.sizeHint().width() + 40 for b in buttons]))   # wider only for a long text
        root.addWidget(left)
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(24, 20, 24, 20)
        mv.addWidget(label(_("Recent projects"), "h1"))
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        host = QWidget()
        self.grid = QGridLayout(host)
        self.grid.setSpacing(16)
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        sc.setWidget(host)
        mv.addWidget(sc, 1)
        root.addWidget(mid, 1)
        self.cards: list[ProjectCard] = []
        self.last: tuple[list[ProjectInfo], int] | None = None
        worker.done.connect(self._done)
        if last is not None:
            self.show_projects(*last)
        self.rescan()

    # ------------------------------------------------------------ the projects
    def rescan(self):
        """Read the workspace again (in the worker)."""
        self.ws.setText(prefs.get("workspace"))
        self.worker.submit("home", scan_workspace, prefs.get("workspace"), [r["path"] for r in prefs.recent()])

    def _done(self, channel, _n, result):
        if channel == "home":
            self.show_projects(*result)

    def show_projects(self, infos: list[ProjectInfo], n_ws: int):
        self.last = (infos, n_ws)
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        self.cards = []
        for i, info in enumerate(infos):
            c = ProjectCard(info)
            c.clicked.connect(self.openProject.emit)
            self.grid.addWidget(c, i // 3, i % 3)
            self.cards.append(c)
        if not infos:
            self.grid.addWidget(label(_("No projects in this workspace yet."), "dim"), 0, 0)
        self.count.setText(_("No project in the workspace") if n_ws == 0 else
                           _("1 project in the workspace") if n_ws == 1 else
                           _("{n} projects in the workspace").format(n=n_ws))

    # ------------------------------------------------------------ actions
    def change_ws(self):
        d = QFileDialog.getExistingDirectory(self, _("Workspace"), prefs.get("workspace"))
        if d:
            prefs.set("workspace", d)
            self.rescan()

    def ask_new_project(self, name: str = "", target: str = "") -> tuple[str, str, str] | None:
        """The New project dialog: (folder, name, target), or None."""
        d = NewProjectDialog(prefs.get("workspace"), name, target, self.window())
        if d.exec():
            return d.folder().as_posix(), d.name.text().strip(), d.target.text().strip()
        return None

    def new_project(self):
        got = self.ask_new_project()
        if got:
            self.createProject.emit(*got)

    def open_other(self):
        f, _x = QFileDialog.getOpenFileName(self, _("Open project"), prefs.get("workspace"),
                                            _("Projects (*.astroproj)") + ";;"
                                            + _("Projects of the previous version (project.json)"))
        if f:
            self.openProject.emit(Path(f).parent.as_posix())
