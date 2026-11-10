"""Home: the projects opened last, then the sections. New project, open.

There is no workspace (issue #31): a project lives where the user puts it, and Home first shows the projects opened
last (prefs "recent", up to ``prefs.RECENT_MAX``). Below them, the SECTIONS (Settings › Folders: a list of rows, prefs
"home_sections"), each a folder with its name and a one-line description (a row's "only", a folder-name pattern such as
"ours_*", keeps only the projects whose folder matches it): none at first. The "My projects" row of the settings of older versions (the workspace) is no longer shown.

The folders are read in the ApiWorker (``scan_home``): every sub-folder holding a project.astroproj and, for a
sub-folder holding none, its own sub-folders that do; names starting with "_" or ".", pytest and temporary folders are
skipped (``hidden``). The cards show at once; the state of each project (up to date, outdated, never run) follows
from a second, slower read of the worker (``project_state``: the keys of its stages, so the file lists of its
frames), for the open sections. Nothing is written but the settings.

Classes: ``HomeView`` shows a ``SectionView`` per section, each with its ``ProjectCard`` items; ``NewProjectDialog``.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer, Signal
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
from .i18n import _, hours_text, maybe, plural
from .icons import IconLabel, set_icon
from .widgets import ACCENT, WARN, ElidedLabel, cap_dialog, label

TW, TH = 260, 150
CARD_W, GAP = TW + 20, 16
THUMB = Path("cache") / "thumb.jpg"
RECENT_SHOWN = prefs.RECENT_MAX                      # the recent projects shown above the sections: all those kept

# ---------------------------------------------------------------------------------------------- the sections (prefs)


def section_rows() -> list[dict]:
    """The sections of Home, as Settings › Folders keeps them (a copy): [{"label", "folder", "note", "open", "only"}].
    The workspace row of older settings ("My projects", no folder of its own) is left out (issue #31)."""
    rows = []
    for r in prefs.get("home_sections") or []:
        if not isinstance(r, dict) or r.get("workspace"):
            continue
        rows.append({"label": str(r.get("label") or ""), "folder": str(r.get("folder") or ""),
                     "note": str(r.get("note") or ""), "open": bool(r.get("open", True)),
                     "only": str(r.get("only") or "")})
    return rows


def save_rows(rows: list[dict]):
    """Keep the sections."""
    out = []
    for r in rows:
        d = {"label": r.get("label", ""), "folder": r.get("folder", ""), "note": r.get("note", ""),
             "open": bool(r.get("open", True))}
        if r.get("only"):
            d["only"] = r["only"]
        out.append(d)
    prefs.set("home_sections", out)


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
    state: str = "Never run"                         # English, translated when shown (state_text)
    state_args: dict = field(default_factory=dict)
    thumb: QImage | None = None
    n_workflows: int = 0
    description: str = ""
    title: str = ""                                  # the folder's name when another project of the section has its name


@dataclass
class SectionScan:
    row: dict                                        # the section (section_rows)
    infos: list[ProjectInfo]
    found: bool                                      # its folder is there


@dataclass
class HomeScan:
    sections: list[SectionScan]
    recent: list[tuple[ProjectInfo, str]]            # (project, the name of its section or "")


_HIDE = re.compile(r"^(?:[_.]|pytest|tmp|temp(?![a-z]))|[_\-.](?:tmp|temp)$", re.I)


def hidden(name: str) -> bool:
    """A folder Home never shows: "_qc", ".git", "__pycache__", "pytest-12", "tmp3", "temp", "heart_tmp"."""
    return bool(_HIDE.search(name))


def _same(a: Path, b: Path) -> bool:
    return a.as_posix().rstrip("/").lower() == b.as_posix().rstrip("/").lower()


def _key(d: Path | str) -> str:
    return Path(d).as_posix().rstrip("/").lower()


def _inside(d: Path, folder: str) -> bool:
    return bool(folder) and _key(d).startswith(_key(folder) + "/")


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


#: the output previews a card can show when Post has not left a thumbnail yet: the simple output's picture (the full
#: outputs only keep diagnostic previews, not the final image)
OUTPUT_PREVIEWS = (("render_simple", "output"),)


def _newest_output(root: Path) -> Path | None:
    """The preview of the newest output computed in the project (read only; never a .tmp or trashed result)."""
    best, when = None, ""
    for stage, name in OUTPUT_PREVIEWS:
        for d in (root / "store" / stage).glob("*/manifest.json"):
            f = d.parent / "previews" / f"{name}.jpg"
            if d.parent.name.endswith(".tmp") or not f.is_file():
                continue
            try:
                m = json.loads(d.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if m.get("status") == "done" and str(m.get("finished") or "") >= when:
                best, when = f, str(m.get("finished") or "")
    return best


def _thumb(root: Path) -> QImage | None:
    """The card's picture: the thumbnail Post left (``cache/thumb.jpg``, the composition as last shown), else the
    picture of the newest simple output."""
    f = root / THUMB
    if not f.is_file():
        f = _newest_output(root)
    if f is None or not f.is_file():
        return None
    im = QImage(str(f))
    if im.isNull():
        return None
    im = im.scaled(TW, TH, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    return im.copy((im.width() - TW) // 2, (im.height() - TH) // 2, TW, TH)


def project_info(root: Path) -> ProjectInfo:
    p = project2.load(root)
    # a night is a session with lights: a template's empty Sessions are none (GUI exploration 2)
    info = ProjectInfo(root.as_posix(), p.name, p.target.name or "", sum(1 for s in p.sessions if s.lights),
                       n_workflows=len(p.workflows), description=p.description or "")
    inv = max(_manifests(root, "inventory"), key=lambda d: d.get("finished") or "", default=None)
    m = (inv or {}).get("metrics") or {}
    info.integration_h = m.get("integration_h")
    when = m.get("last_time") or p.created
    if not when:
        when = datetime.fromtimestamp((root / project2.FILE).stat().st_mtime).isoformat()
    info.date = str(when)[:10]
    if _running(root):
        info.state = "Running"
    elif not p.workflows:
        info.state = "No workflow yet"
    elif inv is not None or next(_manifests(root), None) is not None:
        info.state, info.state_args = "workflows", {"n": len(p.workflows)}     # the states follow (project_state)
    info.thumb = _thumb(root)
    return info


def project_state(root: str) -> tuple[str, dict]:
    """The state of a project for its card (worker thread; nothing is written): Running, Never run, every workflow
    up to date, or how many are outdated (computed earlier, a setting changed since) and how many have still to run.
    Reads the keys of its stages, so the file lists of its frames."""
    from .. import api
    d = Path(root)
    p = project2.load(d)
    if _running(d):
        return "Running", {}
    if not p.workflows:
        return "No workflow yet", {}
    st = api.states(p, update=False)
    if not any(x.state == "done" for x in st.values()):
        return "Never run", {}
    by: dict[int, set] = {}
    for (wid, _stage), x in st.items():
        by.setdefault(wid, set()).add(x.state)
    n = len(p.workflows)
    stale = sum(1 for s in by.values() if "stale" in s)
    todo = sum(1 for s in by.values() if s != {"done"} and "stale" not in s)
    if not stale and not todo:
        return "up_to_date", {"n": n}
    return "mixed", {"n": n, "stale": stale, "todo": todo}


def project_states(roots: list[str]) -> list[tuple[str, str, dict]]:
    out = []
    for r in roots:
        try:
            out.append((r, *project_state(r)))
        except Exception:  # noqa: BLE001 - a project whose frames cannot be read keeps its first state
            continue
    return out


def state_text(state: str, args: dict) -> str:
    """The state line of a card, translated."""
    if state == "workflows":
        return plural(args["n"], "{n} workflow", "{n} workflows")
    if state == "up_to_date":
        return _("1 workflow up to date") if args["n"] == 1 else _("All {n} workflows up to date").format(**args)
    if state == "mixed":
        parts = [plural(args["n"], "{n} workflow", "{n} workflows")]
        if args.get("stale"):
            parts.append(plural(args["stale"], "{n} outdated", "{n} outdated"))
        if args.get("todo"):
            parts.append(plural(args["todo"], "{n} to run", "{n} to run"))
        return " · ".join(parts)
    return _(state).format(**args)


def _dirs(folder: Path) -> list[Path]:
    """The sub-folders of a folder Home may show (by name; none when it cannot be read)."""
    try:
        with os.scandir(folder) as it:
            found = [Path(e.path) for e in it if not hidden(e.name) and e.is_dir()]
    except OSError:
        return []
    return sorted(found, key=lambda p: p.name.lower())


def _is_project(d: Path) -> bool:
    try:
        return (d / project2.FILE).is_file()
    except OSError:
        return False


def find_projects(folder: Path, only: str = "") -> list[Path]:
    """The projects of a section's folder: its sub-folders holding a project.astroproj and, for a sub-folder holding
    none, its own sub-folders that do (two levels: <group>/<project>). Hidden names are skipped at both; with
    ``only`` (a pattern such as "ours_*"), the projects whose folder name does not match it too."""
    from fnmatch import fnmatch
    out = []
    for d in _dirs(folder):
        if _is_project(d):
            out.append(d)
        else:
            out += [g for g in _dirs(d) if _is_project(g)]
    return [d for d in out if not only or fnmatch(d.name.lower(), only.lower())]


def _mtime(d: Path) -> float:
    try:
        return (d / project2.FILE).stat().st_mtime
    except OSError:
        return 0.0


def _read(d: Path) -> ProjectInfo:
    try:
        return project_info(d)
    except (OSError, ValueError, KeyError, TypeError):
        return ProjectInfo(d.as_posix(), d.name, state="Cannot be read")


def _titles(infos: list[ProjectInfo]):
    """Projects of a section that share a name are told apart by their folders' names."""
    seen: dict[str, int] = {}
    for i in infos:
        seen[i.name.lower()] = seen.get(i.name.lower(), 0) + 1
    for i in infos:
        i.title = Path(i.root).name if seen[i.name.lower()] > 1 else ""


def scan_home(rows: list[dict], recent: list[str]) -> HomeScan:
    """Every section's projects (worker thread): the recently opened first, then the newest; and the recent projects
    shown above the sections."""
    cache: dict[str, ProjectInfo] = {}

    def info_of(d: Path) -> ProjectInfo:
        k = _key(d)
        if k not in cache:
            cache[k] = _read(d)
        return cache[k]

    rank = {_key(r): n for n, r in enumerate(recent)}
    sections = []
    for row in rows:
        folder = row.get("folder") or ""
        try:
            found = bool(folder) and Path(folder).is_dir()
        except OSError:
            found = False
        dirs = find_projects(Path(folder), row.get("only") or "") if found else []
        dirs.sort(key=lambda d: (rank.get(_key(d), len(rank)), -_mtime(d)))
        infos = [info_of(d) for d in dirs]
        _titles(infos)
        sections.append(SectionScan(dict(row), infos, found))
    shown: list[tuple[ProjectInfo, str]] = []
    for r in recent:
        d = Path(r)
        if len(shown) >= RECENT_SHOWN:
            break
        if hidden(d.name) or any(_same(d, Path(i.root)) for i, _w in shown) or not _is_project(d):
            continue
        where = next((s.row.get("label", "") for s in sections if _inside(d, s.row.get("folder", ""))), "")
        shown.append((info_of(d), where))
    return HomeScan(sections, shown)


def known_projects(rows: list[dict], recent: list[str]) -> list[Path]:
    """Every project Home knows (worker thread): the recent ones, then those of the sections, each once."""
    out: list[Path] = []
    seen: set[str] = set()
    cands = [Path(r) for r in recent]
    for row in rows:
        folder = row.get("folder") or ""
        try:
            if folder and Path(folder).is_dir():
                cands += find_projects(Path(folder), row.get("only") or "")
        except OSError:
            continue
    for d in cands:
        if _key(d) not in seen and _is_project(d):
            seen.add(_key(d))
            out.append(d)
    return out


# ---------------------------------------------------------------------------------------------- cards

def thumb(info: ProjectInfo) -> QPixmap:
    if info.thumb is not None:
        pm = QPixmap.fromImage(info.thumb)
    else:
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
        return plural(info.nights, "{n} night · {h} · {d}", "{n} nights · {h} · {d}", h=hours_text(info.integration_h),
                      d=info.date)
    return plural(info.nights, "{n} night · {d}", "{n} nights · {d}", d=info.date)


def card_tooltip(info: ProjectInfo) -> str:
    return "\n".join([info.root] + ([info.description] if info.description else []))


class ProjectCard(QFrame):
    clicked = Signal(str)

    def __init__(self, info: ProjectInfo, tag: str = "", parent=None):
        """``tag``: the name of the section the project belongs to (the recent projects above the sections)."""
        super().__init__(parent)
        self.info = info
        self.setObjectName("pcard")
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("QFrame#pcard { background: #1c1f26; border: 1px solid #2e333d; border-radius: 8px; }"
                           f"QFrame#pcard:hover {{ border-color: {ACCENT}; }} QLabel {{ background: transparent; }}")
        self.setFixedWidth(CARD_W)
        self.setToolTip(card_tooltip(info))
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(3)
        t = QLabel()
        t.setPixmap(thumb(info))
        v.addWidget(t)
        if tag:
            tg = ElidedLabel(tag, "small", Qt.ElideRight)
            tg.setStyleSheet(f"color: {ACCENT}; font-size: 9pt;")
            v.addWidget(tg)
        if info.title:
            title, sub = info.title, " · ".join(x for x in (info.name, info.target) if x)
        else:
            title, sub = info.name, info.target
        self.title = ElidedLabel(title, mode=Qt.ElideRight)
        self.title.setStyleSheet("font-weight: 600; font-size: 11pt;")
        v.addWidget(self.title)
        v.addWidget(ElidedLabel(sub or " ", "dim", Qt.ElideRight))
        v.addWidget(ElidedLabel(card_line(info), "small", Qt.ElideRight))
        self.state = ElidedLabel(state_text(info.state, info.state_args), "small", Qt.ElideRight)
        v.addWidget(self.state)
        v.addStretch(1)

    def set_state(self, state: str, args: dict):
        self.info.state, self.info.state_args = state, args
        self.state.setText(state_text(state, args))

    def mousePressEvent(self, e):                    # noqa: N802 - Qt's name
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.info.root)


# ---------------------------------------------------------------------------------------------- a section

class _Header(QWidget):
    clicked = Signal()

    def mousePressEvent(self, e):                    # noqa: N802
        if e.button() == Qt.LeftButton:
            self.clicked.emit()


class SectionView(QWidget):
    """A section of Home: its header (name, number of projects, folder; a click opens or closes it), its description
    and its cards."""
    toggled = Signal(bool)

    def __init__(self, title: str, folder: str = "", note: str = "", count: str = "", is_open: bool = True,
                 collapsible: bool = True, warn: bool = False, parent=None):
        super().__init__(parent)
        self.collapsible = collapsible
        self.is_open = is_open
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        self.header = _Header()
        h = QHBoxLayout(self.header)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        self.arrow = IconLabel("chevron_down", 16, "dim")
        self.arrow.setVisible(collapsible)
        h.addWidget(self.arrow)
        self.title = label(title, "h1")
        h.addWidget(self.title)
        self.count = label(count, "warn" if warn else "dim")
        h.addWidget(self.count)
        h.addSpacing(10)
        self.folder = ElidedLabel(folder, "small")
        h.addWidget(self.folder, 1)
        v.addWidget(self.header)
        self.note = label(note, "small", wrap=True)
        self.note.setContentsMargins(24 if collapsible else 0, 0, 0, 0)
        self.note.setVisible(bool(note))
        v.addWidget(self.note)
        if collapsible:
            self.header.setCursor(Qt.PointingHandCursor)
            self.header.setToolTip(_("Click to show or hide these projects"))
            self.header.clicked.connect(lambda: self.set_open(not self.is_open, True))
        self.body = QWidget()
        self.grid = QGridLayout(self.body)
        self.grid.setContentsMargins(0, 8, 0, 0)
        self.grid.setSpacing(GAP)
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        v.addWidget(self.body)
        self.cards: list[ProjectCard] = []
        self.empty: QLabel | None = None
        self._cols = 0
        self.set_open(is_open)

    def set_open(self, on: bool, by_click: bool = False):
        self.is_open = on or not self.collapsible
        self.arrow.set_role("chevron_down" if self.is_open else "chevron_right")
        self.body.setVisible(self.is_open)
        if by_click:
            self.toggled.emit(self.is_open)

    def set_cards(self, cards: list[ProjectCard], empty: str = ""):
        self.cards = cards
        if not cards and empty:
            self.empty = label(empty, "dim")
        self.place(max(self._cols, 1), force=True)

    def place(self, cols: int, force: bool = False):
        if cols == self._cols and not force:
            return
        self._cols = cols
        while self.grid.count():
            self.grid.takeAt(0)
        for i, card in enumerate(self.cards):
            self.grid.addWidget(card, i // cols, i % cols)
        if self.empty is not None:
            self.grid.addWidget(self.empty, 0, 0)

# ---------------------------------------------------------------------------------------------- new project


def folder_name(name: str) -> str:
    """A folder name from a project name (characters Windows refuses become "_")."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name.strip()).rstrip(". ")


def holds_frames(folder: Path) -> bool:
    """Whether a folder holds frames (camera raw, FITS, JPEG, PNG or TIFF files) itself: the folder of a session's
    frames."""
    from ..core.project import frame_kind
    try:
        return any(frame_kind(q.name) for q in folder.iterdir())
    except OSError:
        return False


def projects_location(avoid: str = "") -> str:
    """Where a new project is proposed (GUI exploration 7.1): beside the last project opened, else the last folder
    used, else the home folder; never in ``avoid`` or below it (the folder of the frames an import reads), never in a
    folder of frames (GUI exploration 4: the darks just added to a session were the last folder used)."""
    bad = Path(avoid).resolve() if avoid else None
    last = Path(prefs.last_folder())
    for c in [Path(r["path"]).parent for r in prefs.recent()] + [last, Path.home()]:
        try:
            if not c.is_dir():
                continue
            r = c.resolve()
        except OSError:
            continue
        if bad is not None and (r == bad or bad in r.parents):
            continue
        if c is last and holds_frames(c):
            continue
        return c.as_posix()
    return (bad.parent if bad is not None else Path.home()).as_posix()


class NewProjectDialog(QDialog):
    """Name and folder of a project not named yet, asked once, at its first Save or Run; the folder is <location>/<name>,
    the location chosen with Browse… (where the projects are at first: ``projects_location``)."""

    def __init__(self, location: str, name: str = "", parent=None):
        super().__init__(parent)
        self.location = Path(location)
        self.setWindowTitle(_("Name the project"))
        self.setMinimumWidth(520)
        cap_dialog(self)
        v = QVBoxLayout(self)
        v.addWidget(label(_("Name the project"), "h1"))
        rows = []
        row = QHBoxLayout()
        lb = label(_("Name"), "dim")
        rows.append(lb)
        row.addWidget(lb)
        self.name = QLineEdit()
        self.name.setPlaceholderText("NGC 7000")
        row.addWidget(self.name, 1)
        v.addLayout(row)
        row = QHBoxLayout()
        lb = label(_("Folder"), "dim")
        rows.append(lb)
        row.addWidget(lb)
        self.folder_lb = ElidedLabel("", "small")
        row.addWidget(self.folder_lb, 1)
        self.browse = QPushButton(_("Browse…"))
        self.browse.setToolTip(_("The folder the project's own folder is made in"))
        self.browse.clicked.connect(self._browse)
        row.addWidget(self.browse)
        v.addLayout(row)
        width = max(80, *(r.sizeHint().width() + 8 for r in rows))
        for r in rows:
            r.setFixedWidth(width)
        self.exists = label(_("A folder with this name already exists there."), "warn")
        self.exists.setContentsMargins(width + 6, 0, 0, 0)
        v.addWidget(self.exists)
        bb = QHBoxLayout()
        bb.addStretch(1)
        c = QPushButton(_("Cancel"))
        c.clicked.connect(self.reject)
        self.ok = QPushButton(_("Save"))
        self.ok.setObjectName("primary")
        self.ok.setDefault(True)
        self.ok.clicked.connect(self.accept)
        bb.addWidget(c)
        bb.addWidget(self.ok)
        v.addLayout(bb)
        self.name.textChanged.connect(self._update)
        self.name.setText(name)
        self._update()

    def folder(self) -> Path:
        return self.location / folder_name(self.name.text())

    def _browse(self):
        """The location chosen (issue #31: the project decides where it lives, not a workspace)."""
        d = QFileDialog.getExistingDirectory(self, _("Location of the project"), self.location.as_posix())
        if d:
            self.location = Path(d)
            prefs.remember_folder(d)
            self._update()

    def _update(self):
        f = folder_name(self.name.text())
        self.folder_lb.setText(self.location.as_posix() + "/" + (f or "…"))
        taken = bool(f) and self.folder().exists()
        self.exists.setVisible(taken)
        self.ok.setEnabled(bool(f) and not taken)


def ask_project_name(parent, name: str = "") -> tuple[Path, str] | None:
    """The project's folder and name, as the user says them, or None. Proposed where the projects are, never in the
    folder of the frames last added (GUI exploration 4)."""
    d = NewProjectDialog(projects_location(), name, parent)
    return (d.folder(), d.name.text().strip()) if d.exec() else None


# ---------------------------------------------------------------------------------------------- the view

class HomeView(QWidget):
    openProject = Signal(str)                        # project folder
    newProject = Signal()                            # New project: the graph at once, the project without a name
    openImages = Signal()                            # images as layers of Post, the project made for them

    def __init__(self, worker, last=None, parent=None):
        """``last``: what a previous Home showed (its HomeScan), shown until the new scan is back."""
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
        buttons = []
        for text, fn, primary, role in ((_("New project"), self.newProject.emit, True, "new"),
                                        (_("Open images…"), self.openImages.emit, False, "block_image"),
                                        (_("Open project…"), self.open_other, False, "open")):
            b = QPushButton(text)                    # compact names (decision 12)
            set_icon(b, role, 16, "on_accent" if primary else "normal")
            if primary:
                b.setObjectName("primary")
            b.clicked.connect(lambda _x=False, f=fn: f())
            lv.addWidget(b)
            buttons.append(b)
        lv.addStretch(1)
        left.setFixedWidth(max([300] + [b.sizeHint().width() + 40 for b in buttons]))   # wider only for a long text
        root.addWidget(left)
        self.recent_cards: list[ProjectCard] = []
        self.sections: list[SectionView] = []
        self.last: HomeScan | None = None
        self._resolved: set[str] = set()             # roots whose state the worker read
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        host = QWidget()
        self.lay = QVBoxLayout(host)
        self.lay.setContentsMargins(24, 20, 24, 20)
        self.lay.setSpacing(26)
        self.lay.addStretch(1)
        self.scroll.setWidget(host)
        self.scroll.viewport().installEventFilter(self)
        root.addWidget(self.scroll, 1)
        worker.done.connect(self._done)
        if isinstance(last, HomeScan):
            self.show_scan(last)
        self.rescan()

    # ------------------------------------------------------------ the projects
    def rescan(self):
        """Read the recent projects and the sections' folders again (in the worker)."""
        self._resolved = set()
        self.worker.submit("home", scan_home, section_rows(), [r["path"] for r in prefs.recent()])

    def all_cards(self) -> list[ProjectCard]:
        return self.recent_cards + [c for s in self.sections for c in s.cards]

    def _done(self, channel, _n, result):
        if channel == "home" and isinstance(result, HomeScan):
            self.show_scan(result)
            self._ask_states()
        elif channel == "home.states":
            for root, state, args in result:
                self._resolved.add(root)
                for c in self.all_cards():
                    if c.info.root == root:
                        c.set_state(state, args)

    def _ask_states(self):
        """The states of the cards shown (open sections, recent projects) not read yet: one request for all of them
        (a newer request replaces one still waiting)."""
        roots = []
        shown = self.recent_cards + [c for s in self.sections if s.is_open for c in s.cards]
        for c in shown:
            r = c.info.root
            if c.info.state == "workflows" and r not in self._resolved and r not in roots:
                roots.append(r)
        if roots:
            self.worker.submit("home.states", project_states, roots)

    def show_scan(self, scan: HomeScan):
        self.last = scan
        for s in self.sections:
            self.lay.removeWidget(s)
            s.hide()
            s.deleteLater()
        self.sections, self.recent_cards = [], []
        at = 0
        if scan.recent:
            sec = SectionView(_("Recent projects"), collapsible=False)
            self.recent_cards = [self._card(info, maybe(where)) for info, where in scan.recent]
            sec.set_cards(self.recent_cards)
            self._add(sec, at)
            at += 1
        for s in scan.sections:
            row = s.row
            if not row.get("folder"):
                continue
            n = len(s.infos)
            count = plural(n, "{n} project", "{n} projects") if s.found else _("Folder not found")
            sec = SectionView(maybe(row.get("label", "")) or Path(row["folder"]).name, row["folder"],
                              maybe(row.get("note", "")), count, bool(row.get("open", True)), warn=not s.found)
            cards = [self._card(i) for i in s.infos]
            if not s.found:
                empty = _("This folder is not there: is its disk plugged in?")
            else:
                empty = _("No projects in this folder.")
            sec.set_cards(cards, empty)
            sec.toggled.connect(lambda on, sc=s: self._toggled(sc, on))
            self._add(sec, at)
            at += 1
        if not scan.recent and not any(s.row.get("folder") for s in scan.sections):
            sec = SectionView(_("Recent projects"), collapsible=False)
            sec.set_cards([], _("No project opened yet: create one, or open one with Open project…"))
            self._add(sec, at)
        self._place()

    def _card(self, info: ProjectInfo, tag: str = "") -> ProjectCard:
        c = ProjectCard(info, tag)
        c.clicked.connect(self.openProject.emit)
        return c

    def _add(self, sec: SectionView, at: int):
        self.sections.append(sec)
        self.lay.insertWidget(at, sec)

    def _toggled(self, s: SectionScan, on: bool):
        """A section opened or closed by a click on its header: kept in the settings; its states are read."""
        s.row["open"] = on
        rows = section_rows()
        for r in rows:
            if _key(r["folder"]) == _key(s.row.get("folder", "")) and r["label"] == s.row.get("label"):
                r["open"] = on
        save_rows(rows)
        if on:
            self._ask_states()

    def columns(self) -> int:
        m = self.lay.contentsMargins()
        width = self.scroll.viewport().width() - m.left() - m.right()
        return max(1, (width + GAP) // (CARD_W + GAP))

    def _place(self):
        cols = self.columns()
        for s in self.sections:
            s.place(cols)

    def eventFilter(self, obj, e):                   # noqa: N802
        if obj is self.scroll.viewport() and e.type() == QEvent.Resize:
            self._place()
        return False

    # ------------------------------------------------------------ actions
    def open_other(self):
        f, _x = QFileDialog.getOpenFileName(self, _("Open project"), prefs.last_folder(),
                                            # one kind of file for the user: a project, of this version or older
                                            _("FLOAT project") + " (*.astroproj project.json)",
                                            options=QFileDialog.HideNameFilterDetails)   # no file name shown
        if f:
            prefs.remember_folder(Path(f).parent.parent)       # the folder the project's folder is in
            self.openProject.emit(Path(f).parent.as_posix())
