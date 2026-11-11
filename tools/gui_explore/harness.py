"""GUI exploration harness: the real window, offscreen, driven by real gestures, in throw-away projects.

Safety (always on, whatever a script does):
- the settings file, the app's home folder, the projects and the scratch are in a fresh temporary folder;
- no GPU: CUDA_VISIBLE_DEVICES is -1 (an empty value does not hide the GPUs here) before anything imports torch;
- no run can start: the worker spawn raises ``RunsBlocked`` (a Run shows as a start that failed: not an anomaly);
  opt-in, GUI_EXPLORE_CPU_RUNS=1 set before the harness is imported: runs really start, forced on the CPU (the plan
  file rewritten to device "cpu", the worker with CUDA_VISIBLE_DEVICES=-1) — for tiny synthetic projects only;
- message boxes, file dialogs and context menus never block: they are recorded (and answered from queues).

Gestures go through Qt events, as a user's would: mouse presses, moves with the button held, releases, double clicks,
right clicks (context menus), keys, and drag and drop (``QDrag.exec`` is replaced by the same enter/move/drop events
a real drop sends, at the point the script chose).

Anomalies: ``ex.note(severity, title, detail, shot=True)`` appends a line to ``<out>/anomalies.jsonl`` with a screenshot.

    from harness import Explorer
    ex = Explorer("palette", lang="fr")
    root = ex.project("blank")                  # or "session", "image", "session+image", "two_flows", "mono", "legacy"
    w = ex.open(root)
    ex.drag_palette("calibration", ex.ghost_center(0))
    ...
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # a training may share the machine: few threads...
if sys.platform == "win32":             # ...and below-normal priority (inherited by the run workers it starts)
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
TMP = Path(tempfile.mkdtemp(prefix="gui_explore_", dir=os.environ.get("TEMP")))
os.environ["ASTRO_GPU_SETTINGS"] = str(TMP / "settings.json")
os.environ["ASTRO_GPU_HOME"] = str(TMP / "home")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QContextMenuEvent, QDrag, QDragEnterEvent, QDragMoveEvent, QDropEvent, QMouseEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMenu, QMessageBox, QWidget  # noqa: E402
from shiboken6 import isValid  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import prefs  # noqa: E402
from astro_gpu.client.services.worker import Gpu, Machine  # noqa: E402
from astro_gpu.client.widgets import QSS  # noqa: E402
from astro_gpu.core import project2, runner2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)
OUT_BASE = Path(os.environ.get("GUI_EXPLORE_OUT", "F:/astro_essais/A_REGARDER/exploration4"))   # one folder a round
MACHINE = Machine([Gpu("cuda:0", "NVIDIA RTX 4090", 24.0), Gpu("cuda:1", "NVIDIA RTX 3070", 8.0)], 16, 64.0)


class RunsBlocked(RuntimeError):
    pass


def _blocked(*a, **k):
    raise RunsBlocked("exploration: runs are blocked (no GPU, no computation)")


_real_spawn = runner2._spawn_worker
CPU_RUNS = os.environ.get("GUI_EXPLORE_CPU_RUNS") == "1"


def _cpu_spawn(root, plan_file, visible, log_file, python=None, notes=None):
    """Opt-in (GUI_EXPLORE_CPU_RUNS=1, for tiny data only): the run's worker really starts, forced on the CPU: its plan
    file says device "cpu" and the worker sees no GPU (CUDA_VISIBLE_DEVICES=-1), whatever the window chose."""
    d = json.loads(Path(plan_file).read_text(encoding="utf-8"))
    d["device"], d["visible"], d["inside"] = "cpu", "-1", "cpu"
    Path(plan_file).write_text(json.dumps(d), encoding="utf-8")
    return _real_spawn(root, plan_file, "-1", log_file, python or sys.executable, notes)


# nothing computes, whatever the window asks (default); GUI_EXPLORE_CPU_RUNS=1: CPU runs of tiny projects
runner2._spawn_worker = _cpu_spawn if CPU_RUNS else _blocked


class Explorer:
    def __init__(self, area: str, lang: str = "fr", size=(1600, 950)):
        self.area = area
        self.out = OUT_BASE / area
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = self.out / "anomalies.jsonl"
        self.app = QApplication.instance() or QApplication([])
        self.app.setStyleSheet(QSS + appmod.EXTRA_QSS)
        self.ws = TMP / "ws"
        self.ws.mkdir(exist_ok=True)
        prefs.load()
        prefs.set("home_sections", [{"label": "Bench", "folder": self.ws.as_posix(), "note": "", "open": True}])
        prefs.set("last_folder", self.ws.as_posix())
        prefs.set("cache_dir", (TMP / "scratch").as_posix())
        appmod.apply_language(lang)
        self.lang = lang
        appmod.machine_info = lambda: MACHINE
        self.size = size
        self.popups: list[tuple] = []                 # (kind, title, text) of every message box asked
        self.file_queue: list = []                     # answers of the file dialogs, in order ("" or [] when empty)
        self.file_asks: list[str] = []
        self.menus: list[list[dict]] = []              # every context menu shown: [{text, enabled, tip, sub}]
        self.menu_pick: str | None = None              # the action to trigger in the next menu (a part of its text)
        self.modals: list[tuple] = []                  # every modal dialog opened: (class, title)
        self.dialog_handler = None                     # what to do with the next modal dialog (else Cancel)
        self.drop_at: tuple[QWidget, QPoint] | None = None
        self.drops: list[dict] = []
        self.w: appmod.Main | None = None
        self._n = 0
        self._patch()

    # ------------------------------------------------------------ never blocking
    def _patch(self):
        ex = self

        def box(kind):
            def f(*a, **k):
                ex.popups.append((kind, str(a[1]) if len(a) > 1 else "", str(a[2]) if len(a) > 2 else ""))
                return QMessageBox.Yes if kind == "question" else QMessageBox.Ok
            return f
        for kind in ("information", "warning", "critical", "question"):
            setattr(QMessageBox, kind, staticmethod(box(kind)))
        QMessageBox.exec = lambda self: (ex.popups.append(("box", self.windowTitle(), self.text())), QMessageBox.Ok)[1]

        def fd(kind):
            def f(*a, **k):
                ex.file_asks.append(f"{kind}: {a[1] if len(a) > 1 else ''}")
                ans = ex.file_queue.pop(0) if ex.file_queue else ([] if kind == "files" else "")
                if kind in ("file", "files", "save"):
                    return (ans, "")
                return ans
            return f
        QFileDialog.getOpenFileName = staticmethod(fd("file"))
        QFileDialog.getOpenFileNames = staticmethod(fd("files"))
        QFileDialog.getExistingDirectory = staticmethod(fd("folder"))
        QFileDialog.getSaveFileName = staticmethod(fd("save"))

        # menus and modal dialogs run their own event loop: a timer inside it records them and closes them (a menu:
        # its items, then the action asked; a modal dialog: ``dialog_handler(d)`` when set, else Cancel)
        self._popup_timer = QTimer()
        self._popup_timer.setInterval(30)
        self._popup_timer.timeout.connect(self._popup_check)
        self._popup_timer.start()

        def drag_exec(self_drag, *a, **k):
            mime = self_drag.mimeData()
            target = ex.drop_at
            ex.drop_at = None
            rec = {"mime": mime.text() if mime.hasText() else "", "accepted": False}
            ex.drops.append(rec)
            if target is None:
                return Qt.IgnoreAction
            widget, pos = target
            rec["accepted"] = ex._drop(widget, pos, mime)
            return Qt.CopyAction if rec["accepted"] else Qt.IgnoreAction
        QDrag.exec = drag_exec

    def _popup_check(self):
        m = QApplication.activePopupWidget()
        if isinstance(m, QMenu) and m.isVisible():
            self.menus.append(self.menu_items(m))
            pick, self.menu_pick = self.menu_pick, None
            act = self._find_action(m, pick) if pick else None
            m.close()
            if act is not None and act.isEnabled():
                QTimer.singleShot(0, act.trigger)
            return
        d = QApplication.activeModalWidget()
        if isinstance(d, QDialog) and d.isVisible() and not getattr(d, "_explored", False):
            d._explored = True
            self.modals.append((d.__class__.__name__, d.windowTitle()))
            h, self.dialog_handler = self.dialog_handler, None
            if h is not None:
                QTimer.singleShot(0, lambda: h(d))
            else:
                QTimer.singleShot(0, d.reject)

    def menu_items(self, m: QMenu) -> list[dict]:
        out = []
        for a in m.actions():
            if a.isSeparator():
                out.append({"text": "---"})
                continue
            d = {"text": a.text().replace("&&", "&"), "enabled": a.isEnabled(), "tip": a.toolTip(), "checked": a.isChecked()}
            if a.menu() is not None:
                d["sub"] = self.menu_items(a.menu())
            out.append(d)
        return out

    def _find_action(self, m: QMenu, part: str):
        """The action whose text (its shortcut left out) is ``part``, else the first that holds it: "Vers Post" is
        the block's own item, not "Sans étoiles vers Post"."""
        def flat(menu):
            for a in menu.actions():
                yield from (flat(a.menu()) if a.menu() is not None else (a,))
        acts = list(flat(m))
        low = part.lower()
        return next((a for a in acts if a.text().replace("&&", "&").split("\t")[0].lower() == low), None) or \
            next((a for a in acts if low in a.text().replace("&&", "&").lower()), None)

    def _drop(self, widget: QWidget, pos: QPoint, mime: QMimeData) -> bool:
        pf = QPointF(pos)
        enter = QDragEnterEvent(pos, Qt.CopyAction | Qt.MoveAction, mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(widget, enter)
        self.settle(2)
        move = QDragMoveEvent(pos, Qt.CopyAction | Qt.MoveAction, mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(widget, move)
        self.settle(2)
        if not move.isAccepted() and not enter.isAccepted():
            QApplication.sendEvent(widget, QEvent(QEvent.DragLeave))
            self.settle()
            return False
        drop = QDropEvent(pf, Qt.CopyAction | Qt.MoveAction, mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(widget, drop)
        self.settle(4)
        return drop.isAccepted()

    # ------------------------------------------------------------ data and projects
    def raws(self, folder: str | Path, n: int, ext: str = ".ARW", start: int = 5850) -> Path:
        """Fake camera raw files (they are listed and counted, never decoded: nothing runs)."""
        d = TMP / "data" / folder
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"DSC0{start + i}{ext}").write_bytes(b"\0" * (1000 + i))
        return d

    def fits_frames(self, folder: str, n: int, filt: str = "", h: int = 64, w: int = 96) -> Path:
        """Small real FITS frames, with a FILTER keyword when ``filt`` (a monochrome session)."""
        from astropy.io import fits
        d = TMP / "data" / folder
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(len(folder))
        for i in range(n):
            hd = fits.Header()
            if filt:
                hd["FILTER"] = filt
            hd["EXPTIME"] = 300.0
            fits.writeto(d / f"Light_{filt or 'L'}_{i:03d}.fits", rng.normal(1000, 20, (h, w)).astype(np.uint16),
                         hd, overwrite=True)
        return d

    def image_file(self, name: str = "stack.fits", stretched: bool = False) -> Path:
        """A small linear stack (FITS, 3 planes) or a stretched picture (PNG)."""
        d = TMP / "data" / "images"
        d.mkdir(parents=True, exist_ok=True)
        f = d / name
        rng = np.random.default_rng(7)
        if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            from PIL import Image
            Image.fromarray((rng.random((120, 160, 3)) * 255).astype(np.uint8)).save(f)
        else:
            from astropy.io import fits
            fits.writeto(f, (0.1 + rng.normal(0, 0.01, (3, 120, 160))).astype(np.float32), overwrite=True)
        return f

    def project(self, kind: str, name: str | None = None) -> Path:
        """A project of a typical state, saved in the bench folder:
        blank · session (one session, 15 raws) · session_empty_wf (a session and an empty workflow) · image (one
        image workflow) · session+image · two_sessions · two_flows (two parallel flows) · mono (Ha and OIII
        sessions, their workflows) · legacy (a workflow of options, "Full") · jpg_mistake (14 raws and a JPEG)."""
        from astro_gpu.core import workflow as wfl
        self._n += 1
        name = name or f"{kind}_{self._n}"
        root = self.ws / name
        p = project2.create(root, name, Target(name="M 31"))
        p.stack_mode = "per_session"
        if kind in ("session", "session_empty_wf", "session+image", "two_sessions", "two_flows", "legacy",
                    "jpg_mistake"):
            p.sessions.append(project2.Session(1, "Session 1", lights=[str(self.raws(f"{name}/n1", 15))]))
        if kind in ("two_sessions", "two_flows"):
            p.sessions.append(project2.Session(2, "Session 2", lights=[str(self.raws(f"{name}/n2", 12, start=6000))]))
        if kind == "jpg_mistake":
            j = TMP / "data" / name / "phone" / "DSC05869.JPG"
            j.parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(j)
            p.sessions[0].lights.append(str(j))
        if kind == "mono":
            for i, f in enumerate(("Ha", "OIII")):
                p.sessions.append(project2.Session(i + 1, f, lights=[str(self.fits_frames(f"{name}/{f}", 6, f))],
                                                   filter=f))
        if kind == "legacy":
            p.add_workflow("Full", wfl.TEMPLATES["Full"])
        project2.save(p)
        if kind in ("session_empty_wf", "two_flows", "session+image", "jpg_mistake", "two_sessions"):
            w = p.add_workflow("Workflow", wfl.TEMPLATES["Classic stack"])
            p.set_blocks(w.id, [])
            if kind == "two_flows":
                w.sessions = [1]
                w2 = p.add_workflow("Flow", wfl.TEMPLATES["Classic stack"])
                w2.sessions = [2]
                p.set_blocks(w2.id, [])
        if kind in ("image", "session+image"):
            w = p.add_workflow("Image", wfl.TEMPLATES["Classic stack"])
            w.source = {"path": str(self.image_file()), "stretched": False}
            p.set_blocks(w.id, [])
        if kind == "mono":
            for sid, f in ((1, "Ha"), (2, "OIII")):
                w = p.add_workflow(f, wfl.TEMPLATES["Classic stack"])
                w.sessions = [sid]
                p.set_blocks(w.id, [wfl.node(b) for b in ("calibration", "registration", "stacking")])
        project2.save(p)
        return root

    # ------------------------------------------------------------ the window
    def open(self, root: Path | None = None) -> appmod.Main:
        """The main window, on Home, or on a project."""
        if self.w is None:
            self.w = appmod.Main()
            self.w.resize(*self.size)
            self.w.show()
            self.w.worker.wait(20)
            self.settle()
        if root is not None:
            self.w.open_project(root)
            self.settle(10)
        return self.w

    def new_project(self, name: str = "Nouveau", template: str | None = None) -> Path:
        """A new project, as a folder given at once (``template``: the key of the card chosen on its blank page, None:
        the blank page)."""
        root = self.ws / name
        self.open()
        self.w.create_project(str(root), name, "M 31")
        self.settle(10)
        if template:
            from astro_gpu.client.gallery import GALLERY
            self.w.graph.template_chosen(next(t for t in GALLERY if t["key"] == template))
            self.settle(10)
        return root

    @property
    def s(self):
        return self.w.s

    @property
    def page(self):
        return self.w.graph

    @property
    def view(self):
        return self.w.graph.graph

    def settle(self, n: int = 6):
        for _ in range(n):
            QApplication.processEvents()
            if self.w is not None:
                self.w.worker.wait(2)
            QApplication.processEvents()
            time.sleep(0.01)

    def dialogs(self) -> list[QDialog]:
        return [x for x in QApplication.topLevelWidgets() if isinstance(x, QDialog) and x.isVisible()]

    def close_dialogs(self):
        for d in self.dialogs():
            d.reject()
        self.settle()

    # ------------------------------------------------------------ what is drawn
    def blocks(self) -> list:
        from astro_gpu.client.graph import BlockItem
        return [it for it in self.view.scene().items() if isinstance(it, BlockItem)]

    def block(self, key: str, pid: int | None = None, sid: int | None = None):
        """The drawn block of a key ("calibration", "session", "image", "fond~2"...), of a workflow or a session."""
        for it in self.blocks():
            n = it.node
            if n.key != key:
                continue
            if sid is not None and n.sid != sid:
                continue
            if pid is not None and pid not in [q.pid for q in n.paths]:
                continue
            return it
        return None

    def point_of(self, item, where: str = "center") -> QPoint:
        r = item.sceneBoundingRect()
        if where == "cross":                           # the × itself: the bounding rect has the port's and the review
            return self.view.mapFromScene(item.mapToScene(item._x_rect().center()))   # dot's margins
        p = {"center": r.center(), "left": QPointF(r.left() + 12, r.center().y()),
             "port": QPointF(r.right() - 1, r.center().y())}[where]
        return self.view.mapFromScene(p)

    def ghosts(self) -> list:
        return list(getattr(self.view, "ghosts", []))

    def new_flow_zone(self):
        return getattr(self.view, "new_flow_ghost", None)

    def texts(self, root: QWidget | None = None) -> list[str]:
        """Every text a visible widget of ``root`` (the window) shows, and the graph's drawn texts."""
        from PySide6.QtWidgets import QAbstractButton, QLabel
        root = root or self.w
        out = []
        for x in [root] + root.findChildren(QWidget):
            if not x.isVisible():
                continue
            if isinstance(x, (QLabel, QAbstractButton)) and x.text():
                out.append(x.text().replace("&&", "&") if isinstance(x, QAbstractButton) else x.text())
            if x.toolTip():
                out.append(x.toolTip())
        for it in self.view.scene().items() if self.w is not None and self.w.graph is not None else []:
            for attr in ("title", "summary", "text"):
                v = getattr(it, attr, None)
                if isinstance(v, str) and v:
                    out.append(v)
            if it.toolTip():
                out.append(it.toolTip())
        return out

    # ------------------------------------------------------------ gestures
    def _mouse(self, widget, kind, pos: QPoint, button=Qt.LeftButton, buttons=None):
        if not isValid(widget):                        # the widget went with a rebuild (a drop rebuilds the palette)
            return
        e = QMouseEvent(kind, QPointF(pos), QPointF(widget.mapToGlobal(pos)), button,
                        button if buttons is None else buttons, Qt.NoModifier)
        QApplication.sendEvent(widget, e)

    def click(self, widget, pos: QPoint | None = None, button=Qt.LeftButton):
        QTest.mouseClick(widget, button, Qt.NoModifier, pos if pos is not None else widget.rect().center())
        self.settle()

    def double_click(self, widget, pos: QPoint | None = None):
        QTest.mouseDClick(widget, Qt.LeftButton, Qt.NoModifier, pos if pos is not None else widget.rect().center())
        self.settle()

    def hover(self, widget, pos: QPoint):
        QTest.mouseMove(widget, pos)
        self._mouse(widget, QEvent.MouseMove, pos, Qt.NoButton, Qt.NoButton)
        self.settle(2)

    def click_block(self, item, where: str = "center"):
        vp = self.view.viewport()
        p = self.point_of(item, where)
        self.hover(vp, p)
        self.click(vp, p)

    def double_click_block(self, item):
        self.double_click(self.view.viewport(), self.point_of(item))

    def right_click_block(self, item, pick: str | None = None) -> list[dict]:
        """The block's context menu as the user opens it: its items (and ``pick`` triggered, a part of its text)."""
        vp = self.view.viewport()
        p = self.point_of(item)
        self.menu_pick = pick
        n = len(self.menus)
        QApplication.sendEvent(vp, QContextMenuEvent(QContextMenuEvent.Mouse, p, vp.mapToGlobal(p)))
        self.settle()
        return self.menus[-1] if len(self.menus) > n else []

    def key(self, k, widget=None, mod=Qt.NoModifier):
        widget = widget or self.view.viewport()
        if self.w is not None:                         # offscreen the window is not active: shortcuts need it
            self.w.activateWindow()
            QApplication.setActiveWindow(self.w) if hasattr(QApplication, "setActiveWindow") else None
            self.settle(1)
        widget.setFocus()
        QTest.keyClick(widget, k, mod)
        self.settle()

    def undo(self):
        self.key(Qt.Key_Z, self.w, Qt.ControlModifier)

    def redo(self):
        self.key(Qt.Key_Y, self.w, Qt.ControlModifier)

    def card(self, block: str):
        return self.page.cards.get(block)

    def drag_mouse(self, widget, a: QPoint, b: QPoint, drop_widget=None, drop_pos: QPoint | None = None):
        """Press at ``a``, move with the button held to ``b``, release: a drag the widget starts (QDrag) ends as a
        drop on ``drop_widget`` at ``drop_pos`` (default: ``widget`` at ``b``)."""
        self.drop_at = (drop_widget or widget, drop_pos if drop_pos is not None else b)
        self._mouse(widget, QEvent.MouseButtonPress, a)
        for t in (0.25, 0.5, 0.75, 1.0):
            p = QPoint(int(a.x() + (b.x() - a.x()) * t), int(a.y() + (b.y() - a.y()) * t))
            self._mouse(widget, QEvent.MouseMove, p, Qt.NoButton, Qt.LeftButton)
        self._mouse(widget, QEvent.MouseButtonRelease, b, Qt.LeftButton, Qt.NoButton)
        self.drop_at = None
        self.settle(6)

    def drag_palette(self, block: str, to: QPoint) -> dict | None:
        """A palette block dragged with the mouse onto the graph at ``to`` (viewport coordinates). Returns the drop
        record (None: the card did not start a drag, e.g. greyed)."""
        c = self.card(block)
        if c is None:
            return None
        n = len(self.drops)
        self.drag_mouse(c, QPoint(20, 12), QPoint(60, 14), self.view.viewport(), to)
        return self.drops[-1] if len(self.drops) > n else None

    def drag_palette_hover(self, block: str, to: QPoint):
        """The drag held over ``to`` without dropping: the places the graph shows meanwhile (screenshot them)."""
        c = self.card(block)
        mime = QMimeData()
        mime.setText("block:" + block)
        vp = self.view.viewport()
        QApplication.sendEvent(vp, QDragEnterEvent(to, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
        QApplication.sendEvent(vp, QDragMoveEvent(to, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
        self.settle(3)
        return c

    def drag_leave(self):
        QApplication.sendEvent(self.view.viewport(), QEvent(QEvent.DragLeave))
        self.settle()

    def double_click_card(self, block: str):
        c = self.card(block)
        if c is not None:
            self.double_click(c)

    def ghost_center(self, gh) -> QPoint:
        return self.view.mapFromScene(gh.sceneBoundingRect().center())

    def drag_block_to_palette(self, item):
        """A placed block dragged out onto the palette (to remove it)."""
        vp = self.view.viewport()
        a = self.point_of(item)
        bar = self.page.palette_bar.viewport()
        self.drag_mouse(vp, a, QPoint(a.x() + 40, a.y()), bar, bar.rect().center())

    def pull_port_to_post(self, item):
        """A block's port pulled onto the Post block."""
        vp = self.view.viewport()
        a = self.point_of(item, "port")
        post = self.view.final_item
        b = self.view.mapFromScene(post.sceneBoundingRect().center())
        self._mouse(vp, QEvent.MouseButtonPress, a)
        for t in (0.3, 0.6, 1.0):
            self._mouse(vp, QEvent.MouseMove, QPoint(int(a.x() + (b.x() - a.x()) * t), int(a.y() + (b.y() - a.y()) * t)),
                        Qt.NoButton, Qt.LeftButton)
        self._mouse(vp, QEvent.MouseButtonRelease, b, Qt.LeftButton, Qt.NoButton)
        self.settle(6)

    # ------------------------------------------------------------ evidence
    def shot(self, name: str, widget: QWidget | None = None) -> str:
        self.settle(2)
        f = self.out / f"{name}.png"
        (widget or self.w).grab().save(str(f))
        return str(f)

    def note(self, severity: str, title: str, detail: str = "", shot: bool | str = True, steps: list | None = None):
        """An anomaly for the user to judge. severity: bloquant | gênant | mineur | question."""
        f = ""
        if isinstance(shot, str) and Path(shot).is_file():
            f = shot                                   # a screenshot taken at the moment it shows
        elif shot:
            f = self.shot(shot if isinstance(shot, str) else f"{len(self.notes()) + 1:03d}_{_slug(title)}")
        rec = {"area": self.area, "severity": severity, "title": title, "detail": detail, "steps": steps or [],
               "shot": f, "lang": self.lang, "t": time.strftime("%H:%M:%S")}
        with self.log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("ANOMALY", severity, title, flush=True)

    def notes(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(x) for x in self.log.read_text(encoding="utf-8").splitlines() if x.strip()]

    def guard(self, title: str, fn, *a, **k):
        """Run a step; an exception raised by the window is an anomaly (with its traceback), not the end of the
        exploration."""
        try:
            return fn(*a, **k)
        except RunsBlocked:
            return None
        except Exception as e:  # noqa: BLE001
            self.note("bloquant", f"Exception: {title}", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2500:]}")
            return None

    def close(self):
        if self.w is not None:
            try:
                if self.w.s is not None:
                    self.w.s.dirty = False if hasattr(self.w.s, "dirty") else None
            except Exception:  # noqa: BLE001
                pass
            self.w.close()
            self.settle()


def _slug(t: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in t.lower())
    return "_".join(x for x in keep.split("_") if x)[:50]


def stray_qt_errors():
    """Collect Qt warnings (printed by Qt): call once at start to see them in the script's output."""
    from PySide6.QtCore import qInstallMessageHandler

    def h(mode, ctx, msg):
        if "propagateSizeHints" in msg:
            return
        print("QT:", msg, flush=True)
    qInstallMessageHandler(h)


if __name__ == "__main__":                            # a smoke check of the harness itself
    import faulthandler
    faulthandler.dump_traceback_later(40, exit=True)
    ex = Explorer("_smoke")
    root = ex.project("session+image")
    ex.open(root)
    print("blocks:", [it.node.key for it in ex.blocks()])
    it = ex.block("image")
    print("menu:", [a.get("text") for a in ex.right_click_block(it)])
    z = ex.new_flow_zone()
    print("drop:", ex.drag_palette("image", ex.ghost_center(z)))
    print("paths:", [(q.name, q.source) for q in ex.s.paths])
    print("dialogs:", [d.__class__.__name__ for d in ex.dialogs()], "popups:", ex.popups)
    ex.shot("smoke")
    ex.close()
    print("OK", TMP)
