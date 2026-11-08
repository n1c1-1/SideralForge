"""A scripted tour of the client, offscreen, in English then in French, with a screenshot of every screen and dialog.

    python tools/client_walkthrough.py [--short] [--out FOLDER] [--folder FOLDER]

The tour (the real application, ``astro_gpu.client.app.Main``, on the real engine, on the CPU):
1. Home;
2. a project created from scratch on the raw frames of heart_a7s (8 frames: Calibration > Advanced settings), its
   folder imported;
3. Frames (sessions, browser), then Graph: a workflow from the New workflow dialog, a second one, a step added, Show
   all;
4. Run to Stacking, then the whole Classic stack workflow, with the plan, the progress and the recap;
5. Inspect on every block, the Selection chart, the histogram, a block not computed yet;
6. the Replace / Try as variant choice on a computed block, a variant made then discarded;
7. Post: the Stacking output sent to Post and computed, two layers, a filter, a mask; then Export;
8. the project closed and opened again, an edit, Undo and Redo;
9. a project of an older version imported (F:/astro_essais/projets/heart_a7s, a test copy);
10. Settings: the language switched to French; the screens and dialogs again, in French (with a Replace on Gradient
    and its run), then back to English.

Every screenshot is checked by code: no text cut short (a label or button narrower than its text), and, in French, no
text shown in English that has a French translation; the texts asked in French with no translation are listed too.
The tour ends with no error and no warning (Qt messages, Python warnings and exceptions are caught and counted). The
report (``report.json`` next to the shots) also gives the durations and the longest stall of the GUI thread while
each run went on.

``--short`` (the test) skips the runs, the export and the import of an older version: the project is created,
its frames read, its workflows edited, every screen shown in both languages.

``--project FOLDER`` is the other tour, the one of the features, on a COMPUTED project (a copy of the test project
F:/astro_essais/tests/heart_small, built by tools/make_test_project.py), in English then in French:
- Inspect, every tab of every block in the common viewer: the calibration masters and the darks and flats browsable
  with their measures, the calibrated lights; the registration charts (quality, drift, drift path, distribution) and
  the frame table; Selection with a frame picked; the stack, its rejection map, coverage and weights; Astrometry, the
  stack with its RA / Dec grid, and the survey; Colour before / after, the sensor response, the per-star fit against
  Gaia; Denoise, Stars, Gradient; Output, the final image;
- Send to Post from Inspect (an image layer), a filter nested under its layer (neutral settings), the export to FITS;
- Graph: the method cards with their duration on the GPU and on the CPU, a breakpoint (pause after a block, the
  review in the Run window, Continue when nothing is left to compute), taken off again.
Nothing is computed (a Continue that would start a run is not clicked); the copy is changed (Post's layers, the
export folder, the reviews), so make a fresh copy for each tour.

Rules: CPU only (CUDA_VISIBLE_DEVICES=-1); the settings file and the engine's state folder are temporary; nothing is
deleted outside the walkthrough's folder (default F:/astro_essais/tests/walkthrough) and temporary folders; never
F:/astro_projets.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ESSAIS = Path("F:/astro_essais")
RAW = ESSAIS / "donnees" / "heart_a7s"                               # the heart_a7s raw frames (lights, darks, flats)
PREVIOUS = ESSAIS / "projets" / "heart_a7s"                          # a test copy of a v1 project
FORBIDDEN = Path("F:/astro_projets")                                 # the user's projects: never opened


def setup_environment(tmp: Path):
    """Before Qt and the engine are imported: offscreen, CPU only, private settings and engine state."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    if Path("C:/Windows/Fonts").is_dir():
        os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                       # an empty value does not hide the GPUs here
    os.environ["ASTRO_GPU_SETTINGS"] = str(tmp / "settings.json")
    home = tmp / "home"
    home.mkdir(parents=True, exist_ok=True)
    real = Path.home() / ".astro_gpu" / "camera_profiles"
    if real.is_dir() and not (home / "camera_profiles").exists():
        shutil.copytree(real, home / "camera_profiles")             # read there, written here
    os.environ["ASTRO_GPU_HOME"] = str(home)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def raw_lights() -> Path:
    """The lights folder of heart_a7s."""
    return RAW / "lights"


# ---------------------------------------------------------------------------------------------- checks of a screen
def texts_of(root) -> list[str]:
    """Every text a visible widget shows (labels, buttons, lists, tables, tabs, tooltips), and the graph's blocks."""
    from PySide6.QtWidgets import QAbstractButton, QComboBox, QGraphicsView, QLabel, QListWidget, QTabBar, QTableWidget, QWidget

    from astro_gpu.client.widgets import ElidedLabel
    out = []
    for w in [root] + root.findChildren(QWidget):
        if w is not root and not w.isVisibleTo(root):
            continue
        if isinstance(w, ElidedLabel):
            out.append(w.full_text())
        elif isinstance(w, QLabel):
            out.append(re.sub(r"<[^>]+>", "", w.text()))
        elif isinstance(w, QAbstractButton):
            out.append(w.text())
        elif isinstance(w, QComboBox):
            out += [w.itemText(i) for i in range(w.count())]
        elif isinstance(w, QTabBar):
            out += [w.tabText(i) for i in range(w.count())]
        elif isinstance(w, QListWidget):
            out += [w.item(i).text() for i in range(min(w.count(), 80))]
        elif isinstance(w, QTableWidget):
            out += [w.horizontalHeaderItem(c).text() for c in range(w.columnCount()) if w.horizontalHeaderItem(c)]
            out += [w.item(r, c).text() for r in range(min(w.rowCount(), 40)) for c in range(w.columnCount())
                    if w.item(r, c) is not None]
        elif isinstance(w, QGraphicsView) and w.scene() is not None:
            for it in w.scene().items():
                for attr in ("title", "summary", "text"):
                    v = getattr(it, attr, None)
                    if isinstance(v, str):
                        out.append(v)
        if w.toolTip():
            out.append(w.toolTip())
    return [t.strip() for t in out if t and t.strip()]


def untranslated(texts, names=()) -> list[str]:
    """Texts shown in English although the French catalogue has them (French only). ``names``: what the user named
    (workflows, projects), never translated even when a name is also an English text ("Classic stack")."""
    from astro_gpu.client import i18n
    cat = i18n._cat
    out = []
    for t in texts:
        for line in t.splitlines():
            line = line.strip()
            if line in cat and cat[line] != line and line not in out and line not in names:
                out.append(line)
    return out


def cut_texts(root) -> list[str]:
    """Labels, buttons and check boxes narrower than their text (names and paths shortened on purpose excepted)."""
    from PySide6.QtWidgets import QCheckBox, QLabel, QPushButton, QWidget

    from astro_gpu.client.widgets import ElidedLabel
    out = []
    for w in root.findChildren(QWidget):
        if not w.isVisibleTo(root) or isinstance(w, ElidedLabel):
            continue
        if isinstance(w, QLabel) and not w.wordWrap() and w.text() and w.pixmap().isNull():
            if w.width() + 1 < w.sizeHint().width():
                out.append(re.sub(r"<[^>]+>", "", w.text()))
        elif isinstance(w, (QPushButton, QCheckBox)) and w.text():
            if w.width() + 1 < w.sizeHint().width():
                out.append(w.text())
    return out


# ---------------------------------------------------------------------------------------------- the tour
class Walk:
    def __init__(self, out: Path, folder: Path, short: bool, tmp: Path):
        from PySide6.QtCore import QTimer, qInstallMessageHandler
        from PySide6.QtWidgets import QApplication

        from astro_gpu.client import app as appmod
        from astro_gpu.client import prefs
        self.out, self.ws, self.short, self.tmp = out, folder, short, tmp
        self.n = 0
        self.problems: list[str] = []
        self.messages: list[str] = []
        self.shots: list[Path] = []
        self.lang = "en"
        self.expect_box = 0                    # a message box the tour asks for (else one is a problem)
        self.missing_fr: set[str] = set()
        self.timings: dict[str, float] = {}
        self.stalls: dict[str, float] = {}             # the longest stall of the GUI thread while a run went on
        qInstallMessageHandler(self._qt_message)
        warnings.simplefilter("always")
        warnings.showwarning = self._py_warning
        sys.excepthook = self._exception
        import logging
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger().addHandler(_Collect(self.messages))
        prefs.load()
        prefs.set("home_sections", [{"label": "Walkthrough", "folder": folder.as_posix(), "open": True}])
        prefs.set("last_folder", folder.as_posix())                       # where New project offers to go
        prefs.set("cache_dir", (tmp / "scratch").as_posix())
        prefs.set("language", "en")
        prefs.set("device", "auto")
        self.qa = QApplication.instance() or QApplication([])
        self.qa.setStyleSheet(appmod.QSS + appmod.EXTRA_QSS)
        appmod.apply_language("en")
        self.appmod, self.prefs = appmod, prefs
        self.w = appmod.Main()
        self.w.resize(1680, 1000)
        self.w.show()
        self.watchdog = QTimer()
        self.watchdog.setInterval(400)
        self.watchdog.timeout.connect(self._unexpected_modal)
        self.watchdog.start()

    # ------------------------------------------------------------ what goes wrong
    def _qt_message(self, mode, ctx, msg):
        from PySide6.QtCore import QtMsgType
        if mode in (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
            self.messages.append(f"Qt: {msg}")

    def _py_warning(self, message, category, filename, lineno, file=None, line=None):
        self.messages.append(f"{category.__name__}: {message} ({Path(filename).name}:{lineno})")

    def _exception(self, t, v, tb):
        text = "".join(traceback.format_exception(t, v, tb))
        self.messages.append("exception: " + text[-1500:])
        print(text, file=sys.stderr, flush=True)

    def guarded(self, fn, *args):
        """A step: an exception is a problem of the tour, which goes on with the next step."""
        try:
            return fn(*args)
        except Exception:  # noqa: BLE001
            self.problem(f"{fn.__name__}: " + traceback.format_exc(limit=8))
            from PySide6.QtWidgets import QApplication
            m = QApplication.activeModalWidget()
            if m is not None:
                m.reject()
            return None

    def problem(self, text: str):
        self.problems.append(text)
        print("  PROBLEM:", text, flush=True)

    def _unexpected_modal(self):
        from PySide6.QtWidgets import QApplication, QMessageBox
        m = QApplication.activeModalWidget()
        if isinstance(m, QMessageBox) and not self.expect_box:
            self.problem(f"unexpected message: {m.windowTitle()}: {m.text()}")
            m.grab().save(str(self.out / f"unexpected_{self.n:02d}.png"))
            m.reject()

    # ------------------------------------------------------------ events
    def pump(self, sec: float = 0.2):
        end = time.time() + sec
        while time.time() < end:
            self.qa.processEvents()
            time.sleep(0.005)
        self.qa.processEvents()

    def until(self, cond, timeout: float, what: str, beat: str | None = None) -> bool:
        """Process events until ``cond()``. ``beat``: the longest pass of the event loop meanwhile (the GUI thread
        stalled that long) is kept under that name."""
        end = time.time() + timeout
        while time.time() < end:
            t0 = time.perf_counter()
            self.qa.processEvents()
            if beat:
                self.stalls[beat] = max(self.stalls.get(beat, 0.0), time.perf_counter() - t0)
            try:
                if cond():
                    return True
            except RuntimeError:                          # a widget went with a rebuild: look again
                pass
            time.sleep(0.01)
        self.problem(f"timed out after {timeout:.0f} s: {what}")
        return False

    def busy(self) -> bool:
        w = self.w
        if w.worker.busy():
            return True
        if w.s is not None:
            from astro_gpu.client.services.inspect import _services
            sv = _services.get(w.s)
            if sv is not None and sv.busy():
                return True
        if w.render is not None and w.render.busy():
            return True
        return False

    def settle(self, timeout: float = 180.0):
        """Until the workers are idle and the views have built what they show."""
        end = time.time() + timeout
        quiet = 0
        while time.time() < end and quiet < 4:
            self.pump(0.05)
            quiet = 0 if self.busy() else quiet + 1
        self.pump(0.15)

    def wait_ready(self, timeout: float = 240.0):
        self.until(lambda: self.w.s is not None and self.w.s.ready and not self.w.worker.busy(), timeout,
                   "the project is read")
        self.settle()

    # ------------------------------------------------------------ screenshots
    def shot(self, name: str, widget=None):
        self.settle()
        target = widget if widget is not None else self.w
        self.n += 1
        f = self.out / f"{self.n:02d}_{name}_{self.lang}.png"
        target.grab().save(str(f))
        self.shots.append(f)
        cut = cut_texts(target)
        if cut:
            self.problem(f"{f.name}: text cut short: {cut[:6]}")
        if self.lang == "fr":
            left = untranslated(texts_of(target), self.user_names(target))
            if left:
                self.problem(f"{f.name}: English left in French: {left[:6]}")
        print(f"  {f.name}", flush=True)
        return f

    def user_names(self, target) -> set[str]:
        """The names the user gave (or the project of the older version had): workflows and projects."""
        names = set()
        w = self.w
        if w.s is not None:
            names |= {p.name for p in w.s.paths} | {w.s.name}
        rep = getattr(target, "report", None)
        if rep is not None and getattr(rep, "workflows", None):
            names |= set(rep.workflows.values())
        if w.home is not None:
            names |= {c.info.name for c in w.home.all_cards()}
        return names

    def modal(self, trigger, act, what: str):
        """Run ``trigger`` (it opens a modal dialog) and, inside the dialog's event loop, ``act(dialog)``; the dialog
        must be closed by ``act``."""
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        state = {"done": False, "t0": time.time()}

        def poll():
            d = QApplication.activeModalWidget()
            if d is None:
                if time.time() - state["t0"] > 30:
                    self.problem(f"no dialog for: {what}")
                    return
                QTimer.singleShot(50, poll)
                return
            try:
                act(d)
            except Exception:  # noqa: BLE001 - the tour goes on; the problem is reported
                self.problem(f"{what}: " + traceback.format_exc(limit=4))
                if d.isVisible():
                    d.reject()
            state["done"] = True

        QTimer.singleShot(80, poll)
        trigger()
        self.pump(0.1)
        return state["done"]

    # ------------------------------------------------------------ helpers on the views
    def select(self, step: str, pid: int | None = None):
        s = self.w.s
        self.w.select(index=s.index_of(pid) if pid is not None else None, step=step)
        self.pump(0.1)

    def inspect(self, step: str, pid: int | None = None, name: str | None = None, tab: int | None = None):
        self.select(step, pid)
        self.w.set_mode(1)
        tune = self.w.tune
        if tab is not None:
            tune.img_tab = tab
            tune.refresh()
        self.settle()
        if step == "selection":
            self.until(lambda: tune.chart is not None or "registration is computed" in " ".join(texts_of(tune)),
                       120, "the frames chart")
        else:
            self.until(lambda: all(sl is None or sl.image is not None or sl.stack.currentWidget() is not sl.note
                                   or "Loading" not in sl.note.text() and "Chargement" not in sl.note.text()
                                   for sl in tune._slots), 180, f"the image of {step}")
        return self.shot(name or f"inspect_{step}")

    def set_setting(self, panel, stage: str, name: str, value):
        """Edit a setting as its editor does (the panel's ``set_setting``: applied, or Replace / Try as variant)."""
        from astro_gpu.client.panel import CompactRow

        def rows():
            return [r for r in panel.findChildren(CompactRow) if r.r.get("stage") == stage and r.r.get("name") == name]
        self.settle()
        if not rows() and not panel.expert:
            panel._expert(True)
            self.settle()
        found = rows()
        if not found:
            self.problem(f"no setting {stage}.{name} in the panel")
            return
        panel.set_setting(found[0].r, value)
        self.settle()

    def run_dialog(self, start: bool, name: str, timeout: float = 900.0, progress_shot: str | None = None):
        """The plan dialog just opened: its shot, then the run and its recap."""
        w = self.w
        self.until(lambda: w.run.dialog is not None and w.run.dialog.isVisible(), 30, "the plan dialog")
        d = w.run.dialog
        self.shot(f"{name}_plan", d)
        if not start:
            d.close()
            return
        t0 = time.time()
        d.start()
        if progress_shot:
            self.until(lambda: w.run.track is not None and any(j.state == "running" and j.stage == "stack"
                                                               for j in w.run.track.jobs), timeout,
                       "the stack running")
            d.hide()
            self.pump(2.0)
            self.shot(progress_shot)
            d.show()
        ok = self.until(lambda: w.run.track is not None and w.run.track.finished and not w.run.running, timeout,
                        f"the run ({name})", beat=name)
        self.timings[name] = time.time() - t0
        self.settle()
        if ok and w.run.track.status != "done":
            self.problem(f"the run {name} ended {w.run.track.status}")
        d = w.run.dialog or d
        self.shot(f"{name}_recap", d)
        d.close()
        self.settle()

    # ------------------------------------------------------------ the steps
    def step_home(self):
        print("1. Home", flush=True)
        self.settle()
        self.shot("home")

    def step_create(self, folder_name: str, raw: Path):
        print("2. A project from scratch", flush=True)
        w = self.w

        def new(d):
            d.name.setText(folder_name)
            d.target.setText("IC 1805")
            self.shot("new_project", d)
            d.ok.click()
        self.modal(w.home.new_project, new, "New project")
        self.wait_ready()
        self.pump(0.3)
        g = getattr(w.graph, "_gallery", None)            # a new project opens on the template gallery (#43)
        if g is not None and g.isVisible():
            self.shot("gallery_new_project", g)
            # issue #47: the template is chosen before the frames (an import no longer makes a workflow of its own)
            g.pick(next(c.t for c in g.cards if c.t["key"] == "full"))
            g.ok.click()
            self.settle()
        self.shot("graph_new_project")
        sid = w.s.p.sessions[0].id                        # its files, kind by kind, as its Inspect adds them
        for kind, folder in (("lights", raw), ("darks", raw.parent / "darks"), ("flats", raw.parent / "flats"),
                             ("bias", raw.parent / "bias")):
            if folder.is_dir():
                w.s.add_session_entries(sid, kind, [str(folder)])
        self.wait_ready()
        self.select("sources")
        w.set_mode(1)
        w.frames.show_page(0)
        self.settle()
        self.until(lambda: w.frames.data is not None, 120, "the sessions table")
        self.shot("frames_sessions")
        w.frames.show_page(1)
        self.settle()
        self.pump(1.0)
        self.settle()
        self.shot("frames_browse")
        w.frames.show_page(0)

    def step_workflows(self):
        print("3. Graph: workflows and steps", flush=True)
        w = self.w
        w.set_mode(0)
        self.settle()

        def classic(d):
            d.pick(next(c.t for c in d.cards if c.t["key"] == "classic"))
            d.name.setText("Classic stack")
            self.shot("new_workflow", d)
            d.accept()
        self.modal(lambda: w.graph.new_workflow(), classic, "New workflow")
        self.wait_ready()
        s = w.s
        p = s.paths[0]
        # what a user sets before a first test: 8 frames (Calibration > Advanced settings) and the focal length
        self.select("calibration", p.pid)
        self.set_setting(w.graph.panel, "inventory", "max_lights", 8)
        self.shot("panel_calibration_advanced")
        w.graph.panel._expert(False)
        self.select("astrometrie", p.pid)
        self.set_setting(w.graph.panel, "astrometry", "focal_mm", 272.0)
        self.shot("panel_astrometry")
        self.select("calibration", p.pid)

        def second(d):
            d.tabs.setCurrentIndex(d.tab_keys.index("copy"))
            d.pick(d.cards[0].t)                        # a copy of "Classic stack": the same settings
            d.name.setText("Stars and denoise")
            self.shot("new_workflow_second", d)
            d.accept()
        self.modal(lambda: w.graph.new_workflow(), second, "New workflow (second)")
        self.wait_ready()
        q = s.paths[-1]
        w.graph.add_tool("etoiles", q.pid)                  # the Stars card, double-clicked (adds Denoise too)
        self.settle()
        self.shot("graph_step_added")
        w.graph.graph.b_all.setChecked(True)
        self.settle()
        self.shot("graph_show_all")
        w.graph.graph.b_all.setChecked(False)
        self.select("empilement", p.pid)
        self.shot("graph_two_workflows")

    def step_runs(self):
        print("4. Runs", flush=True)
        w = self.w
        s = w.s
        p = s.paths[0]
        self.select("empilement", p.pid)                    # a cheap block first: Calibration to Stacking
        w.set_mode(0)
        w.run_to_here()
        self.run_dialog(True, "run_to_stacking", progress_shot="run_progress")
        self.select("post", p.pid)
        w.run_to_here()
        self.run_dialog(True, "run_classic")
        self.select("empilement", p.pid)
        w.set_mode(0)
        self.shot("graph_after_run")
        w.b_log.setChecked(True)
        self.shot("log")
        w.b_log.setChecked(False)

    def step_inspect(self, computed: bool):
        print("5. Inspect", flush=True)
        w = self.w
        s = w.s
        p = s.paths[0]
        if computed:
            self.inspect("calibration", p.pid, "inspect_calibration")
            self.inspect("calibration", p.pid, "inspect_calibrated_frame", tab=1)
            w.tune.img_tab = 0
            self.inspect("alignement", p.pid)
            self.inspect("selection", p.pid)
            self.inspect("empilement", p.pid)
            hb = next(b for b in w.tune.findChildren(self._button) if b.text() == self._("Histogram"))
            hb.setChecked(True)
            self.settle()
            self.shot("inspect_stack_histogram")
            hb = next(b for b in w.tune.findChildren(self._button) if b.text() == self._("Histogram"))
            hb.setChecked(False)
            for k in ("astrometrie", "couleur", "fond", "post"):
                self.inspect(k, p.pid)
        q = s.paths[1] if len(s.paths) > 1 else p
        self.inspect("debruitage" if computed else "empilement", q.pid, "inspect_pending")
        w.set_mode(0)

    @property
    def _button(self):
        from PySide6.QtWidgets import QPushButton
        return QPushButton

    def _(self, text):
        from astro_gpu.client.i18n import _
        return _(text)

    def step_variant(self):
        print("6. Replace or Try as variant", flush=True)
        w = self.w
        s = w.s
        p = s.paths[0]
        self.select("empilement", p.pid)
        w.set_mode(0)
        w.graph.select(index=s.index_of(p.pid), step="empilement")
        s.scope = [p.pid]
        self.set_setting(w.graph.panel, "stack", "kappa", 4.0)
        if not s.pending_change:
            self.problem("no Replace / Try as variant choice on a computed Stacking")
            return
        self.shot("variant_choice")
        p = s.paths[0]
        w.graph.panel._variant(p)
        self.settle()
        self.shot("variant_made")
        v = next((q for q in s.paths if q.variant_of == p.pid), None)
        if v is not None:
            s.drop_variant(v)
            self.settle()

    def step_post(self):
        print("7. Post and Export", flush=True)
        w = self.w
        s = w.s
        p = s.paths[0]
        w.set_mode(0)
        node = w.graph.graph.graph.node_of(p, "empilement")
        w.graph.send_to_post(node)
        self.settle()
        self.shot("graph_raw_stack")
        raw = s.paths[-1]
        self.select("post", raw.pid)
        w.run_to_here()
        self.run_dialog(True, "run_raw_stack")
        w.set_mode(2)
        self.settle()
        self.until(lambda: w.render is not None and not w.render.busy(), 300, "the Post render")
        self.settle()
        view = w.post.view
        wl = [lay for lay in view.layers if lay["kind"] == "workflow"]
        if len(wl) < 2:
            self.problem(f"Post has {len(wl)} workflow layer(s), two expected")
        top = wl[0]
        view.set_visible(top["id"], True)
        view._set(top, "opacity", 50)                       # percent, as the slider
        view.select(top["id"], "layer")
        self.until(lambda: not w.render.busy(), 300, "the Post render")
        self.shot("post_two_layers")
        base = wl[-1]
        view.select(base["id"], "layer")
        view.add_filter("curves")
        self.until(lambda: not w.render.busy(), 300, "the Post render")
        self.shot("post_filter")
        view.select(top["id"], "layer")
        view.mask_button()
        lay = view.layer(top["id"])
        view.add_part(lay, "luminance")
        self.until(lambda: not w.render.busy(), 300, "the Post render")
        self.shot("post_mask")
        view.b_mask.setChecked(True)
        self.until(lambda: not w.render.busy(), 300, "the Post render")
        self.shot("post_mask_shown")
        view.b_mask.setChecked(False)
        view.select(base["id"], "layer")

        def export(d):
            d.fmt.setCurrentIndex(max(0, d.fmt.findData("tiff16")))
            self.shot("export", d)
            t0 = time.time()
            d.go.click()
            self.until(lambda: bool(d.files) or bool(d.error) or not d.running and d.go.text() == self._("Close"),
                       600, "the export")
            self.timings["export"] = time.time() - t0
            self.shot("export_done", d)
            if not d.files:
                self.problem(f"the export wrote nothing: {d.error or d.status.text()}")
            d.accept()
        self.modal(w.export, export, "Export")
        self.settle()

    def step_reopen(self, root: Path):
        print("8. Close, open again, Undo / Redo", flush=True)
        w = self.w
        w.close_project()
        self.settle()
        self.until(lambda: any(c.info.root == root.as_posix() for c in w.home.all_cards()), 60, "the card of the project")
        self.settle(60)
        self.shot("home_after_close")
        w.open_project(root)
        self.wait_ready()
        self.shot("reopened")
        s = w.s
        q = next((x for x in s.paths if s.has_tool(x.opts, "etoiles")), None)
        if q is None:
            self.problem("no workflow with Stars to edit")
            return
        node = w.graph.graph.graph.node_of(q, "etoiles")
        w.graph.remove_tool(node, [q.pid])
        self.settle()
        self.shot("edited")
        w.undo()
        self.settle()
        if not s.has_tool(s.view(q.pid).opts, "etoiles"):
            self.problem("Undo did not bring Stars back")
        self.shot("undo")
        w.redo()
        self.settle()
        if s.has_tool(s.view(q.pid).opts, "etoiles"):
            self.problem("Redo did not take Stars away again")
        self.shot("redo")
        w.undo()
        self.settle()

    def step_import(self, old: Path, do_import: bool):
        print("9. A project of an older version", flush=True)
        w = self.w

        def imp(d):
            self.until(lambda: d.report is not None or d.note.isVisible(), 300, "the check of the old project")
            self.settle()
            self.shot("import_previous", d)
            if not do_import or d.report is None:
                d.reject()
                return
            t0 = time.time()
            d.b_ok.click()
            self.until(lambda: d.result_root is not None or (d.note.isVisible() and not d._busy), 900,
                       "the import of the old project")
            self.timings["import"] = time.time() - t0
            if d.result_root is None:
                self.problem("the import failed: " + d.note.text())
                d._busy = False
                d.reject()
        self.modal(lambda: w.open_project(old), imp, "Import of a project of an older version")
        if do_import:
            self.wait_ready()
            self.shot("imported_graph")
            if any(p.rendered for p in w.s.paths):
                w.set_mode(2)
                self.until(lambda: w.render is not None and not w.render.busy(), 300, "the Post render")
                self.shot("imported_post")
                w.set_mode(0)

    def step_settings(self, to: str):
        print(f"10. Settings: language {to}", flush=True)
        w = self.w
        from PySide6.QtWidgets import QComboBox

        def switch(d):
            self.shot("settings", d)
            combo = d.pages.widget(0).findChildren(QComboBox)[0]
            if to == "fr":
                from astro_gpu.client import i18n
                i18n._missing.clear()
            combo.setCurrentIndex(combo.findData(to))
            self.pump(0.3)
            self.lang = to
            self.shot("settings", d)
            d.accept()
        self.modal(w.open_settings, switch, "Settings")
        self.settle()

    def french_tour(self, root: Path, computed: bool):
        """The screens and dialogs again, in French (nothing is run)."""
        print("French tour", flush=True)
        w = self.w
        w.show_home()
        self.settle()
        self.shot("home")

        def cancel_new(d):
            d.name.setText("M 42")
            self.shot("new_project", d)
            d.reject()
        self.modal(w.home.new_project, cancel_new, "New project (fr)")
        w.open_project(root)
        self.wait_ready()
        s = w.s
        p = s.paths[0]
        self.select("empilement", p.pid)
        w.set_mode(0)
        self.shot("graph")
        self.select("etoiles", s.paths[1].pid if len(s.paths) > 1 else p.pid)
        self.shot("graph_stars_panel")
        self.select("couleur", p.pid)
        self.shot("graph_color_panel")
        w.graph.graph.b_all.setChecked(True)
        self.shot("graph_show_all")
        w.graph.graph.b_all.setChecked(False)

        def cancel_wf(d):
            self.shot("new_workflow", d)
            d.reject()
        self.modal(lambda: w.graph.new_workflow(), cancel_wf, "New workflow (fr)")
        self.select("sources")
        w.set_mode(1)
        self.until(lambda: w.frames.data is not None, 120, "the sessions table")
        self.shot("frames_sessions")
        w.frames.show_page(1)
        self.settle()
        self.shot("frames_browse")
        w.frames.show_page(0)
        self.step_inspect(computed)
        w.set_mode(0)
        w.run.launch()                      # the plan, not started (#51: the button starts at once)
        self.run_dialog(False, "run_all")
        if computed:
            if w.run.track is not None:                     # the recap of the last run, if this window ran one
                w.run.show_dialog()
                self.settle()
                if w.run.dialog is not None:
                    self.shot("run_recap", w.run.dialog)
                    w.run.dialog.close()
            self.select("empilement", p.pid)
            w.set_mode(0)
            s.scope = [p.pid]
            self.set_setting(w.graph.panel, "stack", "kappa", 4.5)
            if s.pending_change:
                self.shot("variant_choice")
                w.graph.panel._cancel()
            # Replace on a cheap block, and its run: the plan, the progress and the recap in French
            self.select("fond", p.pid)
            self.set_setting(w.graph.panel, "gradient", "degree", 3)
            if s.pending_change:
                self.shot("replace_choice")
                w.graph.panel._replace()
                self.settle()
            self.select("post", p.pid)
            w.run_to_here()
            self.run_dialog(True, "run_gradient")
            w.set_mode(2)
            self.until(lambda: w.render is not None and not w.render.busy(), 300, "the Post render")
            self.shot("post")
            view = w.post.view
            masked = next((lay for lay in view.layers if lay.get("mask")), None)
            if masked is not None:
                view.select(masked["id"], "mask")
                self.shot("post_mask")
            filt = next((lay for lay in view.layers if lay["kind"] == "filter"), None)
            if filt is not None:
                view.select(filt["id"], "layer")
                self.shot("post_filter")

            def cancel_export(d):
                self.shot("export", d)
                d.reject()
            self.modal(w.export, cancel_export, "Export (fr)")
            w.set_mode(0)
        else:
            w.set_mode(2)
            self.settle()
            self.shot("post_empty")
            w.set_mode(0)
        w.b_log.setChecked(True)
        self.shot("log")
        w.b_log.setChecked(False)
        d = w.about_dialog()
        d.show()
        self.shot("about", d)
        d.close()
        if PREVIOUS.is_dir():
            self.step_import(PREVIOUS, False)
        from astro_gpu.client import i18n
        self.missing_fr |= {t for t in i18n.missing() if re.search(r"[A-Za-z]{3}", t)}

    # ------------------------------------------------------------ the features tour (--project): a computed copy
    #: (block, tab, shot): every tab of Inspect the 2026-11-01 review asked for, in the order of the chain
    FEATURE_TABS = (
        ("calibration", "masters", "calibration_masters"),
        ("calibration", "darks", "calibration_darks"),
        ("calibration", "flats", "calibration_flats"),
        ("calibration", "calibrated", "calibration_lights"),
        ("alignement", "reference", "registration_reference"),
        ("alignement", "quality", "registration_quality"),
        ("alignement", "drift", "registration_drift"),
        ("alignement", "track", "registration_track"),
        ("alignement", "distribution", "registration_distribution"),
        ("alignement", "table", "registration_table"),
        ("selection", None, "selection"),
        ("empilement", "stack", "stack"),
        ("empilement", "rejection", "stack_rejection"),
        ("empilement", "coverage", "stack_coverage"),
        ("empilement", "weights", "stack_weights"),
        ("astrometrie", "solution", "astrometry_grid"),
        ("couleur", "colors", "color_before_after"),
        ("couleur", "fit", "color_fit"),
        ("debruitage", "denoised", "denoise"),
        ("etoiles", "starless", "stars_starless"),
        ("etoiles", "stars", "stars_alone"),
        ("fond", "result", "gradient"),
        ("fond", "removed", "gradient_sky_model"),
        ("post", "output", "output_final"),
    )

    @staticmethod
    def _view_done(v) -> bool:
        """A viewer of Inspect shows what it was asked (or says why not); one asked nothing waits only on a message
        that ends with "…" (Loading…, Rendering the final image…)."""
        try:
            if v._want is not None:
                return v.wait(0.02)
            return not (v.canvas.message or "").endswith("…")
        except RuntimeError:                                     # gone with a rebuild
            return True

    def wait_tab(self, what: str, timeout: float = 300.0) -> bool:
        """Until the tab shown has its data (the worker's), its pictures and its compared pictures."""
        tune = self.w.tune
        reading = (self._("Reading the results…"),)
        end = time.time() + timeout
        while time.time() < end:
            self.settle(60)
            views = list(tune.views)
            if all(self._view_done(v) for v in views) and \
                    not any(t in " ".join(texts_of(tune)) for t in reading):
                self.pump(0.3)
                self.settle(30)
                if list(tune.views) == views and all(self._view_done(v) for v in views):
                    return True
            self.pump(0.2)
        self.problem(f"timed out after {timeout:.0f} s: {what}")
        return False

    def show_tab(self, step: str, key: str | None, name: str, pid: int):
        w = self.w
        self.select(step, pid)
        w.set_mode(1)
        tune = w.tune
        if key is not None:
            keys = [v.key for v in tune.visuals(w.s.path, step)]
            if key not in keys:
                self.problem(f"no tab {key!r} in {step}: {keys}")
                return None
            tune.tab_of[step] = key
        tune.refresh()
        if step == "selection":
            self.until(lambda: tune.chart is not None, 120, "the frames chart")
            self.settle()
            tune._pick(10)                                     # a frame picked on the chart: shown below it
        t0 = time.time()
        self.wait_tab(f"{step} / {key}")
        self.timings[f"{name}_{self.lang}"] = time.time() - t0
        return self.shot(name)

    def features_inspect(self):
        print("Features: Inspect", flush=True)
        w = self.w
        p = w.s.paths[0]
        for step, key, name in self.FEATURE_TABS:
            self.guarded(self.show_tab, step, key, name, p.pid)
        w.tune.tab_of.clear()

    def features_post(self, first: bool):
        """Send to Post from Inspect (the stack: an image layer), a filter nested under the workflow's layer, Export
        to FITS. The second time (French) the composition is there already: it is shown, and the dialog cancelled."""
        print("Features: Post", flush=True)
        w = self.w
        p = w.s.paths[0]
        w.set_mode(2)
        self.until(lambda: w.render is not None and not w.render.busy(), 300, "the Post render")
        self.settle()
        self.shot("post_layers")
        if first:
            self.show_tab("empilement", "stack", "stack_send_to_post", p.pid)
            v = w.tune.view
            if v is None or v.image() is None:
                self.problem("no stack in Inspect to send to Post")
            else:
                v._send_post()                                    # the viewer's Send to Post button
                self.settle()
                self.until(lambda: w.mode == 2 and w.render is not None and not w.render.busy(), 300,
                           "the Post render with the image layer")
                self.settle()
                if not any(lay["kind"] == "image" for lay in w.post.view.layers):
                    self.problem("Send to Post made no image layer")
        view = w.post.view
        self.shot("post_image_layer")
        wl = next((lay for lay in view.layers if lay["kind"] == "workflow"), None)
        if wl is None:
            self.problem("no workflow layer in Post")
            return
        if first:
            view.select(wl["id"], "layer")
            view.add_filter("curves")                             # nested under the layer, its settings neutral
            self.until(lambda: not w.render.busy(), 300, "the Post render")
        fl = next((lay for lay in view.layers if lay["kind"] == "filter" and lay.get("target") == wl["id"]), None)
        if fl is None:
            self.problem("no filter nested under the workflow's layer")
        else:
            view.select(fl["id"], "layer")
            self.until(lambda: not w.render.busy(), 300, "the Post render")
        self.shot("post_filter_nested")

        def export(d):
            d.fmt.setCurrentIndex(max(0, d.fmt.findData("fits32f")))
            self.shot("export_fits", d)
            if not first:
                d.reject()
                return
            t0 = time.time()
            d.go.click()
            self.until(lambda: bool(d.files) or bool(d.error) or not d.running and d.go.text() == self._("Close"),
                       600, "the export")
            self.timings["export_fits"] = time.time() - t0
            self.shot("export_fits_done", d)
            if not d.files:
                self.problem(f"the export wrote nothing: {d.error or d.status.text()}")
            d.accept()
        self.modal(w.export, export, "Export to FITS")
        self.settle()

    def features_graph(self, block: str):
        """The method cards (GPU and CPU durations), then a breakpoint on ``block``: the graph, the Run window's
        review, Continue (only when it starts nothing), and the breakpoint taken off."""
        print("Features: Graph", flush=True)
        w = self.w
        s = w.s
        p = s.paths[0]
        w.set_mode(0)
        self.select("debruitage", p.pid)
        self.shot("graph_method_cards")
        self.select(block, p.pid)
        node = w.graph.graph.graph.node_of(p, block)
        w.graph.toggle_pause(node)
        self.settle()
        if s.review_state(p, block) != "paused":
            self.problem(f"the breakpoint on {block} is {s.review_state(p, block)!r}, not 'paused'")
        self.shot("graph_breakpoint")
        w.run.launch()
        self.until(lambda: w.run.dialog is not None and w.run.dialog.isVisible(), 30, "the Run window")
        self.settle()
        d = w.run.dialog
        if not d.paused():
            self.problem("the Run window does not ask for the review")
        self.shot("run_paused", d)
        rest = s.plan_all()
        if not rest.jobs:                                       # Continue starts nothing: it is clicked
            d.b_continue.click()
            self.settle()
            if w.run.dialog is not None:
                self.shot("run_continued", w.run.dialog)
        else:
            print(f"  Continue not clicked: it would run {len(rest.jobs)} job(s)", flush=True)
        if w.run.dialog is not None:
            w.run.dialog.close()
        self.settle()
        node = w.graph.graph.graph.node_of(p, block)
        if s.has_breakpoint(p, block):
            w.graph.toggle_pause(node)
        self.settle()

    def features(self, root: Path):
        """The tour of the features of 2026-11-01 on a computed copy, in English then in French."""
        t0 = time.time()
        g = self.guarded
        w = self.w
        g(self.step_home)
        w.open_project(root)
        self.wait_ready()
        s = w.s
        if s is None or not s.paths:
            self.problem(f"the project {root} did not open")
            return self.report()
        from astro_gpu.client.blocks import block_state
        p = s.paths[0]
        states = {k: block_state(s, p, k) for k in s.block_keys(p)}
        print("  blocks:", states, flush=True)
        if any(v != "done" for v in states.values()):
            self.problem(f"blocks not done in the copy: {states}")
        self.shot("graph")
        for lang, block in (("en", "calibration"), ("fr", "empilement")):
            if lang == "fr":
                g(self.step_settings, "fr")
            g(self.features_inspect)
            g(self.features_post, lang == "en")
            g(self.features_graph, block)
        from astro_gpu.client import i18n
        self.missing_fr |= {t for t in i18n.missing() if re.search(r"[A-Za-z]{3}", t)}
        g(self.step_settings, "en")
        self.w.close()
        self.pump(0.5)
        self.timings["total"] = time.time() - t0
        return self.report()

    # ------------------------------------------------------------ all of it
    def run(self, raw: Path, name: str):
        t0 = time.time()
        root = self.ws / name
        g = self.guarded
        g(self.step_home)
        g(self.step_create, name, raw)
        g(self.step_workflows)
        computed = not self.short
        if computed:
            g(self.step_runs)
        g(self.step_inspect, computed)
        if computed:
            g(self.step_variant)
            g(self.step_post)
        else:
            self.w.run.launch()
            g(self.run_dialog, False, "run_all")
        g(self.step_reopen, root)
        if PREVIOUS.is_dir():
            g(self.step_import, PREVIOUS, not self.short)
        g(self.step_settings, "fr")
        g(self.french_tour, root, computed)
        g(self.step_settings, "en")
        g(self.shot, "graph_back_in_english")
        self.w.close()
        self.pump(0.5)
        self.timings["total"] = time.time() - t0
        return self.report()

    def report(self) -> dict:
        noise = [m for m in self.messages if not _harmless(m)]
        out = {"shots": len(self.shots), "folder": str(self.out), "problems": list(self.problems),
               "untranslated_asked": sorted(self.missing_fr), "messages": noise,
               "timings": {k: round(v, 1) for k, v in self.timings.items()},
               "gui_stall_ms_during_runs": {k: round(1000 * v) for k, v in self.stalls.items()}}
        (self.out / "report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        return out


class _Collect:
    """A logging handler that keeps the warnings and errors."""

    def __init__(self, sink: list):
        import logging
        self.sink = sink
        self.level = logging.WARNING
        self.filters = []
        self.lock = None

    def handle(self, record):
        if record.levelno >= self.level:
            self.sink.append(f"log {record.levelname}: {record.getMessage()}")

    def createLock(self):  # noqa: N802 - logging's names
        pass

    def acquire(self):
        pass

    def release(self):
        pass


#: messages that say nothing about the client:
#: - the offscreen platform has no window manager: it cannot resize a window to its hint or raise it, nor give a
#:   menu the keyboard;
#: - a run is a detached process that outlives the object that launched it (core/runner2.launch, by design): Python
#:   notes it with a ResourceWarning, which it hides by default (the tour shows every warning)
HARMLESS = (re.compile(r"^Qt: This plugin does not support (propagateSizeHints|raise)\(\)$"),
            re.compile(r"^Qt: This plugin does not support grabbing the (keyboard|mouse)$"),   # a menu shown
            re.compile(r"^ResourceWarning: subprocess \d+ is still running \(subprocess\.py:\d+\)$"))


def _harmless(message: str) -> bool:
    """Messages of the offscreen platform and of the detached run (above); every other message counts."""
    return any(rx.match(message) for rx in HARMLESS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--short", action="store_true", help="no run, no export, no import (the test)")
    ap.add_argument("--out", default=str(Path(os.environ.get("TEMP", tempfile.gettempdir())) / "client_walk"))
    ap.add_argument("--folder", default=str(ESSAIS / "tests" / "walkthrough"))
    ap.add_argument("--name", default="Heart")
    ap.add_argument("--project", default="", help="the features tour, on this computed test copy (see above)")
    a = ap.parse_args(argv)
    ws = Path(a.folder)
    project = Path(a.project) if a.project else None
    for folder in (ws, project):
        if folder is not None and FORBIDDEN.as_posix().lower() in folder.resolve().as_posix().lower():
            raise SystemExit("the walkthrough never works in the user's projects")
    if project is not None and not (project / "project.astroproj").is_file():
        raise SystemExit(f"no project in {project}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    tmp = Path(tempfile.mkdtemp(prefix="client_walk_"))
    setup_environment(tmp)
    ws.mkdir(parents=True, exist_ok=True)
    if project is None:
        for leftover in (ws / a.name, ws / PREVIOUS.name):        # the projects of an earlier tour (inside ws)
            if leftover.exists():
                shutil.rmtree(leftover)
    import faulthandler
    faulthandler.dump_traceback_later(900, repeat=True)            # a tour that hangs says where
    walk = Walk(out, ws, a.short, tmp)
    try:
        rep = walk.features(project) if project is not None else walk.run(raw_lights(), a.name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    ok = not rep["problems"] and not rep["messages"] and not rep["untranslated_asked"]
    print("WALKTHROUGH", "OK" if ok else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
