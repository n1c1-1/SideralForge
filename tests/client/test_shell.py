"""The client shell (P9a), offscreen: Home over a scratch workspace, new and v1 projects, the language at runtime,
the settings file, the Log, the dialogs, the ApiWorker, and screenshots.

ASTRO_GPU_SETTINGS points to a temporary file in every test; the user's settings file is checked untouched.
ASTRO_GPU_SHOTS names a folder for the screenshots (default: a temporary folder).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")      # the offscreen platform has no fonts else

pytest.importorskip("PySide6")
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QWidget,
)

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client.dialogs import NewWorkflowDialog, name_dialog  # noqa: E402
from astro_gpu.client.home import NewProjectDialog, card_line, project_info, scan_workspace  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker, Gpu, Machine  # noqa: E402
from astro_gpu.client.settingsdlg import SettingsDialog  # noqa: E402
from astro_gpu.client.widgets import QSS, ElidedLabel  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "astro_gpu" / "client"
MACHINE = Machine([Gpu("cuda:0", "NVIDIA RTX 4060 Ti", 16.0), Gpu("cuda:1", "NVIDIA RTX 3070", 8.0)], 16, 64.0)


def _user_settings() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "astro_gpu" / "settings.json"


def _fingerprint(p: Path):
    return (p.stat().st_mtime_ns, hashlib.sha1(p.read_bytes()).hexdigest()) if p.exists() else None


USER_SETTINGS = _user_settings()
USER_BEFORE = _fingerprint(USER_SETTINGS)


# ---------------------------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="session")
def qapp():
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return a


def _nebula(path: Path, w=640, h=380, seed=3):
    from PIL import Image
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32) + 0.04
    for cx, cy, s, col in ((0.38, 0.42, 70, (0.9, 0.25, 0.35)), (0.62, 0.62, 50, (0.85, 0.3, 0.45)),
                           (0.48, 0.5, 120, (0.35, 0.12, 0.2))):
        g = np.exp(-(((x - cx * w) ** 2 + (y - cy * h) ** 2) / (2 * s * s)))
        img += g[..., None] * np.array(col, np.float32)
    for _i in range(500):
        sx, sy = rng.integers(0, w), rng.integers(0, h)
        img[sy, sx] += rng.uniform(0.3, 1.0)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path, quality=92)


def _project(ws: Path, folder: str, name: str, target: str, nights: int, workflows=(), run=None, thumb=False):
    p = project2.create(ws / folder, name, Target(name=target))
    for i in range(nights):
        p.sessions.append(project2.Session(i + 1, f"2026-08-{16 + i}", lights=[f"F:/raw/{folder}/night{i + 1}"]))
    from astro_gpu.core.workflow import TEMPLATES
    for wf in workflows:
        p.add_workflow(wf, TEMPLATES[wf])
    project2.save(p)
    if run:
        m = ws / folder / "store" / "inventory" / "0123456789abcdef"
        m.mkdir(parents=True)
        (m / "manifest.json").write_text(json.dumps({"stage": "inventory", "status": "done", "finished": "x",
                                                     "metrics": run}), encoding="utf-8")
    if thumb:
        (ws / folder / "cache").mkdir()
        _nebula(ws / folder / "cache" / "thumb.jpg")
    return ws / folder


@pytest.fixture()
def env(tmp_path, monkeypatch, qapp):
    """A scratch workspace and settings file, English, a fixed machine; message boxes recorded, not shown."""
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))
    prefs.load()
    ws = tmp_path / "ws"
    ws.mkdir()
    prefs.set("workspace", ws.as_posix())
    prefs.set("cache_dir", (tmp_path / "scratch").as_posix())
    appmod.apply_language("en")
    monkeypatch.setattr(appmod, "machine_info", lambda: MACHINE)
    shown = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, _k=kind, **k: shown.append((_k, a[1:3])))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append(("box", self.text())))
    _project(ws, "Heart", "HeartSoul", "IC 1805 · Heart nebula", 1,
             ["Classic stack", "Full", "Full + neural stars"],
             run={"integration_h": 3.0, "last_time": "2026-08-17T05:15:59"}, thumb=True)
    _project(ws, "NGC7822", "NGC 7822", "Cepheus · emission nebula", 3, ["Full"])
    _project(ws, "Iris", "Iris", "NGC 7023 · reflection nebula", 1, ["Classic stack"],
             run={"integration_h": 2.5, "last_time": "2026-07-28T03:00:00"})
    _project(ws, "_qc", "QC sheets", "", 0)                                  # skipped: starts with "_"
    (ws / "notes").mkdir()                                                   # skipped: no project.astroproj
    old = ws / "OldM31"                                                      # skipped: a v1 project
    old.mkdir()
    (old / "project.json").write_text(json.dumps({"name": "M31", "inputs": {"lights": ["F:/raw/m31"]}}),
                                      encoding="utf-8")
    windows = []
    yield {"ws": ws, "tmp": tmp_path, "shown": shown, "windows": windows}
    for w in windows:
        w.close()
        w.deleteLater()
    qapp.processEvents()
    appmod.apply_language("en")
    assert _fingerprint(USER_SETTINGS) == USER_BEFORE, "the user's settings file was touched"


def _main(env) -> appmod.Main:
    w = appmod.Main()
    w.resize(1600, 950)
    w.show()
    env["windows"].append(w)
    assert w.worker.wait()
    return w


def _snapshot(folder: Path):
    return sorted((p.relative_to(folder).as_posix(), p.stat().st_mtime_ns, p.stat().st_size)
                  for p in folder.rglob("*"))


def _texts(root: QWidget) -> list[str]:
    """Every text a visible widget shows."""
    out = []
    for w in [root] + root.findChildren(QWidget):
        if not w.isVisibleTo(root) and w is not root:
            continue
        if isinstance(w, ElidedLabel):
            out.append(w.full_text())
        elif isinstance(w, QLabel):
            out.append(re.sub(r"<[^>]+>", "", w.text()))
        elif isinstance(w, QAbstractButton):
            out.append(w.text())
        elif isinstance(w, QComboBox):
            out += [w.itemText(i) for i in range(w.count())]
        elif isinstance(w, QListWidget):
            out += [w.item(i).text() for i in range(w.count())]
        elif isinstance(w, QTableWidget):
            out += [w.horizontalHeaderItem(i).text() for i in range(w.columnCount())]
        if w.toolTip():
            out.append(w.toolTip())
        if hasattr(w, "placeholderText") and w.placeholderText():
            out.append(w.placeholderText())
    return [t.strip() for t in out if t and t.strip()]


def _untranslated(texts) -> list[str]:
    """Texts shown in English although a French translation exists."""
    cat = i18n._cat
    return [t for t in texts if t in cat and cat[t] != t]


def _cut(root: QWidget) -> list[str]:
    """Labels, buttons and check boxes narrower than their text."""
    out = []
    for w in root.findChildren(QWidget):
        if not w.isVisibleTo(root):
            continue
        if isinstance(w, ElidedLabel):
            continue                                 # names and paths: shortened on purpose, whole in the tooltip
        if isinstance(w, QLabel) and not w.wordWrap() and w.text() and w.pixmap().isNull():
            if w.width() + 1 < w.sizeHint().width():
                out.append(w.text())
        elif isinstance(w, (QPushButton, QCheckBox)) and w.text():
            if w.width() + 1 < w.sizeHint().width():
                out.append(w.text())
    return out


class FakeModel:
    """What the New workflow dialog reads of the project model."""

    class P:
        def __init__(self, name, opts):
            self.name, self.opts, self.params = name, opts, {}

    def __init__(self):
        self.paths = [self.P("Classic stack", {"d": 0}), self.P("Full", {"d": 2})]

    def opts_seconds(self, opts):
        return 2880 + 1860 * opts.get("d", 0)

    def unique_name(self, base):
        names = {p.name for p in self.paths}
        n = 2
        out = base
        while out in names:
            out, n = f"{base} {n}", n + 1
        return out


TEMPLATES = {"Classic stack": {"d": 0}, "Classic + denoise": {"d": 1}, "Full": {"d": 2},
             "Full + neural stars": {"d": 3}}


# ---------------------------------------------------------------------------------------------- tests

def test_home_lists_the_workspace_and_skips_underscore(env):
    w = _main(env)
    names = [c.info.name for c in w.home.cards]
    assert sorted(names) == ["HeartSoul", "Iris", "NGC 7822"]
    assert w.home.count.text() == "3 projects in the workspace"
    assert w.windowTitle() == "astro_gpu · Home"
    infos, n = scan_workspace(env["ws"].as_posix(), [])
    assert n == 3 and "QC sheets" not in [i.name for i in infos]


def test_card_shows_nights_integration_date_and_state(env):
    heart = project_info(env["ws"] / "Heart")
    assert (heart.nights, heart.integration_h, heart.date) == (1, 3.0, "2026-08-17")
    assert _fmt(heart) == "3 workflow(s)" and heart.thumb is not None
    assert card_line(heart) == "1 night(s) · 3.0 h · 2026-08-17"
    ngc = project_info(env["ws"] / "NGC7822")
    assert ngc.state == "Never run" and ngc.thumb is None
    appmod.apply_language("fr")
    assert card_line(heart) == "1 nuit(s) · 3,0 h · 2026-08-17"


def _fmt(info):
    return i18n._(info.state).format(**info.state_args)


def test_recent_projects_come_first(env):
    prefs.add_recent((env["ws"] / "Iris").as_posix(), "Iris", "")
    infos, _n = scan_workspace(env["ws"].as_posix(), [r["path"] for r in prefs.recent()])
    assert infos[0].name == "Iris"


def test_new_project_writes_a_v2_folder(env):
    w = _main(env)
    root = env["ws"] / "M42"
    assert w.create_project(root.as_posix(), "M42", "Orion nebula")
    d = json.loads((root / "project.astroproj").read_text(encoding="utf-8"))
    assert d["format"] == 2 and d["name"] == "M42" and d["target"]["name"] == "Orion nebula"
    assert d["scratch_dir"] == (env["tmp"] / "scratch" / "M42").as_posix()
    assert project2.load(root).name == "M42"
    assert w.project is not None and w.windowTitle() == "astro_gpu · M42"
    assert w.screens.currentWidget() is w.proj_screen
    assert prefs.recent()[0]["path"] == root.as_posix()
    saved = json.loads(Path(os.environ["ASTRO_GPU_SETTINGS"]).read_text(encoding="utf-8"))
    assert saved["recent"][0]["name"] == "M42"
    # the dialog: the folder follows the name; a taken folder is refused
    dlg = NewProjectDialog(env["ws"].as_posix())
    dlg.name.setText("Iris")
    assert not dlg.ok.isEnabled() and not dlg.exists.isHidden()
    dlg.name.setText("Rosette: night 2")
    assert dlg.ok.isEnabled() and dlg.folder() == env["ws"] / "Rosette_ night 2"


def test_open_a_v1_folder_writes_nothing(env):
    w = _main(env)
    old = env["ws"] / "OldM31"
    before = _snapshot(old)
    assert not w.open_project(old)
    assert _snapshot(old) == before
    assert env["shown"] and "previous version" in env["shown"][-1][1]
    assert w.project is None


def test_open_and_close_a_project(env):
    w = _main(env)
    assert w.open_project(env["ws"] / "Heart")
    before = _snapshot(env["ws"] / "Heart")
    assert "HeartSoul" in w.title.text() and "IC 1805" in w.title.text()
    assert w.status.text() == "3 workflow(s)"
    assert w.machine_lb.text() == "NVIDIA RTX 4060 Ti"
    for i in (1, 2, 0):
        w.set_mode(i)
        assert w.modes.currentIndex() == i and w.mode_btns[i].isChecked()
    w.set_device("cpu")
    assert w.machine_lb.text() == "CPU only"
    file_menu = {a.text(): a for a in w.menuBar().actions()[0].menu().actions()}
    file_menu["Home"].trigger()                               # the project stays open behind Home
    assert w.project is not None and w.screens.currentWidget() is w.home
    assert w.open_project(env["ws"] / "Heart") and w.screens.currentWidget() is w.proj_screen
    file_menu = {a.text(): a for a in w.menuBar().actions()[0].menu().actions()}
    file_menu["Close project"].trigger()
    assert w.project is None and w.screens.currentWidget() is w.home
    assert _snapshot(env["ws"] / "Heart") == before          # opening never writes into the project


def test_french_and_back_rebuild_the_menus(env, qapp):
    w = _main(env)
    w.open_project(env["ws"] / "Heart")
    w.show_home()
    titles = lambda: [a.text() for a in w.menuBar().actions()]  # noqa: E731
    assert titles() == ["File", "Edit", "View", "Run", "Help"]
    d = SettingsDialog(w.worker, w.machine, parent=w)
    d.languageChanged.connect(w.change_language)
    d.show()
    combo = d.pages.widget(0).findChildren(QComboBox)[0]
    combo.setCurrentIndex(combo.findData("fr"))
    qapp.processEvents()
    assert titles() == ["Fichier", "Édition", "Affichage", "Exécuter", "Aide"]
    assert d.windowTitle() == "Paramètres" and d.nav.item(0).text() == "Général"
    assert w.windowTitle() == "astro_gpu · Accueil" and w.screens.currentWidget() is w.home
    assert w.home.count.text() == "3 projets dans le répertoire de travail"
    assert [m.text() for m in w.mode_btns] == ["Graphe", "Inspection", "Post-traitement"]
    assert prefs.get("language") == "fr"
    saved = json.loads(Path(os.environ["ASTRO_GPU_SETTINGS"]).read_text(encoding="utf-8"))
    assert saved["language"] == "fr"
    combo = d.pages.widget(0).findChildren(QComboBox)[0]
    combo.setCurrentIndex(combo.findData("en"))
    qapp.processEvents()
    assert titles() == ["File", "Edit", "View", "Run", "Help"]
    assert d.windowTitle() == "Settings"
    d.close()


def test_prefs_go_only_to_the_override_file(env, monkeypatch, tmp_path):
    f = Path(os.environ["ASTRO_GPU_SETTINGS"])
    prefs.set("units", "px")
    assert json.loads(f.read_text(encoding="utf-8"))["units"] == "px"
    assert prefs.settings_file() == f
    # under pytest, without the override, nothing is read or written
    monkeypatch.delenv("ASTRO_GPU_SETTINGS")
    assert prefs.settings_file() is None
    prefs.load()
    prefs.set("units", "arcsec")
    assert json.loads(f.read_text(encoding="utf-8"))["units"] == "px"
    assert _fingerprint(USER_SETTINGS) == USER_BEFORE


def test_no_mockup_leftovers():
    bad = [re.compile(p) for p in (r"alt2", r"alt3", r"data\.json", r"\bDemo\b", r"\bdemo\b", r"sys\.path\.(insert|append)",
                                   r"mockup/", r"^\s*(from|import)\s+(project|model|blocks|graph|panel|steps|prefs|"
                                   r"i18n|widgets|viewer|home|dialogs)\b")]
    found = []
    for f in CLIENT.rglob("*"):
        if f.suffix not in (".py", ".json") or "__pycache__" in f.parts:
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if any(b.search(line) for b in bad):
                found.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()[:100]}")
    assert not found, "\n".join(found)
    for f in (CLIENT / "locale" / "fr").glob("*.json"):
        cat = json.loads(f.read_text(encoding="utf-8"))
        assert not [k for k, v in cat.items() if re.search(r"(?i)mockup|maquette|\bdemo|démo", k + v)], f.name


def test_settings_hide_what_the_engine_lacks_outside_spec_mode(env, qapp):
    prefs.set("spec_mode", False)
    d = SettingsDialog(None, MACHINE)
    d.show()
    qapp.processEvents()
    assert [d.nav.item(i).text() for i in range(d.nav.count())] == [
        "General", "Folders & disk", "Compute", "Catalogs & astrometry", "Display"]
    d.show_page("compute")
    page = d.pages.currentWidget()
    shown = _texts(page)
    assert "Memory limit" not in shown and "Benchmark" not in shown and "Device" in shown
    dev = page.findChildren(QComboBox)[0]
    assert [dev.itemText(i) for i in range(dev.count())] == ["Automatic", "NVIDIA RTX 4060 Ti", "NVIDIA RTX 3070",
                                                             "CPU only"]
    dev.setCurrentIndex(2)
    assert prefs.get("device") == "cuda:1"
    prefs.set("spec_mode", True)
    d.rebuild()
    qapp.processEvents()
    assert d.nav.count() == 7 and d.nav.item(4).text() == "File formats  ◇"
    d.show_page("compute")
    spins = [s for s in d.pages.currentWidget().findChildren(QWidget) if type(s).__name__ == "QSpinBox"]
    assert spins and all(not s.isEnabled() and s.isVisibleTo(d) for s in spins)
    d.show_page("formats")
    assert all(not c.isEnabled() for c in d.pages.currentWidget().findChildren(QComboBox))
    d.close()


def test_worker_delivers_the_newest_request_of_a_channel(qapp):
    w = ApiWorker()
    got, failed = [], []
    w.done.connect(lambda ch, n, r: got.append((ch, r)))
    w.failed.connect(lambda ch, n, e: failed.append((ch, e)))
    w.submit("a", time.sleep, 0.3)                 # running when the next ones arrive: its result is dropped
    time.sleep(0.05)
    w.submit("a", lambda: "b")                     # waiting: replaced by the next one
    w.submit("a", lambda: "c")
    w.submit("other", lambda: 1)
    w.submit("bad", lambda: 1 / 0)
    assert w.wait(10)
    assert ("a", "c") in got and ("a", "b") not in got and ("a", None) not in got and ("other", 1) in got
    assert failed and failed[0][0] == "bad" and "ZeroDivisionError" in failed[0][1]
    w.stop()


def test_log_shows_the_history_newest_first_and_exports(env, tmp_path):
    w = _main(env)
    w.open_project(env["ws"] / "Heart")
    w.project.log("Project saved")
    w.b_log.setChecked(True)
    rows = w.logp.rows()
    assert [r[1] for r in rows] == ["Project saved", "Project opened"]
    w.logp.append_console("[stack] done")
    out = tmp_path / "log.txt"
    w.logp.write(out)
    text = out.read_text(encoding="utf-8")
    assert "Project opened" in text and "[stack] done" in text
    w.change_language("fr")
    assert [r[1] for r in w.logp.rows()] == ["Projet enregistré", "Projet ouvert"]


def test_new_workflow_dialog_on_the_model(qapp):
    d = NewWorkflowDialog(FakeModel(), templates=TEMPLATES)
    assert len(d.cards) == 6 and d.name.text() == "New workflow"
    d.pick(2)
    assert d.name.text() == "Full 2" and d.opts == {"d": 2}
    d.pick(5)
    assert d.name.text() == "Full (copy)"


def test_french_covers_every_shell_screen(env, qapp):
    """Home, project screen with the Log, every Settings page in Spec mode, the dialogs: no text left in English."""
    prefs.set("spec_mode", True)
    w = _main(env)
    w.change_language("fr")
    assert w.worker.wait()
    texts = _texts(w)
    w.open_project(env["ws"] / "Heart")
    w.b_log.setChecked(True)
    qapp.processEvents()
    texts += _texts(w)
    menus = [a.menu() for a in w.menuBar().actions()]
    while menus:
        m = menus.pop()
        texts += [m.title()] + [a.text() for a in m.actions()]
        menus += [a.menu() for a in m.actions() if a.menu() is not None]
    d = SettingsDialog(w.worker, w.machine, parent=w)
    d.show()
    w.worker.wait()
    for i in range(d.nav.count()):
        d.nav.setCurrentRow(i)
        qapp.processEvents()
        texts += _texts(d)
    for dlg in (NewProjectDialog(env["ws"].as_posix(), "Iris"), NewWorkflowDialog(FakeModel(), templates=TEMPLATES),
                name_dialog(w, "Rename", "Full")[0], w.about_dialog(), w.import_box(env["ws"] / "OldM31")):
        dlg.show()
        qapp.processEvents()
        texts += _texts(dlg) + [dlg.windowTitle()]
        dlg.close()
    assert not i18n.missing(), sorted(i18n.missing())
    assert not _untranslated(texts), _untranslated(texts)


def _shots_dir(tmp_path) -> Path:
    d = Path(os.environ.get("ASTRO_GPU_SHOTS") or tmp_path / "shots")
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_shots(env, qapp, tmp_path):
    """home_en, home_fr, settings (each page), the log and the dialogs, in English and in French; no text cut."""
    out = _shots_dir(tmp_path)
    prefs.set("spec_mode", True)
    cut = {}

    def shot(widget, name):
        widget.show()
        qapp.processEvents()
        widget.grab().save(str(out / f"{name}.png"))
        assert (out / f"{name}.png").stat().st_size > 1000
        c = _cut(widget)
        if c:
            cut[name] = c

    for lang in ("en", "fr"):
        appmod.apply_language(lang)
        w = _main(env)
        prefs.add_recent((env["ws"] / "Heart").as_posix(), "HeartSoul", "IC 1805 · Heart nebula")
        w.build_menus()
        w.show_home()
        w.worker.wait()
        shot(w, f"home_{lang}")
        prefs.set("workspace", (env["tmp"] / "empty").as_posix())        # a workspace not created yet
        prefs.set("recent", [])
        w.show_home()
        w.worker.wait()
        shot(w, f"home_empty_{lang}")
        assert not w.home.cards and w.home.count.text() == i18n._("No project in the workspace")
        prefs.set("workspace", env["ws"].as_posix())
        menu = w.menuBar().actions()[0].menu()
        menu.adjustSize()
        shot(menu, f"menu_file_{lang}")
        for spec in (False, True):
            prefs.set("spec_mode", spec)
            d = SettingsDialog(w.worker, w.machine, parent=w)
            d.show()
            for i, key in enumerate(d.keys):
                d.nav.setCurrentRow(i)
                w.worker.wait()
                shot(d, f"settings{'_spec' if spec else ''}_{i + 1}_{key}_{lang}")
            d.close()
        w.open_project(env["ws"] / "Heart")
        w.project.log("Project saved")
        w.b_log.setChecked(True)
        w.logp.set_console("[stack] 721 frames · drizzle ×1.5 · pixfrac 0.70\n[stack] done in 33 min 52 s")
        shot(w, f"log_{lang}")
        for dlg, name in ((NewProjectDialog(env["ws"].as_posix(), "Rosette", "NGC 2237"), "new_project"),
                          (NewProjectDialog(env["ws"].as_posix(), "Iris"), "new_project_taken"),
                          (NewWorkflowDialog(FakeModel(), templates=TEMPLATES), "new_workflow"),
                          (name_dialog(w, "Rename", "Full + neural stars")[0], "rename"),
                          (w.about_dialog(), "about"), (w.import_box(env["ws"] / "OldM31"), "previous_version")):
            shot(dlg, f"dialog_{name}_{lang}")
            dlg.close()
        w.close()
    assert not cut, cut


def test_python_m_starts_the_client(tmp_path):
    env = dict(os.environ, ASTRO_GPU_SETTINGS=str(tmp_path / "settings.json"), QT_QPA_PLATFORM="offscreen",
               ASTRO_GPU_HOME=str(tmp_path / "home"))
    env.pop("PYTEST_CURRENT_TEST", None)
    (tmp_path / "ws").mkdir()
    (tmp_path / "settings.json").write_text(json.dumps({"workspace": (tmp_path / "ws").as_posix()}), encoding="utf-8")
    p = subprocess.Popen([sys.executable, "-m", "astro_gpu.client"], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(4)
        alive = p.poll() is None
    finally:
        p.kill()
        _out, err = p.communicate(timeout=20)
    assert alive, err[-2000:]
    assert _fingerprint(USER_SETTINGS) == USER_BEFORE


def test_viewer_shows_an_image_and_the_curtain(qapp):
    from astro_gpu.client.viewer import StaticSource, Viewer
    v = Viewer()
    v.resize(400, 300)
    a = np.zeros((120, 160, 3), np.uint8)
    a[40:80, 60:100] = 255
    v.set_source(StaticSource(a))
    v.compare = StaticSource(a // 2)
    appmod.apply_language("fr")
    img = v.grab().toImage()
    assert img.width() == 400 and img.pixelColor(200, 150).red() > 0
    appmod.apply_language("en")
