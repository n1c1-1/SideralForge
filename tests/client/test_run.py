"""The Run part of the client (plan dialog, run service, progress on the graph, the console, the recap), offscreen.

Most tests run toy stages (inventory → stack → denoise (the 'denoise' option: steps, checkpoints) → color →
render_prep) in a real detached worker on the CPU: the worker builds the same toy graph from the spec
'test_run:toy_rules' (this module, put on its PYTHONPATH). The slow tests run the real stages on a copy of the
test project (tests/conftest.py) on the CPU, and take the screenshots (ASTRO_GPU_SHOTS names the folder; default
%TEMP%/run_shots).

ASTRO_GPU_SETTINGS points to a temporary file in every test; the user's settings file is checked untouched.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")      # the offscreen platform has no fonts else

pytest.importorskip("PySide6")
import psutil  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from astro_gpu import api  # noqa: E402
from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client.blocks import Graph  # noqa: E402
from astro_gpu.client.graph import BlockItem, GraphPage  # noqa: E402
from astro_gpu.client.logview import LogPanel  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.runs import RunDialog, RunOverlay, RunStatus, build_rows, row_state, run_blocks  # noqa: E402
from astro_gpu.client.services.run import (  # noqa: E402
    DONE,
    FAILED,
    INTERRUPTED,
    NOT_STARTED,
    RUNNING,
    SKIPPED,
    RunController,
    RunTrack,
    duration_text,
    engine_text,
    render_line,
    size_text,
)
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.client.widgets import QSS  # noqa: E402
from astro_gpu.core import project as project_mod  # noqa: E402
from astro_gpu.core import runner2  # noqa: E402
from astro_gpu.core.params import P, StageParams  # noqa: E402
from astro_gpu.core.project import list_frames  # noqa: E402
from astro_gpu.core.stage import Stage, StageError  # noqa: E402
from astro_gpu.core.store import KeyRules, Store  # noqa: E402
from astro_gpu.core.workflow import Chains  # noqa: E402

try:
    from conftest import TEST_PROJECT, needs_test_project  # noqa: E402
except ImportError:          # a detached worker imports this module for toy_rules alone: tests/ is not on its path
    TEST_PROJECT, needs_test_project = Path("."), pytest.mark.skip(reason="no tests/conftest.py")

HERE = Path(__file__).parent
SPEC = "test_run:toy_rules"
F_COPY = TEST_PROJECT                                   # only read: the runs work on copies
SHOTS = Path(os.environ.get("ASTRO_GPU_SHOTS") or Path(os.environ.get("TEMP", ".")) / "run_shots")


# ---------------------------------------------------------------------------------------------- toy stages
class InvParams(StageParams):
    max_lights: int = P(0, label="max lights", help="", ge=0)


class Inv(Stage):
    name = "inventory"
    title = "Inventory"
    Params = InvParams

    def run(self, ctx, p):
        files = list_frames(ctx.project.config.inputs.lights)
        ctx.save_json("frames.json", {"lights": [{"path": str(f)} for f in files]})
        ctx.metric("n_lights", len(files))


class ToyStackParams(StageParams):
    scale: float = P(1.5, label="Scale", help="", ge=1.0, le=3.0)
    workers: int = P(4, label="workers", help="", ge=1)
    exclude: list[str] = P([], label="exclude", help="", level="internal")


class ToyStack(Stage):
    name = "stack"
    title = "Stack"
    deps = ("inventory",)
    Params = ToyStackParams

    def run(self, ctx, p):
        n = len(ctx.load_json("inventory", "frames.json")["lights"])
        ctx.save_json("stack.json", {"n": n, "scale": p.scale})


class BParams(StageParams):
    steps: int = P(8, label="steps", help="", ge=1)
    delay: float = P(0.0, label="delay", help="seconds per step", ge=0)
    fail: bool = P(False, label="fail", help="")


class B(Stage):
    """A checkpointed training (the Denoise block's): steps of `delay` seconds, resumed from its last step after a
    stop or a kill."""

    name = "denoise"
    title = "B"
    deps = ("stack",)
    Params = BParams

    def run(self, ctx, p):
        if p.fail:
            raise StageError("b failed on purpose")
        ck = ctx.ckpt / "i.json"
        start = json.loads(ck.read_text())["i"] if ck.exists() else 0
        for i in range(start, p.steps):
            ck.write_text(json.dumps({"i": i + 1}))
            time.sleep(p.delay)
            ctx.progress((i + 1) / p.steps, f"step {i + 1}", force=True)        # raises Interrupted on STOP
        ctx.save_json("b.json", {"steps": p.steps, "resumed_from": start})


class CParams(StageParams):
    k: float = P(1.0, label="k", help="")


class C(Stage):
    name = "color"
    title = "C"
    deps = ("stack", "denoise")          # denoise optional
    Params = CParams

    def run(self, ctx, p):
        steps = ctx.load_json("denoise", "b.json")["steps"] if ctx.has("denoise") else None
        ctx.save_json("c.json", {"n": ctx.load_json("stack", "stack.json")["n"], "steps": steps})


class E(Stage):
    name = "render_prep"
    title = "E"
    deps = ("color",)
    Params = CParams

    def run(self, ctx, p):
        ctx.save_json("e.json", ctx.load_json("color", "c.json"))


def toy_rule(options, end, reference, learned_layer):
    return ["inventory", "stack"] + (["denoise"] if options.get("denoise") else []) + ["color", "render_prep"]


def toy_rules() -> KeyRules:
    """Also built by the detached worker (plan.rules = 'test_run:toy_rules')."""
    chains = Chains([Inv(), ToyStack(), B(), C(), E()], extra={"render_prep": ("denoise",)},
                    optional={"color": ("denoise",), "render_prep": ("denoise",)},
                    blocks={"calibration": ("inventory",), "stacking": ("stack",), "denoise": ("denoise",),
                            "color": ("color",), "output": ("render_prep",)}, rule=toy_rule)
    return KeyRules(chains, auto_fields={}, path_fields={}, non_key={"stack": ("workers",)},
                    implied=lambda options, stage: {})


# ---------------------------------------------------------------------------------------------- fixtures
def _user_settings() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "astro_gpu" / "settings.json"


def _fingerprint(p: Path):
    return (p.stat().st_mtime_ns, hashlib.sha1(p.read_bytes()).hexdigest()) if p.exists() else None


USER_SETTINGS = _user_settings()
USER_BEFORE = _fingerprint(USER_SETTINGS)


@pytest.fixture(scope="session")
def qapp():
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return a


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    appmod.apply_language("en")
    shown = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, _k=kind, **k: shown.append((_k, a[1:3])))
    state = tmp_path / "state"                           # a private ASTRO_GPU_HOME: this process and the workers
    monkeypatch.setattr(project_mod, "STATE", state)
    monkeypatch.setenv("ASTRO_GPU_HOME", str(state))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(HERE), os.environ.get("PYTHONPATH", "")]))
    yield {"tmp": tmp_path, "shown": shown}
    appmod.apply_language("en")
    assert _fingerprint(USER_SETTINGS) == USER_BEFORE, "the user's settings file was touched"


def raw_folder(folder: Path, n: int = 3) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"IMG_{i:03d}.ARW").write_bytes(b"\0" * (100 + 10 * i))
    return folder


def beat_limit(load: float) -> float:
    """What the GUI thread may stall for: 200 ms, on a machine that is not busy. The model reads the states and replans
    from the manifests when a stage ends (tens of ms), and the graph page rebuilds: on a machine whose cores are busy
    with other jobs (other tests, project runs) that takes longer, and the limit goes up to 600 ms."""
    return 0.2 if load < 40 else 0.6


class Beat:
    """The GUI thread's heartbeat: the longest time between two ticks of a 10 ms timer."""

    def __init__(self):
        self.max, self.last = 0.0, None
        psutil.cpu_percent(None)
        self.timer = QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.tick)

    def tick(self):
        now = time.perf_counter()
        if self.last is not None:
            self.max = max(self.max, now - self.last)
        self.last = now

    def start(self):
        self.max, self.last = 0.0, None
        self.timer.start()

    def skip(self):
        """Do not count the time since the last tick (the test itself took a screenshot)."""
        self.last = None

    def stop(self):
        self.timer.stop()
        self.load = psutil.cpu_percent(None)                  # the machine's load since the beat started

    @property
    def limit(self) -> float:
        return beat_limit(getattr(self, "load", 0.0))


def pump(cond, timeout: float = 90.0, beat: Beat | None = None) -> bool:
    """Process events until ``cond()`` (the GUI thread is never blocked: a worker thread reads the events)."""
    end = time.time() + timeout
    while time.time() < end:
        QApplication.processEvents()
        if cond():
            return True
        time.sleep(0.005)
    return False


def toy_project(folder: Path, steps: int = 8, delay: float = 0.0, fail: bool = False):
    """A saved project of the toy graph: Classic (no b) and Denoise (with b), three raws, on the CPU."""
    p = api.create_project(folder, "toy")
    api.use_rules(p, SPEC)
    api.add_session(p, lights=[raw_folder(folder.parent / "raw" / "lights")])
    api.add_workflow(p, "Plain", "Classic stack")
    w = api.add_workflow(p, "Trained", "Classic + denoise")
    api.set_param(p, [w.id], "denoise", "steps", steps)
    api.set_param(p, [w.id], "denoise", "delay", delay)
    if fail:
        api.set_param(p, [w.id], "denoise", "fail", True)
    p.devices["main"] = "cpu"
    api.save(p)
    return p


class World:
    """A model, its Log panel and a run controller on one project."""

    def __init__(self, p, history=None):
        self.p = p
        self.worker = ApiWorker()
        self.model = ProjectModel(p, self.worker, readonly=True, gpu=False, device="cpu", history=history)
        assert self.worker.wait(120) and self.model.ready, self.model.error
        self.logp = LogPanel(self.model)
        self.ctl = RunController(self.model, None, lambda: self.logp)
        self.beat = Beat()

    @property
    def track(self) -> RunTrack:
        return self.ctl.track

    def run_all(self, timeout: float = 120.0):
        """Run all through the dialog, and wait for the end."""
        d = self.ctl.launch()
        d.start()
        assert pump(lambda: self.track is not None and self.track.finished and not self.ctl.running, timeout), \
            "the run did not end"
        return d

    def console(self) -> str:
        return self.logp.console.toPlainText()

    def close(self):
        self.beat.stop()
        self.ctl.close()
        self.model.close()
        self.worker.stop()
        self.logp.deleteLater()
        QApplication.processEvents()


@pytest.fixture
def toy(env):
    p = toy_project(env["tmp"] / "toy")
    w = World(p)
    yield w
    w.close()


def make_world(tmp: Path, **kw) -> World:
    return World(toy_project(tmp / "toy", **kw))


def texts_of(root: QWidget) -> list[str]:
    out = []
    for w in [root] + root.findChildren(QWidget):
        if not w.isVisibleTo(root) and w is not root:
            continue
        if isinstance(w, QLabel):
            out.append(re.sub(r"<[^>]+>", "", w.text()))
        elif isinstance(w, QAbstractButton):
            out.append(w.text())
        if w.toolTip():
            out.append(w.toolTip())
    return [t.strip() for t in out if t and t.strip()]


NAMES = {"Classic stack", "Classic + denoise", "Full", "Full + neural stars", "Plain", "Trained"}


def untranslated(texts) -> list[str]:
    """Texts shown in English although a French translation exists (the names the user typed are never translated)."""
    cat = i18n._cat
    return [t for t in texts if t in cat and cat[t] != t and t not in NAMES]


def ev(kind: str, t: float, run: str = "r1", **kw) -> dict:
    return {"t": t, "type": kind, "run": run, **kw}


def job(key, stage, block, wf=(1,), est=60.0, deps=(), measured=False, disk=0.5, run_wf=None) -> runner2.Job:
    return runner2.Job(key, stage, block, list(wf), run_wf or list(wf)[0], {d: d for d in deps}, est, measured, disk)


def done_keys(root: Path) -> dict[str, list[str]]:
    st = Store(root)
    out = {}
    for sd in sorted((root / "store").iterdir()):
        if sd.is_dir() and not sd.name.startswith("_"):
            out[sd.name] = sorted(d.name for d in sd.iterdir() if d.is_dir() and not d.name.endswith(".tmp")
                                  and st.done(sd.name, d.name))
    return out


def snapshot(folder: Path):
    return sorted((p.relative_to(folder).as_posix(), p.stat().st_mtime_ns, p.stat().st_size)
                  for p in folder.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------------------------- the track (no Qt)
def _failed_run_events(t0: float = 1000.0) -> list[dict]:
    """a (done) → b (fails) → c (skipped), and d (done, independent of b)."""
    return [
        ev("run_start", t0, stages=["a", "b", "c", "d"], jobs=4, device="cpu", pid=4242),
        ev("plan", t0, jobs=[{"key": "ka", "stage": "a", "block": "stacking", "wf": [1], "est_s": 60.0}], skipped={}),
        ev("stage_start", t0 + 1, stage="a", key="ka", wf=[1], block="stacking"),
        ev("progress", t0 + 2, stage="a", key="ka", frac=0.5, msg="step 5", eta=3.0),
        ev("progress", t0 + 3, stage="a", key="ka", frac=0.55, msg="step 6", eta=3.0),     # same line: no new console line
        ev("stage_end", t0 + 4, stage="a", key="ka", status="done", duration=3.0),
        ev("stage_start", t0 + 4, stage="b", key="kb", wf=[1], block="denoise"),
        ev("log", t0 + 5, stage="b", key="kb", level="error", msg="out of memory"),
        ev("stage_end", t0 + 5, stage="b", key="kb", status="failed", duration=1.0, error="out of memory"),
        ev("stage_end", t0 + 5, stage="c", key="kc", status="skipped", reason="upstream failed", upstream=["b"]),
        ev("stage_start", t0 + 6, stage="d", key="kd", wf=[2], block="stacking"),
        ev("stage_end", t0 + 7, stage="d", key="kd", status="done", duration=1.0),
        ev("run_end", t0 + 8, status="failed", done=["ka", "kd"], failed=["kb"], skipped=["kc"], interrupted=[],
           not_started=[]),
    ]


def _failed_plan():
    return [job("ka", "a", "stacking"), job("kb", "b", "denoise", deps=["ka"]),
            job("kc", "c", "color", deps=["kb"]), job("kd", "d", "stacking", wf=(2,), est=30.0)]


def test_the_track_follows_the_events_of_a_failed_run():
    t = RunTrack(_failed_plan(), "r1")
    assert [j.state for j in t.jobs] == ["waiting"] * 4 and t.status == "planned" and not t.finished
    first = t.apply(_failed_run_events()[:4])
    assert t.status == "running" and t.current().stage == "a" and t.current().frac == 0.5
    assert first.stage_changed and first.progressed and not first.ended
    assert 0.0 < t.fraction() < 0.5 and t.remaining() > 60.0               # b, c, d wait: their estimates add up
    rest = t.apply(_failed_run_events()[4:])
    assert rest.ended and t.finished and t.status == "failed"
    assert [j.state for j in t.jobs] == [DONE, FAILED, SKIPPED, DONE]
    b, c = t.by_key["kb"], t.by_key["kc"]
    assert b.error == "out of memory" and c.upstream == ["b"]
    assert t.culprits(c) == [b] and t.culprits(t.by_key["ka"]) == []
    assert t.counts[DONE] == 2 and t.counts[FAILED] == 1 and t.counts[SKIPPED] == 1
    assert t.fraction() == pytest.approx(1.0) and t.remaining() == 0.0 and t.elapsed() == pytest.approx(8.0)
    kinds = [x[1] for x in t.lines]
    assert kinds == ["run_start", "stage_start", "progress", "stage_done", "stage_start", "log", "stage_failed",
                     "stage_skipped", "stage_start", "stage_done", "run_end"]
    # the events of another run are ignored
    assert not RunTrack(_failed_plan(), "r1").apply([ev("stage_start", 1.0, run="other", stage="a", key="ka")]).lines


def test_a_track_built_from_the_events_alone_and_the_culprit_without_dependencies():
    t = RunTrack()
    t.apply([ev("plan", 1.0, jobs=[{"key": "ka", "stage": "a", "block": "stacking", "wf": [1], "est_s": 5.0},
                                   {"key": "kb", "stage": "b", "block": "denoise", "wf": [1], "est_s": 5.0}]),
             ev("stage_start", 2.0, stage="a", key="ka"), ev("stage_end", 3.0, stage="a", key="ka", status="failed",
                                                              error="boom"),
             ev("stage_end", 3.0, stage="b", key="kb", status="skipped", upstream=["a"])])
    assert [j.state for j in t.jobs] == [FAILED, SKIPPED] and t.run_id == "r1"
    assert t.culprits(t.by_key["kb"]) == [t.by_key["ka"]]                  # by the stage names of the event


def test_a_stopped_run_and_a_lost_worker_and_a_run_that_never_started():
    t = RunTrack([job("ka", "a", "stacking"), job("kb", "b", "denoise", deps=["ka"])], "r1")
    t.apply([ev("run_start", 1.0, pid=1), ev("stage_start", 2.0, stage="a", key="ka"),
             ev("stage_end", 3.0, stage="a", key="ka", status="interrupted", duration=1.0),
             ev("run_end", 3.0, status="interrupted", done=[], failed=[], skipped=[], interrupted=["ka"],
                not_started=["kb"])])
    assert t.status == "interrupted" and [j.state for j in t.jobs] == [INTERRUPTED, NOT_STARTED] and not t.lost
    lost = RunTrack([job("ka", "a", "stacking"), job("kb", "b", "denoise", deps=["ka"])], "r1")
    lost.apply([ev("run_start", 1.0), ev("stage_start", 2.0, stage="a", key="ka")])
    b = lost.end_lost()
    assert lost.lost and lost.finished and lost.status == "interrupted" and b.ended
    assert [j.state for j in lost.jobs] == [INTERRUPTED, NOT_STARTED] and lost.lines[-1][1] == "run_lost"
    never = RunTrack([job("ka", "a", "stacking")], "r1")
    never.abort("ImportError: no module named x")
    assert never.status == "failed" and never.jobs[0].state == NOT_STARTED and never.lines[-1][1] == "not_started"
    assert "no module named x" in render_line(never.lines[-1])


def test_row_states_and_the_progress_numbers():
    assert row_state([]) == "waiting" and row_state([DONE, DONE]) == DONE and row_state([DONE, RUNNING]) == RUNNING
    assert row_state([DONE, "waiting"]) == RUNNING and row_state([SKIPPED, NOT_STARTED]) == SKIPPED
    assert row_state([NOT_STARTED]) == NOT_STARTED and row_state([DONE, FAILED]) == FAILED
    assert duration_text(0.0) == "≤ 1 min" and duration_text(600, False) == "≈ 10 min" and duration_text(600) == "10 min"
    assert duration_text(3 * 3600 + 120, False) == "≈ 3 h 02"


# ---------------------------------------------------------------------------------------------- language
def test_numbers_and_the_engine_texts_follow_the_language():
    assert size_text(7.6) == "7.6" and size_text(0.05) == "0.05"
    assert engine_text("training step 120/6000, loss 0.0123") == "training step 120/6000, loss 0.0123"
    appmod.apply_language("fr")
    assert size_text(7.6) == "7,6" and size_text(0.05) == "0,05"
    assert engine_text("training step 120/6000, loss 0.0123") == "entraînement, étape 120/6000, perte 0,0123"
    assert engine_text("reading flat darks 3/20") == "lecture des darks de flat 3/20"
    solved = 'solved: centre RA 322.1234 Dec +57.5000, 1.520"/px, field 1.80 x 1.20 deg, residual 0.210 px on 312 stars'
    assert engine_text(solved) == ('résolu : centre AD 322,1234 Dec +57,5000, 1,520"/px, champ 1,80 x 1,20 deg, résidu '
                                   '0,210 px sur 312 étoiles')
    assert engine_text("40/40 frames aligned, residual 0.31 px, FWHM 2.80 px, transparency p10/p90 0.95/1.00") == \
        "40/40 images alignées, résidu 0,31 px, FWHM 2,80 px, transparence p10/p90 0,95/1,00"
    assert engine_text("loading the stack") == "chargement de l'empilement"
    assert engine_text("pass 2: 5/40 frames, rejected 0.31 %") == "passe 2 : 5/40 images, rejetés 0,31 %"
    assert engine_text('searching the field at 1.52"/px (shifted centres)') == \
        'recherche du champ à 1,52"/px (centres décalés)'
    assert "decimal" not in engine_text("a message nobody translated yet")
    t = RunTrack(_failed_plan(), "r1")
    t.apply(_failed_run_events())
    text = "\n".join(render_line(x, str.upper) for x in t.lines)
    assert "A · terminé en 3 min" not in text and "A · terminé en ≤ 1 min" in text
    assert "B · en échec : out of memory" in text and "C · non exécuté, B a échoué" in text
    assert "Exécution en échec : 1 en échec, 1 non exécutées" in text
    assert untranslated([render_line(x).split("  ", 1)[1] for x in t.lines]) == []


def test_every_text_of_the_run_part_has_its_french():
    """Every string the two modules show, and the engine's progress messages, are in run.json (or an older file)."""
    import ast
    appmod.apply_language("fr")
    cat = i18n._cat
    missing = []
    for f in (HERE.parents[1] / "astro_gpu" / "client" / "runs.py",
              HERE.parents[1] / "astro_gpu" / "client" / "services" / "run.py"):
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_" and n.args \
                    and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                if n.args[0].value not in cat:
                    missing.append(n.args[0].value)
    from astro_gpu.client.runs import STATE_TIP
    from astro_gpu.client.services.run import PATTERNS
    missing += [v for v in STATE_TIP.values() if v not in cat]
    missing += [t for _rx, t, _tr in PATTERNS if t not in cat]
    for what in ("{stage} · error: {msg}", "{stage} · warning: {msg}", "{stage} · {msg}"):
        missing += [what] if what not in cat else []
    assert not missing, missing


# ---------------------------------------------------------------------------------------------- the plan dialog
def test_the_plan_dialog_lists_the_blocks_the_sharing_workflows_the_durations_and_the_disk(toy):
    d = toy.ctl.launch()
    assert isinstance(d, RunDialog) and d.phase() == "plan" and d.isVisible() and not d.isModal()
    rows = [(r.row.title, r.row.who) for r in d.rows]
    # the graph's order: the first workflow's blocks, then what the second one adds
    assert rows == [("Calibration", ["Plain", "Trained"]), ("Stacking", ["Plain", "Trained"]),
                    ("Color", ["Plain"]), ('Output "Plain"', []), ("Denoise", ["Trained"]),
                    ("Color", ["Trained"]), ('Output "Trained"', [])]
    # a block every workflow shares says so, as the panel's scope does; else the workflows by name
    assert [r.who.full_text() for r in d.rows[:3]] == ["both workflows", "both workflows", "Plain"]
    plan = toy.model.plan_all()
    assert len(plan) == 7 and [j.stage for j in plan.jobs].count("stack") == 1       # a shared block is planned once
    assert d.summary.text() == f"Duration: ≈ {i18n.hm(plan.seconds)} · disk: {size_text(plan.disk_gb)} GB"
    assert not plan.measured and all(r.dur.text().startswith("≈") or r.dur.text() == "≤ 1 min" for r in d.rows)
    assert d.b_run.isEnabled() and d.b_run.isVisible() and d.b_cancel.text() == "Cancel" and not d.b_stop.isVisible()
    assert toy.ctl.track is None and not toy.ctl.running
    # a title for each way in
    assert d.windowTitle() == "Run project"
    d.close()
    toy.model.select(step="empilement")
    d2 = toy.ctl.run_to_here()
    assert d2.windowTitle() == 'Run "Plain" to Stacking' and [r.row.title for r in d2.rows] == ["Calibration", "Stacking"]
    d2.close()


def test_the_plan_names_the_workflows_that_cannot_run_and_warns_about_the_disk(toy):
    plan = toy.model.plan_all()
    plan.skipped = {2: "the earlier layer belongs to another stack"}
    d = toy.ctl.open_plan(plan, lambda: i18n._("Run project"))
    assert pump(lambda: toy.ctl.free_gb() is not None, 30)
    texts = texts_of(d)
    assert '"Trained" is not run: the earlier layer belongs to another stack' in texts
    assert not any("Not enough free disk space" in t for t in texts)
    plan.jobs[0].disk_gb = 1e9                                                   # more than any drive
    toy.ctl.updated.emit()
    assert any(t.startswith("Not enough free disk space: ") for t in texts_of(d))
    d.close()
    empty = toy.ctl.open_plan(api.Plan(), lambda: i18n._("Run project"))
    assert empty.summary.text() == "Nothing to run" and not empty.b_run.isEnabled() and not empty.scroll.isVisible()
    empty.close()


def test_the_dialog_in_french(toy):
    appmod.apply_language("fr")
    plan = toy.model.plan_all()
    plan.jobs[0].disk_gb = 7.6
    toy.model.plan_all = lambda: plan
    d = toy.ctl.launch()
    t = texts_of(d)
    assert t[0] == "Exécuter le projet"
    assert any(x.startswith("Durée : ≈ ") and x.endswith(f" · disque : {size_text(plan.disk_gb)} Go") and
               "," in size_text(plan.disk_gb) for x in t), t
    assert 'Sortie « Plain »' in t and "Annuler" in t and "Exécuter" in t
    assert untranslated(t) == []
    d.close()


# ---------------------------------------------------------------------------------------------- breakpoints
def in_process(monkeypatch):
    """The run service starts the worker in this process, the toy stages and all (a detached worker is level 3)."""
    from astro_gpu.client.services import run as run_service

    def start(snapshot, plan, device):
        info = runner2.prepare_run(snapshot, plan, "cpu")
        assert runner2.run_plan(snapshot.root, info.plan_file, rules=toy_rules(), echo=False) == "done"
        return info
    monkeypatch.setattr(run_service, "_start_job", start)
    real_init = run_service.RunFollower.__init__
    monkeypatch.setattr(run_service.RunFollower, "__init__",
                        lambda self, project, worker, track, info, offset=None, interval=20, parent=None:
                        real_init(self, project, worker, track, info, offset, 20, parent))        # polls every 20 ms


def finish(w, timeout: float = 60.0):
    assert pump(lambda: w.track is not None and w.track.finished and not w.ctl.running, timeout), "the run did not end"


def states_of(w):
    return {(v.pid, s): w.model.state(v, s) for v in w.model.paths for s in w.model.all_stages(v)}


def test_a_breakpoint_stops_the_run_after_its_block_and_continue_goes_on(toy, monkeypatch):
    in_process(monkeypatch)
    m, ctl = toy.model, toy.ctl
    plain, trained = m.view(1), m.view(2)
    m.set_breakpoint([plain], "calibration")
    assert m.p.workflow(1).breakpoints == ["calibration"] and m.p.workflow(2).breakpoints == []
    # the plan: Calibration only, and it says why
    d = ctl.launch()
    assert [j.stage for j in d.plan.jobs] == ["inventory"] and [r.row.title for r in d.rows] == ["Calibration"]
    assert d.pause_note.isVisibleTo(d) and d.pause_note.text() == "Stops after Calibration for your review"
    assert d.pause_more.isVisibleTo(d) and d.pause_more.text() == "Workflow: Plain"      # one of the two workflows
    assert not d.review.isVisibleTo(d) and d.b_run.isEnabled()
    assert ctl.pause is not None and ctl.pause.block == "calibration" and ctl.pause.pids == [1]
    # the graph before the run: the dot is there, red (the run will stop), on the block both workflows share
    g = Graph(m, None)
    assert g.breakpoint(g.node_of(plain, "calibration")) == ("pending", 1, 2)
    assert g.breakpoint(g.node_of(plain, "empilement")) is None and g.breakpoint(g.root) is None
    d.start()
    finish(toy)
    t = toy.track
    assert t.status == "done" and [j.stage for j in t.jobs] == ["inventory"]
    assert states_of(toy)[(1, "inventory")] == "done" and states_of(toy)[(1, "stack")] == "pending"
    # the recap: paused, with its two buttons; the graph says paused
    assert d.phase() == "end" and d.paused() and d.review.isVisibleTo(d)
    assert d.review_text.text() == "Paused for your review at Calibration: check it in Inspect, then Continue."
    assert d.b_continue.isVisibleTo(d) and d.b_continue.text() == "Continue"
    assert d.b_inspect.isVisibleTo(d) and d.b_inspect.text() == "Open in Inspect"
    assert d.summary.text() == "Paused for your review" and not d.pause_note.isVisibleTo(d)
    assert [r.mark.state for r in d.rows] == [DONE]
    assert Graph(m, None).breakpoint(Graph(m, None).node_of(plain, "calibration")) == ("paused", 1, 2)
    assert m.review_state(plain, "calibration") == "paused" and m.review_state(trained, "calibration") is None
    assert "Paused for your review at Calibration: check it in Inspect, then Continue." in toy.console()
    assert m.history[-1][1] == "{text}"
    assert m.history[-1][2]["text"].startswith("Run paused for your review at Calibration: 1 stage in")
    # Continue: the review is marked, and the run goes on to the end, in a new dialog, at once
    d.b_continue.click()
    assert ctl.pause is None and ctl.running and ctl.dialog is not d and m.is_reviewed(plain, "calibration")
    finish(toy)
    assert sorted(j.stage for j in toy.track.jobs) == sorted(["stack", "color", "render_prep", "denoise", "color",
                                                               "render_prep"])
    assert all(s == "done" for s in states_of(toy).values())
    d2 = ctl.dialog
    assert not d2.paused() and not d2.review.isVisibleTo(d2) and d2.summary.text().startswith("Done in ")
    assert Graph(m, None).breakpoint(Graph(m, None).node_of(plain, "calibration")) == ("reviewed", 1, 2)
    assert len(m.plan_all()) == 0 and m.pause_point() is None


def test_each_continue_runs_on_to_the_next_pause_and_the_plan_counts_them(toy, monkeypatch):
    in_process(monkeypatch)
    m, ctl = toy.model, toy.ctl
    plain = m.view(1)
    m.set_breakpoint([plain], "calibration")
    m.set_breakpoint([plain], "empilement")
    m.set_breakpoint([plain], "post")
    d = ctl.launch()
    assert d.pause_note.text() == "Stops after Calibration for your review"
    assert d.pause_more.text() == "Workflow: Plain · 2 more pauses after it: Stacking, Output"
    assert ctl.pause.later == ["empilement", "post"]
    d.start()
    finish(toy)
    assert [j.stage for j in toy.track.jobs] == ["inventory"] and d.paused()
    assert d.review_more.text() == "Workflow: Plain · 2 more pauses after it: Stacking, Output"
    d.b_continue.click()                                       # to the second pause
    finish(toy)
    d2 = ctl.dialog
    assert [j.stage for j in toy.track.jobs] == ["stack"] and d2.paused()
    assert d2.review_text.text() == "Paused for your review at Stacking: check it in Inspect, then Continue."
    assert d2.review_more.text() == "Workflow: Plain · 1 more pause after it: Output"
    assert [m.review_state(plain, k) for k in ("calibration", "empilement", "post")] == ["reviewed", "paused", "pending"]
    d2.b_continue.click()                                      # to the third (the Output of Plain)
    finish(toy)
    d3 = ctl.dialog
    assert [j.stage for j in toy.track.jobs] == ["color", "render_prep"] and d3.paused()
    assert d3.review_text.text() == "Paused for your review at Output: check it in Inspect, then Continue."
    assert d3.review_more.text() == "Workflow: Plain"
    d3.b_continue.click()                                      # nothing is left but the other workflow
    finish(toy)
    assert all(s == "done" for s in states_of(toy).values())
    assert [m.review_state(plain, k) for k in ("calibration", "empilement", "post")] == ["reviewed"] * 3


def test_reviewed_is_tied_to_the_results_of_the_block(toy, monkeypatch):
    in_process(monkeypatch)
    m, ctl = toy.model, toy.ctl
    plain = m.view(1)
    m.set_breakpoint([plain], "calibration")
    d = ctl.launch()
    d.start()
    finish(toy)
    d.b_continue.click()
    finish(toy)
    assert m.review_state(plain, "calibration") == "reviewed" and m.pause_point() is None
    before = m.review_key(plain, "calibration")
    # a setting of Calibration changes: its results are no longer these ones, so the breakpoint applies again
    m.apply_change(m.paths, {"kind": "param", "stage": "inventory", "name": "max_lights", "value": 2})
    assert m.review_key(plain, "calibration") != before and not m.is_reviewed(plain, "calibration")
    assert m.review_state(plain, "calibration") == "pending"
    pause = m.pause_point()
    assert pause is not None and pause.block == "calibration" and not pause.done
    plan, pause = m.plan_run()
    assert [j.stage for j in plan.jobs] == ["inventory"] and pause is not None
    # the old results are back with Undo: the same results, already reviewed, the run goes through
    assert m.undo()
    assert m.review_key(plain, "calibration") == before and m.review_state(plain, "calibration") == "reviewed"
    assert m.pause_point() is None and m.plan_run()[1] is None
    # Continue while the block is Outdated (settings changed during the review) marks nothing, and pauses again
    m.apply_change(m.paths, {"kind": "param", "stage": "inventory", "name": "max_lights", "value": 2})
    stale = m.review_key(plain, "calibration")
    d2 = ctl.launch()
    assert [j.stage for j in d2.plan.jobs][:1] == ["inventory"] and d2.pause_note.text().startswith("Stops after")
    d2.start()
    finish(toy)
    assert d2.paused() and m.review_state(plain, "calibration") == "paused" and m.review_key(plain, "calibration") == stale
    m.apply_change(m.paths, {"kind": "param", "stage": "inventory", "name": "max_lights", "value": 1})
    assert m.review_state(plain, "calibration") == "pending"
    ctl.pause.pids = [1]
    m.mark_reviewed(ctl.pause)                                  # the block is not computed now: nothing is marked
    assert not m.is_reviewed(plain, "calibration") and m.pause_point().block == "calibration"


def test_a_block_done_and_not_reviewed_opens_the_dialog_on_the_pause(toy, monkeypatch):
    in_process(monkeypatch)
    m, ctl = toy.model, toy.ctl
    toy.run_all()                                              # everything computed, no breakpoint yet
    assert len(m.plan_all()) == 0
    plain = m.view(1)
    keys_before = {(v.pid, s): m.stage_key(v, s) for v in m.paths for s in m.all_stages(v)}
    states_before = states_of(toy)
    m.set_breakpoint([plain], "empilement")                    # a breakpoint changes no key and makes nothing Outdated
    assert {(v.pid, s): m.stage_key(v, s) for v in m.paths for s in m.all_stages(v)} == keys_before
    assert {(v.pid, s): api.stage_key(m.p, v.pid, s) for v in m.paths for s in m.all_stages(v)} == keys_before
    assert states_of(toy) == states_before and len(m.plan_all()) == 0
    assert m.review_state(plain, "empilement") == "paused"
    inspected = []
    ctl.openInspect.connect(lambda pid, block: inspected.append((pid, block)))
    d = ctl.launch()
    assert d.phase() == "plan" and d.paused() and d.review.isVisibleTo(d) and not d.pause_note.isVisibleTo(d)
    assert not d.b_run.isVisibleTo(d) and d.b_cancel.text() == "Close" and d.summary.text() == "Waiting for your review"
    assert d.review_text.text() == "Paused for your review at Stacking: check it in Inspect, then Continue."
    # Open in Inspect: the block is selected for the window, which routes it (the graph page's openBlock)
    m.select(step="sources")
    d.b_inspect.click()
    assert inspected == [(1, "empilement")] and m.sel_step == "empilement" and m.path is plain and m.scope == [1]
    assert not d.isVisible()
    # Continue: it is marked, and there is nothing left to run
    d = ctl.launch()
    d.b_continue.click()
    assert m.review_state(plain, "empilement") == "reviewed" and not ctl.running
    assert ctl.dialog.summary.text() == "Nothing to run" and not ctl.dialog.review.isVisibleTo(ctl.dialog)


def test_open_in_inspect_uses_the_windows_graph_page(toy, monkeypatch):
    """The window's GraphPage already routes ``openBlock`` to Inspect: the recap selects the block there and emits it."""
    in_process(monkeypatch)
    m = toy.model
    toy.run_all()
    window = QWidget()                                         # the main window: it holds its GraphPage as .graph
    page = GraphPage(m)
    page.setParent(window)
    window.graph = page
    toy.ctl.window = window
    opened = []
    page.openBlock.connect(lambda: opened.append(m.sel_step))
    m.set_breakpoint([m.view(2)], "debruitage")
    d = toy.ctl.launch()
    assert d.paused() and toy.ctl.pause.pids == [2]
    m.select(index=0, step="sources")
    d.b_inspect.click()
    assert opened == ["debruitage"] and m.path is m.view(2) and m.scope == [2]
    toy.ctl.window = None
    window.deleteLater()


def test_run_to_here_stops_at_the_first_review_before_the_block(toy, monkeypatch):
    in_process(monkeypatch)
    m, ctl = toy.model, toy.ctl
    plain = m.view(1)
    m.set_breakpoint([plain], "calibration")
    m.select(step="empilement")
    d = ctl.run_to_here()
    assert d.windowTitle() == 'Run "Plain" to Stacking'
    assert [j.stage for j in d.plan.jobs] == ["inventory"]
    assert d.pause_note.text() == "Stops after Calibration for your review"
    d.start()
    finish(toy)
    assert d.paused()
    d.b_continue.click()                                       # the same way: on to Stacking, the block asked for
    finish(toy)
    assert [j.stage for j in toy.track.jobs] == ["stack"] and not ctl.dialog.paused()
    assert states_of(toy)[(1, "stack")] == "done" and states_of(toy)[(1, "color")] == "pending"
    # a review after the block asked for does not matter
    m.set_breakpoint([plain], "post")
    d3 = ctl.run_to_here()
    assert d3.pause is None and len(d3.plan.jobs) == 0
    d3.close()


def test_several_workflows_pausing_at_one_block_are_planned_together_and_the_earliest_block_goes_first(toy):
    m = toy.model
    plain, trained = m.view(1), m.view(2)
    m.set_breakpoint([plain, trained], "empilement")
    m.set_breakpoint([trained], "calibration")
    pause = m.pause_point()
    assert (pause.block, pause.pids, pause.later, pause.done) == ("calibration", [2], ["empilement"], False)
    plan, _ = m.plan_run()
    assert [j.stage for j in plan.jobs] == ["inventory"]
    m.set_breakpoint([trained], "calibration", False)
    pause = m.pause_point()
    assert (pause.block, pause.pids, pause.later) == ("empilement", [1, 2], [])
    plan, pause2 = m.plan_run()
    assert pause2.block == "empilement" and [j.stage for j in plan.jobs] == ["inventory", "stack"]    # shared: once
    assert plan.jobs[1].wf == [1, 2] and set(plan.targets) == {(1, "stack"), (2, "stack")}
    d = toy.ctl.launch()
    assert d.pause_note.text() == "Stops after Stacking for your review" and not d.pause_more.isVisibleTo(d)   # both
    d.close()
    m.set_breakpoint([trained], "empilement", False)
    d = toy.ctl.launch()
    assert d.pause_note.text() == "Stops after Stacking for your review" and d.pause_more.text() == "Workflow: Plain"
    d.close()
    assert [Graph(m, None).breakpoint(n)[:3] for n in Graph(m, None).nodes if n.key == "empilement"] == [("pending", 1, 2)]


def test_a_review_is_kept_with_the_project_and_the_breakpoints_too(env, monkeypatch):
    in_process(monkeypatch)
    p = toy_project(env["tmp"] / "toy")
    worker = ApiWorker()
    m = ProjectModel(p, worker, gpu=False, device="cpu")                  # one that saves
    assert worker.wait(120) and m.ready
    ctl = RunController(m, None, None)
    plain = m.view(1)
    m.set_breakpoint([plain], "calibration")
    m.set_breakpoint([m.view(2)], "post")
    m.flush()
    assert api.open_project(p.root).workflows[0].breakpoints == ["calibration"]           # saved with the definition
    assert api.open_project(p.root).workflows[1].breakpoints == ["output"]
    d = ctl.launch()
    d.start()
    assert pump(lambda: ctl.track is not None and ctl.track.finished and not ctl.running, 60)
    d.b_continue.click()
    assert m.is_reviewed(plain, "calibration")
    assert pump(lambda: ctl.track.finished and not ctl.running, 60)               # on to the Output of Trained
    assert m.review_state(m.view(2), "post") == "paused"
    state = json.loads((Path(p.root) / "state.json").read_text(encoding="utf-8"))
    assert state["reviewed"] == [m.review_key(plain, "calibration")] and "last_keys" in state   # beside the old keys
    ctl.close()
    m.close()
    # another session: the review is read again by the worker, the run goes through Calibration
    worker2 = ApiWorker()
    q = api.open_project(p.root)
    api.use_rules(q, SPEC)
    m2 = ProjectModel(q, worker2, gpu=False, device="cpu")
    assert worker2.wait(120) and m2.ready
    assert m2.is_reviewed(m2.view(1), "calibration") and m2.review_state(m2.view(1), "calibration") == "reviewed"
    assert m2.pause_point().block == "post" and m2.pause_point().pids == [2]
    m2.close()


def test_nothing_changes_for_a_project_without_breakpoints(toy):
    m = toy.model
    assert m.pause_point() is None
    plan, pause = m.plan_run()
    assert pause is None and [j.key for j in plan.jobs] == [j.key for j in m.plan_all().jobs]
    d = toy.ctl.launch()
    assert toy.ctl.pause is None and not d.pause_note.isVisibleTo(d) and not d.review.isVisibleTo(d) and not d.paused()
    assert d.b_run.isVisibleTo(d)
    d.close()
    assert Graph(m, None).breakpoint(Graph(m, None).root) is None
    assert all(Graph(m, None).breakpoint(n) is None for n in Graph(m, None).nodes)


def test_the_pause_in_french(toy, monkeypatch):
    in_process(monkeypatch)
    appmod.apply_language("fr")
    m, ctl = toy.model, toy.ctl
    plain = m.view(1)
    for k in ("calibration", "empilement", "post"):
        m.set_breakpoint([plain], k)
    d = ctl.launch()
    t = texts_of(d)
    assert "S'arrête après Calibration pour votre vérification" in t, t
    assert "Workflow : Plain · 2 autres pauses ensuite : Empilement, Sortie" in t, t
    assert untranslated(t) == []
    d.start()
    finish(toy)
    seen = texts_of(d)
    assert "En pause pour votre vérification à Calibration : vérifiez-le dans Inspection, puis cliquez sur Continuer." in seen
    assert "Continuer" in seen and "Ouvrir dans Inspection" in seen and "En pause pour vérification" in seen
    assert "Workflow : Plain · 2 autres pauses ensuite : Empilement, Sortie" in seen
    assert "En pause pour votre vérification à Calibration : vérifiez-le dans Inspection, puis cliquez sur Continuer." \
        in toy.console()
    assert m.history[-1][2]["text"].startswith("Exécution en pause pour votre vérification à Calibration : 1 étape en")
    assert untranslated(seen) == [], untranslated(seen)
    d.b_continue.click()
    finish(toy)
    d2 = ctl.dialog
    assert "Workflow : Plain · 1 autre pause ensuite : Sortie" in texts_of(d2)
    assert untranslated(texts_of(d2)) == []
    # the zero / one rule of French: "0 étape" and "1 étape" are singular, "2 étapes" is not
    one, many = "{n} more pause after it: {blocks}", "{n} more pauses after it: {blocks}"
    assert i18n.plural(0, one, many, blocks="x") == "0 autre pause ensuite : x"
    assert i18n.plural(1, one, many, blocks="x") == "1 autre pause ensuite : x"
    assert i18n.plural(2, one, many, blocks="x") == "2 autres pauses ensuite : x"


def test_every_plural_of_the_pause_has_its_french_in_both_forms():
    """The two English forms of every `plural()` of the modules that make the pause, and the texts of the pause
    (Continue, Open in Inspect...), are in the French catalogue, with two different forms."""
    import ast
    appmod.apply_language("fr")
    cat = i18n._cat
    root = HERE.parents[1] / "astro_gpu" / "client"
    missing, forms = [], 0
    for name in ("runs.py", "services/run.py", "graph.py", "blocks.py", "model.py"):
        for n in ast.walk(ast.parse((root / name).read_text(encoding="utf-8"))):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "plural" and len(n.args) >= 3:
                one, many = (a.value for a in n.args[1:3] if isinstance(a, ast.Constant))
                missing += [s for s in (one, many) if s not in cat]
                forms += 1
                assert one == many or cat.get(one) != cat.get(many), (one, many)
    assert not missing and forms >= 6, (missing, forms)
    for text in ("Continue", "Open in Inspect", "Waiting for your review", "Paused for your review"):
        assert text in cat and cat[text] != text


# ---------------------------------------------------------------------------------------------- a real run
def test_run_all_follows_a_detached_worker_without_blocking_the_gui(toy, monkeypatch):
    threads = []
    real = api.read_events
    monkeypatch.setattr(api, "read_events", lambda *a, **k: (threads.append(threading.current_thread()), real(*a, **k))[1])
    toy.beat.start()
    d = toy.ctl.launch()
    seen_running = []
    toy.ctl.updated.connect(lambda: seen_running.append(toy.ctl.status_text()))
    d.start()
    assert toy.ctl.running and toy.track.jobs and d.phase() == "run" and d.b_run.isHidden()
    assert pump(lambda: toy.track.finished and not toy.ctl.running, 120), "the run did not end"
    toy.beat.stop()
    assert threads and all(t is not threading.main_thread() for t in threads), "events were read in the GUI thread"
    assert toy.beat.max < toy.beat.limit, f"the GUI thread stalled for {toy.beat.max * 1000:.0f} ms"
    t = toy.track
    assert t.status == "done" and [j.state for j in t.jobs] == [DONE] * 7 and t.counts[DONE] == 7
    assert any(s.startswith("Starting the run") or s.startswith("Running") for s in seen_running)
    # the engine says the same: every stage of every workflow is done, and the worker is gone
    assert api.running(toy.p) is None and not (Path(toy.p.root) / "logs" / "run.lock").exists()
    assert all(toy.model.state(v, s) == "done" for v in toy.model.paths for s in toy.model.all_stages(v))
    assert all(v.computed for v in toy.model.paths)
    # the dialog became the recap: every row green, the sentence, Close
    assert d.phase() == "end" and all(r.mark.state == DONE for r in d.rows) and d.b_done.isVisible()
    recap = texts_of(d.recap)
    assert recap == ["7 stages computed in " + i18n.hm(t.elapsed()) + "."] or recap[0].startswith("7 stages computed in")
    assert d.summary.text().startswith("Done in ")
    # the history line and the console
    assert toy.model.history[-1][1:] == ("Run: {n} stages · {t}", {"n": 7, "t": i18n.hm(t.elapsed())})
    text = toy.console()
    assert "Starting the run: 7 stages" in text and "Run started: 7 stages" in text
    assert "Run finished: 7 stages in" in text and "Stacking · done in" in text and text.count("Stacking · started") == 1
    # nothing left to run
    assert len(toy.model.plan_all()) == 0
    d2 = toy.ctl.launch()
    assert d2.summary.text() == "Nothing to run" and not d2.b_run.isEnabled()
    d2.close()
    toy.ctl.begin(api.Plan())                                                  # an empty plan never starts
    assert not toy.ctl.running


def test_run_to_here_runs_only_the_blocks_it_needs(toy):
    toy.model.select(step="empilement")
    d = toy.ctl.run_to_here()
    d.start()
    assert pump(lambda: toy.track.finished and not toy.ctl.running, 120)
    assert [j.stage for j in toy.track.jobs] == ["inventory", "stack"] and toy.track.status == "done"
    classic = toy.model.view(1)
    assert [toy.model.state(classic, s) for s in toy.model.all_stages(classic)] == ["done", "done", "pending", "pending"]
    assert d.windowTitle() == 'Run "Plain" to Stacking'


def test_failed_job_skips_its_downstream_blocks_and_says_so_in_plain_words(env):
    w = make_world(env["tmp"], fail=True)
    try:
        d = w.run_all()
        t = w.track
        assert t.status == "failed"
        by = {(j.stage, tuple(j.wf)): j.state for j in t.jobs}
        assert by[("denoise", (2,))] == FAILED and by[("color", (2,))] == SKIPPED and by[("render_prep", (2,))] == SKIPPED
        assert by[("color", (1,))] == DONE and by[("render_prep", (1,))] == DONE     # the other workflow still completes
        recap = texts_of(d.recap)
        assert "Denoise failed: b failed on purpose" in recap, recap
        assert 'Not run because Denoise failed: Color, Output "Trained".' in recap
        assert "The blocks that did not depend on it were computed." in recap
        assert [r.mark.state for r in d.rows] == [DONE, DONE, DONE, DONE, FAILED, SKIPPED, SKIPPED]
        assert d.summary.text() == "Failed" and d.b_log.isVisible()
        # the engine's own states, and the graph's stripes, say it too
        v = w.model.view(2)
        assert [w.model.state(v, s) for s in ("denoise", "color", "render_prep")] == ["failed", "skipped", "skipped"]
        g = Graph(w.model, None)
        assert [g.state(g.node_of(v, k)) for k in ("debruitage", "couleur", "post")] == ["failed", "skipped", "skipped"]
        assert g.state(g.node_of(w.model.view(1), "post")) == "done"
        text = w.console()
        assert "Denoise · failed: b failed on purpose" in text and "Color · not run, Denoise failed" in text
        assert "Run failed: 1 failed, 2 not run" in text
        assert w.model.history[-1][1] == "Run failed: {failed} failed, {skipped} not run"
        logged = []
        w.ctl.logRequested.connect(lambda: logged.append(1))
        d.b_log.click()
        assert logged
    finally:
        w.close()


def _slow_denoise_running(w, frac: float = 0.1):
    jb = lambda: next(j for j in w.track.jobs if j.stage == "denoise")                      # noqa: E731
    assert pump(lambda: w.track is not None and w.track.jobs and jb().state == RUNNING and jb().frac > frac, 90)
    return jb


def _checkpoint(w) -> int:
    tmp = next((Path(w.p.root) / "store" / "denoise").glob("*.tmp"))
    return json.loads((tmp / "_ckpt" / "i.json").read_text())["i"]


def test_stop_keeps_the_results_and_the_next_run_resumes_the_checkpoint(env):
    w = make_world(env["tmp"], steps=120, delay=0.03)
    root = Path(w.p.root)
    try:
        # the blocks the two workflows share first: they are done, and must stay exactly as they are
        w.model.select(step="empilement")
        d = w.ctl.run_to_here()
        d.start()
        assert pump(lambda: w.track.finished and not w.ctl.running, 120)
        d.close()
        before = done_keys(root)
        snap = snapshot(root / "store" / "inventory") + snapshot(root / "store" / "stack")
        assert before["inventory"] and before["stack"]
        # everything now: the denoise is slow; Stop it a little after it started
        d = w.ctl.launch()
        d.start()
        jb = _slow_denoise_running(w)
        assert d.b_stop.isEnabled()
        d.b_stop.click()
        assert w.ctl.phase == "stopping" and d.b_stop.text() == "Stopping…" and not d.b_stop.isEnabled()
        assert pump(lambda: w.track.finished and not w.ctl.running, 60)
        t = w.track
        assert t.status == "interrupted" and not t.lost and jb().state == INTERRUPTED
        assert snapshot(root / "store" / "inventory") + snapshot(root / "store" / "stack") == snap
        after = done_keys(root)
        assert all(set(k) <= set(after[s]) for s, k in before.items())                  # nothing done was lost
        assert w.model.state(w.model.view(2), "denoise") == "interrupted"
        stopped_at = _checkpoint(w)
        assert 0 < stopped_at < 120
        assert d.phase() == "end" and d.summary.text() == "Run stopped"
        assert any(x.startswith("Stopped. What was computed is kept") for x in texts_of(d.recap))
        assert "Stopped during Denoise." in texts_of(d.recap)
        text = w.console()
        assert "Run stopped: " in text and "Stop asked" in text and "Denoise · stopped" in text
        assert w.model.history[-1][1] == "Run stopped: {n} stages done"
        # the next run plans the denoise again and goes on from the checkpoint, not from the start
        d.close()
        d = w.ctl.launch()
        assert [j.stage for j in d.plan.jobs if j.stage == "denoise"] == ["denoise"]
        d.start()
        assert pump(lambda: w.track.finished and not w.ctl.running, 120) and w.track.status == "done"
        out = json.loads((Store(root).dir("denoise", api.stage_key(w.p, 2, "denoise")) / "b.json").read_text())
        assert out["resumed_from"] >= stopped_at and out["steps"] == 120
        assert not list((root / "store" / "denoise").glob("*.tmp"))
    finally:
        w.close()


def test_killing_the_worker_marks_the_run_interrupted_and_the_next_run_resumes(env):
    w = make_world(env["tmp"], steps=120, delay=0.03)
    root = Path(w.p.root)
    try:
        d = w.ctl.launch()
        d.start()
        jb = _slow_denoise_running(w)
        pid = w.track.pid
        assert pid and psutil.pid_exists(pid) and api.running(w.p) is not None
        psutil.Process(pid).kill()
        assert pump(lambda: w.track.finished and not w.ctl.running, 30), "the lost worker was not noticed"
        t = w.track
        assert t.lost and t.status == "interrupted" and jb().state == INTERRUPTED
        assert api.running(w.p) is None                      # a stale lock is not a run
        assert (root / "logs" / "run.lock").exists()
        assert w.model.state(w.model.view(2), "denoise") == "interrupted"
        assert any("without warning" in x for x in texts_of(d.recap)) and "stopped without warning" in w.console()
        killed_at = _checkpoint(w)
        d.close()
        d = w.ctl.launch()                                   # the stale lock is replaced; the denoise resumes
        d.start()
        assert pump(lambda: w.track.finished and not w.ctl.running, 120) and w.track.status == "done"
        out = json.loads((Store(root).dir("denoise", api.stage_key(w.p, 2, "denoise")) / "b.json").read_text())
        assert out["resumed_from"] >= killed_at > 0
        assert not (root / "logs" / "run.lock").exists()
    finally:
        w.close()


def test_closing_and_reopening_the_project_reattaches_to_the_run(env):
    w = make_world(env["tmp"], steps=150, delay=0.04)
    root = Path(w.p.root)
    try:
        d = w.ctl.launch()
        d.start()
        _slow_denoise_running(w, 0.05)
        run_id, n_jobs = w.track.run_id, len(w.track.jobs)
        w.ctl.close()                                        # the project is closed: the run goes on without us
        w.model.close()
        assert api.running(w.p) is not None
    finally:
        w.beat.stop()
        w.worker.stop()
    p2 = api.open_project(root)
    api.use_rules(p2, SPEC)
    w2 = World(p2, history=[])
    try:
        assert not w2.ctl.running
        w2.ctl.attach()
        assert pump(lambda: w2.ctl.running, 30), "the run in progress was not found"
        t = w2.ctl.track
        assert t.run_id == run_id and len(t.jobs) == n_jobs and t.running
        assert any(j.state == DONE for j in t.jobs) and w2.ctl.status_text().startswith("Running")
        assert "Run started: " in w2.console() and "Stacking · done in" in w2.console()
        sw = w2.ctl.status_widget()
        assert isinstance(sw, RunStatus) and not sw.isHidden() and sw.b_stop.isEnabled()
        d2 = w2.ctl.show_dialog()                            # F5 while it runs: the progress, not a second plan
        assert d2 is not None and d2.phase() == "run" and w2.ctl.launch() is d2 and d2.windowTitle() == "Run in progress"
        assert not w2.ctl.begin(w2.ctl.plan)
        got = []
        w2.ctl.finished.connect(lambda status, text: got.append((status, text)))
        assert pump(lambda: not w2.ctl.running and w2.ctl.track.finished, 120)
        assert got and got[0][0] == "done" and got[0][1].startswith("Run finished: ")
        assert sw.isHidden() and api.running(p2) is None
        assert all(w2.model.state(v, s) == "done" for v in w2.model.paths for s in w2.model.all_stages(v))
        assert w2.model.history[-1][1] == "Run: {n} stages · {t}"
    finally:
        w2.close()


def test_a_run_that_is_already_going_is_followed_instead_of_started_twice(env):
    # the other run lasts long enough (300 steps of 50 ms) to be still going when this window asks to start, even on a
    # loaded machine (the detached worker can take seconds to start); with a short one the window's own start could
    # come after its end and rightly start a second run
    w = make_world(env["tmp"], steps=300, delay=0.05)
    try:
        other = api.open_project(w.p.root)                  # a run started by another window, or the command line
        api.use_rules(other, SPEC)
        info = api.start_run(other, api.plan_all(other), "cpu")
        assert pump(lambda: api.running(w.p) is not None, 60)
        d = w.ctl.launch()
        d.start()                                           # refused by the engine: this window follows that run
        assert pump(lambda: w.ctl.track is not None and w.ctl.track.run_id == info.run_id, 60)
        assert w.ctl.track.jobs and w.ctl.dialog is None            # the plan dialog is gone: the run has its own
        assert pump(lambda: w.track.finished and not w.ctl.running, 180)
        assert w.track.status == "done", (w.console(), [(j.stage, j.state, j.error) for j in w.track.jobs])
        assert w.track.run_id == info.run_id                        # one run only: the other one, followed to its end
        runs = [e for e in api.read_events(w.p, 0)[0] if e.get("type") == "run_start"]
        assert len(runs) == 1, runs
        assert "Run finished" in w.console()
    finally:
        w.close()


def test_a_start_that_fails_is_said_and_nothing_stays_running(env, monkeypatch):
    w = make_world(env["tmp"])
    try:
        def refuse(p, plan, device=None):
            raise OSError("the disk is full")
        monkeypatch.setattr(api, "start_run", refuse)
        d = w.ctl.launch()
        d.start()
        assert pump(lambda: w.track.finished and not w.ctl.running, 30)
        assert w.track.status == "failed" and "The run could not start: the disk is full" in w.console()
        assert d.phase() == "end" and d.summary.text() == "Failed"
    finally:
        w.close()


# ---------------------------------------------------------------------------------------------- the graph and the status bar
def test_the_graph_shows_the_running_blocks_and_their_progress(env):
    w = make_world(env["tmp"], steps=100, delay=0.05)
    root = QWidget()
    root.setObjectName("root")
    try:
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        page = GraphPage(w.model)
        v.addWidget(page)
        root.resize(1400, 760)
        root.show()
        w.ctl.watch_graph(page.graph)
        status = w.ctl.status_widget()
        v.addWidget(status)
        w.worker.wait(30)
        for _i in range(10):
            QApplication.processEvents()
        ov = page.graph.findChildren(RunOverlay)[0]
        assert ov.fraction_of(page.graph.graph.node_of(w.model.view(2), "debruitage")) is None      # nothing runs yet
        assert status.isHidden()
        d = w.ctl.launch()
        d.start()
        jb = lambda: next(j for j in w.track.jobs if j.stage == "denoise")                  # noqa: E731
        assert pump(lambda: w.track is not None and w.track.jobs and jb().state == RUNNING and jb().frac > 0.3, 90)
        items = []

        def running_shown():                                  # the model reads the states at most every 1.5 s
            items[:] = [it for it in page.graph.scene().items() if isinstance(it, BlockItem)]
            return any(it.state == "running" for it in items)
        assert pump(running_shown, 10)
        running = [it for it in items if it.state == "running"]
        assert [it.node.key for it in running] == ["debruitage"], [(it.node.key, it.state) for it in items]
        f = ov.fraction_of(running[0].node)
        assert f is not None and 0.2 < f <= 1.0
        assert all(ov.fraction_of(it.node) is None for it in items if it.state != "running")
        shot = ov.parent().grab()                             # the overlay paints without error
        assert not shot.isNull()
        assert not status.isHidden() and status.bar.value() > 0 and status.text.text().startswith("Running")
        assert "Details" in texts_of(status) and "Stop" in texts_of(status)
        status.b_stop.click()
        assert pump(lambda: w.track.finished and not w.ctl.running, 60)
        for _i in range(20):
            QApplication.processEvents()
        assert [it.state for it in page.graph.scene().items() if isinstance(it, BlockItem)
                and it.node.key == "debruitage"] == ["interrupted"]
        assert ov.fraction_of(running[0].node) is None and status.isHidden()
    finally:
        root.close()
        w.close()


def test_the_console_survives_a_rebuilt_panel_and_a_change_of_language(toy):
    toy.run_all()
    english = toy.console()
    assert toy.track.status == "done", english                  # the whole console when a stage failed
    assert "Run finished: 7 stages in" in english
    # the Log panel is built again with the project screen (a change of language): the console is filled again
    appmod.apply_language("fr")
    toy.logp = LogPanel(toy.model)
    assert toy.console() == ""
    toy.ctl.retranslate()
    french = toy.console()
    assert "Exécution terminée : 7 étapes en" in french and "Empilement · terminé en" in french
    assert "Run finished" not in french and "Starting" not in french
    assert untranslated([x.split("  ", 1)[1] for x in french.splitlines()]) == []
    # the export of the log writes the console after the history
    out = Path(toy.p.root).parent / "log.txt"
    toy.logp.write(out)
    assert "Exécution terminée" in out.read_text(encoding="utf-8")


def test_a_dialog_in_every_phase_has_no_english_left_in_french(toy):
    """Plan, run, stopping, done, failed, stopped, lost: every text shown has its French."""
    appmod.apply_language("fr")
    plan = toy.model.plan_all()
    plan.jobs[1].disk_gb = 7.6
    toy.model.plan_all = lambda: plan
    d = toy.ctl.launch()
    seen = [texts_of(d)]
    jobs = plan.jobs
    ctl = toy.ctl
    ctl.track = RunTrack.of_plan(plan, "r1")
    ctl.track.plan = plan
    ctl.track.apply([ev("run_start", time.time() - 600, device="cpu", pid=1),
                     ev("stage_start", time.time() - 590, stage=jobs[0].stage, key=jobs[0].key),
                     ev("stage_end", time.time() - 580, stage=jobs[0].stage, key=jobs[0].key, status="done", duration=10.0),
                     ev("stage_start", time.time() - 570, stage=jobs[1].stage, key=jobs[1].key),
                     ev("progress", time.time() - 560, stage=jobs[1].stage, key=jobs[1].key, frac=0.4,
                        msg="training step 120/6000, loss 0.0123", eta=1200.0)])
    ctl.phase = "running"
    ctl.updated.emit()
    assert d.phase() == "run"
    seen.append(texts_of(d))
    assert any(x.startswith("En cours · ") and "reste ≈" in x for x in seen[-1]), seen[-1]
    assert any("entraînement, étape 120/6000, perte 0,0123" in x for x in seen[-1]), seen[-1]
    ctl.phase = "stopping"
    ctl.updated.emit()
    seen.append(texts_of(d))
    assert "Arrêt en cours…" in seen[-1]
    sw = ctl.status_widget()
    assert "Détails" in texts_of(sw) and "Arrêter" in texts_of(sw) and sw.text.text().startswith("Arrêt")
    # the ends
    for how in ("done", "failed", "interrupted", "lost"):
        t = RunTrack.of_plan(plan, "r1")
        t.plan = plan
        names = [j.stage for j in jobs]
        base = [ev("run_start", 100.0, pid=1), ev("stage_start", 101.0, stage=names[0], key=jobs[0].key),
                ev("stage_end", 102.0, stage=names[0], key=jobs[0].key, status="done", duration=1.0)]
        if how == "done":
            for j in jobs[1:]:
                base += [ev("stage_start", 103.0, stage=j.stage, key=j.key),
                         ev("stage_end", 104.0, stage=j.stage, key=j.key, status="done", duration=1.0)]
            base.append(ev("run_end", 110.0, status="done", done=[j.key for j in jobs], failed=[], skipped=[],
                           interrupted=[], not_started=[]))
        elif how == "failed":
            base += [ev("stage_start", 103.0, stage=jobs[1].stage, key=jobs[1].key),
                     ev("stage_end", 104.0, stage=jobs[1].stage, key=jobs[1].key, status="failed", error="out of memory"),
                     ev("run_end", 110.0, status="failed", done=[jobs[0].key], failed=[jobs[1].key],
                        skipped=[j.key for j in jobs[2:]], interrupted=[], not_started=[])]
            base[3:3] = []
            for j in jobs[2:]:
                base.insert(-1, ev("stage_end", 104.5, stage=j.stage, key=j.key, status="skipped",
                                   upstream=[jobs[1].stage]))
        elif how == "interrupted":
            base += [ev("stage_start", 103.0, stage=jobs[1].stage, key=jobs[1].key),
                     ev("stage_end", 104.0, stage=jobs[1].stage, key=jobs[1].key, status="interrupted"),
                     ev("run_end", 110.0, status="interrupted", done=[jobs[0].key], failed=[], skipped=[],
                        interrupted=[jobs[1].key], not_started=[j.key for j in jobs[2:]])]
        t.apply(base)
        if how == "lost":
            t.end_lost()
        ctl.track = t
        ctl.phase = "idle"
        ctl.updated.emit()
        assert d.phase() == "end"
        seen.append(texts_of(d))
    flat = [x for s in seen for x in s]
    assert untranslated(flat) == [], untranslated(flat)
    assert not [x for x in flat if re.search(r"\b(failed|stopped|computed|Not run|running|started)\b", x)
                and x not in ("out of memory",)], [x for x in flat if re.search(r"\b(failed|stopped|computed|Not run)\b", x)]
    assert any(x.endswith(f"disque : {size_text(plan.disk_gb)} Go") and "," in size_text(plan.disk_gb)
               for x in seen[0]), [x for x in seen[0] if "Go" in x or "Dur" in x]


def test_the_run_blocks_entry_opens_the_plan_of_the_block(toy):
    items = toy.model.pending_upto(toy.model.view(1), ["stack"])
    d = run_blocks(toy.model, None, items, "Run to Stacking", None, controller=toy.ctl)
    assert d is not None and [r.row.title for r in d.rows] == ["Calibration", "Stacking"]
    d.close()
    assert run_blocks(toy.model, None, [], "x", None, controller=toy.ctl) is None
    rows = build_rows(toy.model, toy.model.plan_all().jobs)
    assert [r.title for r in rows][0] == "Calibration"


# ---------------------------------------------------------------------------------------------- the real stages
real = needs_test_project
#: the store folders a Classic stack workflow reads (its own chain, and the survey its Astrometry block shows): what a
#: copy that runs it needs
CLASSIC_STORE = ("inventory", "calibrate", "register", "stack", "astrometry", "surveys", "catalogs", "color",
                 "background_poly", "render_simple")


def _classic_copy(tmp: Path) -> Path:
    """A copy of the test project holding the results of the Classic stack chain (not the scratch, not the stages of
    Full alone): a run in it writes into the copy only."""
    dst = tmp / "heart"
    dst.mkdir()
    for f in F_COPY.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    for name in ("inputs",):
        if (F_COPY / name).is_dir():
            shutil.copytree(F_COPY / name, dst / name)
    for sd in (F_COPY / "store").iterdir():
        if sd.is_dir() and (sd.name in CLASSIC_STORE or sd.name.startswith("_")):
            shutil.copytree(sd, dst / "store" / sd.name)
        elif sd.is_file():
            (dst / "store").mkdir(exist_ok=True)
            shutil.copy2(sd, dst / "store" / sd.name)
    return dst


def _grab(w: QWidget, name: str):
    SHOTS.mkdir(parents=True, exist_ok=True)
    QApplication.processEvents()
    w.grab().save(str(SHOTS / f"{name}.png"))


def _real_world(root: Path) -> World:
    """The project in ``root`` with the four workflows, on the CPU: the model never saves (the plan carries the
    definition), the runs write the new results into ``root``."""
    from astro_gpu.core import workflow as wfl
    p = api.open_project(root)
    for name, template, params in (("Classic stack", "Classic stack", None),
                                   ("Classic + denoise", "Classic + denoise", {"denoise": {"iterations": 6000}}),
                                   ("Full + neural stars", "Full + neural stars", {"denoise": {"iterations": 6000}})):
        assert template in wfl.TEMPLATES
        api.add_workflow(p, name, template, params)
    p.devices["main"] = "cpu"
    return World(p)


def _shell(w: World):
    """A window's worth of the client: the graph page, and a status bar holding the run's progress."""
    root = QWidget()
    root.setObjectName("root")
    v = QVBoxLayout(root)
    v.setContentsMargins(0, 0, 0, 0)
    page = GraphPage(w.model)
    v.addWidget(page)
    bar = QFrame()
    bar.setStyleSheet("QFrame { background: #1c1f26; border-top: 1px solid #2a2f38; }")
    bh = QHBoxLayout(bar)
    bh.setContentsMargins(8, 2, 8, 2)
    bh.addWidget(QLabel(i18n._("{n} workflow(s)").format(n=len(w.model.paths))), 1)
    sw = w.ctl.status_widget()
    sw.rebuild()
    bh.addWidget(sw)
    v.addWidget(bar)
    root.resize(1680, 934)
    root.show()
    w.ctl.watch_graph(page.graph)
    for _i in range(10):
        w.worker.wait(30)
        QApplication.processEvents()
    return root, page


@real
@pytest.mark.slow
def test_a_real_render_simple_job_runs_with_the_heartbeat_under_200_ms_and_the_screenshots(env, monkeypatch):
    """Run the Classic stack of a copy of the test project for real on the CPU: background_poly and render_simple.
    The stripes turn green and the layer appears. Takes the screenshots in English and in French."""
    w = _real_world(_classic_copy(env["tmp"]))
    root = None
    try:
        classic = w.model.view(next(v.pid for v in w.model.paths if v.name == "Classic stack"))
        for n, lang in enumerate(("en", "fr")):
            appmod.apply_language(lang)
            w.ctl.retranslate()
            w.logp.deleteLater()
            w.logp = LogPanel(w.model)
            w.logp.setFixedHeight(200)
            w.logp.resize(1400, 200)
            if root is not None:
                root.close()
            w.model.select(index=w.model.index_of(classic.pid), step="post")
            root, page = _shell(w)
            # the plan of Run all after a change of the stacking for every workflow (nothing is saved, nothing runs)
            assert w.model.set_trunk_param("stack", "kappa", 4.0)
            for _i in range(10):
                w.worker.wait(30)
                QApplication.processEvents()
            plan = w.model.plan_all()
            assert plan and len([j for j in plan.jobs if j.stage == "stack"]) == 1          # shared: planned once
            assert any(len(j.wf) == 4 for j in plan.jobs if j.stage == "stack")
            d = w.ctl.launch()
            assert pump(lambda: w.ctl.free_gb() is not None, 30)
            d.refresh()
            _grab(d, f"run_plan_{lang}")
            assert any(len(r.row.who) == 4 for r in d.rows)
            assert all(r.dur.text().startswith("≈") or r.dur.text() == "≤ 1 min" for r in d.rows)
            assert untranslated(texts_of(d)) == [], untranslated(texts_of(d))
            d.close()
            w.model.undo()
            for _i in range(10):
                w.worker.wait(30)
                QApplication.processEvents()
            # a late, cheap setting of the Classic workflow: Run to its Output only runs two short jobs
            w.model.set_param(classic, "background_poly", "object_clip", 2.5 + (int(time.time()) % 100000 + n) / 1e6)
            for _i in range(10):
                w.worker.wait(30)
                QApplication.processEvents()
            d = w.ctl.run_to_here()
            jobs = [(j.stage, j.measured) for j in d.plan.jobs]
            assert [s for s, _m in jobs] == ["background_poly", "render_simple"], jobs
            w.beat.start()
            d.start()
            shot = False
            end = time.time() + 600
            while time.time() < end and not (w.track is not None and w.track.finished and not w.ctl.running):
                QApplication.processEvents()
                time.sleep(0.005)
                cur = w.track.current() if w.track is not None else None
                if cur is not None and cur.frac > 0.2 and not shot:
                    for _i in range(15):
                        QApplication.processEvents()
                        time.sleep(0.02)
                    w.beat.skip()
                    _grab(root, f"run_graph_{lang}")
                    _grab(w.logp, f"run_console_{lang}")
                    w.beat.skip()
                    shot = True
            w.beat.stop()
            assert w.track.status == "done", w.console()
            print(f"\nthe GUI thread's longest pause during the real run ({lang}): {w.beat.max * 1000:.0f} ms "
                  f"(machine load {w.beat.load:.0f} %, limit {w.beat.limit * 1000:.0f} ms)")
            assert w.beat.max < w.beat.limit, f"the GUI thread stalled for {w.beat.max * 1000:.0f} ms"
            for _i in range(30):
                QApplication.processEvents()
            cls = w.model.view(classic.pid)
            assert cls.computed and cls.rendered, "the layer does not appear"
            assert all(w.model.state(cls, s) == "done" for s in w.model.all_stages(cls))
            items = [it for it in page.graph.scene().items() if isinstance(it, BlockItem) and it.node.paths
                     and it.node.key != "final" and cls in it.node.paths]
            assert items and all(it.state == "done" for it in items), [(it.node.key, it.state) for it in items]
            _grab(d, f"run_recap_{lang}")
            _grab(w.logp, f"run_console_{lang}_end")
            assert untranslated(texts_of(d)) == []
            assert untranslated([x.split("  ", 1)[1] for x in w.console().splitlines()]) == []
    finally:
        if root is not None:
            root.close()
        w.close()
        appmod.apply_language("en")


@real
@pytest.mark.slow
def test_screenshots_of_the_failed_run_and_the_stopped_run_on_the_real_plan(env):
    """A failed run and a stopped run on the real four-workflow plan, from synthetic events (no stage is run):
    the recaps in English and French."""
    w = _real_world(F_COPY)
    try:
        plan = w.model.plan_all()
        jobs = plan.jobs
        assert len(jobs) >= 3
        now = time.time() - 3000
        stars = [j for j in jobs if j.stage == "stars"] or jobs[1:2]
        bad = stars[0]
        up = {bad.key} | {j.key for j in jobs if bad.key in j.deps.values()}
        for kind in ("failed", "stopped"):
            ctl = w.ctl
            ctl.plan, ctl.title = plan, lambda: i18n._("Run project")
            t = RunTrack.of_plan(plan, "r1")
            t.plan = plan
            evs = [ev("run_start", now, pid=1)]
            tt = now
            done_keys_, failed, skipped, interrupted, notstarted = [], [], [], [], []
            for j in jobs:
                tt += 60
                if j.key == bad.key:
                    evs.append(ev("stage_start", tt, stage=j.stage, key=j.key))
                    if kind == "failed":
                        evs.append(ev("log", tt + 30, stage=j.stage, key=j.key, level="error",
                                      msg="CUDA out of memory while measuring the stars"))
                        evs.append(ev("stage_end", tt + 30, stage=j.stage, key=j.key, status="failed",
                                      error="CUDA out of memory while measuring the stars"))
                        failed.append(j.key)
                    else:
                        evs.append(ev("progress", tt + 30, stage=j.stage, key=j.key, frac=0.4, msg="far halo of the stars"))
                        evs.append(ev("stage_end", tt + 30, stage=j.stage, key=j.key, status="interrupted"))
                        interrupted.append(j.key)
                    continue
                if bad.key in j.deps.values() or (set(j.deps.values()) & up):
                    up.add(j.key)
                    if kind == "failed":
                        evs.append(ev("stage_end", tt, stage=j.stage, key=j.key, status="skipped", reason="upstream failed",
                                      upstream=[bad.stage]))
                        skipped.append(j.key)
                    else:
                        notstarted.append(j.key)
                    continue
                if kind == "stopped" and bad.key in done_keys_:
                    notstarted.append(j.key)
                    continue
                evs += [ev("stage_start", tt, stage=j.stage, key=j.key),
                        ev("stage_end", tt + 50, stage=j.stage, key=j.key, status="done", duration=50.0)]
                done_keys_.append(j.key)
            evs.append(ev("run_end", tt + 100, status="failed" if kind == "failed" else "interrupted", done=done_keys_,
                          failed=failed, skipped=skipped, interrupted=interrupted, not_started=notstarted))
            t.apply(evs)
            ctl.track, ctl.phase = t, "idle"
            for lang in ("en", "fr"):
                appmod.apply_language(lang)
                d = ctl._make_dialog()
                d.show()
                QApplication.processEvents()
                _grab(d, f"run_recap_{kind}_{lang}")
                if lang == "fr":
                    assert untranslated(texts_of(d)) == [], untranslated(texts_of(d))
                d.close()
                ctl.dialog = None
            appmod.apply_language("en")
    finally:
        w.close()

B3 = Path(os.environ.get("TEMP", ".")) / "b3_shots"


@real
def test_real_pause_after_calibration_runs_for_real_continues_and_the_shots_en_fr(env, monkeypatch):
    """The real client (the main window, offscreen) on the definition of the test project, in a scratch folder
    (the copy itself is only read: its results are other tests'), on the CPU: Pause here for review on Calibration and
    Registration; Run all stops after Calibration (the real stages run in a detached worker); the recap asks for the
    review; Open in Inspect shows the block; Continue runs Registration and stops again. The breakpoints and the reviews
    are saved with the project and read again. Screenshots in English and in French under %TEMP%/b3_shots."""
    from PySide6.QtWidgets import QMenu

    from astro_gpu.client import graph as graphmod
    from astro_gpu.client.services.worker import Gpu, Machine
    from astro_gpu.core import project2
    B3.mkdir(parents=True, exist_ok=True)
    ws = env["tmp"] / "ws"
    ws.mkdir()
    prefs.set("workspace", ws.as_posix())                                    # never the user's own workspace
    prefs.set("cache_dir", (env["tmp"] / "scratch").as_posix())
    prefs.set("device", "cpu")
    monkeypatch.setattr(appmod, "machine_info", lambda: Machine([Gpu("cuda:0", "NVIDIA RTX 4060 Ti", 16.0)], 16, 64.0))
    proj = env["tmp"] / "heart"                                             # a scratch project: the definition of the copy
    proj.mkdir()
    shutil.copy(F_COPY / "project.astroproj", proj / "project.astroproj")
    before = _fingerprint(F_COPY / "project.astroproj")
    w = appmod.Main()
    w.resize(1600, 950)
    w.show()
    menus = []

    class Menu(QMenu):
        def exec(self, *a, **k):
            menus.append(self)
    monkeypatch.setattr(graphmod, "QMenu", Menu)

    def settle(n=15):
        for _i in range(n):
            w.worker.wait(30)
            QApplication.processEvents()
            time.sleep(0.01)

    def shot(widget, name):
        QApplication.processEvents()
        assert widget.grab().save(str(B3 / f"{name}_{i18n.language()}.png"))

    def both(name, widget):
        """The same state in English and in French (the project screen is built again for the language)."""
        for lang in ("en", "fr"):
            w.change_language(lang)
            settle()
            shot(widget() if callable(widget) else widget, name)
        w.change_language("en")
        settle()

    def graph():
        return w.graph.graph.graph

    def block(pid, key):
        return next(it for it in w.graph.graph.scene().items() if isinstance(it, BlockItem)
                    and it.node is graph().node_of(w.s.view(pid), key))
    try:
        assert w.worker.wait(60)
        assert w.open_project(proj)
        s = w.s
        w.set_device("cpu")
        s.add_path("Classic stack", {"debruitage": False, "etoiles": "garder", "appris": False, "fond": "simple"})
        w.set_mode(0)
        settle()
        assert s.ready and len(s.paths) == 2 and not w.run.running
        # 1. the context menu of a block, and the dot once chosen
        for lang in ("en", "fr"):
            w.change_language(lang)
            settle()
            w.graph.graph.block_menu(graph().node_of(s.view(1), "calibration"), None)
            menus[-1].show()
            QApplication.processEvents()
            names = [a.text() for a in menus[-1].actions()]
            assert i18n._("Pause here for review") in names
            shot(menus[-1], "1_block_menu")
            menus[-1].hide()
        w.change_language("en")
        settle()
        for key in ("calibration", "alignement"):
            w.graph.graph.block_menu(graph().node_of(s.view(1), key), None)
            next(a for a in menus[-1].actions() if a.text() == "Pause here for review").trigger()
            settle()
        assert [x.breakpoints for x in s.p.workflows] == [["calibration", "registration"]] * 2
        assert block(1, "calibration").bp == ("pending", 2, 2) and block(1, "alignement").bp == ("pending", 2, 2)
        both("2_graph_breakpoints", lambda: w)
        # 2. Run all: the plan stops after Calibration
        w.launch()
        d = w.run.dialog
        assert d.pause_note.text() == "Stops after Calibration for your review" and d.pause_more.text().startswith(
            "1 more pause after it: Registration")
        assert [j.stage for j in d.plan.jobs] == ["inventory", "calibrate"] and not d.review.isVisibleTo(d)
        both("3_plan_dialog", lambda: w.run.dialog)
        # 3. Run: the real calibration, in a detached worker
        d = w.run.dialog
        d.b_run.click()
        assert pump(lambda: w.run.track is not None and w.run.track.finished and not w.run.running, 400), \
            "the run did not end"
        settle(30)
        t = w.run.track
        assert t.status == "done" and [j.stage for j in t.jobs] == ["inventory", "calibrate"], t.error
        d = w.run.dialog
        assert d.paused() and d.review.isVisibleTo(d) and d.b_continue.isVisibleTo(d) and d.b_inspect.isVisibleTo(d)
        assert d.review_text.text() == "Paused for your review at Calibration: check it in Inspect, then Continue."
        assert [s.state(s.view(1), x) for x in ("inventory", "calibrate", "register", "stack")] == \
            ["done", "done", "pending", "pending"]
        assert block(1, "calibration").bp == ("paused", 2, 2) and block(1, "alignement").bp == ("pending", 2, 2)
        both("4_paused_recap", lambda: w.run.dialog)
        both("5_graph_paused", lambda: w)
        for lang in ("en", "fr"):                          # the menu of the paused block: checked, with Mark as reviewed
            w.change_language(lang)
            settle()
            w.graph.graph.block_menu(graph().node_of(s.view(1), "calibration"), None)
            menus[-1].show()
            QApplication.processEvents()
            checked = [a for a in menus[-1].actions() if a.isChecked()]
            assert [a.text() for a in checked] == [i18n._("Pause here for review")]
            assert i18n._("Mark as reviewed") in [a.text() for a in menus[-1].actions()]
            shot(menus[-1], "5b_block_menu_paused")
            menus[-1].hide()
        # 4. Open in Inspect: the window shows the block in Inspect
        w.change_language("en")
        settle()
        d = w.run.dialog
        d.show()
        d.b_inspect.click()
        settle()
        assert w.mode == 1 and s.sel_step == "calibration" and not d.isVisible()
        shot(w, "6_inspect_calibration")
        w.set_mode(0)
        settle()
        # 5. Continue: the review is marked, Registration runs for real, and the run stops again at its breakpoint
        w.launch()
        d = w.run.dialog
        assert d.paused() and d.phase() == "plan" and d.summary.text() == "Waiting for your review"
        d.b_continue.click()
        assert w.run.running or w.run.track.finished
        assert pump(lambda: w.run.track.finished and not w.run.running, 400), "the second run did not end"
        settle(30)
        t = w.run.track
        assert t.status == "done" and [j.stage for j in t.jobs] == ["register"], t.error
        d = w.run.dialog
        assert d.paused() and d.review_text.text() == \
            "Paused for your review at Registration: check it in Inspect, then Continue."
        assert [s.review_state(s.view(1), k) for k in ("calibration", "alignement")] == ["reviewed", "paused"]
        assert [s.state(s.view(1), x) for x in ("calibrate", "register", "stack")] == ["done", "done", "pending"]
        both("7_second_pause_recap", lambda: w.run.dialog)
        both("8_graph_reviewed_and_paused", lambda: w)
        assert block(1, "calibration").bp == ("reviewed", 2, 2) and block(1, "alignement").bp == ("paused", 2, 2)
        # 6. saved with the project, and read again: the definition has the breakpoints, state.json the review
        s.flush()
        assert [x.breakpoints for x in project2.load(proj).workflows] == [["calibration", "registration"]] * 2
        state = json.loads((proj / "state.json").read_text(encoding="utf-8"))
        assert state["reviewed"] == [s.review_key(s.view(1), "calibration")] or len(state["reviewed"]) == 2
        assert s.is_reviewed(s.view(1), "calibration") and not s.is_reviewed(s.view(1), "alignement")
    finally:
        w.close()
        w.deleteLater()
        QApplication.processEvents()
    assert _fingerprint(F_COPY / "project.astroproj") == before            # the copy was only read
    shots = sorted(p.name for p in B3.glob("*.png") if not p.name.startswith("fit_"))      # fit_*: test_graph's
    assert len(shots) == 17, shots

