"""P11: the Run part of the client (plan dialog, run service, progress on the graph, the console, the recap), offscreen.

Most tests run toy stages (inventory → stack → denoise (the 'denoise' option: steps, checkpoints) → color →
render_prep) in a real detached worker on the CPU: the worker builds the same toy graph from the spec
'test_run:toy_rules' (this module, put on its PYTHONPATH). The slow tests run the real stages on a copy of the
heart_a7s v2 project (F:/astro_essais/v2/heart_a7s_p11) on the CPU, and take the screenshots (ASTRO_GPU_SHOTS names the
folder; default %TEMP%/p11_shots).

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

HERE = Path(__file__).parent
SPEC = "test_run:toy_rules"
F_COPY = Path("F:/astro_essais/v2/heart_a7s_p11")
SHOTS = Path(os.environ.get("ASTRO_GPU_SHOTS") or Path(os.environ.get("TEMP", ".")) / "p11_shots")


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
    with other jobs (the other agents' tests, the runs of P8) that takes longer, and the limit goes up to 600 ms."""
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
    w = make_world(env["tmp"], steps=100, delay=0.04)
    try:
        other = api.open_project(w.p.root)                  # a run started by another window, or the command line
        api.use_rules(other, SPEC)
        info = api.start_run(other, api.plan_all(other), "cpu")
        assert pump(lambda: api.running(w.p) is not None, 60)
        d = w.ctl.launch()
        d.start()                                           # refused by the engine: this window follows that run
        assert pump(lambda: w.ctl.running and w.ctl.track is not None and w.ctl.track.run_id == info.run_id, 30)
        assert w.ctl.track.jobs and w.ctl.dialog is None            # the plan dialog is gone: the run has its own
        assert pump(lambda: w.track.finished and not w.ctl.running, 120) and w.track.status == "done"
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


def test_the_mockups_run_blocks_door_opens_the_plan_of_the_block(toy):
    items = toy.model.pending_upto(toy.model.view(1), ["stack"])
    d = run_blocks(toy.model, None, items, "Run to Stacking", None, controller=toy.ctl)
    assert d is not None and [r.row.title for r in d.rows] == ["Calibration", "Stacking"]
    d.close()
    assert run_blocks(toy.model, None, [], "x", None, controller=toy.ctl) is None
    rows = build_rows(toy.model, toy.model.plan_all().jobs)
    assert [r.title for r in rows][0] == "Calibration"


# ---------------------------------------------------------------------------------------------- the real stages
real = pytest.mark.skipif(not (F_COPY / "project.astroproj").is_file(),
                          reason="the heart_a7s_p11 copy of the v2 test project is not on this machine")


def _grab(w: QWidget, name: str):
    SHOTS.mkdir(parents=True, exist_ok=True)
    QApplication.processEvents()
    w.grab().save(str(SHOTS / f"{name}.png"))


def _real_world(tmp: Path) -> World:
    """The copy of heart_a7s with the four workflows, on the CPU: the model never saves (the plan carries the
    definition), the runs write the new results into the copy."""
    from astro_gpu.core import workflow as wfl
    p = api.open_project(F_COPY)
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
    """Run the Classic stack of the heart_a7s copy for real on the CPU: background_poly and render_simple. The
    stripes turn green and the layer appears. Takes the screenshots in English and in French."""
    monkeypatch.setenv("ASTRO_GPU_HOME", str(F_COPY.parent / "heart_a7s_p11_home"))
    monkeypatch.setattr(project_mod, "STATE", F_COPY.parent / "heart_a7s_p11_home")
    before = done_keys(F_COPY)
    w = _real_world(env["tmp"])
    root = None
    created: list[tuple[str, str]] = []
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
            created += [(j.stage, j.key) for j in d.plan.jobs]
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
        # the copy goes back to what it was: the results of this test are removed (they belong to the copy)
        after = done_keys(F_COPY)
        for stage, keys in after.items():
            for k in keys:
                if k not in before.get(stage, []):
                    shutil.rmtree(F_COPY / "store" / stage / k, ignore_errors=True)
        for stage, k in created:
            shutil.rmtree(F_COPY / "store" / stage / f"{k}.tmp", ignore_errors=True)


@real
@pytest.mark.slow
def test_screenshots_of_the_failed_run_and_the_stopped_run_on_the_real_plan(env):
    """A failed run and a stopped run on the real four-workflow plan, from synthetic events (no stage is run):
    the recaps in English and French."""
    w = _real_world(env["tmp"])
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
