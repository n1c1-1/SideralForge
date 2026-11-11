"""The GUI exploration's points on the run window, the status bar and the checks before a run (RAPPORT 4.2, 4.3, 6.1 to
6.11 and the run_status notes), offscreen on scratch projects: nothing is computed, no GPU."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from astro_gpu import api  # noqa: E402
from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client.logview import event_text  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.runs import RunDialog, build_rows  # noqa: E402
from astro_gpu.client.services.run import (  # noqa: E402
    DONE,
    RunController,
    RunTrack,
    render_line,
    run_here_refusal,
    run_key,
)
from astro_gpu.client.services.worker import ApiWorker, Gpu, Machine  # noqa: E402
from astro_gpu.client.settingsdlg import auto_device, device_items  # noqa: E402
from astro_gpu.core import project as project_mod  # noqa: E402
from astro_gpu.core import project2, runner2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    state = tmp_path / "state"
    monkeypatch.setattr(project_mod, "STATE", state)
    monkeypatch.setenv("ASTRO_GPU_HOME", str(state))
    appmod.apply_language("en")
    yield tmp_path
    appmod.apply_language("en")


def raws(folder: Path, n: int = 3, ext: str = "ARW") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"IMG_{i:03d}.{ext}").write_bytes(b"\0" * (100 + i))
    return folder


GRAPH = [{"block": "calibration"}, {"block": "registration"}, {"block": "stacking"}, {"block": "gradient"},
         {"block": "gradient", "name": "gradient~2"}]


def project(tmp: Path, blocks=GRAPH, sessions: int = 1, lights: bool = True, mode: str = "together",
            current_block: str = "", flows: int = 1):
    tmp = tmp / f"p{len(list(tmp.glob('p*')))}"                    # a project of its own for each call of a test
    p = project2.create(tmp / "P", "P", Target(name="M 31"))
    p.stack_mode = mode
    for i in range(sessions):
        p.sessions.append(project2.Session(i + 1, f"night {i + 1}",
                                           lights=[str(raws(tmp / "raw" / f"n{i}"))] if lights else []))
    for k in range(flows):
        w = p.add_workflow("Workflow" if not k else f"Flow {k + 1}", {})
        if flows > 1 and sessions == flows:                         # a flow per session: nothing shared
            w.sessions = [k + 1]
        p.set_blocks(w.id, list(blocks))
    p.current_block = current_block
    project2.save(p)
    return p


class World:
    def __init__(self, p):
        self.worker = ApiWorker()
        self.m = ProjectModel(p, self.worker, readonly=True, gpu=False, device="cpu")
        assert self.worker.wait(120) and self.m.ready, self.m.error
        self.ctl = RunController(self.m, None, None, self.worker)

    def close(self):
        self.ctl.close()
        self.m.close(save=False)
        self.worker.stop()
        QApplication.processEvents()


@pytest.fixture
def world(env):
    made = []

    def make(**kw):
        w = World(project(env, **kw))
        made.append(w)
        return w
    yield make
    for w in made:
        w.close()


# ---------------------------------------------------------------------------------------------- 4.2: the events
def test_the_events_of_a_block_placed_twice_go_to_its_own_job(tmp_path):
    """4.2: the stage code names itself "gradient"; the job of the second Gradient ("gradient~2") gets its events, not
    the first one (it stayed "not started" after a run that computed it)."""
    j = runner2.RunJournal(tmp_path / "logs", "r1", echo=False)
    jobs = [runner2.Job("k1", "gradient", "gradient", [1], 1, {}), runner2.Job("k2", "gradient~2", "gradient~2", [1], 1,
                                                                             {"gradient": "k1"}),
            runner2.Job("k3", "calibrate@2", "calibration@2", [1], 1, {})]
    for job in jobs:
        j.job = job
        j.emit("stage_start", stage=job.stage.partition("~")[0].partition("@")[0])
        j.emit("stage_end", stage=job.stage.partition("~")[0].partition("@")[0], status="done", duration=1.0)
    events = [json.loads(x) for x in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["key"] for e in events] == ["k1", "k1", "k2", "k2", "k3", "k3"]
    assert events[2]["block"] == "gradient~2" and events[4]["block"] == "calibration@2"
    t = RunTrack(jobs, "r1")
    t.apply(events)
    assert [x.state for x in t.jobs] == [DONE, DONE, DONE]


# ---------------------------------------------------------------------------------------------- the rows
def test_the_run_window_names_its_rows_as_the_graph_names_its_blocks(world):
    """6.1: a session's Calibration is "Calibration · night 1", never "inventory@1"; 4.1: "Gradient 1", "Gradient 2";
    #10: no row for the Session blocks; #30: a row's tooltip is the block and its time, no stage of the engine."""
    w = world(sessions=2, mode="per_session")
    plan = w.m.plan_all()
    titles = [r.title for r in build_rows(w.m, plan.jobs)]
    assert "Calibration · night 1" in titles and "Calibration · night 2" in titles
    assert "Gradient 1" in titles and "Gradient 2" in titles
    assert not any("@" in x or "~" in x or x in ("Session", "inventory", "sessions") for x in titles), titles
    d = w.ctl.launch()
    assert all("@" not in r.toolTip() and "~" not in r.toolTip() and "Inventory" not in r.toolTip() for r in d.rows)
    assert d.rows[0].toolTip() == f"{d.rows[0].row.title}: {d.rows[0].dur.text()}"
    assert not any(r.row.title in ("Session", "night 1", "night 2") for r in d.rows)
    d.close()


def test_blocks_not_placed_are_not_rows_and_the_output_is_not_a_row_of_its_own(world):
    """6.3: Calibration and Registration not placed are computed within Stacking (no rows after it); 6.2: the
    Output hidden in the wire of the last block is that block's row."""
    w = world(blocks=[{"block": "stacking"}])
    plan = w.m.plan_all()
    rows = build_rows(w.m, plan.jobs)
    assert [r.title for r in rows] == ["Stacking"]
    assert {j.stage for j in plan.jobs} <= {"inventory", "calibrate", "register", "stack", "render_simple"}
    assert len(rows[0].keys) == len(plan.jobs)


def test_the_recap_names_the_flow_when_two_rows_share_a_name_and_is_written_once(world):
    """#34: "Stacking (Workflow), Stacking (Flow 2)"; #35: a rebuild (a change of language) writes the recap once."""
    w = world(blocks=[{"block": "stacking"}], flows=2, sessions=2)
    plan = w.m.plan_all()
    t = RunTrack.of_plan(plan)
    t.plan = plan
    t.abort("PermissionError: [Errno 13] Permission denied: 'x'")
    d = RunDialog(w.m, plan, "Run", track=t)
    names = sorted(d._named().values())
    assert names == ["Stacking (Flow 2)", "Stacking (Workflow)"], names
    d.rebuild()
    QApplication.processEvents()
    d.rebuild()
    shown = [lb.text() for lb in d.recap.findChildren(QLabel) if lb.isVisibleTo(d)]
    assert len(shown) == len(set(shown)), shown
    d.close()


def test_the_time_left_of_a_run_is_the_sum_of_its_rows(world):
    """#116: the status bar said "28 min left" over rows adding up to 17: it counted the running job's ETA, the running
    row its whole estimate (the done Inventory in it too). Both are the jobs' ``left`` now: the rows add up to it."""
    import re
    w = world()
    plan = w.m.plan_all()
    est = {"inventory": 60.0, "calibrate": 240.0, "register": 120.0, "stack": 180.0, "gradient": 120.0,
           "gradient~2": 120.0, "render_simple": 60.0}
    for j in plan.jobs:
        j.est_s = est[j.stage]
    k = {j.stage: j.key for j in plan.jobs}
    t = RunTrack.of_plan(plan, "r1")
    t.plan = plan
    t.apply([{"t": 100.0, "type": "run_start", "run": "r1", "device": "cpu", "pid": 1},
              {"t": 101.0, "type": "stage_start", "run": "r1", "stage": "inventory", "key": k["inventory"]},
              {"t": 102.0, "type": "stage_end", "run": "r1", "stage": "inventory", "key": k["inventory"], "status": "done",
               "duration": 1.0},
              {"t": 103.0, "type": "stage_start", "run": "r1", "stage": "calibrate", "key": k["calibrate"]},
              # half its bar, and an ETA far above its estimate (a straight line through a bar filled by phases)
              {"t": 600.0, "type": "progress", "run": "r1", "stage": "calibrate", "key": k["calibrate"], "frac": 0.5,
               "eta": 1080.0}])
    d = RunDialog(w.m, plan, "Run", track=t)
    d.refresh()
    rows = {r.row.title: r.dur.text() for r in d.rows}
    # Calibration: half of the estimate's half and half of the ETA, 0.5 x 120 + 0.5 x 1080 = 10 min; Gradient 2 holds
    # the Output
    assert rows == {"Calibration": "≈ 10 min", "Registration": "≈ 2 min", "Stacking": "≈ 3 min", "Gradient 1": "≈ 2 min",
                    "Gradient 2": "≈ 3 min"}, rows
    total = sum(int(re.search(r"(\d+) min", x).group(1)) for x in rows.values())
    assert t.remaining() == pytest.approx(60.0 * total)
    w.ctl.track, w.ctl.phase = t, "running"
    assert w.ctl.status_text().endswith(f"≈ {total} min left") and d.summary.text().endswith(f"≈ {total} min left")
    w.ctl.track, w.ctl.phase = None, "idle"
    d.close()


def _calibrating(w, now: float):
    """A run of ``world()``'s plan: Inventory done in 1 min, Calibration 5 min into its job, half its bar, an ETA of 18
    min (its time left 10 min, as in #116), the rest waiting: 20 min left after it."""
    plan = w.m.plan_all()
    est = {"inventory": 60.0, "calibrate": 240.0, "register": 120.0, "stack": 180.0, "gradient": 120.0,
           "gradient~2": 120.0, "render_simple": 60.0}
    for j in plan.jobs:
        j.est_s = est[j.stage]
    k = {j.stage: j.key for j in plan.jobs}
    t = RunTrack.of_plan(plan, "r1")
    t.plan, t.groups = plan, w.ctl.groups(plan)
    t.apply([{"t": now - 400.0, "type": "run_start", "run": "r1", "device": "cpu", "pid": 1},
             {"t": now - 370.0, "type": "stage_start", "run": "r1", "stage": "inventory", "key": k["inventory"]},
             {"t": now - 310.0, "type": "stage_end", "run": "r1", "stage": "inventory", "key": k["inventory"],
              "status": "done", "duration": 60.0},
             {"t": now - 300.0, "type": "stage_start", "run": "r1", "stage": "calibrate", "key": k["calibrate"]},
             {"t": now - 1.0, "type": "progress", "run": "r1", "stage": "calibrate", "key": k["calibrate"], "frac": 0.5,
              "eta": 1080.0, "msg": "frame 6/12"}])
    return plan, t, k


def test_a_block_has_one_progress_in_its_row_the_window_and_the_graph_and_the_status_bar_says_the_run(world):
    """#136: at the same moment the status bar said "Running: Astrometry · 32 %" (the whole run's share, read as the
    block's), the window's line "Astrometry · 22 %" (the running job's own bar) and the row's bar 8 % (its jobs' bars
    weighted by their estimates). The row, the line and the block's bar in the graph now show one figure, in time: what
    the block took over that and its time left, so that its percentage and its time left agree."""
    import time

    from astro_gpu.client.blocks import Graph
    from astro_gpu.client.runs import RunOverlay
    w = world()
    plan, t, k = _calibrating(w, time.time())
    d = RunDialog(w.m, plan, "Run", track=t)
    d.refresh()
    row = next(r for r in d.rows if r.row.title == "Calibration")
    assert row.dur.text() == "≈ 10 min"                          # 0.5 x 120 + 0.5 x 1080 s still to go
    # Inventory's 60 s and Calibration's 300 s gone, 600 s to go: 360 / 960
    assert row.bar.value() == pytest.approx(375, abs=2)
    assert d.detail.text() == "Calibration · 38 % · frame 6/12"
    w.ctl.track, w.ctl.phase = t, "running"
    try:
        node = Graph(w.m).node_of(w.m.view(plan.jobs[0].run_wf), "calibration")
        assert RunOverlay.fraction_of(SimpleNamespace(c=w.ctl), node) == pytest.approx(0.375, abs=0.002)
        # the run: 360 s gone, 600 + 600 s to go
        assert w.ctl.status_text() == "Running: Calibration · 23 % of the run · ≈ 20 min left"
    finally:
        w.ctl.track, w.ctl.phase = None, "idle"
        d.close()


def test_a_stopped_row_says_it_was_stopped_and_how_long_it_ran_never_its_estimate(world):
    """#138: after Stop, the stopped row said "≈ 6 min", its first estimate, after 8 min of work. It says it was
    stopped and the time it ran, as a done row says its duration; its bar how far it got. A worker that ended without
    a word on the job (killed) counts until the run's end."""
    import time
    w = world()
    now = time.time()
    for said in (True, False):
        plan, t, k = _calibrating(w, now)
        end = [{"t": now + 180.0, "type": "stage_end", "run": "r1", "stage": "calibrate", "key": k["calibrate"],
                "status": "interrupted", "duration": 480.0}] if said else []
        t.apply(end + [{"t": now + 180.0, "type": "run_end", "run": "r1", "status": "interrupted",
                        "interrupted": [] if said else [k["calibrate"]],
                        "not_started": [j.key for j in plan.jobs if j.stage not in ("inventory", "calibrate")]}])
        d = RunDialog(w.m, plan, "Run", track=t)
        d.refresh()
        rows = {r.row.title: r for r in d.rows}
        cal = rows["Calibration"]
        # Inventory 60 s and Calibration 480 s; 600 s it still had to go: 540 / 1140
        assert cal.dur.text() == "stopped · 9 min" and cal.mark.state == "interrupted", (said, cal.dur.text())
        assert cal.bar.value() == pytest.approx(474, abs=2)
        assert all(r.dur.text() == "—" for title, r in rows.items() if title != "Calibration")
        d.close()


def test_a_failed_row_says_it_failed_and_how_long_it_ran_never_its_estimate(world):
    """As a stopped row (#138): a block that failed after 8 min said "≈ 4 min", its estimate."""
    import time
    w = world()
    now = time.time()
    plan, t, k = _calibrating(w, now)
    t.apply([{"t": now + 180.0, "type": "stage_end", "run": "r1", "stage": "calibrate", "key": k["calibrate"],
              "status": "failed", "duration": 480.0, "error": "ValueError: no frame"},
             {"t": now + 180.0, "type": "run_end", "run": "r1", "status": "failed",
              "not_started": [j.key for j in plan.jobs if j.stage not in ("inventory", "calibrate")]}])
    d = RunDialog(w.m, plan, "Run", track=t)
    d.refresh()
    cal = {r.row.title: r for r in d.rows}["Calibration"]
    assert cal.mark.state == "failed" and cal.dur.text() == "failed · 9 min", cal.dur.text()
    d.close()


def test_a_long_recap_scrolls_in_its_box_never_cut(world):
    """GUI exploration 4: a large project's run not started listed its 72 blocks in one paragraph that ran out of the
    box, cut mid-name. The recap's lines are in a scroll area of bounded height; every name is in its text."""
    from astro_gpu.client.runs import RECAP_MAX_H
    w = world(blocks=GRAPH, flows=16, sessions=16)
    plan = w.m.plan_all()
    t = RunTrack.of_plan(plan)
    t.plan = plan
    t.abort("PermissionError: [Errno 13] Permission denied: 'x'")
    d = RunDialog(w.m, plan, "Run", track=t)
    d.resize(740, 700)
    d.show()
    QApplication.processEvents()
    d.refresh()
    QApplication.processEvents()
    texts = [lb.text() for lb in d.recap_area.widget().findChildren(QLabel)]
    line = next(x for x in texts if x.startswith("Not started: "))
    assert all(name in line for name in d._named().values())
    assert d.recap_area.height() <= RECAP_MAX_H and d.recap_area.verticalScrollBar().maximum() > 0
    d.close()


# ---------------------------------------------------------------------------------------------- 4.3: Run to here
def test_run_to_here_runs_the_block_placed_twice_and_is_refused_on_a_session(world):
    w = world()
    p = w.m.path
    w.m.select(step="fond~2")
    d = w.ctl.run_to_here()
    assert d is not None and "gradient~2" in [j.stage for j in d.plan.jobs]
    assert d.windowTitle() == 'Run "Workflow" to Gradient 2'
    d.close()
    w.m.select(step="fond")
    d1 = w.ctl.run_to_here()
    assert "gradient~2" not in [j.stage for j in d1.plan.jobs] and "gradient" in [j.stage for j in d1.plan.jobs]
    d1.close()
    w.m.select(step="sources")                      # #29: a Session block is not computed: refused, with the reason
    assert w.ctl.run_to_here() is None and run_here_refusal(w.m, "sources")
    assert run_here_refusal(w.m, "fond~2") == "" and run_key("composition") == "etoiles"
    assert run_key("crop") == "alignement"                      # the night test pass: Crop computes nothing
    assert p is w.m.path


# ---------------------------------------------------------------------------------------------- 6.4: cannot start
def test_a_run_that_cannot_start_says_why_in_plain_words_in_the_window_and_the_log(world, monkeypatch):
    from astro_gpu.client.services import run as run_service
    w = world()
    history = []
    w.m.log = lambda template, **a: history.append((template, a))

    def refuse(*_a):
        raise PermissionError(13, "Permission denied", "C:/P/logs")
    monkeypatch.setattr(run_service, "_start_job", refuse)
    appmod.apply_language("fr")
    d = w.ctl.launch()
    d.start()
    assert w.worker.wait(30)
    QApplication.processEvents()
    assert w.ctl.track.never_started and not w.ctl.running
    texts = [lb.text() for lb in d.findChildren(QLabel) if lb.isVisibleTo(d)]
    assert "Exécution non démarrée" in texts
    assert any("accès refusé" in x for x in texts) and not any("Permission" in x or "Errno" in x for x in texts)
    # exploration 2: the real cause follows the plain sentence in the history and the console (never in the window)
    assert history and history[-1][0] == "Run not started: {why} · cause: {detail}"
    assert event_text(("12:00:00", history[-1][0], history[-1][1]))[1] == \
        "Exécution non démarrée : le dossier du projet ne peut pas être écrit : accès refusé · cause : " \
        "PermissionError: [Errno 13] Permission denied: 'C:/P/logs'"
    assert w.ctl.track.lines[-1][1] == "start_detail" and "Permission denied" in render_line(w.ctl.track.lines[-1])
    d.close()


# ---------------------------------------------------------------------------------------------- 6.5, 6.7, #15
def test_run_says_why_it_is_greyed_without_a_block_and_without_lights(world):
    w = world(blocks=[])
    fake = SimpleNamespace(s=w.m)
    assert appmod.no_block(w.m.p.workflows[0])
    assert appmod.Main.run_refusal(fake, w.m.cost()) == "No block: place one"
    w2 = world(blocks=[{"block": "stacking"}], lights=False)
    assert appmod.Main.run_refusal(SimpleNamespace(s=w2.m), w2.m.cost()).startswith("Nothing to compute")
    w2.m.select(step="empilement")
    d = w2.ctl.run_to_here()                       # 6.7: no invented figures, Run greyed with its reason
    assert d.plan.jobs and all(r.dur.text() == "" for r in d.rows)
    assert d.summary.text() == "Nothing to compute" and not d.b_run.isEnabled() and d.b_run.toolTip()
    d.close()


# ---------------------------------------------------------------------------------------------- control round
def test_a_switched_off_block_is_said_so_and_its_pause_waits_for_it(world):
    """Round 3: Run to here on a switched-off Gradient 2 said "is no longer in the graph"; a pause kept on it was
    announced as a stop while the run passed it silently. The reason is right, the pause says it waits, the run window
    says there is no stop there."""
    from astro_gpu.client.blocks import Graph
    from astro_gpu.client.graph import BlockItem
    w = world()
    m = w.m
    p = m.path
    m.set_breakpoint([p], "fond~2")
    assert m.set_cut([p], "fond~2", True)
    p = m.path
    assert m.waiting_breakpoints() == [(p.pid, "fond~2")]
    m.select(step="fond~2")
    d = w.ctl.run_to_here()
    shown = [lb.text() for lb in d.findChildren(QLabel) if lb.isVisibleTo(d)]
    assert "Gradient 2 is switched off: D switches it on again" in shown, shown
    assert not any("no longer in the graph" in x for x in shown) and not d.b_run.isEnabled()
    d.close()
    m.select(step="empilement")
    d = w.ctl.launch()
    assert d.pause_off.isVisibleTo(d) and d.pause_off.text().startswith("No stop at Gradient 2: switched off")
    assert not d.pause_note.isVisibleTo(d)                          # no stop announced
    d.close()
    node = Graph(m).node_of(p, "fond~2")
    bp = Graph(m).breakpoint(node)
    item = BlockItem(SimpleNamespace(), node, "Gradient 2", "", "done", False, False, True, bp=bp)
    assert "waits for the block to be switched on again" in item.toolTip()
    assert "the run stops here" not in item.toolTip()


def test_a_fusion_without_lights_has_nothing_to_run_as_f5_says(world):
    """Round 3: Shift+F5 on a Combine channels of a project without lights opened the run window where F5 said in the status line
    that there is nothing to compute: a Combine channels's data are its channels'."""
    w = world(lights=False)
    fusion = SimpleNamespace(source={"fusion": {"inputs": []}})
    assert not w.m.has_data() and not w.m.has_data(fusion)
    w2 = world()
    assert w2.m.has_data(fusion)


def test_a_block_placed_twice_has_its_own_method_and_astrometry_says_what_it_will_do(world):
    """Round 3: Gradient 2's Method list was greyed "Set by the options of the workflow" (its method is its node's:
    chosen on its cards, one Undo step); the Astrometry block said "solved" before anything was computed."""
    from astro_gpu.client.blocks import Graph
    from astro_gpu.client.model import method_of
    w = world(blocks=GRAPH[:3] + [{"block": "astrometry"}] + GRAPH[3:])
    m = w.m
    p = m.path
    assert method_of(p, "fond~2") == method_of(p, "fond") == "simple"
    assert m.set_node_method([p], "fond~2", "rbf")
    p = m.path
    assert method_of(p, "fond~2") == "rbf" and method_of(p, "fond") == "simple"
    g = Graph(m)
    assert g.summary(g.node_of(p, "fond~2")) != g.summary(g.node_of(p, "fond"))
    assert g.summary(g.node_of(p, "astrometrie")) == "solves the field · Gaia DR3"
    m.undo()
    assert method_of(m.path, "fond~2") == "simple"


def test_a_block_placed_twice_has_the_settings_of_the_first_its_own_values(world):
    """Round 4: Gradient 2's panel had no setting (no surface degree, no advanced settings): its catalogue was looked up
    under "fond~2". It lists Gradient 1's settings, read from and written to its own stage ("gradient~2")."""
    from astro_gpu.client.settings import needed_stages, step_settings
    w = world()
    m = w.m
    p = m.path
    for st in needed_stages(m, p, "fond") | needed_stages(m, p, "fond~2"):
        m.params(p, st)
    assert w.worker.wait(60)
    QApplication.processEvents()
    one, two = step_settings(m, p, "fond"), step_settings(m, p, "fond~2")
    names = [(t, [r["name"] for r in rs]) for t, rs in two]
    assert names == [(t, [r["name"] for r in rs]) for t, rs in one] and "degree" in names[0][1], names
    assert all(r["stage"] == "gradient~2" for _t, rs in two for r in rs)
    assert all(r["stage"] == "gradient" for _t, rs in one for r in rs)


def test_unreadable_lights_are_not_counted_as_used(env):
    """Round 3: five .fits that are not FITS files: Run refused, the Session said "5 / 5 used". Every light is read
    (the first card), the message names the first and the count is kept for the Session."""
    from astro_gpu.client.model import session_checks
    p = project2.create(env / "P", "P", Target(name="M 31"))
    d = env / "bad"
    d.mkdir()
    files = []
    for i in range(5):
        f = d / f"light_{i:03d}.fits"
        f.write_bytes(b"not a fits file" * 10)
        files.append(str(f))
    ses = project2.Session(1, "night 1", lights=[str(d)])
    out = session_checks(p, ses, {"lights": files})
    assert len(out) == 1 and out[0][1]["name"] == "light_000.fits" and out[0][1]["unread"] == 5
    m = SimpleNamespace(_facts=SimpleNamespace(checks={1: out}))
    assert ProjectModel.unreadable_lights(m, 1) == 5


# ---------------------------------------------------------------------------------------------- 6.6
def test_the_project_reopens_on_the_block_selected_and_a_new_one_on_the_session(world):
    w = world(current_block="fond~2")
    assert w.m.sel_step == "fond~2"
    w.m.select(step="empilement")
    assert w.m.p.current_block == "empilement"
    assert project2.Project2.from_dict(w.m.p.to_dict(), w.m.p.root).current_block == "empilement"
    assert world(blocks=[{"block": "stacking"}], current_block="fond").m.sel_step == "sources"   # gone: the Session
    assert world(sessions=2, mode="per_session").m.sel_step == "session@1"


def test_a_closed_project_leaves_no_model_in_memory(env):
    """GUI exploration 4: memory grew at each close and reopening. A model's lambda on its own signal, and the Inspect
    service whose weak key was its model (the service holding it), kept every closed project alive."""
    import gc
    import weakref

    from PySide6.QtCore import QEvent

    from astro_gpu.client.services.inspect import service_of
    p = project(env)
    win = appmod.Main()
    try:
        assert win.open_project(p.root)
        assert win.worker.wait(120)
        service_of(win.s)
        ref = weakref.ref(win.s)
        assert win.close_project(discard=True)
        for _i in range(3):                       # the event loop's deferred deletions
            QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            QApplication.processEvents()
        gc.collect()
        assert ref() is None
    finally:
        win.close()


def test_the_project_reopens_on_its_recombine_block(env):
    """GUI exploration 4: Combine layers selected, saved and reopened, the Session was selected (a Combine layers is a Post layer,
    not a node of the graph). Its layer is selected again; without a Stars block, the Session."""
    stars = GRAPH[:3] + [{"block": "denoise"}, {"block": "stars"}]
    p = project(env, blocks=stars, current_block="composition")
    wid = p.workflows[0].id
    p.post = project2.upgraded_post({"layers": [{"kind": "workflow", "pid": wid, "part": "combined", "id": 7, "name": "Workflow · Combine layers"}]})
    project2.save(p)
    w = World(p)
    try:
        assert w.m.sel_step == "composition" and w.m.sel_layer == 7
    finally:
        w.close()
    w = World(project(env, blocks=GRAPH, current_block="composition"))
    try:
        assert w.m.sel_step == "sources" and w.m.sel_layer is None
    finally:
        w.close()


# ---------------------------------------------------------------------------------------------- 6.8, 6.9
def test_automatic_is_the_gpu_it_picks_and_both_lists_are_the_same():
    m = Machine([Gpu("cuda:0", "RTX 3070", 8.0), Gpu("cuda:1", "RTX 4090", 24.0)], 16, 64.0)
    assert auto_device(m) == "cuda:1" and auto_device(Machine([], 8, 16.0)) == "cpu" and auto_device(None) is None
    assert [t for t, _d, _tip in device_items(m)] == ["Automatic", "RTX 3070", "RTX 4090", "CPU only"]
    assert "RTX 4090" in device_items(m)[0][2]
    prefs.set("device", "auto")
    assert appmod.Main._device(SimpleNamespace(machine=m)) == (True, "cuda:1")       # estimated on the GPU it uses
    prefs.set("device", "cpu")
    assert appmod.Main._device(SimpleNamespace(machine=m)) == (False, "cpu")


# ---------------------------------------------------------------------------------------------- 6.10
def test_the_checks_finish_on_an_unreadable_frame_and_say_the_foreign_files(env):
    from astro_gpu.client.frames import FramesView, check_text
    from astro_gpu.client.services.inspect import project_frames
    p = project2.create(env / "P", "P", Target(name="M 31"))
    lights = raws(env / "raw" / "n1", 3)                       # not real raws: none of them can be read
    (env / "phone.JPG").write_bytes(b"\xff\xd8\xff" + b"\0" * 50)
    p.sessions.append(project2.Session(1, "night 1", filter="OIII", lights=[str(lights), str(env / "phone.JPG")]))
    w = p.add_workflow("Workflow", {})
    p.set_blocks(w.id, [{"block": "stacking"}])
    project2.save(p)
    out = project_frames(p, None)                              # it ends (it stopped on the first unreadable frame)
    row = out.sessions[0]
    assert row.lights == 4 and row.kept == 3 and row.filter == "OIII"       # the JPEG is not kept
    keys = [c.key for c in out.checks]
    assert "unreadable" in keys and "file_types" in keys and "ok" not in keys
    text = check_text(next(c for c in out.checks if c.key == "unreadable"))
    assert text.startswith("night 1: IMG_000.ARW cannot be read")
    ses = {"name": "night 1", "lights": 4, "kept": 4, "flats": 0, "darks": 0, "bias": 0, "filter": "OIII"}
    assert FramesView._cells(ses, row, osc=True)[6] == "OIII"


def test_the_run_window_warns_about_a_file_of_another_type(world):
    """6.10: a JPEG among the lights is said before the run too (it was on the Session block and its panel only)."""
    w = world(blocks=[{"block": "stacking"}])
    jpg = Path(w.m.p.sessions[0].lights[0]).parent / "phone.JPG"
    jpg.write_bytes(b"\xff\xd8\xff" + bytes(50))
    w.m.p.sessions[0].lights.append(str(jpg))
    w.m._start_warm()                                          # the frames read again, as after an import
    assert w.worker.wait(60)
    QApplication.processEvents()
    assert w.m.file_warnings(1)
    d = w.ctl.launch()
    texts = [lb.text() for lb in d.findChildren(QLabel) if lb.isVisibleTo(d)]
    assert any(t.startswith("night 1: phone.JPG is not a frame the engine reads") for t in texts), texts
    d.close()


# ---------------------------------------------------------------------------------------------- words
def test_the_console_counts_blocks_and_the_log_follows_the_language():
    """#3: the console counts blocks; #31: a block's name written in French reads in English; 6.11."""
    assert render_line((0.0, "starting", {"n": 3})).endswith("Starting: 3 blocks")
    t = RunTrack([runner2.Job("a", "inventory", "calibration", [1], 1, {}), runner2.Job("b", "stack", "stacking", [1],
                                                                                         1, {"inventory": "a"})])
    t.groups = [["a", "b"]]
    t.apply([{"t": 1.0, "type": "run_start", "run": "r"},
             {"t": 2.0, "type": "stage_end", "run": "r", "key": "a", "stage": "inventory", "status": "done"},
             {"t": 3.0, "type": "stage_end", "run": "r", "key": "b", "stage": "stack", "status": "done"},
             {"t": 4.0, "type": "run_end", "run": "r", "status": "done"}])
    assert t.block_count()[DONE] == 1 and "1 block in" in render_line(t.lines[-1])
    row = ("12:00:00", '{block} added to "{name}"', {"block": "Empilement", "name": "W"})
    assert event_text(row)[1] == 'Stacking added to "W"'
    assert event_text(("1", '{block} added to "{name}"', {"block": "Débruitage 2", "name": "W"}))[1] == \
        'Denoise 2 added to "W"'
    i18n.set_language("fr")
    try:
        assert event_text(row)[1] == 'Empilement ajouté à « W »' or "Empilement" in event_text(row)[1]
        assert i18n._("its Image block has no file yet") != "its Image block has no file yet"
        assert render_line((0.0, "starting", {"n": 1})).endswith("Démarrage : 1 bloc")
    finally:
        i18n.set_language("en")


def test_every_new_text_has_its_french():
    """The texts of this work, in run.json (or an older file)."""
    import ast
    i18n.set_language("fr")
    try:
        cat = i18n._cat
        root = Path(__file__).parents[2] / "astro_gpu"
        missing = []
        for f in ("client/runs.py", "client/services/run.py", "client/settingsdlg.py", "client/frames.py"):
            for n in ast.walk(ast.parse((root / f).read_text(encoding="utf-8"))):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("_", "plural") \
                        and n.args:
                    for a in (n.args if n.func.id == "plural" else n.args[:1]):
                        if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value not in cat:
                            missing.append(a.value)
        from astro_gpu.client.services.run import STAGE_TITLES, start_failure
        missing += [v for v in STAGE_TITLES.values() if v not in cat]
        for why in ("Errno 28 No space left", "PermissionError: x", "FileNotFoundError: x", "has no folder yet", "?"):
            missing += [start_failure(why)] if start_failure(why) not in cat else []
        assert not missing, missing
        assert api is not None
    finally:
        i18n.set_language("en")
