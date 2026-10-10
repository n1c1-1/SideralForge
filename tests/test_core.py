"""Engine tests: parameter resolution, staleness, resume after interruption, journal."""
from __future__ import annotations

import json

import pytest

from astro_gpu.core.journal import Journal, read_events
from astro_gpu.core.params import ADVANCED, Auto, P, StageParams, params_hash, resolve
from astro_gpu.core.project import Inputs, Project
from astro_gpu.core.runner import Pipeline, request_stop
from astro_gpu.core.stage import Interrupted, Stage


class AParams(StageParams):
    k: float = P(1.0, label="k", help="a factor", ge=0)
    n: int = P(3, label="n", help="a count", level=ADVANCED)


class StageA(Stage):
    name = "a"
    title = "A"
    Params = AParams
    runs = 0

    def auto(self, ctx):
        return {"k": Auto(2.0, "because")}

    def run(self, ctx, p):
        StageA.runs += 1
        ctx.save_json("out.json", {"k": p.k, "n": p.n})
        ctx.metric("k", p.k)


class BParams(StageParams):
    steps: int = P(10, label="steps", help="steps")


class StageB(Stage):
    name = "b"
    title = "B"
    deps = ("a",)
    Params = BParams
    runs = 0
    stop_at: int | None = None

    def run(self, ctx, p):
        StageB.runs += 1
        ck = ctx.ckpt / "i.json"
        start = json.loads(ck.read_text())["i"] if ck.exists() else 0
        for i in range(start, p.steps):
            if StageB.stop_at is not None and i == StageB.stop_at:
                StageB.stop_at = None
                raise Interrupted("test stop")
            ck.write_text(json.dumps({"i": i + 1}))
        ctx.save_json("out.json", {"resumed_from": start, "k": ctx.load_json("a", "out.json")["k"]})


@pytest.fixture
def project(tmp_path):
    StageA.runs = StageB.runs = 0
    return Project.create(tmp_path / "proj", "test", Inputs(lights=[]))


def test_resolve_priority():
    p, origin = resolve(AParams, {"k": Auto(2.0, "r")}, {})
    assert p.k == 2.0 and origin["k"] == "auto" and origin["n"] == "default"
    p, origin = resolve(AParams, {"k": Auto(2.0, "r")}, {"k": 5.0})
    assert p.k == 5.0 and origin["k"] == "user"
    with pytest.raises(ValueError):
        resolve(AParams, {}, {"nope": 1})
    assert params_hash(AParams(k=1)) != params_hash(AParams(k=2))


def test_describe_has_metadata():
    d = {x["name"]: x for x in AParams.describe()}
    assert d["n"]["level"] == "advanced" and d["k"]["ge"] == 0 and d["k"]["help"]


def test_run_and_up_to_date(project):
    pl = Pipeline(project, [StageA(), StageB()], Journal(project.logs, echo=False))
    assert pl.plan() == ["a", "b"]
    assert pl.run() == "done"
    assert [s.state for s in pl.status()] == ["done", "done"]
    assert pl.plan() == []
    assert pl.run() == "done" and StageA.runs == 1 and StageB.runs == 1


def test_param_change_makes_downstream_stale(project):
    pl = Pipeline(project, [StageA(), StageB()], Journal(project.logs, echo=False))
    pl.run()
    project.set_override("a", "k", 7.0)
    st = {s.name: s for s in pl.status()}
    assert st["a"].state == "stale" and "parameters" in st["a"].reason
    assert pl.plan() == ["a", "b"]
    pl.run()
    assert json.loads((project.stage_dir("b") / "out.json").read_text())["k"] == 7.0


def test_resume_after_interruption(project):
    pl = Pipeline(project, [StageA(), StageB()], Journal(project.logs, echo=False))
    StageB.stop_at = 6
    assert pl.run() == "interrupted"
    assert {s.name: s.state for s in pl.status()}["b"] == "interrupted"
    assert pl.run() == "done"
    out = json.loads((project.stage_dir("b") / "out.json").read_text())
    assert out["resumed_from"] == 6
    assert StageA.runs == 1


def test_stop_request(project):
    class Slow(Stage):
        name = "slow"
        title = "slow"

        def run(self, ctx, p):
            request_stop(ctx.project)
            ctx.progress(0.5, "half")

    pl = Pipeline(project, [Slow()], Journal(project.logs, echo=False))
    assert pl.run() == "interrupted"


def test_journal_events(project):
    pl = Pipeline(project, [StageA()], Journal(project.logs, echo=False))
    pl.run()
    evs, pos = read_events(project.logs / "events.jsonl")
    types = [e["type"] for e in evs]
    assert types[0] == "run_start" and "stage_end" in types and types[-1] == "run_end"
    more, pos2 = read_events(project.logs / "events.jsonl", pos)
    assert more == [] and pos2 == pos


def test_preview_accepts_grey_and_channel_first(project):
    import numpy as np

    class P(Stage):
        name = "p"
        title = "p"

        def run(self, ctx, p):
            ctx.preview("grey", np.random.rand(40, 60))
            ctx.preview("chw", np.random.rand(3, 40, 60))
            ctx.preview("hwc", (np.random.rand(40, 60, 3) * 255).astype(np.uint8))

    pl = Pipeline(project, [P()], Journal(project.logs, echo=False))
    assert pl.run() == "done"
    assert len(list((project.stage_dir("p") / "previews").glob("*.jpg"))) == 3


def test_new_parameter_at_default_keeps_results(project):
    pl = Pipeline(project, [StageA(), StageB()], Journal(project.logs, echo=False))
    pl.run()

    class AParams2(AParams):
        extra: bool = P(True, label="extra", help="new option, default = old behaviour")

    class StageA2(StageA):
        Params = AParams2

    pl2 = Pipeline(project, [StageA2(), StageB()], Journal(project.logs, echo=False))
    assert [s.state for s in pl2.status()] == ["done", "done"]
    project.set_override("a", "extra", False)
    assert pl2.status()[0].state == "stale"
