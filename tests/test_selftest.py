"""The compute devices (core/devices.py), the card test (core/selftest.py), a stage sent to the CPU in a run (the plan's
``cpu_jobs``), the memory check of the Stack stage and the tiles of the networks on a small card. No card is used:
fake devices and small synthetic inputs on the CPU (the real test on a card: python -m astro_gpu.core.selftest)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from astro_gpu.core import devices, runner2, selftest
from astro_gpu.core.stage import Context
from astro_gpu.stages import denoise as dn
from astro_gpu.stages import star_removal as sr

TESTS_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTS_DIR)
import test_runner2 as tr  # noqa: E402  (its toy stages and project)
from test_runner2 import proj, state  # noqa: E402, F401  (its fixtures)


# ---------------------------------------------------------------------------------------------- the devices
def test_device_names_and_kinds():
    assert all(devices.is_device(x) for x in ("cuda:0", "cuda", "xpu:1", "mps", "privateml:0"))
    assert not any(devices.is_device(x) for x in ("cpu", "gpu2", "auto", ""))
    assert devices._kind_of("AMD Radeon RX 7800 XT") == "amd"
    assert devices._kind_of("Intel(R) Arc(TM) A770") == "intel"
    assert devices._kind_of("NVIDIA GeForce RTX 4060 Ti") == "nvidia"
    assert devices._kind_of("Apple GPU") == "apple"


def test_a_cuda_card_of_amd_is_beta(monkeypatch):
    class Prop:
        name, total_memory = "AMD Radeon RX 7800 XT", 16 * 2 ** 30

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: Prop)
    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)
    d = devices.list_devices()[0]
    assert (d.device, d.kind, d.beta, round(d.memory_gb)) == ("cuda:0", "amd", True, 16)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)


def test_an_nvidia_card_is_not_beta(monkeypatch):
    class Prop:
        name, total_memory = "NVIDIA GeForce RTX 4060 Ti", 8 * 2 ** 30

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: Prop)
    assert not devices.list_devices()[0].beta


def test_an_intel_card_is_found_and_beta(monkeypatch):
    class Prop:
        name, total_memory, driver_version = "Intel(R) Arc(TM) A770", 8 * 2 ** 30, "1.23"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 1)
    monkeypatch.setattr(torch.xpu, "get_device_properties", lambda i: Prop)
    monkeypatch.setattr(torch.xpu, "memory_reserved", lambda i: 2 * 2 ** 30)
    infos = devices.list_devices()
    assert [(d.device, d.kind, d.beta) for d in infos] == [("xpu:0", "intel", True)]
    assert devices.available("xpu:0") and devices.free_memory("xpu:0") == 6 * 2 ** 30


def test_device_env_of_the_other_cards():
    assert runner2.device_env("xpu:0") == ("-1", "xpu:0")
    assert runner2.device_env("mps") == ("-1", "mps")
    assert runner2.device_env("privateml:0") == ("-1", "privateml:0")


def test_a_stage_on_a_card_this_process_has_not_runs_on_the_cpu():
    ctx = Context.__new__(Context)
    ctx.device_name = "xpu:0"
    assert ctx.device.type == ("xpu" if devices.available("xpu:0") else "cpu")


def test_helpers_do_nothing_on_the_cpu():
    devices.empty_cache("cpu")
    devices.synchronize("cpu")
    assert devices.free_memory("cpu") is None and devices.total_memory("cpu") is None
    assert devices.usable("cpu") == "cpu"


def test_out_of_memory_is_recognised():
    assert devices.is_out_of_memory(torch.OutOfMemoryError("CUDA out of memory"))
    assert devices.is_out_of_memory(RuntimeError("PI_ERROR_OUT_OF_RESOURCES"))
    assert not devices.is_out_of_memory(RuntimeError("something else"))


# ---------------------------------------------------------------------------------------------- the verdicts
def fake_op(card_value=1.0, card_sleep=0.0, cpu_sleep=0.02, error=None):
    """An operation: on the 'cpu' it gives 1.0, on the card ``card_value``, or raises ``error``."""
    def op(dev):
        if dev == "cpu":
            time.sleep(cpu_sleep)
            return {"a": np.array([1.0])}
        time.sleep(card_sleep)
        if error is not None:
            raise error
        return {"a": np.array([card_value])}
    return op


def test_the_card_is_kept_when_it_is_right_and_faster():
    r = selftest.check("drizzle", "cuda:0", fake_op())
    assert r["verdict"] == selftest.OK and r["card_s"] < r["cpu_s"]


def test_a_wrong_result_or_a_nan_sends_the_stage_to_the_cpu():
    assert selftest.check("drizzle", "cuda:0", fake_op(card_value=1.5))["verdict"] == selftest.WRONG
    assert selftest.check("denoise", "cuda:0", fake_op(card_value=float("nan")))["verdict"] == selftest.WRONG


def test_a_card_that_is_slower_goes_to_the_cpu():
    r = selftest.check("post", "cuda:0", fake_op(card_sleep=0.05, cpu_sleep=0.0))
    assert r["verdict"] == selftest.SLOWER


def test_any_exception_or_out_of_memory_means_cpu():
    r = selftest.check("post", "cuda:0", fake_op(error=RuntimeError("the driver said no")))
    assert r["verdict"] == selftest.ERROR and "driver said no" in r["error"]
    r = selftest.check("post", "cuda:0", fake_op(error=torch.OutOfMemoryError("CUDA out of memory")))
    assert r["verdict"] == selftest.ERROR and r["error"] == "out of memory"


def test_a_model_that_is_not_there_says_nothing_about_the_card():
    def absent(dev):
        raise FileNotFoundError("the star removal network is missing")

    r = selftest.check("stars", "cuda:0", absent)
    assert r["verdict"] == selftest.ABSENT
    assert selftest.judge([{"stage": "star_removal", "verdict": r["verdict"], "op": "stars"}])["star_removal"] == "card"


def ok(name):
    return {"op": name, "stage": selftest.OPERATIONS[name], "verdict": selftest.OK, "card_s": 0.1, "cpu_s": 1.0,
            "diff": 0.0, "tolerance": 1e-3, "error": ""}


def test_judge_needs_every_operation_of_a_stage():
    res = [ok("drizzle"), {**ok("sampling"), "verdict": selftest.WRONG}, ok("denoise"), ok("stars"), ok("post")]
    stages = selftest.judge(res)
    assert stages["stack"] == "cpu" and stages["denoise"] == "card" and stages["post"] == "card"


def test_the_profile_keeps_the_card_and_driver_and_says_why():
    res = [ok("drizzle"), {**ok("sampling"), "verdict": selftest.SLOWER}, ok("denoise"), ok("stars"), ok("post")]
    prof = selftest.make_profile("cuda:0", "RTX 4060 Ti", "610.88", res, auto=True)
    assert json.loads(json.dumps(prof))["card"] == "RTX 4060 Ti"                # goes into the settings' JSON
    assert selftest.still_valid(prof, "RTX 4060 Ti", "610.88")
    assert not selftest.still_valid(prof, "RTX 4060 Ti", "620.00")              # a new driver: tested again
    assert not selftest.still_valid(prof, "RTX 3070", "610.88")
    assert not selftest.still_valid(None, "RTX 4060 Ti", "610.88")
    profiles = {"cuda:0": prof, "auto": prof}
    assert selftest.profile_of(profiles, "auto") is prof and selftest.profile_of(profiles, "cuda:0") is prof
    assert selftest.profile_of(profiles, "cuda:1") is None
    assert selftest.stage_reason(prof, "stack") == selftest.REASONS[selftest.SLOWER]
    assert selftest.stage_reason(prof, "denoise") == ""
    assert selftest.stage_on_cpu(profiles, "auto", "stack") and not selftest.stage_on_cpu(profiles, "auto", "post")


def test_cpu_jobs_of_a_plan():
    res = [ok("drizzle"), {**ok("sampling"), "verdict": selftest.ERROR}, ok("denoise"), ok("stars"), ok("post")]
    prof = selftest.make_profile("cuda:0", "x", "1", res)

    class J:
        def __init__(self, key, stage):
            self.key, self.stage = key, stage

    class P:
        jobs = [J("a", "stack"), J("b", "stack@2"), J("c", "denoise")]

    jobs = selftest.cpu_jobs(prof, P)
    assert set(jobs) == {"a", "b"} and "the card failed" in jobs["a"]
    assert selftest.cpu_jobs(None, P) == {}


# ---------------------------------------------------------------------------------------------- the child process
class FakeProc:
    def __init__(self, lines):
        self.stdout = iter(lines)

    def poll(self):
        return 0

    def kill(self):
        pass


def _child(monkeypatch, lines):
    monkeypatch.setattr(selftest.subprocess, "Popen", lambda *a, **kw: FakeProc(lines))


def test_a_child_that_dies_counts_its_unfinished_operations_as_failed(monkeypatch):
    lines = [json.dumps({"ready": True}) + "\n", json.dumps(ok("drizzle")) + "\n"]
    _child(monkeypatch, lines)
    res = {r["op"]: r for r in selftest.run_test("xpu:0")}
    assert res["drizzle"]["verdict"] == selftest.OK
    assert res["denoise"]["verdict"] == selftest.ERROR and res["post"]["verdict"] == selftest.ERROR
    assert selftest.judge(list(res.values()))["denoise"] == "cpu"


def test_a_child_that_cannot_start_is_an_error_not_a_verdict(monkeypatch):
    _child(monkeypatch, [])
    with pytest.raises(RuntimeError):
        selftest.run_test("xpu:0")


def test_no_card_nothing_to_test(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert selftest.run_test("cuda:0") == []


# ---------------------------------------------------------------------------------------------- the operations, on the CPU
@pytest.mark.parametrize("op", ["drizzle", "sampling", "classic", "post"])
def test_operations_run_and_give_finite_numbers(op):
    out = selftest.OPS[op]("cpu")
    assert all(np.isfinite(v).all() for v in out.values())


def test_the_denoise_operation_runs_on_the_cpu():
    if not dn.MODEL.exists():
        pytest.skip("no denoising model here")
    assert np.isfinite(selftest.op_denoise("cpu")["image"]).all()


# ---------------------------------------------------------------------------------------------- a stage on the CPU in a run


def test_a_job_the_profile_sends_to_the_cpu_runs_there(tmp_path, monkeypatch, proj):  # noqa: F811
    plan = tr.plan_all(proj)
    job = plan.jobs[0]
    info = runner2.prepare_run(proj, plan, "cuda:0", {job.key: selftest.STAGE_ON_CPU.format(why="the card failed")})
    body = json.loads(open(info.plan_file, encoding="utf-8").read())
    assert body["cpu_jobs"] == {job.key: selftest.STAGE_ON_CPU.format(why="the card failed")}
    w = runner2._Worker(Path(proj.root), body, tr.RULES, False)
    assert w.device_of(job) == "cpu"
    assert w.device_of(plan.jobs[-1]) == w.device


# ---------------------------------------------------------------------------------------------- the memory of the Stack stage
def test_stack_memory_follows_the_output_pixels():
    assert devices.stack_memory(12.0) == pytest.approx(2.0)
    assert devices.stack_memory(48.0) == pytest.approx(8.0)


def test_a_stack_that_does_not_fit_goes_to_the_cpu(tmp_path, monkeypatch, proj):  # noqa: F811
    from astro_gpu import api
    from astro_gpu.core import estimate as est
    monkeypatch.setattr(api.devices, "available", lambda d: True)
    monkeypatch.setattr(api.devices, "free_memory", lambda d: 3 * 2 ** 30)
    monkeypatch.setattr(api, "_keys", lambda p: tr.K(p))
    monkeypatch.setattr(api, "_store", lambda p: None)
    monkeypatch.setattr(est, "facts", lambda keys, wf, store=None: est.Facts(grid_mpx=48.0))   # 8 GB for 3 GB free
    plan = tr.plan_all(proj)
    out = api._stack_on_cpu(proj, plan, "cuda:0")
    stacks = [j.key for j in plan.jobs if j.stage.split("@")[0] == "stack"]
    assert set(out) == set(stacks) and "needs about 8.0 GB" in next(iter(out.values()))
    monkeypatch.setattr(est, "facts", lambda keys, wf, store=None: est.Facts(grid_mpx=12.0))   # 2 GB: it fits
    assert api._stack_on_cpu(proj, plan, "cuda:0") == {}


# ---------------------------------------------------------------------------------------------- tiles on a small card
class EchoUNet(torch.nn.Module):
    """Stands for the denoising network: its output cancels the normalised input, so the tiles give the image back; it
    raises a memory error for tiles larger than ``limit``."""

    def __init__(self, limit=None):
        super().__init__()
        self.limit = limit

    def forward(self, x):
        if self.limit is not None and x.shape[-1] > self.limit:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3], device=x.device)


def _denoise_input():
    x = np.random.default_rng(0).normal(1.0, 0.1, (3, 300, 400)).astype(np.float32)
    nm = np.ones((300, 400), np.float32)
    return x, nm, np.ones(3, np.float32), np.full(3, 0.1, np.float32)


def test_denoise_keeps_the_tile_on_a_4_gb_card(monkeypatch):
    seen = []
    orig = dn._apply_net
    monkeypatch.setattr(dn.devices, "free_memory", lambda d: 4 * 2 ** 30)
    monkeypatch.setattr(dn, "_apply_net", lambda *a, **kw: seen.append(a[6]) or orig(*a, **kw))
    x, nm, bk, sg = _denoise_input()
    out = dn.apply_net(EchoUNet(), x, nm, bk, sg, torch.device("cpu"))
    assert seen == [512] and np.allclose(out, x, atol=1e-5)


def test_denoise_shrinks_the_tile_when_memory_is_short(monkeypatch):
    seen = []
    orig = dn._apply_net
    monkeypatch.setattr(dn.devices, "free_memory", lambda d: 200 * 2 ** 20)
    monkeypatch.setattr(dn, "_apply_net", lambda *a, **kw: seen.append(a[6]) or orig(*a, **kw))
    x, nm, bk, sg = _denoise_input()
    out = dn.apply_net(EchoUNet(), x, nm, bk, sg, torch.device("cpu"))
    assert seen[0] < 512 and np.allclose(out, x, atol=1e-5)


def test_denoise_halves_the_tile_on_a_memory_error():
    x, nm, bk, sg = _denoise_input()
    out = dn.apply_net(EchoUNet(limit=320), x, nm, bk, sg, torch.device("cpu"))    # a tile of 512+pad does not pass
    assert np.allclose(out, x, atol=1e-5)


def test_the_denoise_error_passes_through_when_no_tile_helps():
    x, nm, bk, sg = _denoise_input()
    with pytest.raises(torch.OutOfMemoryError):
        dn.apply_net(EchoUNet(limit=16), x, nm, bk, sg, torch.device("cpu"))


class EchoNaf(torch.nn.Module):
    """Stands for the star network: the stretched input unchanged, a memory error above ``limit`` tiles in a batch."""

    def __init__(self, limit=None):
        super().__init__()
        self.limit, self.batches = limit, []

    def forward(self, x):
        self.batches.append(x.shape[0])
        if self.limit is not None and x.shape[0] > self.limit:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return x


def _stars_img():
    return np.random.default_rng(1).normal(0.0, 1.0, (3, 300, 500)).astype(np.float32)


def test_the_stars_network_passes_fewer_tiles_on_a_small_card(monkeypatch):
    monkeypatch.setattr(sr.devices, "free_memory", lambda d: 200 * 2 ** 20)
    net = EchoNaf()
    out = sr.apply(net, _stars_img(), torch.device("cpu"))
    assert max(net.batches) < 16 and np.allclose(out, _stars_img(), atol=1e-3)


def test_the_stars_network_keeps_the_batch_on_a_4_gb_card(monkeypatch):
    monkeypatch.setattr(sr.devices, "free_memory", lambda d: 4 * 2 ** 30)
    net = EchoNaf()
    sr.apply(net, np.pad(_stars_img(), ((0, 0), (0, 0), (0, 3000)), mode="reflect"), torch.device("cpu"))
    assert max(net.batches) == 16


def test_the_stars_network_retries_with_a_smaller_batch_after_a_memory_error():
    net = EchoNaf(limit=4)
    img = np.pad(_stars_img(), ((0, 0), (0, 0), (0, 2000)), mode="reflect")          # more tiles than a batch of 4
    out = sr.apply(net, img, torch.device("cpu"))
    assert 4 in net.batches and np.allclose(out, img, atol=1e-3)


