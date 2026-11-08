"""The self-test of the card: nobody can try every card on the market, so the program tries the user's.

For each heavy operation of the engine (the drizzle deposit and the sampling of the prediction of the stack, the Denoise
network on one tile, the Stars network on one tile, the Post render), on a small synthetic input: the operation is run
on the card and on the CPU, the results are compared within a tolerance and both are timed. A stage keeps the card only
if every operation of it gives the right result and is faster there. Any exception, out-of-memory included, sends the
stage to the CPU. A stage that keeps the card runs exactly as it does without the self-test: nothing changes for it.

The operations run in a child process of their own (``python -m astro_gpu.core.selftest <device>``, one JSON line per
operation): a card that crashes or hangs takes the child with it, never the program, and the operations it did not
finish count as failed. ``run_test`` is that parent side (no Qt, no torch needed in the caller beyond the device list).

The result, a profile, is kept per device in the settings, with the card's name and driver (``client/services/selftest``
runs the test once in the background at first launch and when the card or its driver changed). ``cpu_jobs`` turns a
profile into the jobs of a plan that go to the CPU (core/runner2.py, the plan's ``cpu_jobs``).
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from . import devices

#: the operations tested, in order, and the stage each one decides (Stack has two)
OPERATIONS = {"drizzle": "stack", "sampling": "stack", "classic": "stack", "denoise": "denoise", "stars": "star_removal",
              "post": "post"}
STAGES = ("stack", "denoise", "star_removal", "post")
#: the largest difference allowed between the card and the CPU, in the unit each operation returns its result in: the
#: image's own scale for the sums and the render (rounding of float32 and of the order of the additions), the noise of
#: the image for the networks (autocast on the card: bfloat16); set from what a sound card gives (see tests/test_selftest)
TOLERANCE = {"drizzle": 5e-3, "sampling": 5e-3, "classic": 5e-3, "denoise": 0.3, "stars": 0.05, "post": 8e-3}
#: seconds one operation may take before the card counts as gone (its child process is then ended)
OP_TIMEOUT = 300.0
#: the free share of the card's memory below which it is busy with something else: a test launched by itself waits
BUSY_FREE_SHARE = 0.7

#: the verdicts: the card is kept for 'ok' and for 'absent' (a model this installation does not have: the stage
#: cannot run anywhere, and that says nothing about the card)
OK, WRONG, SLOWER, ERROR, ABSENT = "ok", "wrong", "slower", "error", "absent"
#: the reason, in English (translated in Settings and in the run window, client/locale)
REASONS = {WRONG: "the result was wrong on the card", SLOWER: "the card was slower than the processor",
           ERROR: "the card failed"}
STAGE_ON_CPU = "the card test sends this block to the processor: {why}"


# ---------------------------------------------------------------------------------------------- the operations
# Each returns {name: array} in a unit where the tolerance means something (TOLERANCE), computed on ``dev``. The input is
# the same on every device (a fixed seed, built on the host).

def _drizzle_input():
    H, W, S, pix = 1200, 1800, 1.5, 0.8
    rng = np.random.default_rng(1)
    val = rng.normal(1.0, 0.05, (H, W)).astype(np.float32)
    keep = np.ones((H, W), bool)
    keep[::97, ::89] = False
    return H, W, S, pix, val, keep


def op_drizzle(dev: str) -> dict[str, np.ndarray]:
    """The drizzle deposit of one frame (stages/stack.py) on a small output grid."""
    import torch

    from ..stages.stack import drizzle_deposit
    H, W, S, pix, val, keep = _drizzle_input()
    d = torch.device(dev)
    yy, xx = torch.meshgrid(torch.arange(H, device=d, dtype=torch.float32),
                            torch.arange(W, device=d, dtype=torch.float32), indexing="ij")
    Xo, Yo = (xx + 0.8) * S - 0.5, (yy + 0.3) * S - 0.5
    ch = torch.tensor([[0, 1], [1, 2]], device=d)[yy.long() % 2, xx.long() % 2]         # an RGGB pattern
    Ho, Wo = int(H * S), int(W * S)
    num, den = torch.zeros(3 * Ho * Wo, device=d), torch.zeros(3 * Ho * Wo, device=d)
    drizzle_deposit(num, den, torch.from_numpy(val).to(d), Xo, Yo, 1.0, torch.from_numpy(keep).to(d), ch, pix * S / 2, S,
                    Ho, Wo)
    return {"num": num.cpu().numpy(), "den": den.cpu().numpy()}


def op_sampling(dev: str) -> dict[str, np.ndarray]:
    """The model of the stack sampled where a frame's samples land (stages/stack.py ``Prediction``)."""
    import torch

    from ..stages.stack import Prediction
    Ho, Wo = 1300, 1900
    d = torch.device(dev)
    rng = np.random.default_rng(2)
    den = torch.from_numpy(rng.uniform(0.8, 1.2, (3, Ho, Wo)).astype(np.float32)).to(d).reshape(-1)
    num = den * torch.from_numpy(rng.normal(1.0, 0.1, (3, Ho, Wo)).astype(np.float32)).to(d).reshape(-1)
    model = Prediction(num, den, 20, Ho, Wo, d)
    H, W = 1200, 1800
    yy, xx = torch.meshgrid(torch.arange(H, device=d, dtype=torch.float32),
                            torch.arange(W, device=d, dtype=torch.float32), indexing="ij")
    ch = torch.tensor([[0, 1], [1, 2]], device=d)[yy.long() % 2, xx.long() % 2]
    pr = model.sample((xx + 0.8) * 1.05, (yy + 0.3) * 1.05, ch)
    return {k: (v / 20.0 if k == "cover" else v).cpu().numpy() for k, v in pr.items()}


def op_classic(dev: str) -> dict[str, np.ndarray]:
    """The Classic method of the Stacking block (core/stack_classic.py): frames debayered, interpolated and combined
    with rejection, on a few small frames."""
    from .stack_classic import ClassicParams, stack_classic
    H, W, n = 300, 400, 8
    rng = np.random.default_rng(3)
    frames = rng.normal(1000.0, 30.0, (n, H, W)).astype(np.float32)
    coef = np.zeros((n, 2, 10))
    coef[:, 0, 0], coef[:, 0, 1], coef[:, 1, 0], coef[:, 1, 2] = W / 2 + 0.4, W / 2, H / 2 - 0.3, H / 2
    res = stack_classic(lambda i: frames[i], coef, params=ClassicParams(band_pixels=1 << 16), device=dev,
                        used=list(range(n)), ch_of_k=[0, 1, 1, 2], H=H, W=W)
    return {"full": np.asarray(res.full) / 1000.0}


_NETS: dict[tuple[str, str], Any] = {}


def _net(kind: str, dev: str):
    """A network, loaded once per device (its loading is not what is timed)."""
    if (kind, dev) not in _NETS:
        import torch
        d = torch.device(dev)
        if kind == "denoise":
            from ..stages.denoise import pretrained
            net = pretrained(d)
            net.eval()
        else:
            from ..stages.star_removal import network
            net = network(d)
        _NETS[(kind, dev)] = net
    return _NETS[(kind, dev)]


def op_denoise(dev: str) -> dict[str, np.ndarray]:
    """The Denoise network over one tile; the result in units of the image's noise (its only scale)."""
    import torch

    from ..stages.denoise import apply_net
    rng = np.random.default_rng(3)
    x = rng.normal(1.0, 0.1, (3, 512, 512)).astype(np.float32)
    nm = np.ones((512, 512), np.float32)
    bk, sg = np.array([1.0, 1.0, 1.0], np.float32), np.array([0.1, 0.1, 0.1], np.float32)
    out = apply_net(_net("denoise", dev), x, nm, bk, sg, torch.device(dev))
    return {"image": ((out - bk[:, None, None]) / sg[:, None, None]).astype(np.float32)}


def op_stars(dev: str) -> dict[str, np.ndarray]:
    """The Stars network over one tile; the result in the network's own stretch (a star of 300 and a sky of 1 alike)."""
    import torch

    from ..stages.star_removal import TILE, apply, fwd
    rng = np.random.default_rng(4)
    img = rng.normal(0.0, 1.0, (3, TILE, TILE)).astype(np.float32)
    for y, x in ((60, 80), (130, 200), (190, 40)):
        img[:, y - 4:y + 4, x - 4:x + 4] += 300.0
    return {"image": fwd(apply(_net("stars", dev), img, torch.device(dev))).astype(np.float32)}


def _post_image() -> np.ndarray:
    rng = np.random.default_rng(5)
    img = np.abs(rng.normal(0.05, 0.01, (3, 1600, 2400))).astype(np.float32)
    img[:, 700:900, 1100:1300] += 0.3
    return img


def op_post(dev: str) -> dict[str, np.ndarray]:
    """The Post render of an image (0..1, the display's own scale)."""
    from ..render.simple import SimpleRenderEngine, SimpleSettings
    if ("post", dev) not in _NETS:
        _NETS[("post", dev)] = SimpleRenderEngine.from_array(_post_image(), device=dev)
    return {"image": _NETS[("post", dev)].render(SimpleSettings(), 1.0).final.cpu().numpy()}


OPS: dict[str, Callable[[str], dict[str, np.ndarray]]] = {
    "drizzle": op_drizzle, "sampling": op_sampling, "classic": op_classic, "denoise": op_denoise, "stars": op_stars,
    "post": op_post}


# ---------------------------------------------------------------------------------------------- one operation
def _timed(fn: Callable[[str], dict], dev: str) -> tuple[dict, float]:
    """The result and the seconds of ``fn(dev)``: the first call warms the device up (shaders, kernels, the network's
    loading: what a run pays once), the time is the best of two calls after it, so that another program's moment
    of load does not count against the card."""
    fn(dev)
    best, out = float("inf"), None
    for _ in range(2):
        t0 = time.perf_counter()
        out = fn(dev)
        devices.synchronize(dev)
        best = min(best, time.perf_counter() - t0)
    return out, best


def compare(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    """The largest absolute difference between two results (inf where only one of them is not a finite number)."""
    worst = 0.0
    for k in a:
        x, y = np.asarray(a[k], np.float64), np.asarray(b[k], np.float64)
        if x.shape != y.shape:
            return float("inf")
        d = np.abs(x - y)
        d[~np.isfinite(d)] = float("inf")
        if d.size:
            worst = max(worst, float(d.max()))
    return worst


def check(name: str, card: str, op: Callable[[str], dict] | None = None, tolerance: float | None = None) -> dict:
    """One operation on ``card`` and on the CPU: {op, stage, verdict, card_s, cpu_s, diff, tolerance, error}."""
    fn = op or OPS[name]
    tol = TOLERANCE.get(name, 5e-3) if tolerance is None else tolerance
    out: dict[str, Any] = {"op": name, "stage": OPERATIONS.get(name, name), "verdict": ERROR, "card_s": None,
                           "cpu_s": None, "diff": None, "tolerance": tol, "error": ""}
    try:
        ref, out["cpu_s"] = _timed(fn, "cpu")
    except FileNotFoundError as e:                     # a model this installation does not have
        return {**out, "verdict": ABSENT, "error": str(e)[:200]}
    except Exception as e:  # noqa: BLE001 - the operation cannot even run on the CPU: the card is not to blame
        return {**out, "verdict": ABSENT, "error": f"{type(e).__name__}: {e}"[:200]}
    try:
        got, out["card_s"] = _timed(fn, card)
    except Exception as e:  # noqa: BLE001 - any failure on the card, out of memory included: the CPU for this stage
        return {**out, "error": ("out of memory" if devices.is_out_of_memory(e) else f"{type(e).__name__}: {e}"[:200])}
    finally:
        devices.empty_cache(card)
    out["diff"] = compare(ref, got)
    if out["diff"] > tol:
        out["verdict"] = WRONG
    elif out["card_s"] >= out["cpu_s"]:
        out["verdict"] = SLOWER
    else:
        out["verdict"] = OK
    return out


# ---------------------------------------------------------------------------------------------- the profile
def judge(results: list[dict]) -> dict[str, str]:
    """{stage: 'card' | 'cpu'}: the card is kept for a stage when every operation of it says so (a stage not tested
    keeps it, as it would without the self-test)."""
    out = {s: "card" for s in STAGES}
    for r in results:
        if r["verdict"] not in (OK, ABSENT):
            out[r["stage"]] = "cpu"
    return out


def make_profile(device: str, card: str, driver: str, results: list[dict], auto: bool = False) -> dict:
    """The profile saved in the settings: the card, its driver, what the test found per operation and per stage."""
    return {"device": device, "card": card, "driver": driver, "auto": auto,
            "when": time.strftime("%Y-%m-%d %H:%M"), "ops": {r["op"]: r for r in results}, "stages": judge(results)}


def profile_of(profiles: dict | None, device: str | None) -> dict | None:
    """The saved profile of ``device`` ('auto': the one made under Automatic), or None."""
    return (profiles or {}).get(device or "auto")


def still_valid(profile: dict | None, card: str, driver: str) -> bool:
    """Whether a saved profile is for this card and this driver (else the card is tested again)."""
    return bool(profile) and profile.get("card") == card and profile.get("driver") == driver and bool(card)


def stage_reason(profile: dict, stage: str) -> str:
    """Why a stage went to the CPU (English, one of REASONS), '' when it kept the card."""
    if (profile.get("stages") or {}).get(stage) != "cpu":
        return ""
    for r in (profile.get("ops") or {}).values():
        if r.get("stage") == stage and r.get("verdict") not in (OK, ABSENT):
            return REASONS.get(r.get("verdict"), REASONS[ERROR])
    return REASONS[ERROR]


def stage_on_cpu(profiles: dict | None, device: str | None, stage: str) -> bool:
    """Whether the profile of ``device`` sends ``stage`` to the CPU."""
    p = profile_of(profiles, device)
    return bool(p) and (p.get("stages") or {}).get(stage) == "cpu"


def cpu_jobs(profile: dict | None, plan: Any) -> dict[str, str]:
    """{key: why} of the jobs of a plan that the profile sends to the CPU (their stage has 'cpu' in it)."""
    if not profile:
        return {}
    from .workflow import base
    out: dict[str, str] = {}
    for j in plan.jobs:
        why = stage_reason(profile, base(j.stage))
        if why:
            out[j.key] = STAGE_ON_CPU.format(why=why)
    return out


# ---------------------------------------------------------------------------------------------- the card
def _short(name: str) -> str:
    return " ".join(w for w in name.split() if w not in ("GeForce", "NVIDIA"))


def driver_version(device: str, name: str) -> str:
    """The driver of a card: what it says of itself, else the system; '' when not known. Runs small programs: not in
    the GUI thread."""
    kw: dict[str, Any] = {"capture_output": True, "text": True, "timeout": 15,
                          "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    if device.startswith("cuda"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], **kw).stdout
            for line in out.splitlines():
                n, _, v = line.rpartition(",")
                if v.strip() and n.strip() and _short(n.strip()) == _short(name):
                    return v.strip()
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    elif device.startswith("xpu"):
        try:
            import torch
            v = getattr(torch.xpu.get_device_properties(device), "driver_version", "")
            if v:
                return str(v)
        except Exception:  # noqa: BLE001 - not known
            pass
    if sys.platform == "win32":
        try:
            ps = ("Get-CimInstance Win32_VideoController | "
                  "ForEach-Object { $_.Name + '|' + $_.DriverVersion }")
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], **kw).stdout
            for line in out.splitlines():
                n, _, v = line.partition("|")
                if v.strip() and (n.strip() == name or _short(n.strip()) == _short(name)):
                    return v.strip()
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return ""


def card_identity(device: str) -> tuple[str, str]:
    """(the card's name, its driver) as a profile keeps them; ('', '') when this machine has no such device."""
    info = next((d for d in devices.list_devices() if d.device == device), None)
    if info is None:
        return "", ""
    return _short(info.name), driver_version(device, info.name)


def card_busy(device: str) -> bool:
    """Whether something else is using the card's memory: a test started by itself then waits for a quieter moment."""
    free, total = devices.free_memory(device), devices.total_memory(device)
    return free is not None and bool(total) and free < BUSY_FREE_SHARE * total


# ---------------------------------------------------------------------------------------------- the parent side
def run_test(device: str, *, on_step: Callable[[str, dict], None] | None = None,
             on_proc: Callable[[Any], None] | None = None, python: str | None = None) -> list[dict]:
    """Run the operations on ``device`` in a child process: the list of their results, a failed one for what the
    child did not finish (it crashed or hung: the card did). Raises RuntimeError when the child could not even start."""
    from .runner2 import device_env
    visible, inside = device_env(device, os.environ.get("CUDA_VISIBLE_DEVICES"))
    if inside == "cpu":
        return []
    root = str(Path(__file__).resolve().parents[2])
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible, "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [python or sys.executable, "-m", "astro_gpu.core.selftest", inside]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, env=env,
                            cwd=root, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if on_proc is not None:
        on_proc(proc)
    lines: queue.Queue[str | None] = queue.Queue()

    def reader():
        for line in proc.stdout:                       # type: ignore[union-attr]
            lines.put(line)
        lines.put(None)

    threading.Thread(target=reader, name="selftest-reader", daemon=True).start()
    ready, got, why = False, {}, "the card test stopped: the card may have crashed"
    while len(got) < len(OPERATIONS):
        try:
            line = lines.get(timeout=OP_TIMEOUT)
        except queue.Empty:
            why = f"the card test stopped: no answer from the card in {int(OP_TIMEOUT)} s"
            break
        if line is None:
            break
        try:
            r = json.loads(line) if line.lstrip().startswith("{") else None
        except ValueError:
            r = None
        if not isinstance(r, dict):
            continue
        if r.get("ready"):
            ready = True
        elif r.get("op"):
            got[r["op"]] = r
            if on_step is not None:
                on_step(r["op"], r)
    if proc.poll() is None:
        proc.kill()
    if not ready:
        raise RuntimeError("the card test could not start")
    return [got.get(n) or {"op": n, "stage": s, "verdict": ERROR, "card_s": None, "cpu_s": None, "diff": None,
                           "tolerance": TOLERANCE.get(n), "error": why}
            for n, s in OPERATIONS.items()]


# ---------------------------------------------------------------------------------------------- the child process
def main(argv: list[str] | None = None) -> int:
    device = (argv or sys.argv[1:] or ["cpu"])[0]
    print(json.dumps({"ready": True}), flush=True)
    for name in OPERATIONS:
        print(json.dumps(check(name, device)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
