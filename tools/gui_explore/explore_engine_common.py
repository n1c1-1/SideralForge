"""Engine exploration (area engine_shapes): helpers shared by the explore_engine_*.py scripts.

CPU only, tiny synthetic data, throw-away projects under F:/astro_essais/bench_engine_explore. Nothing of the user's is
read or written: ASTRO_GPU_HOME (the timing history, the caches) points to a private folder too.

Every scenario runs in its own process under a hard timeout (``drive``): a scenario is a function of a module that
returns its findings, a list of {"kind": "fail"|"warn"|"info", "what": ..., "detail": ...}.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
BENCH = Path(os.environ.get("TEMP", "C:/Temp")) / "engine_explore_bench"
BENCH.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("ASTRO_GPU_HOME", str(BENCH / "_home"))
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

OUT = Path("F:/astro_essais/A_REGARDER/exploration2/engine_shapes")
RESULTS = Path("F:/astro_essais/bench_engine_explore/_results")
PY = "C:/Users/Nicolas/sandbox/ComfyUI/.venv/Scripts/python.exe"


# ---------------------------------------------------------------------------------------------- notes
def note(severity: str, title: str, detail: str = "", steps: list | None = None) -> None:
    """An anomaly for the user (written by hand after a finding was reproduced twice)."""
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"area": "engine_shapes", "severity": severity, "title": title, "detail": detail, "steps": steps or [],
           "shot": "", "lang": "fr", "t": time.strftime("%H:%M:%S")}
    with (OUT / "anomalies.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------------------------- data
_N = 0


def fresh(name: str) -> Path:
    """A new, empty folder for one scenario run (never reused: a suffix of the time)."""
    global _N
    _N += 1
    d = BENCH / "runs" / f"{name}_{time.strftime('%H%M%S')}_{os.getpid()}_{_N}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def linear_fits(folder: Path, name: str = "stack.fits", seed: int = 1, h: int = 120, w: int = 160,
                stars: int = 40) -> Path:
    """A small linear stack (3, h, w): a sky gradient, noise and a few stars."""
    from astropy.io import fits
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([0.1 + 0.0004 * xx + 0.0002 * yy + c * 0.01 for c in range(3)])
    img = img + rng.normal(0, 0.002, (3, h, w))
    for _ in range(stars):
        y0, x0, a = rng.uniform(5, h - 5), rng.uniform(5, w - 5), rng.uniform(0.02, 0.5)
        img += a * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * 1.3 ** 2))[None]
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / name
    fits.writeto(f, img.astype(np.float32), overwrite=True)
    return f


def stretched_png(folder: Path, name: str = "stretched.png", seed: int = 2, h: int = 120, w: int = 160) -> Path:
    from PIL import Image
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = 0.15 + rng.normal(0, 0.02, (h, w, 3))
    for _ in range(30):
        y0, x0, a = rng.uniform(5, h - 5), rng.uniform(5, w - 5), rng.uniform(0.2, 0.8)
        img += a * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * 1.5 ** 2))[..., None]
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / name
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(f)
    return f


def bayer_frames(folder: Path, n: int = 8, h: int = 200, w: int = 300, seed: int = 3, nstars: int = 90,
                 kind: str = "light", start_hour: int = 22) -> Path:
    """Tiny Bayer (RGGB) FITS frames, uint16: sky, noise, stars that drift by a few pixels from frame to frame."""
    from astropy.io import fits
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    sy, sx = rng.uniform(8, h - 8, nstars), rng.uniform(8, w - 8, nstars)
    flux = rng.uniform(800, 20000, nstars)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for i in range(n):
        if kind == "light":
            dy, dx = rng.uniform(-3, 3, 2)
            img = 1000.0 + 0.5 * xx + rng.normal(0, 15, (h, w))
            for y0, x0, a in zip(sy + dy, sx + dx, flux):
                img += a * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * 1.4 ** 2))
        elif kind == "dark":
            img = 1000.0 + rng.normal(0, 10, (h, w))
        elif kind == "flat":
            img = 20000.0 * (1 - 0.1 * ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (h * h)) + rng.normal(0, 50, (h, w))
        else:
            img = 1000.0 + rng.normal(0, 5, (h, w))
        hd = fits.Header()
        hd["BAYERPAT"] = "RGGB"
        hd["EXPTIME"] = 60.0 if kind in ("light", "dark") else 1.0
        hd["GAIN"] = 100
        hd["DATE-OBS"] = f"2026-10-0{1 + seed % 8}T{start_hour + i // 60:02d}:{i % 60:02d}:00"
        hd["INSTRUME"] = "SYNTH"
        hd["IMAGETYP"] = kind.upper()
        fits.writeto(folder / f"{kind}_{i:03d}.fits", np.clip(img, 0, 65535).astype(np.uint16), hd, overwrite=True)
    return folder


# ---------------------------------------------------------------------------------------------- checks
class Findings(list):
    def fail(self, what: str, detail: str = "") -> None:
        self.append({"kind": "fail", "what": what, "detail": detail})

    def warn(self, what: str, detail: str = "") -> None:
        self.append({"kind": "warn", "what": what, "detail": detail})

    def info(self, what: str, detail: str = "") -> None:
        self.append({"kind": "info", "what": what, "detail": detail})


def plan_problems(p, plan, keys=None) -> list[str]:
    """What is wrong with a plan's shape: a key twice, a job before one it reads, a job's stage not in the chain of
    one of its workflows, a non-finite or negative duration."""
    from astro_gpu.core.store import Keys
    k = keys or Keys(p)
    out = []
    seen = {}
    for i, j in enumerate(plan.jobs):
        if j.key in seen:
            out.append(f"job {j.stage} key {j.key} planned twice")
        seen[j.key] = i
        if not np.isfinite(j.est_s) or j.est_s < 0 or not np.isfinite(j.disk_gb) or j.disk_gb < 0:
            out.append(f"job {j.stage}: duration {j.est_s} / disk {j.disk_gb}")
    for i, j in enumerate(plan.jobs):
        for d, dk in j.deps.items():
            if dk in seen and seen[dk] > i:
                out.append(f"job {j.stage} comes before {d}, which it reads")
        for wid in j.wf:
            try:
                ch = k.chain(wid)
            except Exception as e:  # noqa: BLE001
                out.append(f"job {j.stage}: workflow {wid} has no chain ({e})")
                continue
            if j.stage not in ch:
                out.append(f"job {j.stage} lists workflow {wid}, whose chain has no such stage {ch}")
            elif k.key(wid, j.stage) != j.key:
                out.append(f"job {j.stage} lists workflow {wid}, whose key differs")
    # the same stage computed twice with the same key under two names would be a waste: keys are unique anyway
    by_stage_key = {}
    for j in plan.jobs:
        by_stage_key.setdefault((j.key,), []).append(j.stage)
    return out


def run_and_check(p, f: Findings, label: str, expect: str = "done", plan=None) -> dict:
    """Plan everything, run in this process on the CPU, then check the store: every stage of every workflow done with
    its key, no _absent, no .tmp left, a second plan empty, each key computed once (the events)."""
    from astro_gpu.core import project2, runner2
    from astro_gpu.core.store import Keys, Store
    project2.save(p)
    plan = plan if plan is not None else runner2.plan_all(p)
    for e in plan_problems(p, plan):
        f.fail(f"{label}: plan shape", e)
    if plan.skipped:
        f.info(f"{label}: skipped", json.dumps(plan.skipped))
    t0 = time.time()
    info = runner2.prepare_run(p, plan, "cpu")
    status = runner2.run_plan(p.root, info.plan_file, echo=False)
    dt = time.time() - t0
    evs, _ = runner2.read_events(p.root, info.offset, info.run_id)
    errors = [e for e in evs if e.get("type") == "stage_end" and e.get("status") == "failed"]
    res = {"status": status, "seconds": round(dt, 1), "jobs": [j.stage for j in plan.jobs],
           "errors": [(e.get("stage"), e.get("error")) for e in errors]
           + [(e.get("stage"), e.get("msg")) for e in evs if e.get("type") == "log" and e.get("level") == "error"]}
    if status != expect:
        f.fail(f"{label}: run status {status} (expected {expect})",
               json.dumps(res["errors"], ensure_ascii=False)[:3000])
    # each key computed once
    ran = [e.get("key") for e in evs if e.get("type") == "stage_end" and e.get("status") == "done"
           and not e.get("reused")]
    dup = sorted({x for x in ran if x and ran.count(x) > 1})
    if dup:
        f.fail(f"{label}: a key computed twice in one run", str(dup))
    starts = sum(1 for e in evs if e.get("type") == "stage_start")
    res["stage_starts"] = starts
    if starts > len(plan.jobs):
        f.fail(f"{label}: more stages started than jobs planned", f"{starts} > {len(plan.jobs)}")
    q = project2.load(p.root)
    k, store = Keys(q), Store(q.root)
    if status == "done":
        for w in q.workflows:
            for s in k.chain(w):
                key = k.key(w, s)
                m = store.manifest(s, key)
                if m.get("status") != "done" or m.get("key") != key:
                    f.fail(f"{label}: workflow {w.name} stage {s} not done with its key",
                           f"manifest status {m.get('status')} key {m.get('key')} expected {key}; dir "
                           f"{store.dir(s, key)}")
        again = runner2.plan_all(q)
        if len(again):
            f.fail(f"{label}: a second plan after a full run is not empty", str([(j.stage, j.key) for j in again.jobs]))
    if (q.root / "store" / "_absent").exists():
        f.fail(f"{label}: store/_absent was created", str(list((q.root / "store" / "_absent").rglob("*"))[:10]))
    tmps = [str(x.relative_to(q.root)) for x in (q.root / "store").glob("*/*.tmp")]
    if tmps and status == "done":
        f.fail(f"{label}: .tmp folders left after a done run", str(tmps))
    res["tmps"] = tmps
    return res


def keys_of(p) -> dict:
    from astro_gpu.core.store import Keys
    k = Keys(p)
    return {w.id: k.keys(w.id) for w in p.workflows}


# ---------------------------------------------------------------------------------------------- the driver
def main_one(module_globals: dict) -> None:
    """``python explore_engine_x.py <scenario>``: run one scenario, write its findings as JSON."""
    name = sys.argv[1]
    fn = module_globals[name]
    f = Findings()
    t0 = time.time()
    try:
        extra = fn(f)
    except Exception as e:  # noqa: BLE001
        f.fail(f"{name}: exception", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}")
        extra = None
    out = {"scenario": name, "seconds": round(time.time() - t0, 1), "findings": f, "extra": extra}
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("RESULT " + json.dumps(out, ensure_ascii=False, default=str), flush=True)


def drive(script: str, scenarios: list[str], timeout: float = 120.0, tag: str = "") -> list[dict]:
    """Each scenario in its own process (CPU only), killed with its children past ``timeout``."""
    import subprocess

    import psutil
    out = []
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"}
    for s in scenarios:
        t0 = time.time()
        proc = subprocess.Popen([PY, script, s], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                cwd=str(ROOT))
        try:
            text, _ = proc.communicate(timeout=timeout)
            text = text.decode("utf-8", "replace")
            line = next((x for x in text.splitlines() if x.startswith("RESULT ")), None)
            r = json.loads(line[7:]) if line else {"scenario": s, "findings": [
                {"kind": "fail", "what": f"{s}: no result", "detail": text[-3000:]}]}
        except subprocess.TimeoutExpired:
            try:
                for c in psutil.Process(proc.pid).children(recursive=True):
                    c.kill()
            except Exception:  # noqa: BLE001
                pass
            proc.kill()
            proc.communicate()
            r = {"scenario": s, "findings": [{"kind": "timeout", "what": f"{s}: killed after {timeout:.0f} s",
                                              "detail": ""}]}
        r["wall"] = round(time.time() - t0, 1)
        out.append(r)
        bad = [x for x in r["findings"] if x["kind"] in ("fail", "timeout")]
        print(f"{s:50s} {r['wall']:6.1f}s  {'OK' if not bad else 'FAIL ' + str(len(bad))}", flush=True)
        for x in bad:
            print("   -", x["what"], "|", x["detail"][:400].replace("\n", " / "), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{Path(script).stem}{tag}_{time.strftime('%H%M%S')}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out
