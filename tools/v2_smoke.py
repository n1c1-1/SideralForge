"""The v2 path against v1, end to end, on the heart_a7s test set (tests/test_integration_v2.py, slow).

A fresh v2 project is built under ``F:/astro_essais/p7/`` from the raw frames of ``F:/astro_essais/donnees/heart_a7s``
with the parameters of the v1 test project ``F:/astro_essais/projets/heart_a7s`` (denoise 6000 iterations). It is
driven through ``astro_gpu.api`` as the client does: create, scan, add frames, workflows, plan, start_run, follow
``read_events``. The results are then compared with v1's, file by file and bit for bit.

    python tools/v2_smoke.py all            # every phase below, in order (hours: run it in the background)
    python tools/v2_smoke.py setup          # the project and its workflows Full and Classic stack
    python tools/v2_smoke.py plan           # the plan and the sharing checks (nothing runs)
    python tools/v2_smoke.py run            # run (or re-attach to) the planned jobs, follow the events
    python tools/v2_smoke.py extras         # add Classic + denoise and the Raw stack exit, plan, run
    python tools/v2_smoke.py compare        # stack, denoise, render_prep (and the others) against v1, bit for bit
    python tools/v2_smoke.py render         # render_layer against RenderEngine, previews, sanity of the images
    python tools/v2_smoke.py reproduce      # v1's results imported; stages recomputed by v2 from v1's own inputs
    python tools/v2_smoke.py sep            # sep.extract three times on one image: not the same objects (the root cause)
    python tools/v2_smoke.py v1run          # v1 on a COPY of the test project (the disambiguation run)
    python tools/v2_smoke.py report         # print report.json

Options (before the phase): ``--root DIR`` (default ``$ASTRO_TESTS/p7``), ``--against v1copy`` (compare with the run of
``v1run`` instead of the existing v1 results), ``--device cuda:0``.

Rules. The v1 test project is read only (its files are listed before and after). The run's ``ASTRO_GPU_HOME`` is
``<root>/etat`` (a copy of the camera profiles): nothing of the user's state is written. The numeric helpers at the
top need only numpy (tests/test_integration_v2.py uses them on toy stages); everything else imports the engine on
first use, after ASTRO_GPU_HOME is set.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

TESTS = Path(os.environ.get("ASTRO_TESTS", "F:/astro_essais"))
DATA = TESTS / "donnees" / "heart_a7s"
V1_PROJECT = TESTS / "projets" / "heart_a7s"
V1_PROFILES = TESTS / "etat" / "camera_profiles"
DEFAULT_ROOT = TESTS / "p7"
DENOISE_ITERATIONS = 6000                      # the v1 test project's setting
#: bit for bit in the full chain (everything upstream of them is bit for bit too)
BIT_FOR_BIT = ("stack", "denoise")
#: bit for bit when recomputed by the v2 runner from v1's own inputs. In the one full-chain run the astrometry's WCS came
#: out with other last digits (CRPIX 1e-13, SIP terms 1e-8 relative: reproduce.astrometry gives v1's file again from
#: the same stack), the catalogues, the colour matrix and the PSF fit follow at that level, and the photometry of
#: stars amplifies it; sep.extract, which astrometry and optics use, does not return the same objects twice (phase
#: ``sep``). So the full chain is judged on the final picture, within IMAGE_TOLERANCE
ON_V1_INPUTS = ("render_prep",)
REQUIRED_STAGES = BIT_FOR_BIT + ON_V1_INPUTS
#: the final image of the full chain (v2's render_prep) against v1's, same settings, in levels of 255 and dB:
#: measured 34.4 dB, median 3.2, 99th percentile 12.5 (faithful, 1/4); the tolerance leaves a margin
IMAGE_TOLERANCE = {"psnr_db_min": 30.0, "p50_max": 6.0, "p99_max": 20.0}
FULL_STAGES = ("inventory", "calibrate", "register", "stack", "astrometry", "catalogs", "surveys", "color",
               "denoise", "optics", "stars", "starless", "background", "render_prep")
#: the stages Classic stack shares with Full (DESIGN 2.1: inventory to color; surveys is the H-alpha gradient's)
SHARED_WITH_CLASSIC = ("inventory", "calibrate", "register", "stack", "astrometry", "catalogs", "color")
#: the stages recomputed from v1's own inputs to tell which stage makes the chain differ (shortest first, the
#: long photometry and the network-bound astrometry last)
REPRODUCE_STAGES = ("render_prep", "background", "color", "optics", "starless", "stars", "astrometry")


# ================================================================================================== comparing files
# (numpy only)

CHUNK = 1 << 24                                 # elements per step of an array comparison
SKIP_DIRS = {"previews", "_ckpt"}
SKIP_FILES = {"manifest.json"}                  # run ids, dates, key fields: compared field by field instead


def array_diff(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """How two arrays of one shape differ: how many elements, the largest and the RMS difference (absolute, and
    relative to the RMS of ``a``), the largest in units in the last place for float32, and where the largest is."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"equal": False, "why": f"shape/dtype {a.shape} {a.dtype} against {b.shape} {b.dtype}"}
    fa, fb = a.reshape(-1), b.reshape(-1)
    n, ndiff, maxabs, sq, sqa, where = fa.size, 0, 0.0, 0.0, 0.0, 0
    nan_mismatch, maxulp, ulp_gt4 = 0, 0, 0
    for i in range(0, n, CHUNK):
        x, y = fa[i:i + CHUNK], fb[i:i + CHUNK]
        if x.dtype.kind == "f":
            nx, ny = np.isnan(x), np.isnan(y)
            nan_mismatch += int(np.count_nonzero(nx != ny))
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            d[x == y] = 0
            d[nx & ny] = 0
            d[nx != ny] = np.inf
            if x.dtype == np.float32:
                xi, yi = x.view(np.int32).astype(np.int64), y.view(np.int32).astype(np.int64)
                same_sign = (xi >= 0) == (yi >= 0)
                u = np.where(same_sign, np.abs(xi - yi), 1 << 40)
                u[(d == 0)] = 0
                maxulp = max(maxulp, int(u.max())) if u.size else maxulp
                ulp_gt4 += int(np.count_nonzero(u > 4))
        elif x.dtype.kind in "iub":
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        else:
            d = (x != y).astype(np.float64)
        k = int(np.count_nonzero(d))
        if k:
            ndiff += k
            m = float(d.max())
            if m > maxabs:
                maxabs, where = m, i + int(d.argmax())
            fin = np.where(np.isfinite(d), d, 0)
            sq += float(np.sum(fin * fin))
        if x.dtype.kind in "fiub":
            xf = x.astype(np.float64)
            sqa += float(np.sum(np.where(np.isfinite(xf), xf * xf, 0)))
    out: dict[str, Any] = {"equal": ndiff == 0, "elements": n, "differing": ndiff,
                           "differing_pct": round(100 * ndiff / max(n, 1), 4)}
    if ndiff:
        rms_a = (sqa / max(n, 1)) ** 0.5
        out.update(max_abs=maxabs, rms_diff=(sq / max(n, 1)) ** 0.5, rms_ref=rms_a,
                   rel_rms=((sq / max(n, 1)) ** 0.5) / rms_a if rms_a else None,
                   at=[int(v) for v in np.unravel_index(where, a.shape)], nan_mismatch=nan_mismatch)
        if a.dtype == np.float32:
            out.update(max_ulp=maxulp, elements_over_4_ulp=ulp_gt4)
    return out


def files_equal(a: Path, b: Path) -> bool:
    """Whether two files hold the same bytes."""
    if a.stat().st_size != b.stat().st_size:
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            x, y = fa.read(1 << 24), fb.read(1 << 24)
            if x != y:
                return False
            if not x:
                return True


def _json_diff(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'v2' if k in b else 'v1'}")
            else:
                out += _json_diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} against {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _json_diff(x, y, f"{path}[{i}]")
        return out[:20]
    return [] if a == b else [f"{path}: {str(a)[:60]} against {str(b)[:60]}"]


def _tensors_diff(a: Any, b: Any, path: str = "") -> list[str]:
    import torch
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return [f"{path}: {tuple(a.shape)} {a.dtype} against {tuple(b.shape)} {b.dtype}"]
        if torch.equal(a, b):
            return []
        d = (a.double() - b.double()).abs()
        return [f"{path}: {int((d > 0).sum())}/{a.numel()} differ, max {float(d.max()):.3g}"]
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'v2' if k in b else 'v1'}")
            else:
                out += _tensors_diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} against {len(b)}"]
        return [m for i, (x, y) in enumerate(zip(a, b)) for m in _tensors_diff(x, y, f"{path}[{i}]")]
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return [f"{path}: {type(a).__name__} against {type(b).__name__}"]
        d = array_diff(a, b)
        if d["equal"]:
            return []
        return [f"{path}: " + (d.get("why") or f"{d['differing']}/{d['elements']} differ, max {d['max_abs']:.3g}")]
    try:
        same = bool(a == b)
    except ValueError:                                  # another container that does not compare to a bool
        same = str(a) == str(b)
    return [] if same else [f"{path}: {str(a)[:40]} against {str(b)[:40]}"]


def compare_file(a: Path, b: Path) -> dict[str, Any]:
    """{'status': 'identical' | 'different', 'detail': ...} for two files of the same name (a = v1, b = v2)."""
    suf = a.suffix.lower()
    if suf == ".npy":
        if files_equal(a, b):
            return {"status": "identical"}
        d = array_diff(np.load(a), np.load(b))
        return {"status": "identical" if d["equal"] else "different", "detail": d}
    if suf == ".npz":
        za, zb = np.load(a), np.load(b)
        det: dict[str, Any] = {}
        for k in sorted(set(za.files) | set(zb.files)):
            if k not in za.files or k not in zb.files:
                det[k] = {"equal": False, "why": f"only in {'v2' if k in zb.files else 'v1'}"}
                continue
            d = array_diff(za[k], zb[k])
            if not d["equal"]:
                det[k] = d
        return {"status": "different" if det else "identical", **({"detail": det} if det else {})}
    if suf == ".json":
        diffs = _json_diff(json.loads(a.read_text(encoding="utf-8")), json.loads(b.read_text(encoding="utf-8")))
        return {"status": "different" if diffs else "identical", **({"detail": diffs[:10]} if diffs else {})}
    if suf == ".pt":
        import torch
        kw = {"map_location": "cpu", "weights_only": False}
        diffs = _tensors_diff(torch.load(a, **kw), torch.load(b, **kw))
        return {"status": "different" if diffs else "identical", **({"detail": diffs[:10]} if diffs else {})}
    if suf in (".fits", ".fit"):
        from astropy.io import fits
        with fits.open(a) as fa, fits.open(b) as fb:
            ha, hb = dict(fa[0].header), dict(fb[0].header)
            for h in (ha, hb):
                h.pop("DATE", None)
            diffs = _json_diff({k: str(v) for k, v in ha.items()}, {k: str(v) for k, v in hb.items()})
            da, db = fa[0].data, fb[0].data
            if (da is None) != (db is None):
                diffs.append("data present in only one")
            elif da is not None and not np.array_equal(da, db):
                diffs.append("data differ")
        return {"status": "different" if diffs else "identical", **({"detail": diffs[:10]} if diffs else {})}
    same = files_equal(a, b)
    return {"status": "identical" if same else "different", **({} if same else {"detail": "bytes differ"})}


def compare_dirs(a: Path, b: Path, skip_files: set[str] | None = None,
                 skip_content: set[str] | None = None) -> dict[str, dict[str, Any]]:
    """Every file of the two stage folders (a = v1, b = v2), by relative name; previews, checkpoints and manifests
    are not files of the result. ``skip_content``: names only checked for presence (paths that differ by design)."""
    skip_files = SKIP_FILES | set(skip_files or ())

    def names(d: Path) -> set[str]:
        out = set()
        for f in d.rglob("*"):
            rel = f.relative_to(d)
            if f.is_file() and not (set(rel.parts[:-1]) & SKIP_DIRS) and rel.name not in skip_files:
                out.add(rel.as_posix())
        return out
    na, nb = names(a), names(b)
    out: dict[str, dict[str, Any]] = {}
    for n in sorted(na | nb):
        if n not in nb:
            out[n] = {"status": "missing_v2"}
        elif n not in na:
            out[n] = {"status": "missing_v1"}
        elif n in (skip_content or ()):
            out[n] = {"status": "identical", "note": "presence only"}
        else:
            out[n] = compare_file(a / n, b / n)
    return out


def compare_manifests(a: dict, b: dict) -> list[str]:
    """What differs between two manifests that should not: the stage version, the metrics, the parameters (workers and
    device left out: they do not change the results; nor an empty ``exclude``, which v2's stack adds)."""
    def prm(m: dict) -> dict:
        return {k: v for k, v in (m.get("params") or {}).items()
                if k not in ("workers", "device") and not (k == "exclude" and not v)}
    out = []
    if str(a.get("version")) != str(b.get("version")):
        out.append(f"version {a.get('version')} against {b.get('version')}")
    out += ["metrics" + d for d in _json_diff(a.get("metrics") or {}, b.get("metrics") or {})]
    out += ["params" + d for d in _json_diff(prm(a), prm(b))]
    return out


def summarize(cmp: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ident = [n for n, c in cmp.items() if c["status"] == "identical"]
    diff = {n: c for n, c in cmp.items() if c["status"] != "identical"}
    return {"files": len(cmp), "identical": len(ident), "different": sorted(diff), "detail": diff}


def snapshot(root: Path) -> dict[str, list[int]]:
    """Every file under a folder with its size and modification time: to prove a read-only reference is untouched."""
    out = {}
    for f in sorted(root.rglob("*")):
        if f.is_file():
            st = f.stat()
            out[f.relative_to(root).as_posix()] = [st.st_size, st.st_mtime_ns]
    return out


# ================================================================================================== following a run

class Log:
    """Lines to the console and to a file, with the time."""

    def __init__(self, path: Path | None = None):
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def follow(api: Any, p: Any, info: Any, log: Log, poll: float = 5.0, grace: float = 240.0,
           max_s: float = 8 * 3600) -> dict[str, Any]:
    """Follow a run through ``api.read_events`` until its run_end: {'status', 'events', 'seconds'}. A worker that
    disappears without a run_end (no live lock after ``grace`` seconds of silence) is 'lost'."""
    offset, evs_all, t0, last_ev = info.offset, [], time.time(), time.time()
    seen_progress: dict[str, int] = {}
    status = None
    while True:
        evs, offset = api.read_events(p, offset, info.run_id)
        for e in evs:
            evs_all.append(e)
            last_ev = time.time()
            t = e.get("type")
            if t == "stage_start":
                log(f"  start  {e.get('stage')} (key {e.get('key')}, workflows {e.get('wf')})")
            elif t == "stage_end":
                log(f"  end    {e.get('stage')} {e.get('status')} in {float(e.get('duration') or 0):.1f} s"
                    + (f" -- {e.get('error')}" if e.get("error") else ""))
            elif t == "progress":
                s, fr = e.get("stage", ""), float(e.get("frac") or 0)
                step = int(fr * 10)
                if step > seen_progress.get(s, -1) and fr < 1:
                    seen_progress[s] = step
                    log(f"    {s} {100 * fr:.0f} % {e.get('msg') or ''}")
            elif t == "log" and e.get("level") in ("warning", "error"):
                log(f"  {e.get('level')}: [{e.get('stage')}] {str(e.get('msg'))[:200]}")
            elif t == "run_end":
                status = e.get("status")
                log(f"run end: {status} (done {len(e.get('done') or [])}, failed {len(e.get('failed') or [])}, "
                    f"skipped {len(e.get('skipped') or [])}, interrupted {len(e.get('interrupted') or [])}, "
                    f"not started {len(e.get('not_started') or [])})")
        if status is not None:
            break
        if api.running(p) is None and time.time() - last_ev > grace:
            status = "lost"
            log("the worker is gone without a run_end")
            break
        if time.time() - t0 > max_s:
            status = "timeout"
            break
        time.sleep(poll)
    return {"status": status, "events": evs_all, "seconds": time.time() - t0}


def timings_of(events: list[dict]) -> dict[str, dict[str, Any]]:
    """{key: {stage, wf, seconds, status, reused}} from the stage_end events; start and end times as seen."""
    out: dict[str, dict[str, Any]] = {}
    for e in events:
        if e.get("type") == "stage_start" and e.get("key"):
            out.setdefault(e["key"], {})["start"] = e["t"]
        if e.get("type") == "stage_end" and e.get("key"):
            out.setdefault(e["key"], {}).update(stage=e.get("stage"), wf=e.get("wf"), status=e.get("status"),
                                                seconds=float(e.get("duration") or 0.0), end=e["t"],
                                                reused=bool(e.get("reused")))
    return out


# ================================================================================================== the smoke test

class Smoke:
    """The phases. ``Smoke(root).prepare()`` sets the pilot ASTRO_GPU_HOME and imports the engine."""

    def __init__(self, root: Path, device: str = "cuda:0", against: str = "v1"):
        self.root = Path(root)
        self.device = device
        self.against = against
        self.proj = self.root / "heart_a7s"
        self.report_file = self.root / "report.json"
        self.log = Log(self.root / "logs" / "smoke.log")
        self.api: Any = None
        self.p: Any = None
        self.checks: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ plumbing
    def prepare(self) -> None:
        """The pilot ASTRO_GPU_HOME (a copy of the camera profiles) before anything of astro_gpu is imported."""
        if "astro_gpu.core.project" in sys.modules:
            home = Path(os.environ.get("ASTRO_GPU_HOME") or "")
            if home.resolve() != (self.root / "etat").resolve():
                raise RuntimeError("astro_gpu is already imported with another ASTRO_GPU_HOME: run this as a script")
        home = self.root / "etat"
        (home / "camera_profiles").mkdir(parents=True, exist_ok=True)
        for f in V1_PROFILES.glob("*.npz"):
            if not (home / "camera_profiles" / f.name).exists():
                shutil.copy2(f, home / "camera_profiles" / f.name)
        os.environ["ASTRO_GPU_HOME"] = str(home)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from astro_gpu import api
        self.api = api

    def report(self) -> dict[str, Any]:
        try:
            return json.loads(self.report_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save(self, section: str, value: Any) -> None:
        rep = self.report()
        rep[section] = value
        self._write(rep)

    def _write(self, rep: dict[str, Any]) -> None:
        rep["updated"] = datetime.now().isoformat(timespec="seconds")
        tmp = self.report_file.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, self.report_file)

    def check(self, name: str, ok: bool, detail: str = "", required: bool = True) -> bool:
        c = {"name": name, "ok": bool(ok), "detail": detail[:600], "required": required}
        self.checks.append(c)
        self.log(f"[{'ok' if ok else ('FAIL' if required else 'note')}] {name}" + (f": {detail[:400]}" if detail else ""))
        rep = self.report()
        rep.setdefault("checks", {})[name] = c
        self._write(rep)
        return bool(ok)

    def open(self) -> Any:
        self.p = self.api.open_project(self.proj)
        return self.p

    def wf(self, name: str) -> int:
        for w in self.p.workflows:
            if w.name == name:
                return w.id
        raise KeyError(name)

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        api = self.api
        if (self.proj / "project.astroproj").exists():
            self.open()
            self.log(f"project exists: {self.proj} ({len(self.p.workflows)} workflows)")
            return
        self.log(f"creating {self.proj}")
        p = api.create_project(self.proj, "heart_a7s (P7)", {"name": "IC 1805"})
        scan = api.scan_folder(DATA / "lights")
        self.log(f"scan: { {k: len(v) for k, v in scan.frames.items()} }, errors {len(scan.errors)}")
        api.add_frames(p, scan)
        p.scratch_dir = str(self.root / "scratch" / "heart_a7s")
        p.devices["main"] = self.device
        p.description = "P7: the v2 path against v1 on heart_a7s (parameters of F:/astro_essais/projets/heart_a7s)"
        full = api.add_workflow(p, "Full", "Full")
        api.set_param(p, [full.id], "denoise", "iterations", DENOISE_ITERATIONS)
        api.add_workflow(p, "Classic stack", "Classic stack")
        api.save(p)
        self.p = p
        s = p.sessions
        self.check("sessions are the folders of the v1 project",
                   len(s) == 1 and [Path(x).as_posix().lower() for x in s[0].lights] == [(DATA / "lights").as_posix().lower()]
                   and len(s[0].darks) == 1 and len(s[0].flats) == 1,
                   f"{[(x.name, x.lights, x.darks, x.flats) for x in s]}")

    def add_extras(self) -> None:
        api, p = self.api, self.p
        names = {w.name for w in p.workflows}
        if "Classic + denoise" not in names:
            w = api.add_workflow(p, "Classic + denoise", "Classic + denoise")
            api.set_param(p, [w.id], "denoise", "iterations", DENOISE_ITERATIONS)
        if "Raw stack" not in names:
            api.add_exit(p, self.wf("Classic stack"), "stacking", "Raw stack")
        api.save(p)

    # ------------------------------------------------------------------ plan and sharing
    def plan_checks(self, plan: Any, tag: str) -> dict[str, Any]:
        """The plan has each key once, and the stages the workflows share are one job each with the same key."""
        api, p = self.api, self.p
        keys = [j.key for j in plan.jobs]
        self.check(f"{tag}: each job key once", len(keys) == len(set(keys)), f"{len(keys)} jobs")
        by_wf = {w.id: {s: api.stage_key(p, w.id, s) for s in api.chain(p, w.id)} for w in p.workflows}
        distinct = {(s, k) for ks in by_wf.values() for s, k in ks.items()}
        done = {(s, k) for (s, k) in distinct if api._store(p).done(s, k)}
        self.check(f"{tag}: jobs are the distinct keys not done", {(j.stage, j.key) for j in plan.jobs} == distinct - done,
                   f"{len(distinct)} distinct keys, {len(done)} done, {len(plan.jobs)} jobs")
        shared = {}
        for j in plan.jobs:
            if len(j.wf) > 1:
                shared[j.stage] = j.wf
        out = {"jobs": len(plan.jobs), "distinct_keys": len(distinct), "shared": shared,
               "estimate_s": round(plan.seconds), "measured": plan.measured, "disk_gb": round(plan.disk_gb, 2),
               "skipped": plan.skipped}
        return out

    def sharing_checks(self) -> dict[str, Any]:
        """Full and Classic stack: the common stages have one key each, and the others differ."""
        api = self.api
        full, classic = self.wf("Full"), self.wf("Classic stack")
        cf, cc = api.chain(self.p, full), api.chain(self.p, classic)
        same = [s for s in cf if s in cc and api.stage_key(self.p, full, s) == api.stage_key(self.p, classic, s)]
        diff = [s for s in cf if s in cc and s not in same]
        self.check("Full and Classic stack share inventory to color, one key each", tuple(same) == SHARED_WITH_CLASSIC
                   and not diff, f"shared {same}, same stage with another key {diff}")
        def sharers(stage: str) -> list[int]:               # every workflow of the project with this stage and this key
            key = api.stage_key(self.p, full, stage)
            return sorted(w.id for w in self.p.workflows
                          if stage in api.chain(self.p, w.id) and api.stage_key(self.p, w.id, stage) == key)
        self.check("sharing() names Full and Classic stack (and any other workflow with that key) for a shared stage",
                   all(full in sharers(s) and classic in sharers(s) and api.sharing(self.p, full, s) == sharers(s)
                       for s in same), f"stack: {api.sharing(self.p, full, 'stack')}")
        self.check("the output stages are not shared",
                   api.stage_key(self.p, full, "render_prep") != api.stage_key(self.p, classic, "render_simple")
                   and api.sharing(self.p, classic, "render_simple") == [classic])
        return {"chain_full": cf, "chain_classic": cc, "shared": same,
                "keys_full": {s: api.stage_key(self.p, full, s) for s in cf},
                "keys_classic": {s: api.stage_key(self.p, classic, s) for s in cc}}

    def plan_phase(self) -> Any:
        api, p = self.api, self.p
        plan = api.plan_all(p, self.device)
        for j in plan.jobs:
            self.log(f"  job {j.stage:<15} wf {j.wf} est {j.est_s:7.1f} s{'' if j.measured else ' (approx)'} {j.disk_gb:5.2f} GB")
        info = self.plan_checks(plan, "plan")
        if [w.name for w in p.workflows[:2]] == ["Full", "Classic stack"]:
            info["sharing"] = self.sharing_checks()
            fresh = not self.api._store(p).done("inventory", self.api.stage_key(p, self.wf("Full"), "inventory"))
            if fresh:
                self.check("the first plan: Full's 14 stages + 2 of Classic (background_poly, render_simple)",
                           info["jobs"] == 16 and set(info["shared"]) == set(SHARED_WITH_CLASSIC),
                           f"{info['jobs']} jobs, shared stages {sorted(info['shared'])}")
        self.log(f"plan: {len(plan)} jobs, about {plan.seconds / 60:.0f} min{'' if plan.measured else ' (approximate)'}, "
                 f"{plan.disk_gb:.1f} GB")
        return plan

    # ------------------------------------------------------------------ run
    def run_phase(self, tag: str, plan: Any | None = None) -> dict[str, Any] | None:
        """Run the plan (the current one by default), or re-attach to a live run. Records the timings."""
        api, p = self.api, self.p
        live = api.running(p)
        if live is not None:
            self.log(f"re-attaching to the run {live.run_id} (pid {live.pid})")
            info = live
            planned = {j["key"]: j for j in json.loads(Path(live.plan_file).read_text(encoding="utf-8")).get("jobs", [])}
        else:
            plan = plan or api.plan_all(p, self.device)
            if not plan:
                self.log(f"{tag}: nothing to run")
                return None
            planned = {j.key: j.to_dict() for j in plan.jobs}
            info = api.start_run(p, plan, self.device)
            self.log(f"run {info.run_id} started, pid {info.pid}, device {info.device}, {len(plan)} jobs")
            for n in info.notes:
                self.log(f"  note: {n}")
        res = follow(api, p, info, self.log)
        api.refresh(p)
        tm = timings_of(res["events"])
        rows = []
        for k, j in planned.items():
            t = tm.get(k, {})
            rows.append({"key": k, "stage": j["stage"], "wf": j["wf"], "estimate_s": j["est_s"],
                         "estimate_measured": j["measured"], "seconds": t.get("seconds"), "status": t.get("status")})
        data = {"run_id": info.run_id, "status": res["status"], "wall_s": round(res["seconds"], 1), "jobs": rows,
                "plan_estimate_s": round(sum(j["est_s"] for j in planned.values()), 1)}
        self.save(f"run_{tag}", data)
        starts = [e["key"] for e in res["events"] if e.get("type") == "stage_start" and e.get("key")]
        self.check(f"{tag}: one stage_start per job key, events carry run, key and wf",
                   sorted(starts) == sorted(planned) and all(e.get("run") == info.run_id for e in res["events"])
                   and all(e.get("wf") for e in res["events"] if e.get("type") == "stage_start"),
                   f"{len(starts)} starts for {len(planned)} jobs")
        pe = [e for e in res["events"] if e.get("type") == "plan"]
        self.check(f"{tag}: the plan event lists the planned jobs in order",
                   bool(pe) and [j["key"] for j in pe[0]["jobs"]] == list(planned), "")
        ok = self.check(f"{tag}: the run ends done, every job done", res["status"] == "done"
                        and all(r["status"] == "done" for r in rows), f"{res['status']}: "
                        f"{[(r['stage'], r['status']) for r in rows if r['status'] != 'done']}")
        st = api.states(p)
        bad = {k: v.state for k, v in st.items() if v.state != "done"}
        self.check(f"{tag}: every stage of every workflow is done", not bad, str(bad))
        again = api.plan_all(p, self.device)
        self.check(f"{tag}: a second plan is empty", len(again) == 0 and api.cost(p).jobs == 0,
                   f"{len(again)} jobs: {[j.stage for j in again.jobs]}")
        return data if ok else data

    def extras_phase(self) -> None:
        """Classic + denoise (shares Full's denoise) and the Raw stack exit: only their own stages are planned. A
        project that already has them has nothing left to plan or run."""
        fresh = "Classic + denoise" not in {w.name for w in self.p.workflows}
        self.add_extras()
        plan = self.api.plan_all(self.p, self.device)
        w3, w4 = self.wf("Classic + denoise"), self.wf("Raw stack")
        self.check("extras: Classic + denoise shares Full's denoise key",
                   self.api.stage_key(self.p, w3, "denoise") == self.api.stage_key(self.p, self.wf("Full"), "denoise"))
        if fresh:
            stages = sorted((j.stage, tuple(j.wf)) for j in plan.jobs)
            self.check("extras: only the new jobs are planned (denoise is Full's)",
                       stages == sorted([("background_poly", (w3,)), ("render_simple", (w3,)), ("render_simple", (w4,))]),
                       str(stages))
            self.run_phase("extras", plan)
        else:
            self.check("extras: already run, a plan is empty", len(plan) == 0, f"{len(plan)} jobs")

    # ------------------------------------------------------------------ v1 against v2
    def v1_dir(self, stage: str) -> Path:
        base = V1_PROJECT if self.against == "v1" else self.root / "v1" / "heart_a7s"
        return base / "work" / stage

    def compare_phase(self) -> dict[str, Any]:
        api, p = self.api, self.p
        full = self.wf("Full")
        before = snapshot(V1_PROJECT) if self.against == "v1" else None
        res: dict[str, Any] = {"against": self.against}
        for stage in FULL_STAGES:
            out = api.stage_outputs(p, full, stage)
            v1 = self.v1_dir(stage)
            if out is None or not v1.is_dir():
                self.check(f"compare {stage}: both results exist", False, f"v2 {out is not None}, v1 {v1.is_dir()}")
                continue
            skip_content = {"cache.json"} if stage == "calibrate" else set()
            try:
                cmp = compare_dirs(v1, Path(out.dir), skip_content=skip_content)
            except Exception as e:                      # one unreadable file must not hide the other stages
                self.check(f"compare {stage}: the files can be compared", False, f"{type(e).__name__}: {e}",
                           required=stage in BIT_FOR_BIT)
                continue
            s = summarize(cmp)
            m1 = json.loads((v1 / "manifest.json").read_text(encoding="utf-8"))
            mdiff = compare_manifests(m1, out.manifest)
            s["manifest_differences"] = mdiff
            res[stage] = s
            self.check(f"compare {stage}: {s['identical']}/{s['files']} files identical, manifest metrics/params same",
                       not s["different"] and not mdiff,
                       (f"different: {s['different']} " if s["different"] else "") + (f"manifest: {mdiff[:3]}" if mdiff else ""),
                       required=stage in BIT_FOR_BIT)
            for n in s["different"]:
                self.log(f"    {stage}/{n}: {json.dumps(cmp[n], default=str)[:400]}")
        if before is not None:
            after = snapshot(V1_PROJECT)
            self.check("the v1 test project is untouched (file list, sizes, times)", before == after,
                       f"{len(after)} files")
            res["v1_files_checked"] = len(after)
        self.save("compare", res)
        return res

    # ------------------------------------------------------------------ rendering
    def render_phase(self) -> dict[str, Any]:
        """render_layer against RenderEngine; the four layers as previews, their sanity; the sliders."""
        import torch
        api, p = self.api, self.p
        from astro_gpu.render.engine import RenderEngine
        from astro_gpu.render.settings import PRESETS, RenderSettings
        from astro_gpu.render.sliders import KEYS
        dev = self.device if torch.cuda.is_available() else "cpu"
        full = self.wf("Full")
        saved = json.loads((V1_PROJECT / "results" / "natural_v2" / "natural_v2_settings.json").read_text(encoding="utf-8"))
        p.render_bases = {"faithful": saved["settings"], "artistic": dict(PRESETS["Artistic"])}
        out: dict[str, Any] = {"device": dev}
        svc = api.layer_service(dev)
        prep2 = Path(api.stage_outputs(p, full, "render_prep").dir)
        prep1 = self.v1_dir("render_prep")
        eng2 = RenderEngine(prep2, device=dev)
        eng1 = RenderEngine(prep1, device=dev) if prep1.is_dir() else None
        twos = {k: 2.0 for k in KEYS}
        for base, scale in (("faithful", 0.25), ("artistic", 0.25), ("faithful", 0.5), ("faithful", 1.0)):
            t0 = time.time()
            lay = api.render_layer(p, full, base, twos, scale, service=svc)
            dt = time.time() - t0
            s = RenderSettings.from_saved(p.render_bases[base])
            ref = eng2.render(s, scale).final.float().cpu().numpy()
            d = array_diff(ref, lay.rgb)
            res = {"equal": d["equal"], "render_layer_s": round(dt, 2), "diff": None if d["equal"] else d}
            if eng1 is not None:
                ref1 = eng1.render(s, scale).final.float().cpu().numpy()
                d1 = array_diff(ref1, lay.rgb)
                res["against_v1_render_prep"] = {"equal": d1["equal"], "diff": None if d1["equal"] else d1}
                if scale == 0.25:
                    im = self.image_difference(ref1, lay.rgb, f"{base}_v1_v2")
                    res["against_v1_render_prep"]["image"] = im
                    t, lv = IMAGE_TOLERANCE, im["abs_diff_levels"]
                    self.check(f"{base} at 1/4: v2's render_prep gives v1's picture within the tolerance (PSNR >= "
                               f"{t['psnr_db_min']:.0f} dB, median <= {t['p50_max']:.0f}, 99th percentile <= "
                               f"{t['p99_max']:.0f} levels of 255)",
                               im["psnr_db"] >= t["psnr_db_min"] and lv["p50"] <= t["p50_max"] and lv["p99"] <= t["p99_max"],
                               json.dumps(im))
                self.check(f"render_layer({base}, 2) at 1/{round(1 / scale)} equals RenderEngine on v1's render_prep",
                           d1["equal"], "" if d1["equal"] else json.dumps(d1, default=str)[:300], required=False)
            self.check(f"render_layer({base}, 2) at 1/{round(1 / scale)} equals RenderEngine.render({base})",
                       d["equal"], f"{rgb_shape(lay.rgb)} in {dt:.1f} s" if d["equal"] else json.dumps(d, default=str)[:300])
            out[f"{base}_{scale}"] = res
            del ref
        # the artifact the user has: v1's 16-bit TIFF of this project's render, against api.export_workflow
        v1_tif = V1_PROJECT / "results" / "natural_v2" / "natural_v2_full_resolution_16bit.tif"
        if v1_tif.is_file():
            import tifffile
            t0 = time.time()
            files = api.export_workflow(p, full, "faithful", None, {"name": "faithful_check", "formats": ["tiff16"]},
                                        service=svc)
            tif = next(f for f in files if f.suffix == ".tif")
            a, b = tifffile.imread(v1_tif), tifffile.imread(tif)
            d = array_diff(a, b)
            out["export_tiff16"] = {"equal": d["equal"], "shape": list(a.shape), "seconds": round(time.time() - t0, 1),
                                    "diff": None if d["equal"] else d}
            self.check("api.export_workflow(faithful) 16-bit TIFF equals the v1 test project's natural_v2 TIFF",
                       d["equal"], f"{a.shape}" if d["equal"] else json.dumps(d, default=str)[:300], required=False)
            del a, b
        # sliders away from 2 differ from the base (and the table holds)
        b = api.render_layer(p, full, "faithful", None, 0.25, service=svc).rgb
        lo = api.render_layer(p, full, "faithful", {"sky": 0.0}, 0.25, service=svc).rgb
        hi = api.render_layer(p, full, "faithful", {"sky": 4.0}, 0.25, service=svc).rgb
        self.check("Full: the sky slider never darkens when raised, and 4 is brighter than 0 (mean brightness)",
                   lo.mean() <= b.mean() <= hi.mean() and lo.mean() < hi.mean(),
                   f"{lo.mean():.5f} <= {b.mean():.5f} <= {hi.mean():.5f}")
        self.check("Full: the sky slider moves the image between 0 and 2 (it does not: fit_tone_curve flat below "
                   "sky_level 36 on heart_a7s, the same in v1's RenderEngine)", lo.mean() < b.mean(),
                   f"{lo.mean():.5f} against {b.mean():.5f}", required=False)
        del eng1, eng2
        # the four layers as previews
        prev = self.root / "previews"
        prev.mkdir(parents=True, exist_ok=True)
        layers = {"full": (full, "faithful"), "classic_stack": (self.wf("Classic stack"), None)}
        for nm in ("Classic + denoise", "Raw stack"):
            try:
                layers[nm.lower().replace(" + ", "_").replace(" ", "_")] = (self.wf(nm), None)
            except KeyError:
                pass
        sane: dict[str, Any] = {}
        images, frames = {}, {}
        for name, (wid, base) in layers.items():
            lay = api.render_layer(p, wid, base, None, 0.25, service=svc)
            x0, x1, y0, y1 = (v // 4 for v in lay.crop)
            img = lay.rgb[:, y0:y1, x0:x1]
            images[name] = img
            frames[name] = lay.rgb
            save_png(prev / f"{name}.png", img)
            s = sanity(img)
            sane[name] = s
            ok = s["finite"] and s["in_range"] and s["lum_std"] > 0.03 and 0.03 < s["lum_p50"] < 0.7 \
                and s["clipped_white_pct"] < 2.0 and s["clipped_black_pct"] < 15.0
            self.check(f"image {name}: finite, in range, not flat, plausible brightness", ok, json.dumps(s))
        out["sanity"] = sane
        if "raw_stack" in images:
            r = sane["raw_stack"]["median_rgb"]
            self.check("Raw stack exit: no colour cast (channel medians within 5 %)", max(r) / max(min(r), 1e-9) < 1.05,
                       f"{r}, sky {sane['raw_stack']['sky_rgb']}")
        # Classic stack against Full: the same field on the same pixels (the brightest stars of one are peaks of the other)
        if "full" in frames and "classic_stack" in frames:
            m = peak_match(frames["classic_stack"], frames["full"])
            shifted = peak_match(np.roll(frames["classic_stack"], (3, 5), axis=(1, 2)), frames["full"])
            out["classic_stack_peaks_in_full"] = {"matched": m, "control_shifted_by_3_5_px": shifted}
            self.check("Classic stack and Full show the same field on the same pixels (the 200 brightest stars of "
                       "Classic: at least 70 % have a peak within 1.5 px in Full; 5 % or less once shifted)",
                       m >= 0.7 and shifted <= 0.05, f"{m:.2f}, shifted control {shifted:.2f}")
        # the polynomial gradient's numbers
        cl = api.stage_outputs(p, self.wf("Classic stack"), "background_poly")
        if cl is not None:
            out["background_poly"] = cl.metrics
            self.check("polynomial gradient: samples found and a small gradient",
                       0 < cl.metrics.get("sky_fit_blocks_pct", 0) <= 100, json.dumps(cl.metrics))
        # the sliders of a simple layer, and the greyed ones
        sl = {s.key: (s.available, s.reason) for s in api.sliders_of(p, self.wf("Classic stack"))}
        self.check("Classic stack: sky, contrast, saturation offered; detail and stars greyed with a reason",
                   all(sl[k][0] for k in ("sky", "contrast", "saturation")) and not sl["detail"][0] and sl["detail"][1]
                   and not sl["stars_number"][0], str(sl))
        c0 = api.render_layer(p, self.wf("Classic stack"), None, {"sky": 0.0}, 0.25, service=svc).rgb.mean()
        c4 = api.render_layer(p, self.wf("Classic stack"), None, {"sky": 4.0}, 0.25, service=svc).rgb.mean()
        self.check("Classic stack: the sky slider is monotonic", c0 < c4, f"{c0:.4f} < {c4:.4f}")
        # a contact sheet to look at
        sheet(prev / "sheet.png", images)
        out["previews"] = sorted(str(f) for f in prev.glob("*.png"))
        self.save("render", out)
        return out

    def _reproduce_check(self, stage: str, s: dict[str, Any]) -> None:
        same = not s["different"] and not s["manifest_differences"]
        self.check(f"reproduce {stage}: v2 on v1's inputs gives v1's {s['identical']}/{s['files']} files bit for bit",
                   same, (f"different: {s['different']} " if s["different"] else "")
                   + (f"manifest: {s['manifest_differences'][:3]}" if s["manifest_differences"] else ""),
                   required=stage in ON_V1_INPUTS)

    def image_difference(self, v1: np.ndarray, v2: np.ndarray, name: str) -> dict[str, Any]:
        """How two renderings of the same settings differ as images, in 8-bit levels: the median, 99th and 99.9th
        percentile of the absolute difference over the whole image and over the nebula's pixels (luminance above the
        median), and a preview of v1, v2 and the difference (x4)."""
        d = np.abs(np.nan_to_num(v1) - np.nan_to_num(v2)) * 255
        dl = d.mean(0)
        mse = float(np.mean((np.nan_to_num(v1) - np.nan_to_num(v2)) ** 2))
        res = {"abs_diff_levels": {k: round(float(np.percentile(dl, q)), 2) for k, q in
                                   (("p50", 50), ("p90", 90), ("p99", 99), ("p99_9", 99.9))},
               "mean_abs_diff_levels": round(float(dl.mean()), 3),
               "psnr_db": round(float(10 * np.log10(1 / max(mse, 1e-12))), 1)}
        prev = self.root / "previews"
        h, w = dl.shape
        crop = (slice(None), slice(h // 4, h // 4 + 400), slice(w // 4, w // 4 + 600))
        tiles = [v1[crop], v2[crop], np.clip(d[crop].mean(0, keepdims=True).repeat(3, 0) * 4 / 255, 0, 1)]
        save_png(prev / f"{name}_crop.png", np.concatenate(tiles, axis=2))
        save_png(prev / f"{name}_diff_x4.png", np.clip(d.mean(0, keepdims=True).repeat(3, 0) * 4 / 255, 0, 1))
        return res

    # ------------------------------------------------------------------ the same inputs as v1, through the v2 runner
    def reproduce_phase(self, stages: tuple[str, ...] = REPRODUCE_STAGES) -> dict[str, Any]:
        """Where the full-chain comparison differs, which stage is to blame: v1's own results are imported (adopted,
        under the write guard) into ``<root>/imported/heart_a7s``; then, one stage at a time, its result is put aside,
        the stage is recomputed by the v2 runner from v1's own inputs (every other stage stays v1's), and the new
        result is compared with v1's. The adopted result is put back before the next stage, so each stage is judged on
        v1's inputs alone. Stages already judged (in report.json) are skipped."""
        api = self.api
        full = self.wf("Full")
        imp = self.root / "imported" / "heart_a7s"
        if not (imp / "project.astroproj").exists():
            self.log(f"importing the v1 test project (copy) into {imp}")
            rep = api.import_v1(V1_PROJECT, imp, None, progress=lambda m: None)
            self.log(f"import: {sum(1 for r in rep.stages if r.adopted)} stages adopted, "
                     f"{sum(1 for r in rep.stages if not r.adopted)} recomputed, {rep.copy_bytes / 1e9:.1f} GB copied")
        q = api.open_project(imp)
        qw = q.workflows[0].id
        keys_v2 = {s: api.stage_key(self.p, full, s) for s in FULL_STAGES}
        keys_v1 = {s: api.stage_key(q, qw, s) for s in FULL_STAGES}
        store = api._store(q)
        self.check("import: the keys of v1's adopted results are the keys the v2 run computed (every stage of Full)",
                   keys_v1 == keys_v2 and all(store.done(s, keys_v1[s]) for s in FULL_STAGES),
                   str({s: (keys_v1[s], keys_v2[s]) for s in FULL_STAGES if keys_v1[s] != keys_v2[s]}))
        done = (self.report().get("reproduce") or {})
        aside_root, rerun_root = imp.parent / "_adopted", imp.parent / "_rerun"
        for stage in stages:
            if stage in done:                                   # judged earlier: the check is made again from the record
                self._reproduce_check(stage, done[stage])
                continue
            key = keys_v1[stage]
            d, a = store.dir(stage, key), aside_root / stage / key
            if not a.exists():                                  # put v1's result aside; the stage is now missing
                a.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(d), str(a))
            store.forget()
            api.refresh(q)
            if not d.exists():
                plan = api.plan_all(q, self.device)
                self.check(f"reproduce {stage}: the plan is this one stage", [j.stage for j in plan.jobs] == [stage],
                           str([j.stage for j in plan.jobs]))
                info = api.start_run(q, plan, self.device)
                res = follow(api, q, info, self.log)
                self.log(f"reproduce {stage}: run {res['status']} in {res['seconds']:.0f} s")
                if res["status"] != "done" or not d.exists():
                    self.check(f"reproduce {stage}: the v2 run is done", False, str(res["status"]))
                    break
            v1 = V1_PROJECT / "work" / stage
            cmp = compare_dirs(v1, d, skip_content=set())
            s = summarize(cmp)
            m1 = json.loads((v1 / "manifest.json").read_text(encoding="utf-8"))
            s["manifest_differences"] = compare_manifests(m1, store.manifest(stage, key))
            s["seconds_v2"] = store.manifest(stage, key).get("duration")
            s["seconds_v1"] = m1.get("duration")
            done = {**done, stage: s}
            self.save("reproduce", done)
            self._reproduce_check(stage, s)
            for n in s["different"]:
                self.log(f"    {stage}/{n}: {json.dumps(cmp[n], default=str)[:400]}")
            # v1's own result back in the store, the new one kept for a look
            (rerun_root / stage).mkdir(parents=True, exist_ok=True)
            if (rerun_root / stage / key).exists():
                shutil.rmtree(rerun_root / stage / key)
            shutil.move(str(d), str(rerun_root / stage / key))
            shutil.move(str(a), str(d))
            store.forget()
            api.refresh(q)
        return done

    # ------------------------------------------------------------------ the library under the astrometry and optics
    def sep_phase(self) -> dict[str, Any]:
        """``sep.extract`` (astrometry and optics detect stars with it) three times on one unchanged image: the same
        call does not give the same objects twice (a few dozen faint, blended ones). optics' lateral colour grid keeps
        the first 120 objects of each cell, so it changes from one run to the next (measured: one cell of 35, 1.5e-3 px),
        and the astrometry's star list is made the same way. Not a v2 matter: v1 does the same."""
        if "sep" in self.report():
            return self.report()["sep"]
        import sep
        src = V1_PROJECT / "work" / "denoise" / "n2n_avg.npy"
        g = np.array(np.load(src, mmap_mode="r")[1], dtype=np.float32, copy=True)
        sep.set_extract_pixstack(3000000)
        bkg = sep.Background(g, bw=64, bh=64)
        sub = g - bkg.back()
        runs = [sep.extract(sub, 40, err=bkg.globalrms, minarea=6) for _ in range(3)]

        def keyed(o):
            return {(round(float(x), 3), round(float(y), 3)): i for i, (x, y) in enumerate(zip(o["x"], o["y"]))}
        k = [keyed(o) for o in runs]
        common = [q for q in k[0] if q in k[1]]
        out = {"sep_version": sep.__version__, "image": f"{src.name}[1] of the v1 test project",
               "objects": [len(o) for o in runs],
               "only_in_run0_against_run1": sum(1 for q in k[0] if q not in k[1]),
               "only_in_run1_against_run0": sum(1 for q in k[1] if q not in k[0]),
               "common": len(common),
               "common_with_another_value": sum(1 for q in common if not np.array_equal(runs[0][k[0][q]], runs[1][k[1][q]]))}
        self.save("sep", out)
        self.check("sep.extract gives the same objects twice on one image (it does not; astrometry and "
                   "optics detect their stars with it)", len({len(o) for o in runs}) == 1
                   and out["only_in_run0_against_run1"] == 0, json.dumps(out), required=False)
        return out

    # ------------------------------------------------------------------ v1 on a copy
    def v1run_phase(self) -> None:
        """Run v1 itself on a COPY of the test project's definition (no results copied), same GPU, same home."""
        dst = self.root / "v1" / "heart_a7s"
        if not (dst / "project.json").exists():
            dst.mkdir(parents=True, exist_ok=True)
            d = json.loads((V1_PROJECT / "project.json").read_text(encoding="utf-8"))
            d["scratch_dir"] = str(self.root / "scratch" / "v1_heart_a7s")
            d["name"] = "heart_a7s_p7_v1copy"
            (dst / "project.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
        ev = dst / "logs" / "events.jsonl"
        start = ev.stat().st_size if ev.exists() else 0
        log = open(self.root / "logs" / "v1run.log", "ab")
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        t0 = time.time()
        proc = subprocess.Popen([sys.executable, "-m", "astro_gpu.cli", "run", str(dst)], cwd=str(dst), env=env,
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        self.log(f"v1 run on the copy started, pid {proc.pid}")
        seen = start
        while proc.poll() is None:
            time.sleep(10)
            if ev.exists() and ev.stat().st_size > seen:
                with open(ev, "rb") as f:
                    f.seek(seen)
                    data = f.read()
                seen += len(data)
                for line in data.decode("utf-8", "replace").splitlines():
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if e.get("type") in ("stage_start", "stage_end"):
                        self.log(f"  v1 {e['type']} {e.get('stage')} {e.get('status', '')} {e.get('duration', '')}")
        durations = {}
        for s in FULL_STAGES:
            try:
                durations[s] = json.loads((dst / "work" / s / "manifest.json").read_text(encoding="utf-8")).get("duration")
            except (OSError, ValueError):
                pass
        self.save("v1run", {"exit": proc.returncode, "wall_s": round(time.time() - t0, 1), "durations": durations})
        self.check("v1 run on the copy finished", proc.returncode == 0, f"exit {proc.returncode}")

    # ------------------------------------------------------------------ timings
    def timing_table(self) -> list[dict[str, Any]]:
        """v1 against v2 durations per stage, the estimate against the measured time, and the disk the plan announced
        against the size of the stage's folder in the store."""
        rep = self.report()
        v2: dict[str, dict[str, Any]] = {}
        for tag in ("run_first", "run_extras"):
            for r in (rep.get(tag) or {}).get("jobs", []):
                v2.setdefault(r["stage"], r)
        planned = {j["key"]: j for j in (rep.get("plan") or {}).get("jobs", [])}
        rows = []
        for s in (*FULL_STAGES, "background_poly", "render_simple"):
            v1 = None
            try:
                v1 = json.loads((self.v1_dir(s) / "manifest.json").read_text(encoding="utf-8")).get("duration")
            except (OSError, ValueError):
                pass
            r = v2.get(s)
            d = self.proj / "store" / s / r["key"] if r else None
            gb = None if d is None or not d.is_dir() else sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
            est_gb = (planned.get(r["key"]) or {}).get("disk_gb") if r else None
            rows.append({"stage": s, "v1_s": None if v1 is None else round(v1, 1),
                         "v2_s": None if not r or r["seconds"] is None else round(r["seconds"], 1),
                         "estimate_s": None if not r else round(r["estimate_s"], 1),
                         "disk_gb": None if gb is None else round(gb, 2), "disk_estimate_gb": est_gb})
        return rows

    def finish(self) -> int:
        rows = self.timing_table()
        v1w = sum(r["v1_s"] or 0 for r in rows if r["stage"] in FULL_STAGES)
        v2w = sum(r["v2_s"] or 0 for r in rows if r["stage"] in FULL_STAGES)
        self.log("stage            v1 s     v2 s   estimate s   disk GB   announced GB")
        for r in rows:
            cells = ["-" if r[k] is None else str(r[k]) for k in ("v1_s", "v2_s", "estimate_s", "disk_gb", "disk_estimate_gb")]
            self.log(f"{r['stage']:<15} {cells[0]:>7} {cells[1]:>8} {cells[2]:>10} {cells[3]:>9} {cells[4]:>12}")
        self.log(f"Full chain: v1 {v1w / 60:.1f} min, v2 {v2w / 60:.1f} min")
        net = ("astrometry", "catalogs", "surveys")             # Gaia and survey downloads: their time is the network's
        self.log(f"without the network-bound stages {net}: v1 "
                 f"{sum(r['v1_s'] or 0 for r in rows if r['stage'] in FULL_STAGES and r['stage'] not in net) / 60:.1f} min, v2 "
                 f"{sum(r['v2_s'] or 0 for r in rows if r['stage'] in FULL_STAGES and r['stage'] not in net) / 60:.1f} min")
        self.save("timings", rows)
        allc = self.report().get("checks", {})
        failed = [n for n, c in allc.items() if c["required"] and not c["ok"]]
        self.save("failed_required", failed)
        self.log(f"{len(allc)} checks recorded, {len(failed)} required failed")
        for n in failed:
            self.log(f"  FAILED: {n}: {allc[n]['detail'][:300]}")
        return 1 if failed else 0


# ================================================================================================== images

def rgb_shape(a: np.ndarray) -> str:
    return "x".join(str(v) for v in a.shape)


def save_png(path: Path, rgb: np.ndarray) -> None:
    """(3, h, w) float display sRGB in [0, 1] to an 8-bit PNG."""
    from PIL import Image
    a = np.clip(np.nan_to_num(rgb, nan=0.0) * 255 + 0.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(a).save(path)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def sanity(rgb: np.ndarray) -> dict[str, Any]:
    """Numbers that show an image is not broken: finite, in range, spread, brightness, clipping, and its colour
    balance: the median of each channel over the whole image (``median_rgb``: a raw stack without colour calibration
    must not be tinted) and over the darkest 30 % of the pixels (``sky_rgb``)."""
    fin = bool(np.isfinite(rgb).all())
    r = np.nan_to_num(rgb)
    lum = luminance(r)
    dark = lum <= np.percentile(lum, 30)
    sky = [float(np.median(r[c][dark])) for c in range(3)]
    return {"finite": fin, "in_range": bool(r.min() >= -1e-6 and r.max() <= 1 + 1e-6),
            "min": round(float(r.min()), 4), "max": round(float(r.max()), 4), "lum_std": round(float(lum.std()), 4),
            "lum_p1": round(float(np.percentile(lum, 1)), 4), "lum_p50": round(float(np.percentile(lum, 50)), 4),
            "lum_p99": round(float(np.percentile(lum, 99)), 4),
            "clipped_black_pct": round(100 * float((lum <= 1 / 255).mean()), 3),
            "clipped_white_pct": round(100 * float((lum >= 254 / 255).mean()), 3),
            "sky_rgb": [round(v, 4) for v in sky], "median_rgb": [round(float(np.median(r[c])), 4) for c in range(3)]}


def _pool(a: np.ndarray, n: int = 64) -> np.ndarray:
    h, w = a.shape
    f = max(1, min(h, w) // n)
    return a[:h // f * f, :w // f * f].reshape(h // f, f, w // f, f).mean((1, 3))


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation of the pooled luminance of two renderings of the same field (shapes may differ by a pixel)."""
    h, w = min(a.shape[1], b.shape[1]), min(a.shape[2], b.shape[2])
    x, y = _pool(luminance(a[:, :h, :w])).ravel(), _pool(luminance(b[:, :h, :w])).ravel()
    rx, ry = x.argsort().argsort().astype(float), y.argsort().argsort().astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def star_peaks(rgb: np.ndarray, n: int = 200, border: int = 8) -> np.ndarray:
    """The ``n`` strongest local peaks (x, y) of an image's luminance above its smooth background."""
    import cv2
    lum = np.ascontiguousarray(luminance(np.nan_to_num(rgb)), dtype=np.float32)
    mx = cv2.dilate(lum, np.ones((7, 7), np.uint8))
    above = lum - cv2.GaussianBlur(lum, (0, 0), 8)
    c = (lum == mx) & (above > 0)
    c[:border] = c[-border:] = False
    c[:, :border] = c[:, -border:] = False
    ys, xs = np.nonzero(c)
    o = np.argsort(-above[ys, xs])[:n]
    return np.c_[xs[o], ys[o]].astype(float)


def peak_match(a: np.ndarray, b: np.ndarray, n: int = 200, tol: float = 1.5) -> float:
    """The share of the ``n`` strongest peaks of ``a`` that have one of the ``3 n`` strongest peaks of ``b`` within
    ``tol`` pixels: two renderings of one field on the same pixels (a shift of a few pixels brings it to about 0)."""
    from scipy.spatial import cKDTree
    h, w = min(a.shape[1], b.shape[1]), min(a.shape[2], b.shape[2])
    pa, pb = star_peaks(a[:, :h, :w], n), star_peaks(b[:, :h, :w], 3 * n)
    if len(pa) == 0 or len(pb) == 0:
        return 0.0
    d, _ = cKDTree(pb).query(pa)
    return float((d <= tol).mean())


def sheet(path: Path, images: dict[str, np.ndarray], width: int = 800) -> None:
    """The layers side by side (two per row), each reduced to ``width`` pixels, with its name."""
    from PIL import Image, ImageDraw
    tiles = []
    for name, im in images.items():
        h, w = im.shape[1:]
        a = np.clip(np.nan_to_num(im) * 255 + 0.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        t = Image.fromarray(a).resize((width, max(1, round(h * width / w))), Image.LANCZOS)
        ImageDraw.Draw(t).text((8, 6), name, fill=(255, 255, 0))
        tiles.append(t)
    rows = [tiles[i:i + 2] for i in range(0, len(tiles), 2)]
    H = sum(max(t.height for t in r) for r in rows)
    sh = Image.new("RGB", (2 * width, H))
    y = 0
    for r in rows:
        for i, t in enumerate(r):
            sh.paste(t, (i * width, y))
        y += max(t.height for t in r)
    sh.save(path)


# ================================================================================================== main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--against", choices=["v1", "v1copy"], default="v1")
    ap.add_argument("phase", choices=["all", "setup", "plan", "run", "extras", "compare", "render", "reproduce", "sep",
                                      "v1run", "report"])
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    if TESTS.resolve() not in root.parents and root != TESTS.resolve():
        raise SystemExit(f"refusing to write outside {TESTS}: {root}")
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")        # type: ignore[attr-defined]
    sm = Smoke(root, a.device, "v1" if a.against == "v1" else "v1copy")
    if a.phase == "report":
        print(json.dumps(sm.report(), indent=1, default=str))
        return 0
    sm.prepare()
    sm.log(f"phase {a.phase}; root {root}; ASTRO_GPU_HOME {os.environ['ASTRO_GPU_HOME']}")
    ph = a.phase
    if ph in ("all", "setup", "plan", "run", "extras", "compare", "render", "reproduce"):
        sm.setup()
        sm.open()
    if ph in ("all", "plan"):
        plan = sm.plan_phase()
        if plan.jobs or "plan" not in sm.report():      # the first plan stays on record once everything is done
            sm.save("plan", {"jobs": [j.to_dict() for j in plan.jobs], "seconds": plan.seconds, "measured": plan.measured})
    if ph in ("all", "run"):
        sm.run_phase("first")
    if ph in ("all", "extras"):
        sm.extras_phase()
    if ph == "v1run":
        sm.v1run_phase()
    if ph in ("all", "compare"):
        sm.compare_phase()
    if ph in ("all", "render"):
        sm.render_phase()
    if ph in ("all", "reproduce"):
        sm.reproduce_phase()
    if ph in ("all", "sep"):
        sm.sep_phase()
    return sm.finish() if ph in ("all", "compare", "render", "extras", "run", "reproduce", "sep") else 0


if __name__ == "__main__":
    sys.exit(main())
