"""The Astrometry stage before and after Asterion's first solution, on stacks FLOAT already solved (read only).

    python tools/bench/astrometry_stage_compare.py [--out <folder>] [--only <text in the project path>]

For each case below, the stage runs twice on the stack of the stored run, with that run's parameters: FLOAT's own
search (Asterion said not installed), then with Asterion's first solution (``ASTRO_GPU_ASTERION`` /
``ASTERION_CATALOGS``). Each line: the stage's time, the first solver, and the final solution against the other one:
the sky at the centre (arcsec) and at the four corners (px, the largest), the scale ratio, the stars of the fit and the
residual. The corners judged by stars nobody fitted there: Asterion's matched Tycho-2 stars beyond the inscribed
ellipse of the frame (radius > 1, in half-widths), the median distance (px) from their measured position to where each
solution puts them.

Nothing is written beside the stacks: the stage's outputs and scratch go to ``<out>/<case>/<mode>``. No network: the
catalogue comes from the offline Gaia extract (``ASTRO_GPU_GAIA_EXTRACTS``) or the cache, ``ASTRO_GPU_OFFLINE`` is set
and any socket connection raises. The camera's history (instruments.json) is neither read nor written, so both runs
start the same way.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = Path("F:/astro_essais/_a_effacer/agent_asterion/stage_compare")
#: (project, astrometry run): the bench's Heart, Rosette and Seestar, the read-only stacks of E:/astro_float
CASES = [
    ("F:/astro_essais/bench/heart/test_project", "c29c47c4def4e48d"),
    ("F:/astro_essais/bench/rosette/ours_full", "49d8e41d39150cf5"),
    ("F:/astro_essais/bench/seestar/ours_express", "e2d3ff15f68449ea"),
    ("E:/astro_float/Rosette", "13b53cf20dc59b4a"),
    ("E:/astro_float/Pleiades", "e3a939566887055e"),
    ("E:/astro_float/tests_gui/brutes_10", None),
]


def _no_network():
    def refuse(*a, **k):
        raise OSError("no network in this comparison")
    socket.create_connection = refuse
    socket.socket.connect = refuse


def _run_dir(project: Path, run: str | None) -> Path:
    if run:
        return project / "store" / "astrometry" / run
    for m in sorted((project / "store" / "astrometry").glob("*/manifest.json")):
        man = json.loads(m.read_text(encoding="utf-8"))
        if man.get("status") == "done" and (m.parent / "wcs.fits").is_file():
            return m.parent
    raise SystemExit(f"no solved run in {project}")


def stage(project: Path, run: Path, out: Path, asterion_on: bool):
    """(seconds, logs, metrics, the WCS written) of one run of the stage."""
    from astro_gpu.core import asterion
    from astro_gpu.stages import astrometry as am
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    stack = project / "store" / "stack" / man["inputs"]["stack"]
    inv = project / "store" / "inventory" / man["inputs"].get("inventory", "") / "manifest.json"
    inv_metrics = json.loads(inv.read_text(encoding="utf-8")).get("metrics", {}) if inv.is_file() else {}
    out.mkdir(parents=True, exist_ok=True)
    (out / "scratch").mkdir(exist_ok=True)
    logs, metrics = [], {}
    import numpy as np
    ctx = SimpleNamespace(
        load_json=lambda st, name: json.loads((stack / name).read_text(encoding="utf-8")),
        load=lambda st, name, mmap=False: np.load(stack / name, mmap_mode="r" if mmap else None),
        log=lambda m, level="info": logs.append((round(time.monotonic() - t0, 1), level, m)),
        progress=lambda *a, **k: None, metric=metrics.__setitem__, out=lambda n: out / n,
        metric_of=lambda st, key, default=None: inv_metrics.get(key, default) if st == "inventory" else default,
        scratch=out / "scratch", check_stop=lambda: None)
    real = asterion.status
    am.remember_instrument = lambda *a: None
    am.focal_hints = lambda *a: []
    am._GAIA_CACHE.clear()
    if not asterion_on:
        asterion.status = lambda *a, **k: asterion.Status(False, why="off for the comparison", missing=True)
    t0 = time.monotonic()
    try:
        am.Astrometry().run(ctx, am.AstrometryParams(**man["params"]))
        err = ""
    except Exception as e:  # noqa: BLE001 - said in the table
        err = f"{type(e).__name__}: {e}"
    finally:
        asterion.status = real
    secs = time.monotonic() - t0
    w = None
    if (out / "wcs.fits").is_file():
        from astropy.io import fits
        from astropy.wcs import WCS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = WCS(fits.getheader(out / "wcs.fits"))
    (out / "log.json").write_text(json.dumps({"logs": logs, "metrics": metrics, "error": err}, indent=1), "utf-8")
    return secs, logs, metrics, w, err, np.load(stack / "drz_full.npy", mmap_mode="r").shape[-2:]


def corner_px(w, matches, W, H) -> tuple[float, int]:
    import numpy as np
    from astropy.coordinates import SkyCoord
    r = np.hypot((matches[:, 0] - W / 2) / (W / 2), (matches[:, 1] - H / 2) / (H / 2))
    k = r > 1.0
    if not k.any():
        return float("nan"), 0
    x, y = w.world_to_pixel(SkyCoord(matches[k, 2], matches[k, 3], unit="deg"))
    return float(np.median(np.hypot(x - matches[k, 0], y - matches[k, 1]))), int(k.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ASTRO_GPU_HOME", str(args.out / "home"))
    os.environ.setdefault("ASTRO_GPU_SETTINGS", str(args.out / "settings.json"))
    os.environ["ASTRO_GPU_OFFLINE"] = "1"
    _no_network()
    import numpy as np

    from astro_gpu.core import asterion
    if not asterion.status().ok:
        raise SystemExit(asterion.status().why)
    rows = []
    for proj, run_id in CASES:
        if args.only not in proj:
            continue
        project = Path(proj)
        run = _run_dir(project, run_id)
        name = project.name if project.parent.name in ("astro_float", "tests_gui") else \
            f"{project.parent.name}/{project.name}"
        res = {}
        for mode in ("before", "after"):
            res[mode] = stage(project, run, args.out / name.replace("/", "_") / mode, mode == "after")
            secs, logs, m, w, err, _shape = res[mode]
            print(f"{name} {mode}: {secs:.1f} s {m.get('first_solver')} {err}", flush=True)
        (sb, _lb, mb, wb, eb, (H, W)), (sa, _la, ma, wa, ea, _s) = res["before"], res["after"]
        row = {"case": name, "size": f"{W}x{H}", "params": {k: v for k, v in json.loads(
            (run / "manifest.json").read_text(encoding="utf-8"))["params"].items() if k in ("focal_mm", "scale")},
            "before_s": round(sb, 1), "after_s": round(sa, 1), "first_solver": ma.get("first_solver"),
            "first_solution_s": ma.get("first_solution_s"), "error_before": eb, "error_after": ea}
        for tag, m in (("before", mb), ("after", ma)):
            row[f"stars_{tag}"], row[f"residual_{tag}"] = m.get("stars_used"), m.get("residual_px")
        if wb is not None and wa is not None:
            pts = np.array([[W / 2, H / 2], [0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], float)
            sep = wb.pixel_to_world(pts[:, 0], pts[:, 1]).separation(wa.pixel_to_world(pts[:, 0], pts[:, 1])).arcsec
            scale = float(mb["scale_arcsec"])
            row.update(centre_arcsec=round(float(sep[0]), 3), corner_px=round(float(sep[1:].max()) / scale, 3),
                       scale_ratio=round(float(ma["scale_arcsec"]) / scale, 6))
            stk = json.loads((run / "manifest.json").read_text(encoding="utf-8"))["inputs"]["stack"]
            img = np.load(project / "store" / "stack" / stk / "drz_full.npy", mmap_mode="r")
            sol = asterion.solve(np.array(img[1], np.float32), args.out / "work", timeout=10)
            if sol is not None:
                for tag, w in (("before", wb), ("after", wa)):
                    px, n = corner_px(w, sol.matches, W, H)
                    row[f"tycho_corner_px_{tag}"], row["tycho_corner_stars"] = round(px, 2), n
        rows.append(row)
        print(json.dumps(row), flush=True)
    (args.out / "stage_compare.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
