"""Asterion's first solution against the solutions FLOAT already computed, on every stack that has one (read only).

    python tools/bench/astrometry_compare.py [--roots F:/astro_essais/bench E:/astro_float] [--out <folder>]
                                             [--sigma N] [--binning N] [--radius DEG]

For each stack with a FLOAT solution (``store/stack/<key>/drz_full.npy`` and an Astrometry run of that stack,
``store/astrometry/<run>/wcs.fits``, status done), Asterion runs three ways through ``astro_gpu.core.asterion.solve``:
the target position only (the Astrometry block's RA/Dec, else its target name from the shipped catalogues, a radius of
``--radius`` degrees, no scale), the position and the scale (the block's scale when its focal length and pixel size
give it, else FLOAT's solved scale standing for a focal length entered by hand), and blind (neither). Each line: solved
or not, the seconds of the call (the FITS file written included) and of the program alone, the matched / detected
stars, and the difference with FLOAT's stored solution: the sky at the centre and at the four corners of the stack
(arcsec, the largest of the corners), the scale ratio, the difference of the up position angles (deg), whether the
mirror agrees. One setting for every image (Asterion's defaults unless ``--sigma`` / ``--binning`` say otherwise).

Nothing is written beside the stacks: the FITS file of each stack goes to ``<out>/work`` and is removed after; the
table to ``<out>/astrometry_compare.json`` and ``.md``. Environment: ``ASTRO_GPU_HOME`` and ``ASTRO_GPU_SETTINGS`` set to
files under ``<out>`` when not set; the program and its index from ``ASTRO_GPU_ASTERION`` / ``ASTERION_CATALOGS``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = Path("F:/astro_essais/_a_effacer/agent_asterion/compare")
ROOTS = [Path("F:/astro_essais/bench"), Path("E:/astro_float")]


def projects(roots: list[Path]):
    """Every project folder (one holding ``store/astrometry``) up to three levels under the roots."""
    for r in roots:
        for depth in ("*", "*/*", "*/*/*"):
            for p in sorted(r.glob(depth)):
                if (p / "store" / "astrometry").is_dir() and (p / "store" / "stack").is_dir():
                    yield p


def cases(roots: list[Path]) -> list[dict]:
    """One case per stack (the first project that solved it), its newest Astrometry run."""
    seen: dict[str, dict] = {}
    for proj in projects(roots):
        runs = []
        for a in (proj / "store" / "astrometry").iterdir():
            m = a / "manifest.json"
            if not (a / "wcs.fits").is_file() or not m.is_file():
                continue
            try:
                man = json.loads(m.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            stack = proj / "store" / "stack" / str(man.get("inputs", {}).get("stack")) / "drz_full.npy"
            if man.get("status") == "done" and stack.is_file():
                runs.append((man.get("finished", ""), a, man, stack))
        for _fin, a, man, stack in sorted(runs, key=lambda r: r[0], reverse=True):
            key = stack.parent.name
            if key not in seen:
                seen[key] = {"project": proj, "run": a, "manifest": man, "stack": stack}
    return list(seen.values())


def compare(sol, wf, W: int, H: int, float_metrics: dict) -> dict:
    import numpy as np
    pts = np.array([[W / 2, H / 2], [0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], dtype=float)
    a = sol.wcs.pixel_to_world(pts[:, 0], pts[:, 1])
    f = wf.pixel_to_world(pts[:, 0], pts[:, 1])
    sep = a.separation(f).arcsec
    dang = (sol.up_position_angle - float(float_metrics.get("up_position_angle", np.nan)) + 180) % 360 - 180
    return {"centre_arcsec": round(float(sep[0]), 2), "corner_arcsec": round(float(sep[1:].max()), 2),
            "scale_ratio": round(sol.scale / float(float_metrics["scale_arcsec"]), 5),
            "rotation_deg": round(float(dang), 3),
            "mirror_agrees": bool(sol.flipped) == bool(float_metrics.get("parity_flipped"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", type=Path, default=ROOTS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--sigma", type=float)
    ap.add_argument("--binning", type=int)
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from astro_gpu.core.sirilcat import private_home
    private_home(os.environ.get("ASTRO_GPU_HOME") or args.out / "home")     # the installed catalogue stays visible
    os.environ.setdefault("ASTRO_GPU_SETTINGS", str(args.out / "settings.json"))
    os.environ.setdefault("ASTRO_GPU_OFFLINE", "1")
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    from astro_gpu.core import asterion
    from astro_gpu.stages.astrometry import find_offline
    st = asterion.status()
    if not st.ok:
        raise SystemExit(st.why)
    print(f"Asterion {st.version}: {st.program}, index {st.indexes} in {st.catalogs}", flush=True)
    rows = []
    for c in cases(args.roots):
        man = c["manifest"]
        p, m = man.get("params", {}), man.get("metrics", {})
        ra, dec = p.get("ra"), p.get("dec")
        if (ra is None or dec is None) and p.get("target"):
            ra, dec = find_offline(p["target"]) or (None, None)
        S = 1.0
        try:
            S = float(json.loads((c["stack"].parent / "drz_meta.json").read_text(encoding="utf-8"))["scale"])
        except (OSError, ValueError, KeyError):
            pass
        scale = p.get("scale") or (206.265 * p["pixel_um"] / p["focal_mm"] / S if p.get("focal_mm") and p.get("pixel_um")
                                   else None)
        scale_from = "block" if scale else "FLOAT's solution"
        scale = scale or m.get("scale_arcsec")
        t = time.monotonic()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wf = WCS(fits.getheader(c["run"] / "wcs.fits"))
            img = np.asarray(np.load(c["stack"], mmap_mode="r"))   # read once: the runs time Asterion, not the disk
        except OSError as e:                                     # a project removed meanwhile (not by this tool)
            print(f"skipped {c['stack']}: {e}", flush=True)
            continue
        load_s = time.monotonic() - t
        H, W = img.shape[-2:]
        name = f"{c['project'].relative_to(c['project'].anchor)}".replace("\\", "/")
        base = {"project": name, "stack": c["stack"].parent.name, "size": f"{W}x{H}", "target": p.get("target") or "",
                "float_seconds": round(float(man.get("duration") or 0), 1), "float_scale": m.get("scale_arcsec"),
                "float_stars": m.get("stars_used"), "load_seconds": round(load_s, 2)}
        for mode, kw in (("position", {"ra": ra, "dec": dec, "radius": args.radius}),
                         ("position+scale", {"ra": ra, "dec": dec, "radius": args.radius, "scale": scale}),
                         ("blind", {})):
            if mode != "blind" and (ra is None or dec is None):
                continue
            info: dict = {}
            sol = asterion.solve(img, args.out / "work", timeout=args.timeout, sigma=args.sigma, binning=args.binning,
                                 info=info, **kw)
            row = {**base, "mode": mode, "solved": sol is not None, "seconds": round(info.get("seconds", 0), 2),
                   "program_ms": round(float((info.get("timing") or {}).get("totalMillis", 0)), 1),
                   "detected": info.get("detected"), "why": "" if sol is not None else info.get("why", "")}
            if mode == "position+scale":
                row["scale_hint"] = round(float(scale), 4)
                row["scale_from"] = scale_from
            if sol is not None:
                row.update(matched=sol.matched, rms_arcsec=round(sol.rms_arcsec, 3), **compare(sol, wf, W, H, m))
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    (args.out / "astrometry_compare.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = ["| project | stack | size | mode | solved | s (call) | ms (program) | matched/detected | centre \" | corner \" "
             "| scale ratio | rotation deg | mirror | FLOAT s |", "|" + "---|" * 14]
    for r in rows:
        lines.append(f"| {r['project']} | {r['stack'][:6]} | {r['size']} | {r['mode']} | {'yes' if r['solved'] else 'NO'} "
                     f"| {r['seconds']} | {r['program_ms']} | {r.get('matched', '-')}/{r.get('detected', '-')} "
                     f"| {r.get('centre_arcsec', '-')} | {r.get('corner_arcsec', '-')} | {r.get('scale_ratio', '-')} "
                     f"| {r.get('rotation_deg', '-')} | {r.get('mirror_agrees', '-')} | {r['float_seconds']} |")
    (args.out / "astrometry_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
