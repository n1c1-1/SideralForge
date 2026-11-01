"""Do a variant's darks match the lights? The hot pixels and the fixed pattern of each master dark against the ones
the lights themselves show (rosette: B = the Iris night's real darks, C = the sky frames listed as darks).

    python tools/bench/darkcheck.py CASE [--n 40]

The lights' own defects are found as the engine's "hot pixels from the lights" finds them: in sensor coordinates, the
lower quartile of n Base lights spread over the session (the sky drifted: a star covers a pixel in a few frames only).
Each master dark is the median of its frames (the engine's master). On each colour plane, a pixel's excess is its
value minus the median of its 5 x 5 same-colour neighbours.

Per variant (A = no dark: the lights' defects stay as they are):
- levels: the master's median (ADU) against the raw black level and the lights' floor;
- hot pixels: the lights' (excess > thr, thr = 5 x the noise of the lights' quartile map, the engine's threshold),
  the share of them the master also has (excess > thr / 2), the master's own hot pixels (> thr) that the lights do
  not show (subtracted, they would dig a hole);
- the amplitude ratio lights / master on the master's strong hot pixels (> 3 thr): dark current of the night against
  the darks' (a thermometer: dark current doubles every ~6 °C);
- what subtracting the master leaves: hot pixels still above thr, holes below -thr;
- the fixed pattern: row and column profiles of the lights' map and of the master (the mean of each row / column of
  the plane, smooth trend removed): their spread in ADU and their correlation; a pattern the master removes does not
  turn into walking noise when the sky drifts;
- the master's large-scale structure (64 x 64 px blocks, p99 - p1 in ADU): what subtracting it adds to the sky.
Writes <case>/darkcheck.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as C  # noqa: E402

os.environ["USERPROFILE"] = str(C.HOME)
os.environ["ASTRO_GPU_HOME"] = str(C.AG_HOME)
sys.path.insert(0, str(C.REPO))
C.utf8_console()

from scipy.ndimage import median_filter, uniform_filter1d  # noqa: E402

from astro_gpu.io.frames import read_cfa  # noqa: E402

MAD = 1.4826


def rstd(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(MAD * np.median(np.abs(v - np.median(v))))


def plane(a: np.ndarray, k: int) -> np.ndarray:
    i, j = divmod(k, 2)
    return a[..., i::2, j::2]


def stack_stat(paths: list[Path], q: float, label: str) -> np.ndarray:
    """Per-pixel percentile q of the frames (raw ADU), in row bands to keep the memory low."""
    t0 = time.time()
    first = read_cfa(paths[0])
    H, W = first.shape
    st = np.empty((len(paths), H, W), np.uint16)
    st[0] = first.astype(np.uint16)
    for i, p in enumerate(paths[1:], 1):
        st[i] = read_cfa(p).astype(np.uint16)
    out = np.empty((H, W), np.float32)
    for y in range(0, H, 256):
        out[y:y + 256] = np.percentile(st[:, y:y + 256].astype(np.float32), q, axis=0)
    print(f"{label}: {len(paths)} frames, percentile {q:g} in {time.time() - t0:.0f} s", flush=True)
    return out


def excess(a: np.ndarray) -> np.ndarray:
    """Each pixel minus the median of its 5 x 5 same-colour neighbours (the four colour planes)."""
    out = np.empty_like(a)
    for k in range(4):
        v = plane(a, k)
        plane(out, k)[...] = v - median_filter(v, size=5)
    return out


def profiles(a: np.ndarray) -> dict[str, np.ndarray]:
    """Row and column profiles of each colour plane (mean over the plane's columns / rows, robust: the 10-90 %
    trimmed mean), each minus its smooth trend (a 31-sample running mean)."""
    out = {"rows": [], "cols": []}
    for k in range(4):
        v = plane(a, k)
        lo, hi = np.percentile(v, [10, 90])
        m = np.where((v >= lo) & (v <= hi), v, np.nan)
        for name, ax in (("rows", 1), ("cols", 0)):
            pr = np.nanmean(m, axis=ax)
            out[name].append(pr - uniform_filter1d(pr, 31, mode="nearest"))
    return {k: np.concatenate(v) for k, v in out.items()}


def large_scale(a: np.ndarray, b: int = 64) -> float:
    H, W = a.shape
    blk = a[: H // b * b, : W // b * b].reshape(H // b, b, W // b, b)
    med = np.median(blk.transpose(0, 2, 1, 3).reshape(H // b, W // b, -1), axis=-1)
    return float(np.percentile(med, 99) - np.percentile(med, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=[c for c in C.CASES if C.CASES[c].get("variants")])
    ap.add_argument("--n", type=int, default=40, help="lights used (spread over the Base selection)")
    a = ap.parse_args()
    t0 = time.time()
    man = json.loads((C.BENCH / a.case / "manifest.json").read_text(encoding="utf-8"))
    black = float(man["camera"]["black"])
    lights = sorted(C.lights_dir(a.case, "base").iterdir())
    sub = [lights[i] for i in np.linspace(0, len(lights) - 1, min(a.n, len(lights))).astype(int)]
    L = stack_stat(sub, 25, "lights (lower quartile)")
    one = read_cfa(sub[len(sub) // 2])
    sky_noise = float(np.median([rstd(plane(one, k)[::4, ::4] - median_filter(plane(one, k)[::4, ::4], 3))
                                 / np.sqrt(1 + 1 / 9) for k in range(4)]))
    hL = excess(L)
    sig = float(np.median([rstd(plane(hL, k)[::2, ::2]) for k in range(4)]))
    thr = 5.0 * sig
    hotL = hL > thr
    prL = profiles(L)
    res: dict = {"case": a.case, "lights_used": len(sub), "black_level": black,
                 "lights_floor_adu": float(np.median(L)), "light_sky_noise_adu": sky_noise,
                 "lights_map_noise_adu": sig, "thr_adu": thr, "thr_in_sky_noise": thr / sky_noise,
                 "lights_hot": int(hotL.sum()), "lights_hot_ppm": 1e6 * float(hotL.mean()),
                 "lights_pattern_adu": {k: rstd(v) for k, v in prL.items()}, "variants": {}}
    print(f"lights: floor {res['lights_floor_adu']:.1f} ADU, sky noise {sky_noise:.1f}, map noise {sig:.2f}, "
          f"{int(hotL.sum())} hot pixels above {thr:.1f} ADU", flush=True)
    res["variants"]["A"] = {"label": "no dark", "left_hot": int(hotL.sum()),
                            "left_hot_ppm": 1e6 * float(hotL.mean()), "holes": 0}
    for v, ve in (man.get("variants") or {}).items():
        d = (ve.get("calibration") or {}).get("darks")
        if not d:
            continue
        paths = sorted(Path(d["folder"]).iterdir())
        M = stack_stat(paths, 50, f"master dark {v}")
        hM = excess(M)
        sigM = float(np.median([rstd(plane(hM, k)[::2, ::2]) for k in range(4)]))
        hotM = hM > thr
        strong = hM > 3 * thr
        ratio = hL[strong] / hM[strong]
        r = hL - hM
        prM = profiles(M)
        e = {"label": ve["label"], "frames": len(paths), "master_level_adu": float(np.median(M)),
             "master_minus_black_adu": float(np.median(M)) - black, "master_noise_adu": sigM,
             "master_hot": int(hotM.sum()), "master_hot_ppm": 1e6 * float(hotM.mean()),
             "lights_hot_also_in_master_pct": 100 * float((hM[hotL] > thr / 2).mean()) if hotL.any() else None,
             "master_hot_not_in_lights_pct": 100 * float((hL[hotM] < thr / 2).mean()) if hotM.any() else None,
             "strong_master_hot": int(strong.sum()),
             "amplitude_ratio_lights_over_master": float(np.median(ratio)) if strong.any() else None,
             "amplitude_ratio_p25_p75": [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))]
             if strong.any() else None,
             "left_hot": int((r > thr).sum()), "left_hot_ppm": 1e6 * float((r > thr).mean()),
             "holes": int((r < -thr).sum()), "holes_ppm": 1e6 * float((r < -thr).mean()),
             "pattern_adu": {k: rstd(x) for k, x in prM.items()},
             "pattern_corr_with_lights": {k: float(np.corrcoef(prL[k], prM[k])[0, 1]) for k in prL},
             "large_scale_p99_p1_adu": large_scale(M)}
        res["variants"][v] = e
        print(f"{v}: {json.dumps(e)}", flush=True)
        del M, hM, r
    res["lights_large_scale_p99_p1_adu"] = large_scale(L)
    res["seconds"] = round(time.time() - t0, 1)
    out = C.BENCH / a.case / "darkcheck.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
