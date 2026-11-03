"""Bench of the gradient removal methods: Siril (polynomial, RBF; automatic in its development version), GraXpert
(AI), ours (polynomial), on the bench's
stacks, with injected gradients and an injected faint object (docs/chantiers/gradient.md, part 3).

    python tools/bench/gradient.py [--cases heart,iris,...] [--out F:/astro_essais/bench/gradient]

For each case, the input is the stack of the bench's reference workflow (PI-own of v2x_express) with its colour matrix
applied, as our polynomial background sees it, cropped to the area every frame covers, scaled to 0..1 (Siril and
GraXpert expect it). Five versions of it go through every method: as it is, plus a tilted plane, a dome, a corner
glow (each a known gradient), and plus a faint wide object. The background a method removes is input - output.

Measures (results.json, and a table printed):
- gradient recovery: the background found with the injected gradient minus the one found without it, against the
  injected gradient, up to a constant: the RMS of the difference in % of the gradient's RMS (0 = recovered exactly);
- object survival: how much of the injected faint object is still in the output (100 % = kept whole);
- flatness: the robust spread (5th to 95th percentile) of the sky blocks of the output, in % of the input's sky level
  (a method may leave its own pedestal or none: the input's level is the one common scale).
A sheet per case (sheet_<case>.jpg): the input, each output and each removed background, one stretch for all.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BENCH = Path("F:/astro_essais/bench")
CASES = ("heart", "iris", "m51", "m8", "rosette", "seestar")
WORKFLOW = "PI-own"
SIRIL = Path("C:/Users/Nicolas/Tools/siril-1.4.5/siril/bin/siril-cli.exe")
#: Siril's development version (1.5, built from its master branch): the automatic sample-free model is only there
SIRIL_DEV = Path("C:/Users/Nicolas/Tools/siril-1.5-dev-20261102/siril/bin/siril-cli.exe")
GRAXPERT = Path("C:/Users/Nicolas/Tools/GraXpert-3.1.0rc2/GraXpert.exe")
LP_COLOUR = np.array([1.25, 1.0, 0.7])           # light pollution is orange: more red, less blue
LUM = np.array([0.2126, 0.7152, 0.0722])


# ---------------------------------------------------------------------------------------------- the input
def load_case(case: str) -> np.ndarray:
    """The colour-calibrated linear stack of the case's reference workflow, cropped to the fully covered area."""
    from astro_gpu.core import project2
    from astro_gpu.core.store import Keys
    root = BENCH / case / "v2x_express"
    p = project2.load(root)
    w = next(x for x in p.workflows if x.name == WORKFLOW)
    k = Keys(p)
    sd = root / "store"
    stack = np.load(sd / "stack" / k.key(w, "stack") / "drz_full.npy").astype(np.float32)
    weight = np.load(sd / "stack" / k.key(w, "stack") / "drz_weight.npy", mmap_mode="r")[1]
    M = np.load(sd / "color" / k.key(w, "color") / "M.npy").astype(np.float32)
    rgb = np.tensordot(M, stack, 1)
    full = np.asarray(weight) > 0.8 * float(np.median(weight[weight > 0]))
    y0, y1, x0, x1 = inner_box(full)
    m = max(4, int(0.01 * min(y1 - y0, x1 - x0)))          # a margin: the edges of the coverage
    return np.ascontiguousarray(rgb[:, y0 + m:y1 - m, x0 + m:x1 - m])


def inner_box(mask: np.ndarray, fill: float = 0.998) -> tuple[int, int, int, int]:
    """A rectangle (y0, y1, x0, x1) almost wholly inside ``mask``: from the mask's bounding box, the side whose edge
    line is least covered is moved in, one step at a time (a field turned during the night is not a rectangle)."""
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    step = max(1, min(y1 - y0, x1 - x0) // 400)
    while (y1 - y0) > 16 and (x1 - x0) > 16 and mask[y0:y1, x0:x1].mean() < fill:
        edges = {"top": mask[y0, x0:x1].mean(), "bottom": mask[y1 - 1, x0:x1].mean(),
                 "left": mask[y0:y1, x0].mean(), "right": mask[y0:y1, x1 - 1].mean()}
        worst = min(edges, key=edges.get)
        if worst == "top":
            y0 += step
        elif worst == "bottom":
            y1 -= step
        elif worst == "left":
            x0 += step
        else:
            x1 -= step
    return int(y0), int(y1), int(x0), int(x1)


def to_unit(img: np.ndarray) -> tuple[np.ndarray, float, float]:
    """The image in 0..1 (Siril and GraXpert expect it): (img - lo) / span, the brightest stars clipped."""
    lo = float(np.percentile(img, 0.01))
    hi = float(np.percentile(img, 99.99))
    span = (hi - lo) * 1.2
    return np.clip((img - lo) / span, 0, 1).astype(np.float32), lo, span


def sky_level(img: np.ndarray) -> float:
    return float(np.median(np.tensordot(LUM, img[:, ::8, ::8], 1)))


# ---------------------------------------------------------------------------------------------- the injections
def gradients(H: int, W: int, level: float) -> dict[str, np.ndarray]:
    """Three known gradients (3, H, W), light-pollution coloured, of amplitudes typical of a suburban sky."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = x / W, y / H
    tilt = 0.3 * level * (0.8 * u + 0.2 * v)
    dome = 0.2 * level * ((u - 0.4) ** 2 + (v - 0.6) ** 2) / 0.5
    d = np.hypot(u - 1.0, v - 0.0)
    corner = 0.4 * level * np.exp(-d / 0.35)
    return {k: (g[None] * LP_COLOUR[:, None, None]).astype(np.float32)
            for k, g in (("tilt", tilt), ("dome", dome), ("corner", corner))}


def faint_object(H: int, W: int, level: float) -> tuple[np.ndarray, np.ndarray]:
    """A faint wide neutral cloud (an IFN-like glow): its image, and the area where it is measured (within 2 sigma)."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    s = 0.12 * min(H, W)
    r2 = ((x - 0.3 * W) ** 2 + (y - 0.7 * H) ** 2) / s ** 2
    g = 0.08 * level * np.exp(-0.5 * r2)
    return np.repeat(g[None], 3, 0).astype(np.float32), r2 < 4


# ---------------------------------------------------------------------------------------------- the methods
def write_fits(path: Path, img: np.ndarray):
    from astropy.io import fits
    fits.writeto(path, img.astype(np.float32), overwrite=True)


def read_fits(path: Path, like: np.ndarray) -> np.ndarray:
    """A method's output, in the input's orientation (a tool may write its rows bottom up)."""
    from astropy.io import fits
    with fits.open(path) as h:
        a = np.asarray(h[0].data, np.float32)
    if a.ndim == 3 and a.shape[-1] == 3 and a.shape[0] != 3:
        a = a.transpose(2, 0, 1)
    if a.shape != like.shape:
        raise RuntimeError(f"{path.name}: shape {a.shape}, expected {like.shape}")
    s = (slice(None), slice(None, None, 16), slice(None, None, 16))
    if np.corrcoef(a[s].ravel(), like[s].ravel())[0, 1] < np.corrcoef(a[:, ::-1][s].ravel(), like[s].ravel())[0, 1]:
        a = np.ascontiguousarray(a[:, ::-1])
    return a


def siril(img: np.ndarray, work: Path, command: str, exe: Path = SIRIL) -> np.ndarray:
    src, dst = work / "in.fits", work / "out"
    write_fits(src, img)
    script = work / "run.ssf"
    script.write_text(f"requires 1.4.0\nload \"{src.as_posix()}\"\n{command}\nsave \"{dst.as_posix()}\"\n")
    r = subprocess.run([str(exe), "-d", str(work), "-s", str(script)], capture_output=True, text=True, timeout=900)
    out = work / "out.fit"
    if not out.exists():
        out = work / "out.fits"
    if not out.exists():
        raise RuntimeError(f"Siril failed: {r.stdout[-800:]}")
    a = read_fits(out, img)
    out.unlink()
    return a


def graxpert(img: np.ndarray, work: Path) -> np.ndarray:
    src, dst = work / "in.fits", work / "gx_out"
    write_fits(src, img)
    r = subprocess.run([str(GRAXPERT), str(src), "-cli", "-cmd", "background-extraction", "-output", str(dst),
                        "-correction", "Subtraction", "-gpu", "true"], capture_output=True, text=True, timeout=1800,
                       env={k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"})
    out = next((p for p in (work / "gx_out.fits", work / "gx_out.fit") if p.exists()), None)
    if out is None:
        raise RuntimeError(f"GraXpert failed: {(r.stdout + r.stderr)[-800:]}")
    a = read_fits(out, img)
    out.unlink()
    return a


def ours_poly(img: np.ndarray, work: Path) -> np.ndarray:
    from astro_gpu.stages.gradient import GradientParams, fit_poly_sky, poly_sky
    p = GradientParams()
    valid = np.ones(img.shape[1:], bool)
    fit = fit_poly_sky(img, valid, p.degree, p.samples, p.object_clip)
    return img - poly_sky(fit["coefs"], p.degree, *img.shape[1:])


METHODS = {
    "siril_poly2": lambda img, work: siril(img, work, "subsky 2 -samples=20 -tolerance=1.0"),
    "siril_poly4": lambda img, work: siril(img, work, "subsky 4 -samples=20 -tolerance=1.0"),
    "siril_rbf": lambda img, work: siril(img, work, "subsky -rbf -samples=20 -tolerance=1.0 -smooth=0.5"),
    "siril_auto": lambda img, work: siril(img, work, "subsky -auto", SIRIL_DEV),
    "graxpert_ai": graxpert,
    "ours_poly": ours_poly,
}


# ---------------------------------------------------------------------------------------------- the measures
def recovery(bg_with: np.ndarray, bg_without: np.ndarray, g: np.ndarray) -> float:
    """The RMS of (found - injected), up to a constant per channel, in % of the injected gradient's RMS."""
    d = bg_with - bg_without - g
    d -= d.reshape(3, -1).mean(1)[:, None, None]
    g0 = g - g.reshape(3, -1).mean(1)[:, None, None]
    return float(100 * np.sqrt((d[:, ::4, ::4] ** 2).mean()) / np.sqrt((g0[:, ::4, ::4] ** 2).mean()))


def survival(out_with: np.ndarray, out_without: np.ndarray, obj: np.ndarray, area: np.ndarray) -> float:
    kept = (out_with - out_without)[:, area].sum()
    return float(100 * kept / obj[:, area].sum())


def flatness(out: np.ndarray, level: float, block: int = 32) -> float:
    """The spread (p5 to p95) of the sky blocks of an output, in % of ``level`` (the input's sky): block medians of the
    luminance, the objects left out by an iterative clip above the median."""
    L = np.tensordot(LUM, out, 1)
    H, W = L.shape
    h, w = H // block, W // block
    b = np.median(L[:h * block, :w * block].reshape(h, block, w, block).transpose(0, 2, 1, 3).reshape(h, w, -1), -1)
    v = b.ravel()
    keep = np.ones(v.size, bool)
    for _ in range(20):
        med = np.median(v[keep])
        mad = 1.4826 * np.median(np.abs(v[keep] - med)) + 1e-12
        new = v < med + 2.5 * mad
        if (new == keep).all():
            break
        keep = new
    sky = v[keep]
    return float(100 * (np.percentile(sky, 95) - np.percentile(sky, 5)) / max(level, 1e-9))


# ---------------------------------------------------------------------------------------------- the sheet
def sheet(path: Path, img: np.ndarray, outs: dict[str, np.ndarray]):
    """The input, each output and each removed background (asinh stretch, the same for all), reduced."""
    import cv2
    k = max(1, int(np.ceil(max(img.shape[1:]) / 700)))

    def small(a):
        h, w = a.shape[1] // k * k, a.shape[2] // k * k
        return a[:, :h, :w].reshape(3, h // k, k, w // k, k).mean(axis=(2, 4))
    ref = small(img)
    lo, hi = np.percentile(ref, 1), np.percentile(ref, 99.7)

    def show(a, centre=False):
        a = small(a)
        if centre:
            a = a - np.median(a) + lo + 0.2 * (hi - lo)
        x = np.arcsinh(np.clip((a - lo) / (hi - lo), 0, None) * 10) / np.arcsinh(10)
        return (np.clip(x, 0, 1).transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
    tiles = [("input", show(img))]
    sky_in = float(np.median(img))
    for name, out in outs.items():
        # each output with a neutral sky at the input's level: a method may leave its own pedestal per colour, or none
        ped = np.median(out.reshape(3, -1)[:, ::7], axis=1)[:, None, None]
        tiles += [(name, show(out - ped + sky_in)), (name + " model", show(img - out, centre=True))]
    th, tw = tiles[0][1].shape[:2]
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.full((rows * (th + 24), cols * tw, 3), 20, np.uint8)
    for i, (name, t) in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * (th + 24) + 24:r * (th + 24) + 24 + th, c * tw:(c + 1) * tw] = t
        cv2.putText(canvas, name, (c * tw + 6, r * (th + 24) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])


# ---------------------------------------------------------------------------------------------- the run
def run_case(case: str, out_dir: Path, methods: list[str]) -> dict:
    t0 = time.time()
    raw = load_case(case)
    img, lo, span = to_unit(raw)
    del raw
    H, W = img.shape[1:]
    level = sky_level(img)
    grads = gradients(H, W, level)
    obj, area = faint_object(H, W, level)
    res: dict = {"size": [W, H], "sky_level": level, "methods": {}}
    outs_plain = {}
    with tempfile.TemporaryDirectory(prefix=f"grad_{case}_") as td:
        work = Path(td)
        for m in methods:
            fn = METHODS[m]
            r: dict = {}
            try:
                t = time.time()
                base = fn(img, work)
                r["seconds"] = round(time.time() - t, 1)
                outs_plain[m] = base
                bg0 = img - base
                r["flatness_pct"] = round(flatness(base, level), 2)
                for gname, g in grads.items():
                    bg = (img + g) - fn(np.clip(img + g, 0, 1), work)
                    r[f"recovery_{gname}_pct"] = round(recovery(bg, bg0, g), 1)
                r["object_kept_pct"] = round(survival(fn(np.clip(img + obj, 0, 1), work), base, obj, area), 1)
            except Exception as e:  # noqa: BLE001 - one method failing does not stop the bench
                r["error"] = f"{type(e).__name__}: {e}"[:400]
            res["methods"][m] = r
            print(f"  {case} {m}: {r}", flush=True)
    sheet(out_dir / f"sheet_{case}.jpg", img, outs_plain)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--out", default=str(BENCH / "gradient"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")          # our engine on the CPU; GraXpert uses its own GPU
    f = out / "results.json"
    allres = json.loads(f.read_text()) if f.exists() else {}
    for case in a.cases.split(","):
        print(f"{case}", flush=True)
        allres[case] = run_case(case, out, a.methods.split(","))
        f.write_text(json.dumps(allres, indent=1))
    keys = ["recovery_tilt_pct", "recovery_dome_pct", "recovery_corner_pct", "object_kept_pct", "flatness_pct"]
    print("\ncase      method        " + "  ".join(k.replace("_pct", "") for k in keys))
    for case, r in allres.items():
        for m, v in r["methods"].items():
            print(f"{case:9} {m:13} " + "  ".join(f"{v.get(k, float('nan')):>{len(k) - 4}}" for k in keys))
    return 0


if __name__ == "__main__":
    sys.exit(main())
