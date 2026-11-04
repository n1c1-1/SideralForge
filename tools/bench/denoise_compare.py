"""Our denoiser against other tools, on the bench cases, with the measure of denoise_eval.py (docs/chantiers/debruitage.md).

    python tools/bench/denoise_compare.py export                       the half-stacks A and B of each case as FITS
    python tools/bench/denoise_compare.py graxpert                     GraXpert's AI denoising (3.0.2) on A and on B
    python tools/bench/denoise_compare.py topaz                        A and B stretched, 16-bit TIFF, for Topaz
    python tools/bench/denoise_compare.py ours [--model ...] [--adapt ft500@2e-4,ft4000@3e-4]   our denoiser, the same way
    python tools/bench/denoise_compare.py measure                      the measure of every output found

Each tool denoises half-stack A and half-stack B (each on its own, the whole image, linear, before any stretch); the
measure is the one of denoise_eval.py: the denoised A against B and the denoised B against A, the Noise2Noise loss
weighted by the expected noise, over the same held-out zone (the right 20 % of the width), so that the numbers sit
beside those of our methods (F:/astro_essais/denoise_eval/results*.json). The FITS hold the stack's own levels divided
by one factor per case (the brightest value of A and B), so that they span 0..1 as the tools expect.

GraXpert's strength is a blend of its input and its full denoising (checked: strength 0.5 is exactly half of each), so
it runs once at strength 1 and the measure gives its default (0.5), the full one (1.0) and the best strength of each
case (the most favourable setting for GraXpert).

Topaz (a photo tool) works on stretched images: ``topaz`` writes A and B of each case with the background taken off
(a neutral sky), the channels kept in their own proportions, and one stretch known and invertible, the same for A and
B: y = asinh((v + 3 s) / (3 s)) / asinh((vmax + 3 s) / (3 s)), v the level above the background, s the mean noise
(values under -3 s are set to -3 s). The outputs put back in topaz/out under the same name (any suffix, the same for A
and B) are measured after undoing the stretch.

Files: F:/astro_essais/denoise_compare/<case>/{A,B}.fits, meta.json, <tool>_<setting>_{A,B}.fits, results.json;
topaz/in/<case>_{A,B}.tif, topaz/stretch.json, topaz/out/<case>_{A,B}<suffix>.tif.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "bench"))
import denoise_eval as E  # noqa: E402

OUT = Path("F:/astro_essais/denoise_compare")
GRAXPERT = Path.home() / "AppData/Local/Programs/GraXpert/GraXpert.exe"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def meta_of(case: str) -> dict:
    return json.loads((E.DATA / f"bench_{case}" / "pair.json").read_text())


def write_fits(path: Path, img: np.ndarray) -> None:
    from astropy.io import fits
    fits.PrimaryHDU(np.ascontiguousarray(img, np.float32)).writeto(path, overwrite=True)


def read_fits(path: Path) -> np.ndarray:
    from astropy.io import fits
    with fits.open(path) as h:
        d = np.asarray(h[0].data, np.float32)
    return d if d.shape[0] == 3 else np.moveaxis(d, -1, 0)


def export(cases) -> int:
    for case in cases:
        A, B, _N = E.load(case)
        m = meta_of(case)
        bk, sg = (np.asarray(m[k], np.float32)[:, None, None] for k in ("bk", "sg"))
        phys = [np.clip(x * sg + bk, 0, None) for x in (A, B)]
        scale = float(max(p.max() for p in phys))
        d = OUT / case
        d.mkdir(parents=True, exist_ok=True)
        for k, p in zip("AB", phys):
            write_fits(d / f"{k}.fits", p / scale)
        (d / "meta.json").write_text(json.dumps({"case": case, "scale": scale, "bk": m["bk"], "sg": m["sg"],
                                                 "shape": list(A.shape)}, indent=1))
        log(f"{case}: A.fits, B.fits ({A.shape[2]}x{A.shape[1]}), scale {scale:.1f}")
    return 0


def graxpert(cases, strengths) -> int:
    for case in cases:
        d = OUT / case
        for s in strengths:
            for k in "AB":
                out = d / f"graxpert_{s}_{k}"
                if Path(f"{out}.fits").exists():
                    continue
                t = time.time()
                r = subprocess.run([str(GRAXPERT), str(d / f"{k}.fits"), "-cli", "-cmd", "denoising",
                                    "-strength", str(s), "-output", str(out)], capture_output=True, text=True)
                for _ in range(60):                      # the file is written a moment after the program returns
                    if Path(f"{out}.fits").exists():
                        break
                    time.sleep(1)
                found = Path(f"{out}.fits").exists()
                log(f"{case} GraXpert {s} on {k}: {'done' if found else 'NO OUTPUT'} in {time.time() - t:.0f} s")
                if not found:
                    log(r.stdout[-2000:] + r.stderr[-2000:])
    return 0


def topaz_export(cases) -> int:
    import tifffile
    tin = OUT / "topaz" / "in"
    tin.mkdir(parents=True, exist_ok=True)
    (OUT / "topaz" / "out").mkdir(exist_ok=True)
    st_f = OUT / "topaz" / "stretch.json"
    stretch = json.loads(st_f.read_text()) if st_f.exists() else {}
    for case in cases:
        A, B, _N = E.load(case)
        sg = np.asarray(meta_of(case)["sg"], np.float32)[:, None, None]
        s = float(sg.mean())
        vmax = float(max((A * sg).max(), (B * sg).max()))
        top = np.arcsinh((vmax + 3 * s) / (3 * s))
        for k, x in (("A", A), ("B", B)):
            v = np.clip(x * sg, -3 * s, None)
            y = np.arcsinh((v + 3 * s) / (3 * s)) / top
            tifffile.imwrite(tin / f"{case}_{k}.tif", np.round(np.clip(y, 0, 1) * 65535).astype(np.uint16)
                             .transpose(1, 2, 0), photometric="rgb")
        stretch[case] = {"s": s, "vmax": vmax, "sg": sg.ravel().tolist()}
        log(f"{case}: topaz/in/{case}_A.tif, {case}_B.tif")
    st_f.write_text(json.dumps(stretch, indent=1))
    return 0


def topaz_outputs(case: str) -> dict[str, dict[str, np.ndarray]]:
    """The Topaz results of a case put back in topaz/out, in the units of the pairs (the stretch undone), by suffix."""
    import tifffile
    st = json.loads((OUT / "topaz" / "stretch.json").read_text())[case]
    sg = np.asarray(st["sg"], np.float32)[:, None, None]
    s, top = st["s"], np.arcsinh((st["vmax"] + 3 * st["s"]) / (3 * st["s"]))
    found: dict[str, dict[str, np.ndarray]] = {}
    for fa in sorted((OUT / "topaz" / "out").glob(f"{case}_A*.tif*")):
        suffix = fa.name[len(case) + 2:].rsplit(".", 1)[0]
        fb = next((OUT / "topaz" / "out").glob(f"{case}_B{suffix}.tif*"), None)
        if fb is None:
            continue
        found[f"topaz{suffix}"] = {}
        for k, fp in (("A", fa), ("B", fb)):
            y = tifffile.imread(fp).astype(np.float32)
            y = (y[..., :3] / (65535.0 if y.max() > 1.5 else 1.0)).transpose(2, 0, 1)
            found[f"topaz{suffix}"][k] = (3 * s * np.sinh(y * top) - 3 * s) / sg
    return found


def star_mask(A, B):
    """The stars and their surroundings: where the mean of the two halves stands 3 sigma or more above its local median
    (15 px), widened by 4 px. The rest is the sky and the extended objects (nebulae, galaxies)."""
    import cv2
    from scipy.ndimage import median_filter
    m = np.zeros(A.shape[1:], bool)
    for c in range(3):
        x = cv2.GaussianBlur(np.ascontiguousarray(0.5 * (A[c] + B[c])), (0, 0), 1.0)
        m |= x - median_filter(x, 15) > 3.0
    return cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0


def ours(cases, model: Path, device: str, adapt: list[str]) -> int:
    """Our denoiser on the whole of A and of B, written as the tools' outputs (ours_<variant>_{A,B}.fits): the
    pretrained model alone, and adapted to the image (``ftN@lr``, learning on the left 80 % only, as denoise_eval)."""
    from astro_gpu.stages.denoise import UNet
    dev = torch.device(device)
    state = torch.load(model, map_location=dev, weights_only=False)["net"]
    for case in cases:
        d = OUT / case
        m = json.loads((d / "meta.json").read_text())
        bk, sg = (np.asarray(m[k], np.float32)[:, None, None] for k in ("bk", "sg"))
        A, B, N = E.load(case)
        x0 = int(A.shape[2] * E.SPLIT)
        for variant in ["pretrained", *adapt]:
            torch.manual_seed(0)
            net = UNet().to(dev)
            net.load_state_dict(state)
            t = time.time()
            if variant != "pretrained":
                steps, _, lr = variant[2:].partition("@")
                E.learn(net, list(net.parameters()), A, B, N, dev, int(steps), float(lr or 5e-5), x0)
            net.eval()
            for k, x in (("A", A), ("B", B)):
                with torch.no_grad():
                    den = E.apply(net, x, N, dev)
                write_fits(d / f"ours_{variant}_{k}.fits", (den * sg + bk) / m["scale"])
            log(f"{case} ours {variant}: {time.time() - t:.0f} s")
    return 0


def measure(cases) -> int:
    """Each output against the other half, over the held-out zone: everywhere, and without the stars (star_mask)."""
    res_f = OUT / "results.json"
    allres = {}
    for case in cases:
        d = OUT / case
        m = json.loads((d / "meta.json").read_text())
        bk, sg = (np.asarray(m[k], np.float32)[:, None, None] for k in ("bk", "sg"))
        A, B, N = E.load(case)
        x0 = int(A.shape[2] * E.SPLIT)
        zones = {"all": np.ones(A.shape[1:], bool), "no_stars": ~star_mask(A, B)}
        tools = {"none": {"A": A, "B": B}}
        tools.update({t: None for t in sorted({f.stem[:-2] for f in d.glob("*_A.fits")}) if (d / f"{t}_B.fits").exists()})
        if (OUT / "topaz" / "stretch.json").exists():
            tools.update(topaz_outputs(case))
        r = allres.setdefault(case, {})
        for tool, given in tools.items():
            dens = given or {k: (read_fits(d / f"{tool}_{k}.fits") * m["scale"] - bk) / sg for k in "AB"}
            blends = np.round(np.arange(0.0, 1.0001, 0.05), 2) if tool.startswith("graxpert_1.0") else [1.0]
            losses = {z: {} for z in zones}
            for b_ in blends:
                tot = {z: 0.0 for z in zones}
                cnt = {z: 0 for z in zones}
                for k, src, tgt in (("A", A, B), ("B", B, A)):
                    den = b_ * dens[k] + (1 - b_) * src
                    p, t, s_, n = (torch.from_numpy(np.ascontiguousarray(a[:, :, x0:]))[None]
                                   for a in (den, tgt, src, N))
                    L = E.weighted(p, t, s_, n)[0].mean(0).numpy()
                    ok = (N[0, :, x0:] > 0) & (N[0, :, x0:] < 1.6)
                    for z, zm in zones.items():
                        sel = ok & zm[:, x0:]
                        tot[z] += float(L[sel].sum())
                        cnt[z] += int(sel.sum())
                for z in zones:
                    losses[z][float(b_)] = round(tot[z] / max(cnt[z], 1), 4)
            if tool.startswith("graxpert_1.0"):
                for z in zones:
                    best = min(losses[z], key=losses[z].get)
                    r.setdefault("graxpert_0.5", {})[z] = losses[z][0.5]
                    r.setdefault("graxpert_1.0", {})[z] = losses[z][1.0]
                    r.setdefault("graxpert_best", {})[z] = losses[z][best]
                    r.setdefault("graxpert_best_strength", {})[z] = best
                log(f"{case} GraXpert: {r['graxpert_0.5']} (0.5), {r['graxpert_1.0']} (1.0), "
                    f"best {r['graxpert_best']} at {r['graxpert_best_strength']}")
            else:
                r[tool] = {z: losses[z][1.0] for z in zones}
                log(f"{case} {tool}: {r[tool]}")
        r["_stars_share"] = round(float(1 - zones["no_stars"][:, x0:].mean()), 4)
        res_f.write_text(json.dumps(allres, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("export", "graxpert", "topaz", "ours", "measure"))
    ap.add_argument("--model", default=str(E.DATA / "pretrained_unet_v2_mid.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--adapt", default="ft500@2e-4,ft4000@3e-4")
    ap.add_argument("--cases", default=",".join(E.CASES))
    ap.add_argument("--strengths", default="1.0")
    a = ap.parse_args(argv)
    cases = a.cases.split(",")
    if a.cmd == "export":
        return export(cases)
    if a.cmd == "ours":
        return ours(cases, Path(a.model), a.device, [x for x in a.adapt.split(",") if x])
    if a.cmd == "topaz":
        return topaz_export(cases)
    if a.cmd == "graxpert":
        return graxpert(cases, [float(x) for x in a.strengths.split(",")])
    return measure(cases)


if __name__ == "__main__":
    sys.exit(main())
