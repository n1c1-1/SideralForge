"""The training set of our star removal network: starless images of the denoise dataset's fields, made once by
StarNet, then reviewed by hand (docs/chantiers/etoiles.md, the plan of 4 November).

    python tools/bench/stars_dataset.py prepare  [--device cuda:1]     the white balance of every field
    python tools/bench/stars_dataset.py generate [--only name,...] [--device cuda:1]
    python tools/bench/stars_dataset.py review                         the TIFFs to review, one per field

The images are what the Stars block will receive in the chain: the stack minus its sky gradient, white balanced and
denoised.

``prepare``: the white balance of each field (the pairs of one set at several noise levels share their stars and
their colours). The dataset's projects (F:/astro_essais/dataset_projets) are run by the engine up to the Color block on
their whole stack (the target named as the dataset's catalogue names it), with the Gaia stars limited to G 15 so
that the offline Gaia extract is enough (run offline: ASTRO_GPU_OFFLINE=1); the user's projects
(F:/astro_projets, read only: nothing is written there) get the Color stage's white balance computed here from their
own stack, plate solution and Gaia XP stars. Result: F:/astro_essais/stars_dataset_v2/colour.json.

``generate``: for every training pair of ``F:/astro_essais/denoise_dataset`` (the bench's sources held out, as for the
denoiser: ``denoise_dataset.HELD_OUT``; the two Stellina sets left out: EXCLUDED), from its stack: the sky gradient
modelled by the engine's Gradient method (polynomial, its defaults) on the whole stack and taken off both half-stacks;
each channel minus its background over its noise; denoised by the shipped network and averaged; then each channel
times its white balance factor and its noise, over the mean of those (one unit for the three channels: noise sigmas,
colours balanced). StarNet 2 (``--linear``) removes its stars; it takes values in 0..1: the image goes in as (v +
OFFSET) / (top + OFFSET), clipped at 0, and comes back the same way. StarNet is a tool of the data preparation only:
the software never calls it.

``review``: for each field, the deepest level as two 16-bit TIFFs stretched by one known curve (y = asinh((v + OFFSET)
/ SOFT) / asinh((top + OFFSET) / SOFT)): the original and StarNet's starless image, to correct by hand, with nothing
global changed (levels, colours, sharpness). The pixels left as they were keep their full precision on the way back.

Files: F:/astro_essais/stars_dataset_v2/<pair>/{input.npy, starless.npy, meta.json};
F:/astro_essais/stars_dataset_v2/_revue/<field>_{original,starless}.tif and stretch.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "bench"))

DATA = Path("F:/astro_essais/denoise_dataset")
PROJECTS = Path("F:/astro_essais/dataset_projets")
USER = Path("F:/astro_projets")
DATASET_CATALOG = Path("F:/dataset_astro/catalog.csv")
OUT = Path("F:/astro_essais/stars_dataset_v2")
STARNET = Path.home() / "astro_gpu_scratch/tools/starnet2_2.6.2/starnet2_win_2.6.2-0242_ORT_x64_cli/starnet2.exe"
OFFSET = 3.0           # noise sigmas below the background kept (the sky sits at OFFSET / (top + OFFSET))
SOFT = 3.0             # the review stretch's softness, in noise sigmas
LEVELS = ("", "_half", "_quarter", "_eighth")
#: left out after the review of 4 November: set 14 (registered on its hot pixels: no darks, stars smeared into a
#: trail), set 15 (faint target, coloured blotches of uncorrected hot pixels), set 18 (0.5 s frames, most of them not
#: aligned: too few measurable stars for any white balance)
EXCLUDED = ("set_14_", "set_15_", "set_18_")
#: the Gaia stars of the Color block limited to what the offline extract holds (no download)
G_LIMIT = 15.0


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def pairs(only=None) -> list[dict]:
    from denoise_dataset import HELD_OUT
    out = []
    for f in sorted(DATA.glob("*/pair.json")):
        m = json.loads(f.read_text())
        if m["role"] != "train" or m["name"].startswith(HELD_OUT + EXCLUDED):
            continue
        if only and m["name"] not in only:
            continue
        out.append(m)
    return out


def field_of(name: str) -> str:
    for lv in LEVELS[1:]:
        if name.endswith(lv):
            return name[:-len(lv)]
    return name


def write_fits(path: Path, img: np.ndarray) -> None:
    from astropy.io import fits
    fits.PrimaryHDU(np.ascontiguousarray(img, np.float32)).writeto(path, overwrite=True)


def read_fits(path: Path) -> np.ndarray:
    from astropy.io import fits
    with fits.open(path) as h:
        d = np.asarray(h[0].data, np.float32)
    return d if d.shape[0] == 3 else np.moveaxis(d, -1, 0)


def valid_of(wt: np.ndarray, cover_min: float = 0.6) -> np.ndarray:
    """The covered pixels, as the Gradient stage's ``coverage``."""
    from scipy import ndimage
    w = wt[1]
    good = w > cover_min * float(np.median(w[w > 0]))
    lab, n = ndimage.label(~good)
    if n:
        border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        good |= (lab > 0) & ~np.isin(lab, border[border > 0]) & (w > 0)
    return good


def sky_variation_of(full: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """The engine's Gradient model (its default method and settings) of a stack, as a variation of median 0."""
    from astro_gpu.stages.gradient import GradientParams, read_grid, sky_model_of
    H, W = full.shape[1:]
    m = sky_model_of(full, valid, GradientParams())
    var = read_grid(m["grid"], m["info"], H, W).astype(np.float32)
    for c in range(3):
        var[c] -= float(np.median(var[c][valid]))
    return var


# ---------------------------------------------------------------------------------------------- prepare
def user_white_balance(name: str, stack_dir: Path) -> dict:
    """The Color stage's white balance of one of the user's projects, computed here (nothing written there): its
    stack minus its gradient, its Gaia XP stars (its catalogue result), the stage's photometry and fit."""
    import colour as colour_lib

    from astro_gpu.stages.color import XYZ_TO_SRGB, bradford, colour_star_table, star_white_balance, white_xyz
    store = USER / name / "store"
    cats = sorted((store / "catalogs").glob("*/gaia_xp.npz"), key=lambda f: f.stat().st_mtime)
    if not cats:
        return {}
    xp = np.load(cats[-1])
    full = np.load(stack_dir / "drz_full.npy").astype(np.float32)
    wt = np.load(stack_dir / "drz_weight.npy", mmap_mode="r")
    full -= sky_variation_of(full, valid_of(np.asarray(wt, np.float32)))
    table, why = colour_star_table(full, np.asarray(xp["x"], float), np.asarray(xp["y"], float), 5.0, None)
    idx = np.where(why == "")[0]
    if len(idx) < 50:
        return {}
    w_d65 = colour_lib.xy_to_XYZ(colour_lib.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"])
    expected = np.asarray(xp["xyz"], float)[idx] @ (XYZ_TO_SRGB @ bradford(white_xyz(5778.0), w_d65)).T
    wb = star_white_balance(table[idx], expected)
    return {"k": [round(float(v), 4) for v in wb["k"]], "stars": wb["stars"], "source": "computed here (Color stage)"}


def catalogue_targets() -> dict[str, str]:
    """The target of each set, from the dataset's catalogue (its first name: "Sh2-129 / Ou4 Squid" -> "Sh2-129")."""
    import csv
    with open(DATASET_CATALOG, encoding="utf-8-sig") as fh:
        return {r["id"]: r["target"].split(" / ")[0].split(" (")[0].split(" - ")[0].strip()
                for r in csv.DictReader(fh, delimiter=";")}


def star_median_balance(stack_dir: Path) -> dict:
    """The fallback white balance when Gaia is out of reach: the bright unsaturated stars of the stack (its gradient
    removed) measured by the Color stage's photometry, each colour scaled so that their median colour is neutral."""
    import cv2
    from scipy.ndimage import maximum_filter

    from astro_gpu.stages.color import colour_star_table
    full = np.load(stack_dir / "drz_full.npy").astype(np.float32)
    wt = np.asarray(np.load(stack_dir / "drz_weight.npy", mmap_mode="r"), np.float32)
    full -= sky_variation_of(full, valid_of(wt))
    L = cv2.GaussianBlur(full.mean(0), (0, 0), 1.5)
    hp = L - cv2.GaussianBlur(L, (0, 0), 8)
    s = 1.4826 * float(np.median(np.abs(hp[::4, ::4] - np.median(hp[::4, ::4]))))
    # bright but not saturated: the raw peak of every colour under 40 % of its brightest pixels, the Color stage's
    # rule being 45 % (on a wide field the brightest 3000 were all saturated: NGC 7000 at 135 mm)
    peak = maximum_filter(full, size=(1, 3, 3))
    chmax = np.array([np.percentile(full[c][::3, ::3], 99.999) for c in range(3)])[:, None, None]
    unsat = (peak < 0.4 * chmax).all(0)
    del peak
    pk = (hp > 30 * s) & (L == maximum_filter(L, 9)) & unsat
    y, x = np.nonzero(pk)
    order = np.argsort(-hp[y, x])[:3000]
    table, why = colour_star_table(full, x[order].astype(float), y[order].astype(float), 5.0, None)
    t = table[why == ""]
    if len(t) < 50:
        return {}
    k = np.array([np.median(t[:, 1] / t[:, 0]), 1.0, np.median(t[:, 1] / t[:, 2])])
    return {"k": [round(float(v), 4) for v in k], "stars": int(len(t)), "source": "median star made neutral (no Gaia)"}


def prepare(device: str) -> int:
    """The engine's own home (ASTRO_GPU_HOME unset): its offline Gaia extract and sky cache are there; run with
    ASTRO_GPU_OFFLINE=1 so that a field the extract does not cover fails at once instead of waiting on the services."""
    import make_test_project as mtp

    from astro_gpu import api
    targets = catalogue_targets()
    OUT.mkdir(parents=True, exist_ok=True)
    cf = OUT / "colour.json"
    res = json.loads(cf.read_text()) if cf.exists() else {}
    fields = {field_of(m["name"]): m for m in pairs() if not m["name"][len(field_of(m["name"])):]}
    for fld, m in sorted(fields.items()):
        if res.get(fld, {}).get("k"):
            continue
        t = time.time()
        try:
            if fld.startswith("user_"):
                r = user_white_balance(fld[len("user_"):], Path(m["source"]))
            else:
                p = api.open_project(PROJECTS / fld[len("set_"):])
                tgt = targets.get(fld[len("set_"):], "")
                if tgt and p.target.name != tgt:                # the night named it after its folder
                    p.target.name = tgt
                w = next(x for x in p.workflows if x.name == "Stack")
                for k in ("g_deep", "g_faint"):
                    api.set_param(p, [w.id], "catalogs", k, G_LIMIT)
                api.save(p)
                mtp.run(api, p, api.plan_to_block(p, w.id, "color", device), device)
                o = api.stage_outputs(p, w.id, "color")
                if o is None:
                    r = {"why": "the Color block did not finish (see the log)"}
                else:
                    M = np.load(Path(o.dir) / "M.npy")
                    k = np.diag(M) / M[1, 1]
                    r = {"k": [round(float(v), 4) for v in k], "stars": o.metrics.get("white_balance_stars"),
                         "source": str(o.dir)}
        except Exception as e:  # noqa: BLE001 - one field failing does not stop the others
            r = {"why": f"{type(e).__name__}: {e}"[:300]}
        if not r.get("k"):                                   # Gaia out of reach: the stars' median made neutral
            fb = star_median_balance(Path(m["source"]))
            if fb:
                r = {**fb, "gaia_failed": r.get("why", "")}
        r["minutes"] = round((time.time() - t) / 60, 1)
        res[fld] = r
        cf.write_text(json.dumps(res, indent=1))
        log(f"{fld}: {r}")
    ok = sum(1 for v in res.values() if v.get("k"))
    log(f"white balance for {ok} of {len(res)} fields")
    return 0


# ---------------------------------------------------------------------------------------------- generate
def generate(only, device: str) -> int:
    import torch

    from astro_gpu.stages.denoise import apply_net, norm_params, pretrained
    dev = torch.device(device)
    net = pretrained(dev)
    net.eval()
    zero, one = np.zeros(3, np.float32), np.ones(3, np.float32)
    cf = OUT / "colour.json"
    colours = json.loads(cf.read_text()) if cf.exists() else {}
    for m in pairs():                                       # a field still without a white balance: the fallback
        fld = field_of(m["name"])
        if m["name"] == fld and not colours.get(fld, {}).get("k"):
            fb = star_median_balance(Path(m["source"]))
            if fb:
                colours[fld] = {**fb, "gaia_failed": colours.get(fld, {}).get("why", "")}
                cf.write_text(json.dumps(colours, indent=1))
                log(f"{fld}: white balance {fb}")
    for m in pairs(only):
        d = OUT / m["name"]
        if (d / "meta.json").exists():
            continue
        t = time.time()
        src = Path(m["source"])
        wt = np.load(src / "drz_weight.npy").astype(np.float32)
        valid = valid_of(wt)
        var = sky_variation_of(np.load(src / "drz_full.npy").astype(np.float32), valid)
        A = np.load(src / "drz_A.npy").astype(np.float32) - var
        B = np.load(src / "drz_B.npy").astype(np.float32) - var
        del var
        bk, sg, nm = norm_params(A, wt)
        A = (A - bk[:, None, None]) / sg[:, None, None]
        B = (B - bk[:, None, None]) / sg[:, None, None]
        with torch.no_grad():
            img = 0.5 * (apply_net(net, A, nm, zero, one, dev) + apply_net(net, B, nm, zero, one, dev))
        del A, B
        k = np.asarray(colours.get(field_of(m["name"]), {}).get("k") or [1.0, 1.0, 1.0], np.float32)
        gain = k * sg / float(np.mean(k * sg))                  # balanced colours, one unit for the three channels
        img *= gain[:, None, None]
        img[:, nm <= 0] = 0.0                                  # outside the covered field
        t1 = time.time()
        top = float(img.max())
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "input.npy", img.astype(np.float16))
        write_fits(d / "_in.fits", np.clip((img + OFFSET) / (top + OFFSET), 0, 1))
        r = subprocess.run([str(STARNET), "--input", str(d / "_in.fits"), "--output", str(d / "_sl.fits"), "--linear",
                            "--quiet"], capture_output=True, text=True, cwd=str(STARNET.parent))
        if not (d / "_sl.fits").exists():
            log(f"{m['name']}: StarNet FAILED {(r.stdout + r.stderr)[-600:]}")
            continue
        sl = read_fits(d / "_sl.fits") * (top + OFFSET) - OFFSET
        sl[:, nm <= 0] = 0.0
        np.save(d / "starless.npy", sl.astype(np.float16))
        for f in ("_in.fits", "_sl.fits"):
            (d / f).unlink()
        meta = {"name": m["name"], "field": field_of(m["name"]), "shape": list(img.shape), "top": top,
                "offset": OFFSET, "white_balance": k.tolist(), "white_balanced": field_of(m["name"]) in colours
                and bool(colours[field_of(m["name"])].get("k")), "gain": gain.tolist(),
                "prep_s": round(t1 - t, 1), "starnet_s": round(time.time() - t1, 1),
                "units": "noise sigmas (the mean of the three channels' after the white balance), background 0"}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        log(f"{m['name']}: {img.shape[2]}x{img.shape[1]}, prepared {meta['prep_s']} s, StarNet {meta['starnet_s']} s"
            f"{'' if meta['white_balanced'] else ', no white balance'}")
    return 0


# ---------------------------------------------------------------------------------------------- review
def review(dest: Path | None = None, watch: bool = False) -> int:
    """The review TIFFs of every field ready (its deepest level made), into ``dest`` (default OUT/_revue); with
    ``watch``, again every minute as the generation goes on, until every field has its TIFFs (a field already written
    is not written again)."""
    import tifffile
    rv = dest or OUT / "_revue"
    rv.mkdir(parents=True, exist_ok=True)
    sf = rv / "stretch.json"
    stretch = json.loads(sf.read_text()) if sf.exists() else {}
    expected = {field_of(m["name"]) for m in pairs()}
    while True:
        fields: dict[str, str] = {}
        for f in sorted(OUT.glob("*/meta.json")):
            m = json.loads(f.read_text())
            name, fld = m["name"], m["field"]
            if fld not in fields or LEVELS.index(name[len(fld):]) < LEVELS.index(fields[fld][len(fld):]):
                fields[fld] = name
        for fld, name in sorted(fields.items()):
            if stretch.get(fld, {}).get("pair") == name and (rv / f"{fld}_starless.tif").exists():
                continue
            d = OUT / name
            m = json.loads((d / "meta.json").read_text())
            top = np.arcsinh((m["top"] + OFFSET) / SOFT)
            for kind, fname in (("original", "input.npy"), ("starless", "starless.npy")):
                v = np.load(d / fname).astype(np.float32)
                y = np.arcsinh(np.clip(v + OFFSET, 0, None) / SOFT) / top
                tifffile.imwrite(rv / f"{fld}_{kind}.tif", np.round(np.clip(y, 0, 1) * 65535).astype(np.uint16)
                                 .transpose(1, 2, 0), photometric="rgb")
            stretch[fld] = {"pair": name, "top": m["top"], "offset": OFFSET, "soft": SOFT,
                            "white_balance": m.get("white_balance"), "white_balanced": m.get("white_balanced")}
            sf.write_text(json.dumps(stretch, indent=1))
            log(f"{fld}: review TIFFs from {name}")
        if not watch or expected <= set(stretch):
            break
        time.sleep(60)
    log(f"{len(stretch)} fields in {rv}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("prepare", "generate", "review"))
    ap.add_argument("--only", default="")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dest", default="", help="review: where the TIFFs go (default: the dataset's _revue)")
    ap.add_argument("--watch", action="store_true", help="review: again every minute until every field is there")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        return prepare(a.device)
    if a.cmd == "generate":
        return generate([x for x in a.only.split(",") if x] or None, a.device)
    return review(Path(a.dest) if a.dest else None, a.watch)


if __name__ == "__main__":
    sys.exit(main())
