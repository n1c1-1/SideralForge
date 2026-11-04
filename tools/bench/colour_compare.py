"""Our colour calibration against Siril's (PCC, or SPCC for the cameras of its database), on the bench cases, with one
judge for both (docs/chantiers/couleur.md).

    python tools/bench/colour_compare.py [--cases heart,iris,...]

The judge: on each calibrated linear image, the Gaia XP stars of the field are measured the same way (the aperture
photometry of our Color stage, with a local background), and the white balance the image still needs is fitted the
same way as our stage fits its own (``star_white_balance``: for red and blue, a Theil-Sen line through the stars'
measured colour ratio against the one their spectrum gives, read at the white reference). A perfectly balanced image
needs (1, 1, 1): the red and blue numbers below are what is left to correct, in %. Also the spread of the stars around
that line (the median absolute deviation of log(measured ratio / expected ratio)), which does not depend on the white
reference. The expected colours are the Gaia XP spectra in linear sRGB with a white reference made neutral; both
images are judged against two of them: a G2V star (the Sun; our stage's, and SPCC's "Star, type G2(v)") and the
Average Spiral Galaxy (SPCC's default, SWIRE templates), both from Siril's SPCC database (wb_refs). Our image is fitted
by its own stage on the first criterion with the same stars: its number there is near 0 by construction; what the
comparison shows is how far each calibration stands from each white, and how far apart they are.

Images: ours, the Classic stack workflow of ``ours_express`` (the stack minus its gradient, times the Color stage's
factors); Siril's, ``siril_express/color_linear.fit`` (subsky then pcc / spcc, its own plate solution). The stars are
placed on Siril's image through the sky: our pixel -> our WCS -> RA, Dec -> Siril's WCS.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BENCH = Path("F:/astro_essais/bench")
CASES = ("heart", "iris", "m51", "m8", "rosette", "seestar")
OUT = BENCH / "colour_compare.json"
WB_REFS = Path.home() / "AppData/Local/siril-spcc-database/wb_refs"
WHITES = {"g2v": "Star_g2v.json", "galaxy": "Average_spiral_galaxy.json"}
SOURCE = "Classic stack"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def siril_factors(case: str) -> dict:
    """Siril's white balance factors and what it ran (pcc / spcc), from its log."""
    d = BENCH / case / "siril_express"
    text = (d / "post.log").read_text(encoding="utf-8", errors="replace")
    i = text.rfind("White balance factors:")
    k = [float(text[text.index(f"K{c}:", i) + 3:].split()[0]) for c in range(3)] if i >= 0 else None
    cmd = "spcc" if "running command spcc" in text else "pcc"
    return {"method": cmd, "k": k}


def white_of(name: str) -> np.ndarray:
    """The XYZ (Y = 1) of a white reference spectrum of Siril's SPCC database."""
    import colour
    d = json.loads((WB_REFS / WHITES[name]).read_text(encoding="utf-8"))[0]
    lam = np.asarray(d["wavelength"]["value"], float) * (0.1 if d["wavelength"].get("units") == "angstrom" else 1.0)
    v = np.asarray(d["values"]["value"], float)
    keep = (lam >= 360) & (lam <= 830)
    sd = colour.SpectralDistribution(dict(zip(lam[keep], v[keep])))
    sd = sd.interpolate(colour.SpectralShape(360, 830, 1))
    xyz = colour.sd_to_XYZ(sd)
    return xyz / xyz[1]


def judge(img, x, y, xyz, aperture) -> dict:
    """The white balance an image still needs against each white reference, from its Gaia XP stars, and the stars'
    spread."""
    import colour

    from astro_gpu.stages.color import XYZ_TO_SRGB, bradford, colour_star_table, star_white_balance
    table, why = colour_star_table(img, x, y, aperture, None)
    idx = np.where(why == "")[0]
    w_d65 = colour.xy_to_XYZ(colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"])
    m = table[idx]
    out = {"stars": int(len(idx))}
    for name in WHITES:
        to_out = XYZ_TO_SRGB @ bradford(white_of(name), w_d65)
        expected = xyz[idx] @ to_out.T
        k = star_white_balance(m, expected)["k"]
        out[name] = {"red_pct": round(100 * (k[0] - 1), 1), "blue_pct": round(100 * (k[2] - 1), 1)}
    expected = xyz[idx] @ XYZ_TO_SRGB.T
    ok = (m > 0).all(1) & (expected > 0).all(1)
    for c, nm in ((0, "spread_red"), (2, "spread_blue")):
        r = np.log((m[ok, c] / m[ok, 1]) / (expected[ok, c] / expected[ok, 1]))
        out[nm] = round(float(1.4826 * np.median(np.abs(r - np.median(r)))), 3)
    return out


def case_images(api, case: str):
    """(our calibrated image, Siril's, the stars' positions on each, their XYZ, the aperture)."""
    from astropy.io import fits
    from astropy.wcs import WCS

    p = api.open_project(BENCH / case / "ours_express")
    w = next(x for x in p.workflows if x.name == SOURCE)
    d = {st: Path(api.stage_outputs(p, w.id, st).dir) for st in ("stack", "catalogs", "color", "astrometry",
                                                                 "gradient", "register")}
    full = np.load(d["stack"] / "drz_full.npy").astype(np.float32)
    M = np.load(d["color"] / "M.npy")
    ours = np.einsum("ij,jhw->ihw", M, full)                  # the background is the photometry's annulus's
    xp = np.load(d["catalogs"] / "gaia_xp.npz")
    x, y, xyz = np.asarray(xp["x"], float), np.asarray(xp["y"], float), np.asarray(xp["xyz"], float)
    sky = WCS(fits.getheader(d["astrometry"] / "wcs.fits")).pixel_to_world(x, y)
    with fits.open(BENCH / case / "siril_express" / "color_linear.fit") as h:
        siril = np.asarray(h[0].data, np.float32)
        sw = WCS(h[0].header, naxis=2)
    sx, sy = sw.world_to_pixel(sky)
    fwhm = float(api.stage_outputs(p, w.id, "register").metrics.get("fwhm_px") or 4.0)
    aperture = max(4.0, round(0.7 * fwhm, 1))
    return ours, siril, (x, y), (np.asarray(sx, float), np.asarray(sy, float)), xyz, aperture, \
        api.stage_outputs(p, w.id, "color").metrics.get("white_balance")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default=",".join(CASES))
    a = ap.parse_args(argv)
    os.environ.setdefault("ASTRO_GPU_HOME", str(BENCH / "_state_probe"))
    from astro_gpu import api
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for case in a.cases.split(","):
        try:
            ours, siril, (x, y), (sx, sy), xyz, aperture, our_k = case_images(api, case)
            r = {"aperture_px": aperture, "siril": {**siril_factors(case), **judge(siril, sx, sy, xyz, aperture)},
                 "ours": {"k": our_k, **judge(ours, x, y, xyz, aperture)}}
        except Exception as e:  # noqa: BLE001 - one case failing does not stop the others
            r = {"error": f"{type(e).__name__}: {e}"[:300]}
        res[case] = r
        log(f"{case}: {r}")
        OUT.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
