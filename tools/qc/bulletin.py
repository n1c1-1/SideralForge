"""Report card of a processed test set: the measurements the stages recorded, the export's quality checks, and
measurements taken on the rendered image itself (sky flatness and colour cast, star size and roundness). None of
them needs a reference image: they flag what is clearly wrong, they do not say what is beautiful.

    python tools/qc/bulletin.py PROJECT [PROJECT ...] --render NAME [--sheet] [--out FILE.json]

--sheet also writes <project>/results/<NAME>/<NAME>_planche.jpg: the whole image and four 100 % crops (centre, a
corner, the brightest star, the busiest zone), for the eye.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

#: stage metric -> short name in the report card
KEYS = {
    ("inventory", "camera"): "camera", ("inventory", "n_lights"): "lights", ("inventory", "exposure_s"): "pose_s",
    ("register", "aligned"): "aligned", ("register", "fwhm_px"): "fwhm_px", ("register", "rms_px"): "align_rms_px",
    ("astrometry", "scale_arcsec"): "arcsec_px", ("astrometry", "residual_px"): "astrom_res_px",
    ("astrometry", "stars_used"): "astrom_stars", ("stack", "frames_used"): "stacked",
    ("stack", "rejected_pct"): "rejected_pct", ("color", "response_source"): "colour_source",
    ("color", "fit_agreement_pct"): "colour_fit_pct", ("color", "chromaticity_residual"): "colour_chroma_res",
    ("color", "stars"): "colour_stars", ("optics", "psf_residual_pct"): "psf_res_pct",
    ("stars", "prediction_scatter_pct"): "star_pred_scatter_pct", ("stars", "modelled_stars"): "stars_modelled",
    ("background", "flat_error_pct"): "flat_error_pct", ("background", "sky_fit_blocks_pct"): "sky_blocks_pct",
    ("denoise", "noise_reduction"): "noise_reduction", ("starless", "filled_pct"): "starless_filled_pct",
    ("render_prep", "vignetted_pct"): "vignetted_pct",
}


def stage_metrics(proj: Path) -> dict:
    out, status = {}, {}
    for m in sorted((proj / "work").glob("*/manifest.json")):
        try:
            j = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        status[m.parent.name] = j.get("status")
        for (st, k), short in KEYS.items():
            if st == m.parent.name and k in j.get("metrics", {}):
                out[short] = j["metrics"][k]
    out["stages_not_done"] = [k for k, v in status.items() if v != "done"]
    return out


def _read(png: Path) -> np.ndarray:
    a = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    return (a[:, :, ::-1].astype(np.float32) / (65535.0 if a.dtype == np.uint16 else 255.0))


def sky_measures(img: np.ndarray) -> dict:
    """Sky floor per block (8 x 8 grid): its spread (leftover gradient or vignetting) and its colour (cast), in 0-255
    units. The floor of a block is the median of its darkest fifth, stars and nebula mostly left out."""
    H, W, _ = img.shape
    Y = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    n = 8
    lv, col = [], []
    for i in range(n):
        for j in range(n):
            b = (slice(i * H // n, (i + 1) * H // n), slice(j * W // n, (j + 1) * W // n))
            y = Y[b]
            m = y <= np.percentile(y, 20)
            lv.append(float(np.median(y[m])))
            col.append(np.median(img[b][m], axis=0))
    lv, col = np.array(lv) * 255, np.array(col) * 255
    return {"sky_floor_255": round(float(np.median(lv)), 1),
            "sky_spread_255": round(float(np.percentile(lv, 90) - np.percentile(lv, 10)), 1),
            "sky_cast_rg_bg_255": [round(float(np.median(col[:, 0] - col[:, 1])), 1),
                                   round(float(np.median(col[:, 2] - col[:, 1])), 1)]}


def star_measures(img: np.ndarray) -> dict:
    """Size (FWHM, px at sensor scale) and elongation of the isolated, unsaturated stars of the render, from their
    second moments."""
    Y = img.mean(2)
    bg = cv2.medianBlur((np.clip(Y, 0, 1) * 255).astype(np.uint8), 15).astype(np.float32) / 255
    d = Y - bg
    from scipy.ndimage import maximum_filter
    pk = (d == maximum_filter(d, size=11)) & (d > 0.15) & (Y < 0.85)
    ys, xs = np.nonzero(pk)
    r = 6
    ok = (xs > r) & (xs < Y.shape[1] - r) & (ys > r) & (ys < Y.shape[0] - r)
    ys, xs = ys[ok], xs[ok]
    if len(xs) > 3000:
        k = np.argsort(-d[ys, xs])[:3000]
        ys, xs = ys[k], xs[k]
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    fw, el = [], []
    for y, x in zip(ys, xs):
        s = d[y - r:y + r + 1, x - r:x + r + 1].clip(0)
        s = np.where(s > 0.1 * s.max(), s, 0)            # the star, not its neighbours' wings
        if (s > 0.5 * d[y, x]).sum() < 2:
            continue
        w = s.sum()
        mx, my = (s * xx).sum() / w, (s * yy).sum() / w
        sxx, syy = (s * (xx - mx) ** 2).sum() / w, (s * (yy - my) ** 2).sum() / w
        sxy = (s * (xx - mx) * (yy - my)).sum() / w
        t, dt = sxx + syy, np.sqrt(max((sxx - syy) ** 2 + 4 * sxy ** 2, 0))
        a, b = np.sqrt(max((t + dt) / 2, 1e-6)), np.sqrt(max((t - dt) / 2, 1e-6))
        fw.append(2.355 * np.sqrt((a * a + b * b) / 2))
        el.append(1 - b / a)
    if not fw:
        return {"stars_measured": 0}
    return {"stars_measured": len(fw), "star_fwhm_px": round(float(np.median(fw)), 2),
            "star_elongation": round(float(np.median(el)), 3),
            "star_elongation_p90": round(float(np.percentile(el, 90)), 3)}


def sheet(img: np.ndarray, out: Path, title: str) -> None:
    from PIL import Image, ImageDraw
    H, W, _ = img.shape
    u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    ov_w = 1200
    ov = cv2.resize(u8, (ov_w, round(H * ov_w / W)), interpolation=cv2.INTER_AREA)
    c = 400
    Y = img.mean(2)
    ys, xs = np.unravel_index(np.argmax(cv2.GaussianBlur(Y, (0, 0), 3)), Y.shape)
    busy = cv2.blur(np.abs(cv2.Laplacian(Y, cv2.CV_32F)), (c, c))
    by, bx = np.unravel_index(np.argmax(busy[c // 2:-c // 2, c // 2:-c // 2]), busy[c // 2:-c // 2, c // 2:-c // 2].shape)
    centres = {"centre": (H // 2, W // 2), "corner": (c // 2 + 20, c // 2 + 20), "brightest": (ys, xs),
               "busiest": (by + c // 2, bx + c // 2)}
    crops = []
    for k, (y, x) in centres.items():
        y0, x0 = int(np.clip(y - c // 2, 0, max(H - c, 0))), int(np.clip(x - c // 2, 0, max(W - c, 0)))
        crops.append((k, u8[y0:y0 + c, x0:x0 + c]))
    sh = Image.new("RGB", (max(ov_w, 3 * c), ov.shape[0] + 40 + c), (40, 40, 40))
    sh.paste(Image.fromarray(ov), (0, 30))
    dr = ImageDraw.Draw(sh)
    dr.text((8, 8), title, fill=(230, 230, 230))
    for i, (k, cr) in enumerate(crops[:3]):
        sh.paste(Image.fromarray(cr), (i * c, ov.shape[0] + 40))
        dr.text((i * c + 6, ov.shape[0] + 42), k + " 100 %", fill=(255, 255, 0))
    sh.save(out, quality=90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("projects", nargs="+")
    ap.add_argument("--render", required=True)
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    allr = {}
    for p in a.projects:
        proj = Path(p)
        r = {"project": proj.name, **stage_metrics(proj)}
        d = proj / "results" / a.render
        sj = d / f"{a.render}_settings.json"
        if sj.exists():
            qa = json.loads(sj.read_text(encoding="utf-8"))["qa"]
            r.update({k: qa.get(k) for k in ("pure_black_pct", "pure_white_pct", "burnt_highlights_worst_zone_pct",
                                              "dark_dips_at_stars", "dark_dips_blind", "warnings")})
            img = _read(d / f"{a.render}.png")
            r.update(sky_measures(img))
            r.update(star_measures(img))
            if a.sheet:
                sheet(img, d / f"{a.render}_planche.jpg", f"{proj.name} - {a.render}")
        else:
            r["render"] = "missing"
        allr[proj.name] = r
        print(json.dumps(r, ensure_ascii=False), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(allr, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
