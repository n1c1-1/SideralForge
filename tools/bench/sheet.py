"""Side-by-side sheet of the finals: Siril's (final.tif, linked autostretch) against ours (export bench_final, the
engine's suggested rendering), on the same sky.

    python tools/bench/sheet.py CASE LEVEL

Ours is resampled onto Siril's framing through both astrometric solutions (overview), and the crops show the same
sky at our native resolution (Siril's pixels enlarged, bicubic). Crops: the target, the field centre, a corner, a
bright star. Writes <case>/sheet_<level>.jpg.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as C  # noqa: E402

os.environ["USERPROFILE"] = str(C.HOME)
os.environ["ASTRO_GPU_HOME"] = str(C.AG_HOME)
sys.path.insert(0, str(C.REPO))

from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

FONT = "C:/Windows/Fonts/arial.ttf"


def _tif(path: Path) -> np.ndarray:
    import tifffile
    a = tifffile.imread(str(path))
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    a = a[..., :3].astype(np.float32)
    return a / (65535.0 if a.max() > 255 else 255.0)


def _lum_small(a: np.ndarray, f: int) -> np.ndarray:
    H, W = a.shape[:2]
    a = a[: H // f * f, : W // f * f]
    return a.reshape(H // f, f, W // f, f, -1).mean((1, 3, 4)) if a.ndim == 3 else \
        a.reshape(H // f, f, W // f, f).mean((1, 3))


def siril_side(case: str, level: str):
    d = C.BENCH / case / f"siril_{level}"
    img = _tif(d / "final.tif")
    with fits.open(d / "color_linear.fit") as h:
        lin = np.asarray(h[0].data, np.float32)
        wcs = WCS(h[0].header, naxis=2)
    H = lin.shape[-2]
    Ll = lin.mean(0) if lin.ndim == 3 else lin
    a = _lum_small(img, 16)
    b = _lum_small(Ll, 16)
    n = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[: n[0], : n[1]], b[: n[0], : n[1]]
    bf = _lum_small(Ll[::-1], 16)[: n[0], : n[1]]

    def corr(u, v):
        u, v = np.argsort(np.argsort(u.ravel())), np.argsort(np.argsort(v.ravel()))
        return float(np.corrcoef(u, v)[0, 1])
    flipped = corr(a, bf) > corr(a, b)
    return img, wcs, H, flipped


def ours_side(case: str, level: str):
    from astro_gpu import api
    root = C.BENCH / case / f"ours_{level}"
    p = api.open_project(root)
    w = next(w for w in p.workflows if w.name == "Classic stack")
    rs = Path(api.stage_outputs(p, w.id, "render_simple").dir)
    ast = Path(api.stage_outputs(p, w.id, "astrometry").dir)
    prep = json.loads((rs / "prep_simple.json").read_text(encoding="utf-8"))
    x0, x1, y0, y1 = prep["crop"]
    tifs = sorted((root / "export" / "bench_final").glob("*.tif"))
    img = _tif(tifs[-1])
    if img.shape[0] != y1 - y0 or img.shape[1] != x1 - x0:
        raise SystemExit(f"our export {img.shape} does not match the crop {prep['crop']}")
    return img, WCS(fits.getheader(ast / "wcs.fits"), naxis=2), (x0, y0)


def render(case: str, level: str, s_img, s_wcs, s_H, s_flip, o_img, o_wcs, o_org, u0, v0, w, h, scale):
    """Both finals on a display grid: (u, v) display pixels of Siril's final at `scale` display px per Siril px,
    the window starting at Siril-final pixel (u0, v0), size w x h display px."""
    step = 8
    gu = np.arange(0, w + step, step)
    gv = np.arange(0, h + step, step)
    GU, GV = np.meshgrid(gu, gv)
    fx = u0 + GU / scale                      # Siril final pixel
    fy = v0 + GV / scale
    fits_y = (s_H - 1 - fy) if s_flip else fy
    ra, dec = s_wcs.all_pix2world(fx.ravel(), fits_y.ravel(), 0)
    ox, oy = o_wcs.all_world2pix(ra, dec, 0)
    ox = ox.reshape(GU.shape) - o_org[0]
    oy = oy.reshape(GU.shape) - o_org[1]
    # dense maps
    dense = lambda m: ndi.zoom(m, ((h + step) / m.shape[0], (w + step) / m.shape[1]), order=1)[:h, :w]  # noqa: E731
    FX, FY, OX, OY = dense(fx), dense(fy), dense(ox), dense(oy)
    s = np.stack([ndi.map_coordinates(s_img[..., c], [FY, FX], order=3, mode="constant", cval=0) for c in range(3)],
                 -1)
    o = np.stack([ndi.map_coordinates(o_img[..., c], [OY, OX], order=1, mode="constant", cval=0) for c in range(3)],
                 -1)
    return np.clip(s, 0, 1), np.clip(o, 0, 1)


def make(case: str, level: str) -> Path:
    man = json.loads((C.BENCH / case / "manifest.json").read_text(encoding="utf-8"))
    s_img, s_wcs, s_H, s_flip = siril_side(case, level)
    o_img, o_wcs, o_org = ours_side(case, level)
    SH, SW = s_img.shape[:2]
    s_scale = float(np.abs(np.linalg.det(s_wcs.pixel_scale_matrix)) ** 0.5 * 3600)
    o_scale = float(np.abs(np.linalg.det(o_wcs.pixel_scale_matrix)) ** 0.5 * 3600)
    OW = 1000
    ov_scale = OW / SW
    ov = render(case, level, s_img, s_wcs, s_H, s_flip, o_img, o_wcs, o_org, 0, 0, OW, int(SH * ov_scale), ov_scale)
    # crops at our native resolution
    cs = s_scale / o_scale
    cw = 480

    def sky_to_final(ra, dec):
        x, y = s_wcs.all_world2pix([ra], [dec], 0)
        y = (s_H - 1 - y) if s_flip else y
        return float(x[0]), float(y[0])

    crops = []
    obj = C.CASES[case].get("obj") or (man["ra"], man["dec"])
    tx, ty = sky_to_final(*obj)
    centres = [(f"cible ({man['target']})", tx, ty), ("centre du champ", SW / 2, SH / 2),
               ("coin haut gauche", 0.12 * SW, 0.12 * SH)]
    # a bright star: the brightest Gaia star well inside the field
    try:
        from astro_gpu import api
        p = api.open_project(C.BENCH / case / f"ours_{level}")
        w = next(w for w in p.workflows if w.name == "Classic stack")
        cat = np.load(Path(api.stage_outputs(p, w.id, "catalogs").dir) / "gaia_deep.npz")
        x, y = s_wcs.all_world2pix(cat["ra"], cat["dec"], 0)
        y = (s_H - 1 - y) if s_flip else y
        ok = (x > 0.15 * SW) & (x < 0.85 * SW) & (y > 0.15 * SH) & (y < 0.85 * SH) & (cat["g"] > 4)
        if ok.any():
            k = np.where(ok)[0][np.argmin(cat["g"][ok])]
            centres.append((f"étoile brillante (G {cat['g'][k]:.1f})", float(x[k]), float(y[k])))
    except Exception as e:  # noqa: BLE001
        print(f"no bright star crop: {e}")
    for name, cx, cy in centres[:4]:
        half = cw / cs / 2
        u0, v0 = cx - half, cy - half
        crops.append((name, render(case, level, s_img, s_wcs, s_H, s_flip, o_img, o_wcs, o_org, u0, v0, cw, cw, cs)))
    # compose
    gap, top = 12, 46
    ovh = ov[0].shape[0]
    W = 2 * OW + 3 * gap
    rows = (len(crops) + 1) // 2
    Hs = top + ovh + gap + 30 + rows * (cw + 30 + gap) + 40
    sheet = Image.new("RGB", (W, Hs), (24, 24, 24))
    dr = ImageDraw.Draw(sheet)
    f1, f2 = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 16)
    lv = man["levels"][level]
    dr.text((gap, 10), f"{case} · {level} · {lv['count']} poses · {man['camera_label']} · {man['scope']}",
            fill=(235, 235, 235), font=f1)

    def put(a, x, y):
        sheet.paste(Image.fromarray((a * 255 + 0.5).astype(np.uint8)), (x, y))
    put(ov[0], gap, top)
    put(ov[1], 2 * gap + OW, top)
    dr.text((gap + 6, top + 6), "Siril 1.4 (recette de base)", fill=(255, 220, 120), font=f1)
    dr.text((2 * gap + OW + 6, top + 6), "astro_gpu · Classic stack", fill=(120, 220, 255), font=f1)
    y = top + ovh + gap
    dr.text((gap, y), f"Recadrages à 100 % de notre grille ({o_scale:.2f}\"/px) ; Siril ({s_scale:.2f}\"/px) agrandi "
                      f"x{cs:.2f} pour montrer la même zone du ciel. Gauche : Siril, droite : astro_gpu.",
            fill=(200, 200, 200), font=f2)
    y += 30
    for i, (name, (a, b)) in enumerate(crops):
        col = i % 2
        if col == 0 and i:
            y += cw + 30 + gap
        x = gap + col * (OW + gap)
        dr.text((x, y), name, fill=(220, 220, 220), font=f2)
        put(a, x, y + 22)
        put(b, x + cw + 8, y + 22)
    out = C.BENCH / case / f"sheet_{level}.jpg"
    sheet.save(out, quality=90)
    print(f"sheet {out}")
    return out


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
