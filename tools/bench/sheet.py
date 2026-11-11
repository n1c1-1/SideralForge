"""Side-by-side sheet of the finals: Siril's (final.tif, linked autostretch) against ours (export bench_final, the
engine's suggested rendering), on the same sky.

    python tools/bench/sheet.py CASE LEVEL

Ours is resampled onto Siril's framing through both astrometric solutions (overview), and the crops show the same
sky at our native resolution (Siril's pixels enlarged, bicubic). Crops: the target, the field centre, a corner, a
bright star. Writes <case>/sheet_<level>.jpg.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as C  # noqa: E402

C.engine_home()
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


# ---------------------------------------------------------------------------------------------- a case with variants

class Src:
    """An image (H, W, C) and its sky -> pixel mapping (ra, dec arrays -> x, y arrays)."""

    def __init__(self, img: np.ndarray, sky2pix):
        self.img, self.sky2pix = img, sky2pix

    def sample(self, ra, dec, shape, order=1) -> np.ndarray:
        x, y = self.sky2pix(ra, dec)
        x, y = x.reshape(shape), y.reshape(shape)
        return np.stack([ndi.map_coordinates(self.img[..., c], [y, x], order=order, mode="constant", cval=0)
                         for c in range(self.img.shape[-1])], -1)


def _wcs_map(wcs: WCS, flip_H: int | None = None, org=(0.0, 0.0), binf: int = 1):
    def f(ra, dec):
        x, y = wcs.all_world2pix(ra, dec, 0)
        if flip_H is not None:
            y = flip_H - 1 - y
        x, y = x - org[0], y - org[1]
        return (x + 0.5) / binf - 0.5, (y + 0.5) / binf - 0.5
    return f


def _bin(a: np.ndarray, f: int) -> np.ndarray:
    if f <= 1:
        return a
    H, W = a.shape[:2]
    return a[: H // f * f, : W // f * f].reshape(H // f, f, W // f, f, *a.shape[2:]).mean((1, 3))


def _zstretch(L: np.ndarray, block: int) -> np.ndarray:
    """A linear luminance map shown by its own noise: minus a smooth background, over the robust spread of the
    residual; -2.5 sigma black, +5 sigma white."""
    H, W = L.shape
    small = _bin(L, block)
    bg = ndi.zoom(ndi.median_filter(small, 5), (H / small.shape[0], W / small.shape[1]), order=1)[:H, :W]
    r = L[: bg.shape[0], : bg.shape[1]] - bg
    v = r[::3, ::3]
    v = v[np.isfinite(v) & (v != 0)]
    s = 1.4826 * float(np.median(np.abs(v - np.median(v)))) or 1.0
    return np.clip((r / s + 2.5) / 7.5, 0, 1)[..., None].astype(np.float32)


def _linear_lum(kind: str, case: str, level: str, variant: str):
    """(luminance of the raw linear stack, its WCS): ours drz_full, Siril's stack_linear.fit."""
    if kind == "siril":
        with fits.open(C.siril_dir(case, level, variant) / "stack_linear.fit", memmap=False) as h:
            d = np.asarray(h[0].data, np.float32)
            wcs = WCS(h[0].header, naxis=2)
        return (d.mean(0) if d.ndim == 3 else d), wcs
    from astro_gpu import api
    p = api.open_project(C.ours_dir(case, level, variant))
    w = next(w for w in p.workflows if w.name == "Classic stack")
    stk = Path(api.stage_outputs(p, w.id, "stack").dir)
    ast = Path(api.stage_outputs(p, w.id, "astrometry").dir)
    d = np.load(stk / "drz_full.npy", mmap_mode="r")
    return np.asarray(d, np.float32).mean(0), WCS(fits.getheader(ast / "wcs.fits"), naxis=2)


def make_variants(case: str, level: str) -> Path:
    """One sheet for the level: every variant, Siril's and ours, on the framing of the first Siril variant. Rows: the
    finals; the raw linear stacks shown by their own noise (streaks show there); 100 % crops of the target and of a
    corner (finals), and of the same corner in the linear stacks."""
    man = json.loads((C.BENCH / case / "manifest.json").read_text(encoding="utf-8"))
    cols = []
    for v in C.variants_of(case):
        if level not in C.variant_levels(case, v):
            continue
        if C.variant_has_siril(case, v) and (C.siril_dir(case, level, v) / "final.tif").exists():
            cols.append(("siril", v))
        if (C.ours_dir(case, level, v) / "export" / "bench_final").is_dir():
            cols.append(("ours", v))
    ref_v = next(v for k, v in cols if k == "siril")
    s_img, s_wcs, s_H, s_flip = _siril_any(case, level, ref_v)
    SH, SW = s_img.shape[:2]
    s_scale = float(np.abs(np.linalg.det(s_wcs.pixel_scale_matrix)) ** 0.5 * 3600)

    def ref_sky(fx, fy):
        fits_y = (s_H - 1 - fy) if s_flip else fy
        return s_wcs.all_pix2world(fx.ravel(), fits_y.ravel(), 0)

    finals, lins, o_scale = [], [], s_scale
    for kind, v in cols:
        if kind == "siril":
            img, wcs, H, flip = _siril_any(case, level, v)
            finals.append(Src(img, _wcs_map(wcs, H if flip else None)))
        else:
            img, wcs, org = _ours_any(case, level, v)
            finals.append(Src(img, _wcs_map(wcs, None, org)))
            o_scale = float(np.abs(np.linalg.det(wcs.pixel_scale_matrix)) ** 0.5 * 3600)
        L, lw = _linear_lum(kind, case, level, v)
        lins.append((L, lw))
    CW, gap, top, lab = 560, 10, 52, 26
    n = len(cols)
    ov_scale = CW / SW
    ovh = int(SH * ov_scale)

    def grid(u0, v0, w, h, scale):
        GU, GV = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
        ra, dec = ref_sky(u0 + GU / scale, v0 + GV / scale)
        return ra, dec, GU.shape

    # the target: the brightest extended region of the reference's linear colour image (16 px blocks, smoothed)
    with fits.open(C.siril_dir(case, level, ref_v) / "color_linear.fit", memmap=False) as h:
        cl = np.asarray(h[0].data, np.float32)
    cl = cl.mean(0) if cl.ndim == 3 else cl
    sm = ndi.median_filter(_bin(cl, 16), 7)
    m = np.zeros_like(sm, bool)
    m[int(0.15 * sm.shape[0]):int(0.85 * sm.shape[0]), int(0.15 * sm.shape[1]):int(0.85 * sm.shape[1])] = True
    ty, tx = np.unravel_index(np.argmax(np.where(m, sm, -np.inf)), sm.shape)
    tx, ty = (tx + 0.5) * 16, (ty + 0.5) * 16                   # FITS grid of the reference
    ty = (s_H - 1 - ty) if s_flip else ty                        # its final's grid
    cs = s_scale / o_scale
    half = CW / cs / 2
    crops = [("cible : la région la plus brillante de la nébuleuse", tx, ty),
             ("coin haut gauche", 0.15 * SW, 0.15 * SH)]
    # linear maps shown by their own noise, binned to the overview's scale / native for the crop
    rows = []
    ra, dec, shp = grid(0, 0, CW, ovh, ov_scale)
    rows.append(("Images finales (rendus par défaut)", [np.clip(f.sample(ra, dec, shp, 1), 0, 1) for f in finals]))
    ov_lin = []
    for (L, lw) in lins:
        lsc = float(np.abs(np.linalg.det(lw.pixel_scale_matrix)) ** 0.5 * 3600)
        f = max(1, int(round(SW / CW * s_scale / lsc)))
        ov_lin.append(Src(_zstretch(_bin(L, f), 32), _wcs_map(lw, None, (0, 0), f)).sample(ra, dec, shp, 1))
    rows.append(("Piles linéaires, fond étiré par son propre bruit (−2,5σ noir, +5σ blanc) : les traînées s'y voient",
                 [np.repeat(o, 3, -1) for o in ov_lin]))
    for name, cx, cy in crops:
        ra, dec, shp = grid(cx - half, cy - half, CW, CW, cs)
        rows.append((f"100 % : {name} (images finales)", [np.clip(f.sample(ra, dec, shp, 1), 0, 1) for f in finals]))
    cx, cy = 0.15 * SW, 0.15 * SH
    ra, dec, shp = grid(cx - half, cy - half, CW, CW, cs)
    lin_c = []
    for (L, lw) in lins:
        x, y = _wcs_map(lw)(ra, dec)
        x0, x1 = int(max(0, np.nanmin(x) - 40)), int(min(L.shape[1], np.nanmax(x) + 40))
        y0, y1 = int(max(0, np.nanmin(y) - 40)), int(min(L.shape[0], np.nanmax(y) + 40))
        sub = ndi.gaussian_filter(L[y0:y1, x0:x1], 1.0)
        lin_c.append(Src(_zstretch(sub, 16), _wcs_map(lw, None, (x0, y0))).sample(ra, dec, shp, 1))
    rows.append(("100 % : le même coin, pile linéaire étirée par son bruit (lissée 1 px)",
                 [np.repeat(o, 3, -1) for o in lin_c]))
    Wt = n * CW + (n + 1) * gap
    Ht = top + sum(lab + r[1][0].shape[0] + gap for r in rows) + 30
    sheet = Image.new("RGB", (Wt, Ht), (24, 24, 24))
    dr = ImageDraw.Draw(sheet)
    f1, f2 = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 16)
    lv = man["levels"][level]
    dr.text((gap, 12), f"{case} · {level} · {lv['count']} poses · {man['camera_label']} · {man['scope']} · même zone "
                       f"du ciel ; Siril {s_scale:.2f}\"/px, astro_gpu {o_scale:.2f}\"/px", fill=(235, 235, 235), font=f1)
    y = top
    for title, ims in rows:
        dr.text((gap, y + 3), title, fill=(220, 220, 220), font=f2)
        y += lab
        for i, a in enumerate(ims):
            x = gap + i * (CW + gap)
            sheet.paste(Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)), (x, y))
            if title.startswith("Images finales"):
                kind, v = cols[i]
                who = "Siril 1.4" if kind == "siril" else "astro_gpu"
                short = (man.get("variants") or {}).get(v, {}).get("short", "")
                dr.text((x + 6, y + 6), f"{v} · {who}", fill=(255, 220, 120) if kind == "siril" else (120, 220, 255),
                        font=f1)
                dr.text((x + 6, y + 32), short, fill=(230, 230, 230), font=f2)
        y += ims[0].shape[0] + gap
    out = C.BENCH / case / f"sheet_{level}.jpg"
    sheet.save(out, quality=88)
    print(f"sheet {out}")
    return out


def _siril_any(case: str, level: str, variant: str):
    d = C.siril_dir(case, level, variant)
    img = _tif(d / "final.tif")
    with fits.open(d / "color_linear.fit") as h:
        lin = np.asarray(h[0].data, np.float32)
        wcs = WCS(h[0].header, naxis=2)
    H = lin.shape[-2]
    Ll = lin.mean(0) if lin.ndim == 3 else lin
    a, b = _lum_small(img, 16), _lum_small(Ll, 16)
    n = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[: n[0], : n[1]], b[: n[0], : n[1]]
    bf = _lum_small(Ll[::-1], 16)[: n[0], : n[1]]

    def corr(u, v):
        u, v = np.argsort(np.argsort(u.ravel())), np.argsort(np.argsort(v.ravel()))
        return float(np.corrcoef(u, v)[0, 1])
    return img, wcs, H, corr(a, bf) > corr(a, b)


def _ours_any(case: str, level: str, variant: str):
    from astro_gpu import api
    root = C.ours_dir(case, level, variant)
    p = api.open_project(root)
    w = next(w for w in p.workflows if w.name == "Classic stack")
    rs = Path(api.stage_outputs(p, w.id, "render_simple").dir)
    ast = Path(api.stage_outputs(p, w.id, "astrometry").dir)
    prep = json.loads((rs / "prep_simple.json").read_text(encoding="utf-8"))
    x0, x1, y0, y1 = prep["crop"]
    img = _tif(sorted((root / "export" / "bench_final").glob("*.tif"))[-1])
    if img.shape[0] != y1 - y0 or img.shape[1] != x1 - x0:
        raise SystemExit(f"our export {img.shape} does not match the crop {prep['crop']}")
    return img, WCS(fits.getheader(ast / "wcs.fits"), naxis=2), (x0, y0)


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "variants" or len(C.CASES[sys.argv[1]].get("variants") or {}) > 1:
        make_variants(sys.argv[1], sys.argv[2])
    else:
        make(sys.argv[1], sys.argv[2])
