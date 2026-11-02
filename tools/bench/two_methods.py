"""The bench of 2 November, "the two methods": Siril classic and Siril drizzle against our drizzle, our drizzle with
PixInsight's rejection, our classic stack (with and without the white balance before the debayering) and the previous
version, all on the same sky (docs/pilot/BENCH.md, "Banc du 2 novembre : les deux méthodes").

    python tools/bench/two_methods.py setup|run CASE            the workflows (crossed.TODAY) and their run
    python tools/bench/two_methods.py measure CASE              -> F:/astro_essais/bench/<case>/today_express.json
    python tools/bench/two_methods.py tables                    -> docs/pilot/bench/two_methods_tables.md
    python tools/bench/two_methods.py board CASE                -> docs/pilot/shots/2026-11-02/bench/<case>.png

The measures are crossed.measure_crossed's (measure.py's functions); the board reads the final linear images (colour
calibrated, background extracted), puts them on one photometric scale (the flux of the same Gaia stars) and shows
them with one stretch, 100 % crops of a star field and of a faint area.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossed as X  # noqa: E402  (sets the bench environment first)
import cases as C  # noqa: E402
import measure as M  # noqa: E402

LEVEL = "express"
OUT = "today_express"
SIRIL, SIRIL_DRZ = X.SIRIL, "Siril drizzle"
#: the columns, in the order of the tables
STACKS = [SIRIL, SIRIL_DRZ, "PI-own", "PI-int", "CL", "CL-off", "LEG"]
FR = {SIRIL: "Siril classique", SIRIL_DRZ: "Siril drizzle", "PI-own": "Notre drizzle",
      "PI-int": "Notre drizzle, rejet PixInsight", "CL": "Notre classique", "CL-off": "Notre classique sans équilibrage",
      "LEG": "Ancienne version"}
OURS = ["PI-own", "PI-int", "CL", "CL-off", "LEG"]
CASES = ["heart", "iris", "m51", "m8", "seestar", "rosette"]
NAMES = {"heart": "Heart", "iris": "Iris", "m51": "M51", "m8": "M8", "seestar": "Seestar", "rosette": "Rosette"}
DOCS = C.REPO / "docs" / "pilot"
SHOTS = DOCS / "shots" / "2026-11-02" / "bench"


def load(case: str) -> dict:
    return json.loads((C.BENCH / case / f"{OUT}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------- the tables

#: (label, measure key of crossed.MEASURES or a function, higher is better / None)
ROWS = [("Profondeur (mag, 5σ ; plus haut = plus profond)", "depth"),
        ("Bruit fin (mag/arcsec² ; plus haut = moins de bruit)", "fine"),
        ("Bruit large (mag/arcsec²)", "large"),
        ("Signal sur bruit de la cible", "snr"),
        ("Étoiles faibles trouvées", "faint"),
        ("FWHM au centre (″ ; plus bas = plus fines)", "fwhm"),
        ("Rondeur (1 = rondes)", "round"),
        ("Points d'une seule couleur (par degré²)", "dots"),
        ("Erreur de couleur B−R (plus bas = plus fidèle)", "br"),
        ("Erreur de couleur V−R", "gr"),
        ("Point blanc B−R (BP−RP où B−R = 0 ; Gaia : environ 0,8)", "white"),
        ("Planéité du fond (mag/arcsec² ; plus haut = plus plat)", "flat"),
        ("Poses gardées", "frames"),
        ("Taux de rejet (% des échantillons)", "reject"),
        ("Durée de la préparation (min)", "time_pre")]


def _mk(key: str):
    return next(m for m in X.MEASURES if m[0] == key)


def cell_value(m: dict, case: str, name: str, key: str):
    """One cell: the value of ``name`` for this measure (a stack measured once for two names: that one's)."""
    n = m["same_as"].get(name, name)
    e = m["by_image"].get(n)
    if e is None:
        return None
    if key == "white":
        return ((e.get("colours") or {}).get("b_r") or {}).get("white_at_bp_rp")
    path = _mk(key)[2]
    if key == "frames" and n in (SIRIL, SIRIL_DRZ):
        import report as R
        v = R.siril_frames(case, LEVEL if n == SIRIL else f"{LEVEL}_drizzle", "")
        return v if v is not None else json.loads((C.BENCH / case / "manifest.json").read_text(
            encoding="utf-8"))["levels"][LEVEL]["count"]
    if key == "time_pre":
        if n in (SIRIL, SIRIL_DRZ):
            sj = C.siril_dir(case, LEVEL if n == SIRIL else f"{LEVEL}_drizzle", "") / "siril.json"
            return (json.loads(sj.read_text(encoding="utf-8")).get("pre_s") or 0) / 60 if sj.exists() else None
        eg = e.get("engine") or {}
        parts = [eg.get(k) for k in ("calibrate_s", "register_s", "stack_s")]
        return sum(parts) / 60 if all(x is not None for x in parts) else None
    return X._get(e, path)


def fmt_of(key: str) -> str:
    return "{:.3f}" if key in ("br", "gr", "white", "reject") else "{:.2f}" if key == "snr" else \
        "{:.1f}" if key == "time_pre" else _mk(key)[3]


def case_table(case: str, m: dict) -> str:
    cols = [s for s in STACKS if s == SIRIL or s in m["by_image"] or s in m["same_as"]]
    head = "| | " + " | ".join(FR[s] for s in cols) + " |\n|---|" + "---:|" * len(cols) + "\n"
    rows = []
    for label, key in ROWS:
        vals = {s: cell_value(m, case, s, key) for s in cols}
        fmt = fmt_of(key)
        higher = _mk(key)[5] if key != "white" else None
        ok = {s: v for s, v in vals.items() if s in OURS and v is not None and np.isfinite(v)}
        best = None
        if higher is not None and ok:
            best = max(ok, key=lambda s: ok[s]) if higher else min(ok, key=lambda s: ok[s])
        cells = []
        for s in cols:
            v = vals[s]
            t = X._fr(v, fmt) if v is not None else "–"
            cells.append(f"**{t}**" if s == best else t)
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def verdict(m: dict, case: str, a: str, b: str) -> dict:
    """Over the measures of the criterion: how many are better / worse for a than for b beyond the tolerance."""
    up, down = [], []
    for key, _l, path, _f, kind, higher, tol, crit in X.MEASURES:
        if not crit or (key, case) in X.NOT_IN_CRITERION:
            continue
        va, vb = cell_value(m, case, a, key), cell_value(m, case, b, key)
        if va is None or vb is None:
            continue
        j = X.judge(va, vb, kind, higher, tol, key)
        (up if j > 0 else down if j < 0 else []).append(key)
    return {"better": up, "worse": down}


def tables() -> int:
    out, summary = [], {}
    for case in CASES:
        f = C.BENCH / case / f"{OUT}.json"
        if not f.exists():
            continue
        m = load(case)
        man = json.loads((C.BENCH / case / "manifest.json").read_text(encoding="utf-8"))
        n = man["levels"][LEVEL]["count"]
        out.append(f"#### {NAMES[case]} ({n} poses)\n\n" + case_table(case, m))
        eg = (m["by_image"].get("PI-own") or {}).get("engine") or {}
        summary[case] = {"advice": (eg.get("rule") or {}).get("method"), "scale": eg.get("scale"),
                         "pixfrac": eg.get("pixfrac"), "rule": eg.get("rule"),
                         "drizzle_vs_classic": verdict(m, case, "PI-own", "CL"),
                         "int_vs_own": verdict(m, case, "PI-int", "PI-own"),
                         "cl_vs_cloff": verdict(m, case, "CL", "CL-off"),
                         "drz_vs_siril": verdict(m, case, "PI-own", SIRIL),
                         "drz_vs_siril_drz": verdict(m, case, "PI-own", SIRIL_DRZ),
                         "cl_vs_siril": verdict(m, case, "CL", SIRIL),
                         "drz_vs_leg": verdict(m, case, "PI-own", "LEG"),
                         "cl_vs_leg": verdict(m, case, "CL", "LEG")}
    d = DOCS / "bench"
    d.mkdir(parents=True, exist_ok=True)
    (d / "two_methods_tables.md").write_text("\n".join(out), encoding="utf-8")
    (d / "two_methods_verdicts.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"written {d / 'two_methods_tables.md'}")
    for c, s in summary.items():
        print(c, "advice", s["advice"], "scale", s["scale"], "| drizzle vs classic", len(s["drizzle_vs_classic"]["better"]),
              "better", len(s["drizzle_vs_classic"]["worse"]), "worse", s["drizzle_vs_classic"])
    return 0


# ---------------------------------------------------------------------------------------------- the board

def board(case: str) -> Path:
    """One board per case: the seven stacks side by side, 100 % crops of a star field and of a faint area, one
    photometric scale (the flux of the same Gaia stars) and one stretch for all."""
    from PIL import Image, ImageDraw, ImageFont
    from scipy import ndimage as ndi
    import sheet as SH
    m = load(case)
    sv = C.norm_variant(case, None)
    p, _ = X.project(case, LEVEL)
    names = [s for s in STACKS if s == SIRIL or s in m["by_image"]]
    dirs = {}
    for n in names:
        if n in (SIRIL, SIRIL_DRZ):
            continue
        w = X.wf(p, n)
        dirs[n] = {s: Path(o.dir) for s in X.api.chain(p, w.id) if (o := X.api.stage_outputs(p, w.id, s)) is not None}
    first = m["first"]

    def lin(n: str) -> M.Img:
        if n == SIRIL:
            return M.load_siril(case, LEVEL, "color_linear", sv)
        if n == SIRIL_DRZ:
            return M.load_siril(case, f"{LEVEL}_drizzle", "color_linear", sv)
        return M.load_ours(dirs[n], "lin")
    ref = lin(SIRIL)
    geo0 = M.load_ours(dirs[first], "lin")
    st = M.gaia(dirs[first], geo0)
    del geo0
    # the Gaia stars every image has, bright and isolated: the common photometric scale
    x, y = ref.pix(st["ra"], st["dec"])
    sel = ref.inside(x, y, 60) & (st["g"] > 11.5) & (st["g"] < 14.5)
    sel = M.isolated(st, sel, 30 / 3600, 2.5)
    fw = float(m["by_image"][SIRIL]["stars"].get("fwhm_px", 3.0))

    def star_flux(im: M.Img) -> np.ndarray:
        xx, yy = im.pix(st["ra"][sel], st["dec"][sel])
        k = im.scale / ref.scale                      # radii in this image's pixels
        ok = im.inside(xx, yy, 40)
        out = np.full(len(xx), np.nan)
        out[ok] = M.star_flux(im.L, xx[ok], yy[ok], 2.5 * fw / k, 3.5 * fw / k, 5.0 * fw / k)   # exact apertures
        return out
    f_ref = star_flux(ref)
    imgs, kf = {}, {}
    for n in names:
        im = ref if n == SIRIL else lin(n)
        f = star_flux(im)
        ok = np.isfinite(f) & np.isfinite(f_ref) & (f > 0) & (f_ref > 0)
        kf[n] = float(np.median(f_ref[ok] / f[ok]))                # this image's flux in Siril classic's units
        imgs[n] = im
    print(case, "flux scale vs Siril classic:", {n: float(f"{v:.4g}") for n, v in kf.items()}, flush=True)

    # the two places, in Siril classic's pixels: the densest star field (Gaia G < 15) and a faint area
    CROP = 400
    blk = 128
    cover = ref.cover
    stars = ref.inside(x, y, 0) & (st["g"] < 15)
    hist, _, _ = np.histogram2d(y[stars], x[stars], bins=[np.arange(0, (ref.H // blk) * blk + 1, blk),
                                                         np.arange(0, (ref.W // blk) * blk + 1, blk)])
    full = SH._bin(cover.astype(np.float32), blk)                    # the share of each block that is covered
    win = max(2, CROP // blk + 1)
    sm = ndi.uniform_filter(hist, size=win)
    sm = np.where(ndi.minimum_filter(full, size=win) > 0.98, sm, -1)
    iy, ix = np.unravel_index(np.argmax(sm), sm.shape)
    star_c = ((ix + 0.5) * blk, (iy + 0.5) * blk)
    Lb = SH._bin(ref.L, 16)
    Lm = ndi.median_filter(Lb, 7)
    mm = np.zeros_like(Lm, bool)
    mm[int(0.15 * Lm.shape[0]):int(0.85 * Lm.shape[0]), int(0.15 * Lm.shape[1]):int(0.85 * Lm.shape[1])] = True
    mm &= SH._bin(cover.astype(np.float32), 16) > 0.99
    # faint area = the median brightness among the covered positions at the quiet end (15th percentile), away from
    # the stars' places
    vals = np.where(mm, Lm, np.nan)
    target = np.nanpercentile(vals, 15)
    iy, ix = np.unravel_index(np.nanargmin(np.abs(np.where(mm, Lm, np.inf) - target)), Lm.shape)
    faint_c = ((ix + 0.5) * 16, (iy + 0.5) * 16)
    places = [("Un champ d'étoiles", star_c), ("Une zone faible", faint_c)]

    # one stretch: asinh on the surface brightness (Siril classic's units per arcsec²), black at each crop's own sky
    # (the median of the crop of the same stack, so that a residual offset does not show), white and softening from
    # Siril classic's crop; the same numbers for every stack
    def crop(n: str, c: tuple[float, float]) -> np.ndarray:
        im = imgs[n]
        gx, gy = np.meshgrid(np.arange(CROP) - CROP / 2, np.arange(CROP) - CROP / 2)
        ra, dec = ref.wcs.all_pix2world(c[0] + gx.ravel(), c[1] + gy.ravel(), 0)
        px, py = im.wcs.all_world2pix(ra, dec, 0)
        px, py = px.reshape(gx.shape) + im.dx, py.reshape(gx.shape) + im.dy
        x0, x1 = max(0, int(np.floor(px.min())) - 6), min(im.W, int(np.ceil(px.max())) + 7)
        y0, y1 = max(0, int(np.floor(py.min())) - 6), min(im.H, int(np.ceil(py.max())) + 7)
        out = np.stack([ndi.map_coordinates(np.asarray(im.rgb[k, y0:y1, x0:x1]), [py - y0, px - x0], order=3,
                                            mode="nearest") for k in range(3)], -1)
        return out * (kf[n] / im.pixarea * ref.pixarea)           # per Siril-pixel area, in Siril classic's flux units
    shots = {}
    for title, c in places:
        row = {n: crop(n, c) for n in names}
        shots[title] = row
    panels = {}
    for title, row in shots.items():
        r = row[SIRIL]
        lum = r.mean(-1)
        sky = float(np.median(lum))      # (the reference's crop sets the stretch)
        sig = float(1.4826 * np.median(np.abs(lum - sky)))
        white = float(np.percentile(lum, 99.7) - sky)
        a = max(2.0 * sig, 1e-9)
        norm = np.arcsinh(max(white, 10 * a) / a)
        panels[title] = {}
        for n in names:
            d = row[n]
            skyn = np.median(d.reshape(-1, 3), axis=0)                 # each colour's own sky: a neutral background
            v = np.arcsinh(np.clip(d - skyn + 1.0 * sig, 0, None) / a) / norm
            panels[title][n] = (np.clip(v, 0, 1) * 255 + 0.5).astype(np.uint8)

    gap, top, lab = 8, 62, 26
    Wt = len(names) * (CROP + gap) + gap
    Ht = top + len(places) * (CROP + lab + gap) + 10
    img = Image.new("RGB", (Wt, Ht), (24, 24, 24))
    dr = ImageDraw.Draw(img)
    f1, f2 = ImageFont.truetype(SH.FONT, 22), ImageFont.truetype(SH.FONT, 16)
    man = json.loads((C.BENCH / case / "manifest.json").read_text(encoding="utf-8"))
    dr.text((gap, 10), f"{NAMES[case]} · Express, {man['levels'][LEVEL]['count']} poses · {man['camera_label']} · même ciel, "
                       "même échelle de flux, même étirement, fond de chaque couleur ramené au même niveau, 100 % (pixel de Siril)",
            fill=(235, 235, 235), font=f1)
    for i, n in enumerate(names):
        dr.text((gap + i * (CROP + gap) + 4, 38), FR[n], fill=(255, 220, 120) if n in (SIRIL, SIRIL_DRZ) else (120, 220, 255),
                font=f2)
    y0 = top
    for title, _c in places:
        dr.text((gap, y0 + 3), f"{title} : {panels[title][SIRIL].shape[1]} × {panels[title][SIRIL].shape[0]} pixels",
                fill=(220, 220, 220), font=f2)
        y0 += lab
        for i, n in enumerate(names):
            img.paste(Image.fromarray(panels[title][n]), (gap + i * (CROP + gap), y0))
        y0 += CROP + gap
    SHOTS.mkdir(parents=True, exist_ok=True)
    out = SHOTS / f"{case}.png"
    img.save(out)
    print(f"board {out}", flush=True)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "tables":
        sys.exit(tables())
    if cmd == "board":
        for c in sys.argv[2:]:
            board(c)
        sys.exit(0)
    if cmd == "measure":
        for c in sys.argv[2:]:
            r = X.measure_crossed(c, LEVEL, [n for n in OURS], {SIRIL_DRZ: f"{LEVEL}_drizzle"})
            (C.BENCH / c / f"{OUT}.json").write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")
            print(f"written {c} ({r['seconds']:.0f} s)", flush=True)
        sys.exit(0)
    sys.exit(2)
