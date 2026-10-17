"""Export of a rendering (full resolution) and automatic quality checks."""
from __future__ import annotations

import json
from datetime import datetime

import cv2
import numpy as np
import torch

from ..core.params import P, StageParams
from .engine import RenderEngine
from .settings import RenderSettings


class ExportSettings(StageParams):
    name: str = P("render", label="Name", help="Base name of the exported files.")
    orientation: str = P("as captured", label="Orientation", choices=["as captured", "north up", "match reference"],
                         help="Exact rotation by multiples of 90 deg / mirror, never resampled. 'north up' uses the "
                              "astrometric solution; 'match reference' the orientation of the reference image.")
    full_tiff: bool = P(True, label="16-bit TIFF, full resolution",
                        help="The drizzled resolution (finer than the sensor), with an embedded sRGB ICC profile.")
    sensor_png: bool = P(True, label="16-bit PNG at sensor scale",
                         help="Downsampled to the sensor pixel scale (sharper looking, smaller).")
    jpeg: bool = P(True, label="JPEG at sensor scale", help="8-bit JPEG (quality 93) for sharing.")
    layers: bool = P(True, label="Separate layers",
                     help="Starless and stars-only images. Recombine with the stars on top in 'Screen' mode.")
    reference_framing: bool = P(True, label="Reference framing",
                                help="Also export with exactly the framing of the reference image (if any).")


def orientation_ops(project, choice: str) -> tuple[int, bool]:
    """(k rot90, vertical flip) for the export orientation."""
    if choice == "match reference":
        p = project.stage_dir("reference") / "align_ref.json"
        if p.exists():
            d = json.loads(p.read_text())
            if d.get("reference"):
                return int(d["k"]), bool(d["flip"])
        return 0, False
    if choice == "north up":
        m = json.loads((project.stage_dir("astrometry") / "manifest.json").read_text())["metrics"]
        ang = float(m.get("up_position_angle", 0.0))       # position angle of image 'up' (east of north)
        flip = bool(m.get("parity_flipped", False))
        k = int(round(ang / 90.0)) % 4
        return k, flip
    return 0, False


def _orient(im: np.ndarray, crop, k: int, flip: bool) -> np.ndarray:
    x0, x1, y0, y1 = crop
    c = im[:, y0:y1, x0:x1]
    if flip:
        c = c[:, ::-1]
    return np.ascontiguousarray(np.rot90(c, k, axes=(1, 2)))


def _icc() -> bytes:
    from PIL import ImageCms
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def export(project, engine: RenderEngine, s: RenderSettings, e: ExportSettings, progress=lambda f, m: None) -> dict:
    import tifffile
    from PIL import Image
    out_dir = project.results / e.name
    out_dir.mkdir(parents=True, exist_ok=True)
    progress(0.05, "rendering at full resolution")
    lay = engine.render(s, 1.0)
    imgs = {"final": lay.final.cpu().numpy(), "starless": lay.starless.cpu().numpy(), "stars": lay.stars.cpu().numpy()}
    if engine.dev.type == "cuda":
        torch.cuda.empty_cache()
    k, flip = orientation_ops(project, e.orientation)
    icc = _icc()
    files = []
    SC = engine.SC
    names = {"final": e.name, "starless": f"{e.name}_starless", "stars": f"{e.name}_stars"}
    for i, (key, im) in enumerate(imgs.items()):
        if key != "final" and not e.layers:
            continue
        if key == "stars" and not s.stars:
            continue
        progress(0.3 + 0.2 * i, f"writing {names[key]}")
        o = _orient(im, engine.crop, k, flip)
        u16 = (np.clip(o, 0, 1) * 65535 + 0.5).astype(np.uint16).transpose(1, 2, 0)
        if e.full_tiff:
            p = out_dir / f"{names[key]}_full_resolution_16bit.tif"
            tifffile.imwrite(p, u16, photometric="rgb", iccprofile=icc, compression="zlib")
            files.append(p)
        small = np.stack([cv2.resize(c, (round(c.shape[1] / SC), round(c.shape[0] / SC)), interpolation=cv2.INTER_AREA)
                          for c in o])
        if e.sensor_png:
            p = out_dir / f"{names[key]}.png"
            cv2.imwrite(str(p), (np.clip(small, 0, 1) * 65535 + 0.5).astype(np.uint16).transpose(1, 2, 0)[:, :, ::-1])
            files.append(p)
        if e.jpeg:
            p = out_dir / f"{names[key]}.jpg"
            Image.fromarray((np.clip(small, 0, 1) * 255 + 0.5).astype(np.uint8).transpose(1, 2, 0)).save(
                p, quality=93, icc_profile=icc)
            files.append(p)
    ref = project.stage_dir("reference") / "align_ref.json"
    if e.reference_framing and ref.exists() and json.loads(ref.read_text()).get("reference"):
        progress(0.9, "reference framing")
        info = json.loads(ref.read_text())
        for key in ("final", "starless", "stars") if e.layers else ("final",):
            if key == "stars" and not s.stars:
                continue
            crop = remap_to_reference(imgs[key], info)
            p = out_dir / f"{names[key]}_reference_framing.png"
            cv2.imwrite(str(p), (np.clip(crop, 0, 1) * 65535 + 0.5).astype(np.uint16).transpose(1, 2, 0)[:, :, ::-1])
            files.append(p)
    progress(0.95, "quality checks")
    qa = quality_checks(imgs, engine, s)
    meta = {"date": datetime.now().isoformat(timespec="seconds"), "settings": s.model_dump(),
            "export": e.model_dump(), "orientation": {"rot90": k, "flip": flip}, "qa": qa,
            "files": [str(f) for f in files], "info": _jsonable(lay.info),
            "variant": project.config.active_variant or "main",
            "preparation": getattr(getattr(engine, "_cur", engine), "dir", None).name
            if getattr(getattr(engine, "_cur", engine), "dir", None) is not None else ""}
    (out_dir / f"{e.name}_settings.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    progress(1.0, "done")
    return meta


def remap_to_reference(img: np.ndarray, info: dict) -> np.ndarray:
    Hr, Wr = info["ref_shape"]
    vv, uu = np.mgrid[0:Hr, 0:Wr].astype(np.float64)
    a, b = uu / Wr * 2 - 1, vv / Hr * 2 - 1
    terms = [a ** i * b ** j for i in range(4) for j in range(4 - i)]
    mx = sum(c * t for c, t in zip(info["cx"], terms)).astype(np.float32)
    my = sum(c * t for c, t in zip(info["cy"], terms)).astype(np.float32)
    return np.clip(np.stack([cv2.remap(np.ascontiguousarray(c.astype(np.float32)), mx, my, cv2.INTER_LANCZOS4,
                                       borderMode=cv2.BORDER_CONSTANT) for c in img]), 0, 1)


def quality_checks(imgs: dict, engine: RenderEngine, s: RenderSettings) -> dict:
    """Automatic guard rails (cropped useful frame): clipping, burnt highlights, layer blend, dark dips at stars."""
    x0, x1, y0, y1 = engine.crop
    fin = imgs["final"][:, y0:y1, x0:x1]
    sl = imgs["starless"][:, y0:y1, x0:x1]
    out: dict = {}
    out["pure_black_pct"] = round(100 * float((fin.max(0) <= 0).mean()), 4)
    out["pure_white_pct"] = round(100 * float((fin.min(0) >= 1).mean()), 4)
    Y = 0.2126 * fin[0] + 0.7152 * fin[1] + 0.0722 * fin[2]
    Ys = cv2.GaussianBlur(Y, (0, 0), 8)
    out["background_levels_255"] = {f"p{p}": round(255 * float(np.percentile(Ys[::4, ::4], p)), 1) for p in (1, 5, 20, 50)}
    Yh = cv2.GaussianBlur(0.2126 * sl[0] + 0.7152 * sl[1] + 0.0722 * sl[2], (0, 0), 2)
    win = int(200 * engine.SC)
    worst = 0.0
    for yy in range(0, Yh.shape[0] - win, win // 2):
        for xx in range(0, Yh.shape[1] - win, win // 2):
            worst = max(worst, float((Yh[yy:yy + win, xx:xx + win] > 0.92).mean()))
    out["burnt_highlights_worst_zone_pct"] = round(100 * worst, 2)
    if s.stars:
        st = imgs["stars"][:, y0:y1, x0:x1]
        out["layer_blend_error_255"] = round(255 * float(np.abs(1 - (1 - st) * (1 - sl) - fin).mean()), 3)
    # dark dips in the starless layer at Gaia stars G < 17 and blind
    # (measured at sensor scale, like a viewer looking at the exported image)
    from scipy.ndimage import minimum_filter
    SC = engine.SC
    L = cv2.resize(sl.mean(0), (round(sl.shape[2] / SC), round(sl.shape[1] / SC)), interpolation=cv2.INTER_AREA)
    small = cv2.resize((L * 255).astype(np.uint8), None, fx=0.5, fy=0.5)
    bg = cv2.resize(cv2.medianBlur(small, 9), (L.shape[1], L.shape[0])).astype(np.float32) / 255
    d = cv2.GaussianBlur(L, (0, 0), 1.0) - bg
    m = (engine.G < 17)
    xs, ys = (engine.X[m] - x0) / SC, (engine.Y[m] - y0) / SC
    ok = (xs > 5) & (xs < L.shape[1] - 5) & (ys > 5) & (ys < L.shape[0] - 5)
    dm = minimum_filter(d, size=5)[np.round(ys[ok]).astype(int), np.round(xs[ok]).astype(int)]
    out["dark_dips_at_stars"] = int((dm < -0.08).sum())
    out["stars_checked"] = int(ok.sum())
    out["dark_dips_blind"] = int(((d == minimum_filter(d, size=9)) & (d < -0.08)).sum())
    fails = []
    if out["pure_black_pct"] > 0.01:
        fails.append("clipped shadows")
    if out["burnt_highlights_worst_zone_pct"] > 1.0:
        fails.append("burnt highlights")
    if out.get("layer_blend_error_255", 0) > 0.5:
        fails.append("layer blend")
    if out["dark_dips_at_stars"] > max(20, 1e-3 * out["stars_checked"]):
        fails.append("dark dips at stars")
    out["warnings"] = fails
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o
