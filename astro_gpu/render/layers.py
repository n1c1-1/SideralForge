"""Workflow layers for Post: render a workflow's output folder with a base and slider positions, its masks, and export
the composed image.

* ``LayerService.render(folder, base, sliders, scale, region, part)`` -> ``LayerImage(rgb, crop, info)``: the layer
  on the whole stack grid (or the asked region), uncropped, with the useful frame (``crop``). The folder is an output
  stage's result: render_prep / render_prep_learned (``RenderEngine``) or render_simple (``SimpleRenderEngine``).
  A full-resolution region uses the tone fit of the 1/4 rendering (``RenderEngine.render(region=...)``).
* The service keeps the engines it opened and drops the least recently used one when host memory runs short (or a
  byte budget is reached). A GPU out-of-memory error renders again on the CPU, with a note.
* ``masks(folder, scale, region)``: the star mask and the H-alpha map of Post's mask parts.
* ``export_rgb(rgb, out_dir, options, wcs, origin, history)``: TIFF 16 / 32f, PNG 16, JPEG, FITS 32f with the WCS, in
  sRGB, at full size or reduced, in an orientation of v1 (rot90 / flip), with a history file.
"""
from __future__ import annotations

import gc
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..core.params import P, StageParams
from . import ops
from .engine import RenderEngine
from .settings import RenderSettings
from .simple import SimpleRenderEngine
from .sliders import availability, settings_at

NOTE_CPU = "GPU memory full: rendered on the CPU"
PARTS = (None, "stars", "starless")


@dataclass
class LayerImage:
    rgb: np.ndarray                      # (3, h, w) float32 display sRGB in [0, 1]: the grid (or region) at the scale
    crop: tuple[int, int, int, int]      # the useful frame (x0, x1, y0, y1), full-resolution pixels of the grid
    info: dict = field(default_factory=dict)


def layer_kind(folder: str | Path) -> str:
    """engine (render_prep, render_prep_learned), simple (render_simple) or stacking (render_simple pulled from
    Stacking: no Color)."""
    d = Path(folder)
    if (d / "prep_simple.json").exists():
        prep = json.loads((d / "prep_simple.json").read_text(encoding="utf-8"))
        return "stacking" if prep.get("exit") == "stacking" else "simple"
    if (d / "prep.json").exists():
        return "engine"
    raise FileNotFoundError(f"{d} holds no rendering preparation (prep.json or prep_simple.json)")


def has_denoise(folder: str | Path) -> bool:
    """Whether the layer's workflow has Denoise (the noise slider): always for engine layers (Stars needs it)."""
    d = Path(folder)
    if (d / "prep_simple.json").exists():
        return bool(json.loads((d / "prep_simple.json").read_text(encoding="utf-8")).get("denoised"))
    return True


def open_engine(folder: str | Path, device: str = "cpu") -> RenderEngine | SimpleRenderEngine:
    d = Path(folder)
    if layer_kind(d) == "engine":
        return RenderEngine(d, device=device)
    return SimpleRenderEngine(d, device=device)


def folder_bytes(folder: str | Path) -> int:
    """Host memory an engine on this folder takes, about: its arrays."""
    return sum(p.stat().st_size for p in Path(folder).iterdir() if p.suffix in (".npy", ".npz"))


def _smoothstep(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    t = ((x - a) / max(b - a, 1e-12)).clamp(0, 1)
    return t * t * (3 - 2 * t)


class LayerService:
    """The rendering engines of the workflow layers, by folder, the least recently used dropped first.

    ``device``: where engines render ('cpu', 'cuda:N'); ``budget_bytes``: the most host memory the engines may take
    (None: no limit); ``reserve_bytes``: host memory left free when an engine is opened."""

    def __init__(self, device: str = "cpu", budget_bytes: int | None = None, reserve_bytes: int = 2 << 30):
        self.device = device
        self.budget = budget_bytes
        self.reserve = reserve_bytes
        self._eng: OrderedDict[str, RenderEngine | SimpleRenderEngine] = OrderedDict()
        self._size: dict[str, int] = {}
        self._on_cpu: set[str] = set()                     # folders that fell back to the CPU

    # ------------------------------------------------------------------ engines
    @staticmethod
    def _key(folder) -> str:
        return str(Path(folder).resolve())

    def loaded(self) -> list[str]:
        """The folders of the engines kept, least recently used first."""
        return list(self._eng)

    def engine(self, folder) -> RenderEngine | SimpleRenderEngine:
        k = self._key(folder)
        if k in self._eng:
            self._eng.move_to_end(k)
            return self._eng[k]
        need = folder_bytes(k)
        self._make_room(need)
        eng = open_engine(k, "cpu" if k in self._on_cpu else self.device)
        self._eng[k], self._size[k] = eng, need
        return eng

    def _make_room(self, need: int) -> None:
        import psutil
        while self._eng:
            over = self.budget is not None and sum(self._size.values()) + need > self.budget
            short = psutil.virtual_memory().available - need < self.reserve
            if not (over or short):
                break
            self.evict(next(iter(self._eng)))

    def evict(self, folder=None) -> None:
        """Drop one engine (the least recently used one by default), or all with folder='all'."""
        keys = list(self._eng) if folder == "all" else [self._key(folder) if folder else next(iter(self._eng), None)]
        for k in keys:
            if k in self._eng:
                del self._eng[k]
                self._size.pop(k, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def kind(self, folder) -> str:
        return layer_kind(folder)

    def sliders_of(self, folder) -> dict[str, str]:
        """Per slider, '' when the layer offers it, else the reason it is greyed."""
        return availability(layer_kind(folder), has_denoise(folder))

    # ------------------------------------------------------------------ rendering
    def _render(self, folder, s, scale, region=None, fit=None):
        """(Layers, note): rendered on the service's device; on a GPU out-of-memory error, again on the CPU."""
        eng = self.engine(folder)
        try:
            return eng.render(s, scale, region, fit), ""
        except torch.OutOfMemoryError:
            if eng.dev.type != "cuda":
                raise
        k = self._key(folder)
        self.evict(k)
        self._on_cpu.add(k)
        return self.engine(k).render(s, scale, region, fit), NOTE_CPU

    def settings(self, folder, base: Any = None, sliders: dict[str, float] | None = None):
        """The render settings of a layer: its base moved by its slider positions."""
        return settings_at(layer_kind(folder), base, sliders, has_denoise(folder))

    def render(self, folder, base: Any = None, sliders: dict[str, float] | None = None, scale: float = 0.25,
               region=None, part: str | None = None, fit: dict | None = None) -> LayerImage:
        """A workflow layer at a scale (1, 1/2, 1/4...), or a region (x0, x1, y0, y1) of the full-resolution grid.
        ``part``: None, 'stars' (the layer minus its starless image, clipped at 0) or 'starless' (engine layers)."""
        if part not in PARTS:
            raise ValueError(f"unknown layer part {part!r} (allowed: {PARTS})")
        kind = layer_kind(folder)
        if part is not None and kind != "engine":
            raise ValueError("only a layer with a star model splits into stars and starless")
        s = self.settings(folder, base, sliders)
        t0 = time.perf_counter()
        lay, note = self._render(folder, s, scale, region, fit)
        if part == "starless":
            img = lay.starless
        elif part == "stars":
            img = (lay.final - lay.starless).clamp(0, 1)
        else:
            img = lay.final
        rgb = img.float().cpu().numpy()
        eng = self.engine(folder)
        info = {"kind": kind, "scale": lay.scale, "region": lay.info.get("region"), "part": part,
                "grid": [int(eng.W), int(eng.H)], "drizzle_scale": float(eng.SC), "device": str(eng.dev),
                "render_ms": round(1000 * (time.perf_counter() - t0), 1), "note": note,
                "settings": s.model_dump(mode="json")}
        if kind == "engine" and getattr(eng, "last_fit", None) is not None and region is None:
            info["fit"] = eng.last_fit
        return LayerImage(rgb, tuple(int(v) for v in eng.crop), info)

    # ------------------------------------------------------------------ masks
    def masks(self, folder, scale: float = 0.25, region=None) -> dict[str, np.ndarray | None]:
        """Post's data masks, float32 in [0, 1] on the grid at the scale (or the region): 'stars', every drawn star
        with its glow (engine layers; None for simple ones), and 'halpha', the H-alpha emission (0 at its median, 1 at
        its 97th percentile, smoothed)."""
        eng = self.engine(folder)
        f = max(1, int(round(1 / scale)))
        out: dict[str, np.ndarray | None] = {"stars": None, "halpha": None}
        with torch.no_grad():
            if isinstance(eng, RenderEngine):
                lay, _ = self._render(folder, RenderSettings(), scale, region)
                st = _smoothstep(ops.luminance(lay.stars.float()), 0.03, 0.15)
                st = F.max_pool2d(st[None, None], 3, 1, 1)[0, 0]
                out["stars"] = ops.gaussian_blur(st, 1.0).clamp(0, 1).cpu().numpy()
            out["halpha"] = self._halpha(eng, f, region)
        return out

    @staticmethod
    def _emission(eng, f: int, box) -> torch.Tensor:
        """The emission map on a box (x0, x1, y0, y1) of the grid, reduced by f: the engine's emission layer, the
        H-alpha excess of a v1 preparation, the red excess of a simple one."""
        x0, x1, y0, y1 = box
        if isinstance(eng, SimpleRenderEngine):
            lin = eng._host["lin"][:, y0:y1, x0:x1].to(eng.dev).float()
            E = lin[0] - lin[1]
        elif "Eg0" in eng._host:
            E = eng._host["Eg0"][y0:y1, x0:x1].to(eng.dev).float()
        else:
            a = eng._host[next(iter(eng._host))][:, y0:y1, x0:x1].to(eng.dev).float()
            E = a[0] - float(eng.prep.get("alpha", 1.0)) * a[1]
        if f > 1:
            h, w = E.shape[0] // f * f, E.shape[1] // f * f
            E = F.avg_pool2d(E[None, None, :h, :w], f)[0, 0]
        return ops.gaussian_blur(E, 6 / f)

    def _halpha(self, eng, f: int, region) -> np.ndarray:
        valid_host = eng.valid_host if isinstance(eng, SimpleRenderEngine) else eng.valid.cpu()
        # the thresholds come from the whole field at 1/4, so that a region is the crop of the whole map
        E4 = self._emission(eng, 4, (0, eng.W, 0, eng.H))
        v4 = F.avg_pool2d(valid_host[None, None, :E4.shape[0] * 4, :E4.shape[1] * 4].float(), 4)[0, 0] > 0.99
        q0, q1 = ops.percentile(E4, [50, 97], v4.to(E4.device))
        if region is None:
            E = self._emission(eng, f, (0, eng.W // f * f, 0, eng.H // f * f))
        else:
            x0, x1, y0, y1 = (int(v) // f * f for v in region)
            pad = 32 * f
            bx0, by0 = max(0, x0 - pad), max(0, y0 - pad)
            box = (bx0, min(eng.W // f * f, x1 + pad), by0, min(eng.H // f * f, y1 + pad))
            E = self._emission(eng, f, box)[(y0 - by0) // f:(y1 - by0) // f, (x0 - bx0) // f:(x1 - bx0) // f]
        return _smoothstep(E, q0, q1).cpu().numpy()


# ---------------------------------------------------------------------- export
FORMATS = ("tiff16", "tiff32f", "png16", "jpeg", "fits32f")
SIZES = (0, 2048, 4096)                  # long side in pixels; 0 = full resolution


class ExportOptions(StageParams):
    name: str = P("image", label="Name", help="Base name of the exported files.")
    formats: list[str] = P(["tiff16"], label="Formats", choices=list(FORMATS),
                           help="TIFF 16-bit, TIFF 32-bit float, PNG 16-bit, JPEG, FITS 32-bit float (with the "
                                "astrometric solution).")
    size: int = P(0, label="Size", choices=[str(v) for v in SIZES], unit="px",
                  help="Long side of the image; 0 = full resolution (the drizzled grid).")
    rot90: int = P(0, label="Rotation", ge=0, le=3, help="Quarter turns, counter-clockwise (never resampled).")
    flip: bool = P(False, label="Mirror", help="Vertical mirror, before the rotation.")
    jpeg_quality: int = P(93, label="JPEG quality", ge=50, le=100, help="Quality of the JPEG file (93 as in v1).")


def _icc() -> bytes:
    from PIL import ImageCms
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _pixel_map(H: int, W: int, origin, k: int, flip: bool, zoom: float) -> np.ndarray:
    """3 x 3 matrix taking an exported pixel (x, y, 1), 0-based, to the grid pixel it shows."""
    T = np.array([[1.0, 0, origin[0]], [0, 1.0, origin[1]], [0, 0, 1]])
    if flip:
        T = T @ np.array([[1.0, 0, 0], [0, -1.0, H - 1], [0, 0, 1]])
    w = W
    for i in range(k % 4):                  # each quarter turn: rotated (x, y) <- (w - 1 - y, x) before it
        T = T @ np.array([[0, -1.0, w - 1], [1.0, 0, 0], [0, 0, 1]])
        w = H if i % 2 == 0 else W
    if zoom != 1:
        T = T @ np.array([[1 / zoom, 0, 0.5 / zoom - 0.5], [0, 1 / zoom, 0.5 / zoom - 0.5], [0, 0, 1]])
    return T


def oriented_wcs(header, H: int, W: int, origin=(0, 0), k: int = 0, flip: bool = False, zoom: float = 1.0):
    """The WCS header of an exported image: the grid's (``header``) for an image taken at ``origin`` of the grid
    (H x W before orientation), turned k quarter turns, mirrored, resized by ``zoom``. The distortion terms (SIP) are
    kept for a plain crop and dropped otherwise (noted in the header)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    w = WCS(header)
    T = _pixel_map(H, W, origin, k, flip, zoom)
    A, t = T[:2, :2], T[:2, 2]
    cd = w.wcs.cd if w.wcs.has_cd() else w.pixel_scale_matrix
    crpix = np.linalg.solve(A, np.asarray(w.wcs.crpix, float) - t - 1 + A @ np.ones(2))
    h = w.to_header(relax=True)
    for key in [c for c in h if c.startswith(("PC", "CD", "CDELT"))]:
        del h[key]
    pure_shift = np.allclose(A, np.eye(2))
    if not pure_shift:
        for key in [c for c in h if c[:2] in ("A_", "B_") or c[:3] in ("AP_", "BP_")]:
            del h[key]
        for i in (1, 2):
            h[f"CTYPE{i}"] = str(h[f"CTYPE{i}"]).replace("-SIP", "")
        if w.sip is not None:
            h["COMMENT"] = "distortion terms (SIP) left out: the image was turned or resized"
    newcd = cd @ A
    for i in range(2):
        h[f"CRPIX{i + 1}"] = float(crpix[i])
        for j in range(2):
            h[f"CD{i + 1}_{j + 1}"] = float(newcd[i, j])
    return fits.Header(h)


def export_rgb(rgb: np.ndarray, out_dir: str | Path, options: ExportOptions | dict | None = None, wcs=None,
               origin=(0, 0), history: dict | None = None, progress=lambda f, m: None) -> list[Path]:
    """Write a composed image (3, H, W), display sRGB in [0, 1]. ``wcs``: the grid's astrometric solution (a FITS
    header, a WCS or a wcs.fits path) for the FITS file; ``origin``: the grid pixel of rgb[:, 0, 0] (the crop);
    ``history``: written beside the images as <name>_history.json. Returns the files written."""
    import cv2
    import tifffile
    from PIL import Image

    from .export import _orient
    o = options if isinstance(options, ExportOptions) else ExportOptions(**(options or {}))
    bad = [f for f in o.formats if f not in FORMATS]
    if bad:
        raise ValueError(f"unknown export format(s) {bad} (allowed: {FORMATS})")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = np.asarray(rgb, np.float32)
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"expected a (3, H, W) image, got {img.shape}")
    H, W = img.shape[1:]
    img = _orient(img, (0, W, 0, H), o.rot90, o.flip)
    zoom = 1.0
    if o.size and max(img.shape[1:]) > o.size:
        zoom = o.size / max(img.shape[1:])
        size = (max(1, round(img.shape[2] * zoom)), max(1, round(img.shape[1] * zoom)))
        img = np.stack([cv2.resize(c, size, interpolation=cv2.INTER_AREA) for c in img])
    img = np.clip(img, 0, 1)
    hwc = np.ascontiguousarray(img.transpose(1, 2, 0))
    icc = _icc()
    files = []
    for i, fmt in enumerate(o.formats):
        progress(0.1 + 0.8 * i / len(o.formats), f"writing {fmt}")
        if fmt == "tiff16":
            p = out_dir / f"{o.name}_16bit.tif"
            tifffile.imwrite(p, (hwc * 65535 + 0.5).astype(np.uint16), photometric="rgb", iccprofile=icc,
                             compression="zlib")
        elif fmt == "tiff32f":
            p = out_dir / f"{o.name}_32bit_float.tif"
            tifffile.imwrite(p, hwc, photometric="rgb", iccprofile=icc, compression="zlib")
        elif fmt == "png16":
            p = out_dir / f"{o.name}.png"
            cv2.imwrite(str(p), (hwc * 65535 + 0.5).astype(np.uint16)[:, :, ::-1])
        elif fmt == "jpeg":
            p = out_dir / f"{o.name}.jpg"
            Image.fromarray((hwc * 255 + 0.5).astype(np.uint8)).save(p, quality=o.jpeg_quality, icc_profile=icc)
        else:
            from astropy.io import fits
            p = out_dir / f"{o.name}.fits"
            hdr = fits.Header()
            if wcs is not None:
                hdr = oriented_wcs(_wcs_header(wcs), H, W, origin, o.rot90, o.flip, zoom)
            hdr["COLORSPC"] = ("sRGB", "display-referred, non-linear")
            fits.PrimaryHDU(img.astype(np.float32), header=hdr).writeto(p, overwrite=True)
        files.append(p)
    if history is not None:
        p = out_dir / f"{o.name}_history.json"
        p.write_text(json.dumps(history, indent=1, default=str), encoding="utf-8")
        files.append(p)
    progress(1.0, "done")
    return files


def _wcs_header(wcs):
    from astropy.io import fits
    from astropy.wcs import WCS
    if isinstance(wcs, WCS):
        return wcs.to_header(relax=True)
    if isinstance(wcs, (str, Path)):
        return fits.getheader(wcs)
    return wcs
