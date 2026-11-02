"""Interactive rendering engine (GPU).

Works from the outputs of the ``render_prep`` stage. ``render(settings, scale)`` returns the nebula layer, the
star layer and their screen blend as display-referred sRGB tensors in [0, 1].

* ``scale = 1``: full resolution (exports, 100 % view).
* ``scale < 1``: fast preview; the nebula is processed on a reduced image with every spatial parameter scaled,
  the stars are always drawn at full resolution and then reduced, so star sizes and brightness match the export.

The nebula is the calibrated image at full resolution (star footprints guarded), mixed live with the optional layers
of ``render_prep`` (fine detail, fine colour, OIII, reflection tones), so these are ordinary sliders. Intermediate
results are cached by the settings they depend on: moving a star slider only redraws the stars, changing the
saturation only redoes the last colour step.

Regions (``render(s, scale, region=...)``): a part of the grid is rendered with a margin, the statistics of the whole
image (tone curve, white point, detail thresholds) taken from the *fit* of a 1/4 rendering, so that a 100 % view never
renders the whole grid. The fit of each rendering is kept in ``last_fit``; ``fit_of(s)`` gives the 1/4 one.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import ops
from .settings import RenderSettings

XYZ_TO_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660, 1.8760108, 0.0415560],
                        [0.0556434, -0.2040259, 1.0572252]])

LAYER_KEYS = ("fine_k", "fine_lo_pct", "fine_hi_pct", "faint_smooth", "faint_lo_pct", "faint_hi_pct", "resid_boost",
              "colour_detail", "oiii_boost", "oiii_threshold", "oiii_smooth", "sharpen", "sharpen_lo_pct",
              "real_stars", "spike_photo", "spike_g_max")
NEB_LIN_KEYS = LAYER_KEYS + ("ha_boost", "ha_sigma", "glow_k", "glow_sigma", "g_glow", "speck_k", "white_temp", "tint",
                             "star_glow", "glare_nebula", "star_sat", "diffuse_cut", "sky_neutral", "sky_pct")
NEB_KEYS = NEB_LIN_KEYS + ("chroma", "chroma_protect", "hue_target", "hue_shift", "chroma_sigma", "snr_lo", "snr_hi",
                           "sky_level", "dark_level", "toe", "sky_pct", "dark_pct", "white_mult", "white_ignore_pct",
                           "hdr_strength", "hdr_knee", "hdr_r", "hdr_eps", "lc_fine", "lc_medium", "lc_large", "neb_sharp",
                           "sharp_white", "hl_knee", "hl_max", "hl_detail", "hl_dmin", "hl_r1", "hl_r2", "hl_eps",
                           "hl_shoulder", "hl_keep_chroma", "bright_halo_desat")
STAR_KEYS = ("stars", "g_glow", "g_lim", "t_lim", "d10", "d7", "star_sat", "sat_min", "core_white", "t9",
             "white_temp", "tint", "real_stars", "star_glow", "glare_nebula", "diffuse_cut", "spike_photo",
             "spike_g_max", "spike_smooth", "stars_photographed")
# the stages of the nebula, each cached per resolution with the settings it reads: a slider re-renders from the stage
# it moves (the sky from the tone curve, the contrast from the local contrast, the saturation only the last colour step)
NEB_ASTRO_KEYS = LAYER_KEYS + ("diffuse_cut", "ha_boost", "ha_sigma", "glow_k", "glow_sigma", "g_glow")
NEB_PRE_KEYS = NEB_LIN_KEYS + ("chroma_sigma", "snr_lo", "snr_hi", "dark_pct", "sky_pct", "white_ignore_pct")
NEB_TONE_KEYS = NEB_PRE_KEYS + ("sky_level", "dark_level", "toe", "white_mult", "hdr_strength", "hdr_knee", "hdr_r",
                                "hdr_eps")
NEB_DISP_KEYS = tuple(k for k in NEB_KEYS if k not in ("chroma", "chroma_protect", "bright_halo_desat"))
#: the cached images of one resolution (the full resolution ones are dropped when another resolution renders: they are
#: an export's, a gigabyte that would stay)
_STAGES = ("neb", "neb_disp", "neb_tone", "neb_pre", "neb_lin", "neb_astro", "neb_sharp_w", "stars", "stars_draft",
           "mask_stars")
#: the star layer at full resolution, before it is reduced: the same for every resolution (key 0, never dropped)
_FULL = 0
#: intermediate stages not kept for a whole-grid rendering at full resolution (an export: gigabytes used once); a
#: region's view keeps them (a slider moved at 100 % starts from them)
_ONE_SHOT = ("neb_disp", "neb_tone", "neb_pre", "neb_astro", "neb_sharp_w")


class Cancelled(Exception):
    """The caller's ``stop()`` asked to abandon the rendering (a newer request came). The stages already finished stay
    in the cache: the next rendering starts from them."""

#: margin rendered around a region (full-resolution pixels): the widest spatial supports of the rendering (faint area
#: smoothing, glow, large-scale contrast, highlight filters) stay inside it, and it is cropped away
REGION_MARGIN = 256
#: a region whose padded area shows fewer pixels than this takes even its resolution-dependent statistics (the noise)
#: from the fit
REGION_MIN_SHOWN = 4096
#: resolution of the fit a region uses
FIT_SCALE = 0.25


def settings_key(s) -> str:
    """A short digest of the settings a fit depends on: every statistic of the fit is the nebula's (NEB_KEYS), so a
    star slider moved at 100 % keeps the fit."""
    blob = json.dumps({k: getattr(s, k) for k in NEB_KEYS}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


@dataclass
class Layers:
    final: torch.Tensor
    starless: torch.Tensor
    stars: torch.Tensor | None
    scale: float
    info: dict


class RenderEngine:
    # statistics taken from a fit (region renderings, see _stat); a plain rendering measures them all
    _fit_in: dict | None = None
    _fit_f: int = 0
    _local_ok: bool = True
    _stop = None                                         # the caller's stop() during a rendering (see Cancelled)
    _scratch: dict = {}                                  # the one-shot stages of the rendering going on (_ONE_SHOT)

    def __init__(self, prep_dir: str | Path, device: str = "cuda:0", calibrated_white: float = 5778.0):
        self.dir = Path(prep_dir)
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.prep = json.loads((self.dir / "prep.json").read_text())
        self.M = np.array(self.prep["M"], np.float64)
        self.SC = float(self.prep["drizzle_scale"])
        self.crop = self.prep["crop"]                        # x0, x1, y0, y1
        self.calibrated_white = calibrated_white
        # preparation layers stay in host memory; reduced versions are cached on the device, the full resolution is
        # only uploaded while it is being mixed
        # layered preparations (version >= 7) write N.npy and describe the layers in prep.json; an older preparation
        # may share a folder with leftover layer files, so prep.json decides
        self.layered = (self.dir / "N.npy").exists() and "w_cam" in self.prep
        if self.layered:
            names = ["N", "Eg0", "dY", "R"] + [n for n in ("chroma_fine", "dS", "Dglow") if (self.dir / f"{n}.npy").exists()]
            if (self.dir / "O.npy").exists() and self.prep.get("oiii_sigma"):
                names.append("O")
                self.o_vec = torch.tensor(self.prep.get("o_display", self.prep["o"]), dtype=torch.float32,
                                          device=self.dev)
            self._host = {n: torch.from_numpy(np.load(self.dir / f"{n}.npy")) for n in names}
            self.w_cam = torch.tensor(self.prep["w_cam"], dtype=torch.float32, device=self.dev)
            self.lum = torch.tensor(np.array([0.2126, 0.7152, 0.0722]) @ self.M, dtype=torch.float32, device=self.dev)
        elif (self.dir / "astro_prep.npy").exists():         # preparation of version 1: one pre-mixed layer
            self._host = {"astro": torch.from_numpy(np.load(self.dir / "astro_prep.npy"))}
        else:
            raise RuntimeError("these rendering layers were prepared by an older version: run the 'Rendering "
                               "preparation' stage again (about a minute)")
        self.H, self.W = next(iter(self._host.values())).shape[-2:]
        self.valid = torch.from_numpy(np.load(self.dir / "valid.npy")).to(self.dev)
        st = np.load(self.dir / "stars.npz")
        # magnitudes relative to the depth of the data (0 for a field where S/N = 10 is reached at G = 15.5)
        self.X, self.Y, self.G = st["x"], st["y"], st["g"] - float(self.prep.get("depth_shift", 0.0))
        # stars shown by their real light (saturated cores, with their halo and spikes) and the measured spikes of the
        # others: a linear layer added to the drawn stars (preparations of version 18 and later)
        self.real = st["real"] if "real" in st.files else np.zeros(len(self.X), bool)
        sr = self.dir / "Sreal.npy"
        self.Sreal = torch.from_numpy(np.load(sr)) if sr.exists() and "real" in st.files else None
        # measured shape (diamond, spikes) of the drawn stars: drawn with them, attenuated like their wings
        ss_ = self.dir / "Sshape.npy"
        self.Sshape = torch.from_numpy(np.load(ss_)) if ss_.exists() and self.Sreal is not None else None
        # far halo of the stars (glare of the optics), taken out of the nebula: drawn back as a glow
        sh_ = self.dir / "Shalo.npy"
        if sh_.exists():
            self._host["Shalo"] = torch.from_numpy(np.load(sh_))
            if (self.dir / "ShaloL.npy").exists():
                self._host["ShaloL"] = torch.from_numpy(np.load(self.dir / "ShaloL.npy"))
        # photographed spikes (preparations of version 41 and later): the nebula interpolated across the spike strips
        # (dNsp, change from the base), the strips' weight (Wsp; Wsl without the star cores), the magnitude of the star
        # of each strip (Gsp) and the spikes' light averaged along them (Ssp) and raw (Ssp_raw)
        # every star as photographed (learned preparations only)
        sa_ = self.dir / "Sall.npy"
        self.Sall = torch.from_numpy(np.load(sa_)) if sa_.exists() else None
        self.Ssp = self.Ssp_raw = None
        if self.layered and self.Sreal is not None and (self.dir / "Wsp.npy").exists():
            for n in ("Wsp", "Wsl", "Gsp", "dNsp"):
                self._host[n] = torch.from_numpy(np.load(self.dir / f"{n}.npy"))
            self.Ssp = torch.from_numpy(np.load(self.dir / "Ssp.npy"))
            self.Ssp_raw = torch.from_numpy(np.load(self.dir / "Ssp_raw.npy"))
        self.Fm,self.Fpred, self.Fglow = st["flux"].astype(np.float64), st["pred"].astype(np.float64), st["glow_flux"]
        pr = np.load(self.dir / "profile.npz")
        self.r_all, self.prof_raw = pr["r"], pr["prof"]
        self.alpha = float(self.prep["alpha"])
        self.vh = torch.tensor(self.prep["vh"], dtype=torch.float32, device=self.dev)
        self._proxy: dict[int, tuple] = {}
        self._cache: dict[tuple, tuple] = {}
        self._scratch: dict[tuple, tuple] = {}
        self.last_info: dict = {}
        self._fits: dict[int, dict] = {}                    # resolution factor -> statistics last measured there
        self.last_fit: dict | None = None
        self._fit_memo: OrderedDict = OrderedDict()         # (settings key, f) -> fit
        self._view_cache: tuple | None = None               # (key, engine view) of the last region

    # ------------------------------------------------------------------ helpers
    def _stat(self, name: str, f: int, fn, scale_free: bool = True):
        """A statistic of the whole image (percentiles, noise): measured, or taken from the fit of another rendering.
        A region uses the fit of the 1/4 rendering, so that its tone curve and thresholds are those of the whole
        image; a statistic that depends on the resolution (the noise) is measured on the region itself, unless the
        region shows too few pixels or the fit has the region's resolution."""
        fit = self._fit_in
        if fit is not None and name in fit and (scale_free or self._fit_f == f or not self._local_ok):
            v = fit[name]
        else:
            v = fn()
        self._fits.setdefault(f, {})[name] = v
        return v

    def _layer(self, name: str, f: int) -> torch.Tensor:
        """One preparation layer on the device in float32, reduced by an integer factor f (1 = full resolution)."""
        if (name, f) in self._proxy:
            return self._proxy[name, f]
        t = self._host[name].to(self.dev).float()
        if f == 1:
            return t
        h, w = self.H // f * f, self.W // f * f
        r = F.avg_pool2d(t[..., :h, :w].reshape(-1, 1, h, w), f).reshape(*t.shape[:-2], h // f, w // f)
        self._proxy[name, f] = r
        return r

    def _valid(self, f: int) -> torch.Tensor:
        if f == 1:
            return self.valid
        if ("valid", f) not in self._proxy:
            h, w = self.H // f * f, self.W // f * f
            self._proxy["valid", f] = F.avg_pool2d(self.valid[None, None, :h, :w].float(), f)[0, 0] > 0.99
        return self._proxy["valid", f]

    def _shown(self, f: int) -> torch.Tensor:
        """The covered pixels of the cropped frame, the part of the image shown: the statistics of the rendering (tone
        curve, white point, noise, ranges of the detail layers) are measured there, not on the borders the crop
        removes (a bright edge outside it set the Heart's white point 44 % too high)."""
        if ("shown", f) not in self._proxy:
            x0, x1, y0, y1 = self.crop
            inside = torch.zeros_like(self._valid(f))
            inside[y0 // f:y1 // f, x0 // f:x1 // f] = True
            self._proxy["shown", f] = self._valid(f) & inside
        return self._proxy["shown", f]

    def _spike_w(self, s: RenderSettings, f: int):
        """Weight of the photographed spikes against the drawn ones, per pixel of the strips: 'Photographed spikes'
        times the stars brighter than 'Photographed spikes down to' (0.5 mag transition). None = all drawn."""
        if self.Ssp is None or not s.real_stars or s.spike_photo <= 0:
            return None

        def make():
            g = self._host["Gsp"].to(self.dev).float()
            w = s.spike_photo * ((s.spike_g_max - g) / 0.5 + 0.5).clamp(0, 1)
            if f == 1:
                return w
            h, w_ = self.H // f * f, self.W // f * f
            return F.avg_pool2d(w[None, None, :h, :w_], f)[0, 0]
        return self._cached(f"spike_w{f}", ("real_stars", "spike_photo", "spike_g_max"), s, f, make)

    def _shape(self, f: int) -> tuple[int, int]:
        return (self.H, self.W) if f == 1 else (self.H // f, self.W // f)

    def _astro(self, s: RenderSettings, f: int) -> torch.Tensor:
        """Calibrated nebula in camera space: the full-resolution layer plus the optional layers."""
        if not self.layered:
            return self._layer("astro", f)
        shown = self._shown(f)
        N = self._layer("N", f)
        # photographed spikes: the nebula interpolated across their strips; there the fine detail layers (taken from
        # the starless images the shared spike shape was subtracted from) are off and the colour layers interpolated
        # from the strips' edges (switched off, the boosted colours left grey bars across the nebula: Rosette)
        wsp = self._spike_w(s, f)
        keep, wk = 1.0, None
        if wsp is not None:
            N = N + wsp[None] * self._layer("dNsp", f)
            wk = wsp * self._layer("Wsp", f)
            keep = 1 - wk
        # 'Fine detail' = factor on the network's fine scales already in the base (newer preparations), or amount of
        # network detail added to a Wiener base (older ones)
        kf = s.fine_k - 1.0 if self.prep.get("fine_layer") == "scale" else s.fine_k
        sharpen = s.sharpen if "dS" in self._host else 0.0
        new = kf != 0 or s.faint_smooth > 0 or sharpen != 0 or wsp is not None
        if kf != 0:                          # Noise2Noise fine detail (brightness), above a brightness of the signal
            Eg0 = self._layer("Eg0", f)
            q1, q2 = self._stat("fine_q", f, lambda e=Eg0: ops.percentile(e, [s.fine_lo_pct, s.fine_hi_pct], shown))
            wf = ops.gaussian_blur(((Eg0 - q1) / max(q2 - q1, 1e-6)).clamp(0, 1), 6 / f)
            del Eg0
            N = N + kf * (keep * wf * self._layer("dY", f))[None] * self.w_cam[:, None, None]
            del wf
        if sharpen != 0:                     # measured-shape sharpening (brightness), above a brightness of the signal
            Eg0 = self._layer("Eg0", f)
            q1, q2 = self._stat("sharpen_q", f, lambda e=Eg0: ops.percentile(e, [s.sharpen_lo_pct, s.fine_hi_pct], shown))
            ws = ops.gaussian_blur(((Eg0 - q1) / max(q2 - q1, 1e-6)).clamp(0, 1), 6 / f)
            del Eg0
            N = N + sharpen * (keep * ws * self._layer("dS", f))[None] * self.w_cam[:, None, None]
            del ws
        if s.faint_smooth > 0:               # faint areas: only the scales confirmed by the data
            Eg = ops.gaussian_blur(torch.einsum("c,chw->hw", self.lum, N), 8 / f)
            q1, q2 = self._stat("faint_q", f, lambda e=Eg: ops.percentile(e, [s.faint_lo_pct, s.faint_hi_pct], shown))
            wE = ops.gaussian_blur(((Eg - q1) / max(q2 - q1, 1e-6)).clamp(0, 1), 6 / f)[None]
            del Eg
            N = wE * N + (1 - wE) * ops.gaussian_blur(N, s.faint_smooth / f)
            del wE
        # cached / host layers are never modified in place: copy unless a step above already made a new tensor
        astro = N if new else N.clone()
        if s.resid_boost != 0:
            astro += s.resid_boost * _fill_across(self._layer("R", f), wk, f)
        if s.colour_detail > 0 and "chroma_fine" in self._host:
            astro += s.colour_detail * _fill_across(self._layer("chroma_fine", f), wk, f)
        if s.oiii_boost > 0 and "O" in self._host:        # OIII emission above the noise, in its own colour
            astro += s.oiii_boost * self.o_vec[:, None, None] * _fill_across(self._oiii(s, f, shown), wk, f)[None]
        return astro

    def white_balance(self, s: RenderSettings) -> np.ndarray:
        """Diagonal in linear sRGB that makes a blackbody of ``white_temp`` neutral (identity at the calibration)."""
        import colour

        from ..stages.color import bradford, white_xyz
        d65 = colour.xy_to_XYZ(colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"])
        A = XYZ_TO_SRGB @ bradford(white_xyz(self.calibrated_white), d65)
        rgb_t = A @ white_xyz(s.white_temp)
        rgb_c = A @ white_xyz(self.calibrated_white)
        wb = rgb_c / rgb_t
        wb[1] *= 1 - s.tint
        return wb / (0.2126 * wb[0] + 0.7152 * wb[1] + 0.0722 * wb[2])

    def M_eff(self, s: RenderSettings) -> torch.Tensor:
        return torch.tensor(np.diag(self.white_balance(s)) @ self.M, dtype=torch.float32, device=self.dev)

    def _cached(self, name, keys, s, extra, fn, info: dict | None = None):
        """A stage's result, kept per (name, extra) - extra is the resolution - with the settings it was made with.
        ``info``: the rendering's information; the stage's own (``fn(sub)`` writes it) is kept with the result and
        given again when the result is reused, so that a reused stage says what a computed one would."""
        k = tuple(getattr(s, n) for n in keys)
        slot = (name, extra)
        hit = self._cache.get(slot) or self._scratch.get(slot)
        if hit is not None and hit[0] == k:
            if info is not None:
                info.update(hit[2])
            return hit[1]
        if self._stop is not None and self._stop():
            raise Cancelled()
        sub: dict = {}
        v = fn(sub) if info is not None else fn()
        if extra == 1 and self._fit_in is None and name in _ONE_SHOT:
            self._scratch[slot] = (k, v, sub)                  # for this rendering only
        else:
            self._cache[slot] = (k, v, sub)
        if info is not None:
            info.update(sub)
        return v

    def _trim(self, f: int) -> None:
        """Before a rendering at f: the images of the full resolution of another rendering go (see _STAGES)."""
        if f != 1:
            for slot in [sl for sl in self._cache if sl[1] == 1 and sl[0] in _STAGES]:
                del self._cache[slot]

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def render(self, s: RenderSettings, scale: float = 1.0, region=None, fit: dict | None = None, *,
               draft: bool = False, stop=None) -> Layers:
        """The layers at a resolution (``scale`` 1, 1/2, 1/4...).

        ``region`` (x0, x1, y0, y1), in full-resolution pixels of the grid: only that part, rendered with a margin and
        with the statistics of ``fit`` (the ``last_fit`` of a rendering with the same settings; by default the 1/4 one,
        ``fit_of``). A fit without a region renders the whole grid with the fit's statistics.

        ``draft``: a quick look while a slider moves (a reduced scale only): the nebula is the exact one at that scale,
        the stars are drawn on a reduced grid instead of at full resolution. ``stop()``: asked between the stages, True
        abandons the rendering (``Cancelled``); what was finished stays cached."""
        from ..core.determinism import deterministic
        prev = self._stop                        # (fit_of renders inside a region's rendering: its stop stays)
        if stop is not None:
            self._stop = stop
        try:
            with deterministic():                # the same settings give the same image (stars drawn in a fixed order)
                if region is None and fit is None:
                    return self._render(s, scale, draft)
                return self._render_region(s, scale, region, fit)
        finally:
            self._stop = prev

    @torch.no_grad()
    def stars_only(self, s: RenderSettings, scale: float = 1.0, region=None) -> torch.Tensor:
        """The star layer of ``render(s, scale[, region]).stars`` without the nebula (Post's star mask). Its own cache
        and view, so that it never replaces the images of the renderings shown."""
        from ..core.determinism import deterministic
        f = max(1, int(round(1 / scale)))
        with deterministic():
            if region is None:
                def make(sub):
                    hit = self._cache.get(("stars_full", _FULL))
                    key = tuple(getattr(s, n) for n in STAR_KEYS)
                    st = hit[1] if hit is not None and hit[0] == key else self._star_layer(s, sub, 1)
                    if f > 1:
                        h, w = self.H // f * f, self.W // f * f
                        st = F.avg_pool2d(st[None, :, :h, :w], f)[0]
                    return st
                return self._cached("mask_stars", STAR_KEYS, s, f, make, {})
            box, sl, _ = self._region_box(region, f)
            v = self._make_view(box, {"key": "stars", "f": 4, "stats": {}}, f)
            return v._stars(s, f, {})[sl]

    def fit_of(self, s: RenderSettings, scale: float = FIT_SCALE) -> dict:
        """The fit of the rendering of ``s`` at ``scale``: {key, f, stats}, the statistics of the whole image (tone
        curve percentiles, white point, detail thresholds, noise). Remembered from an earlier rendering, or rendered."""
        f = max(1, int(round(1 / scale)))
        k = (settings_key(s), f)
        if k not in self._fit_memo:
            self.render(s, 1.0 / f)
        return self._fit_memo[k]

    def _remember_fit(self, fit: dict) -> None:
        k = (fit["key"], fit["f"])
        self._fit_memo[k] = fit
        self._fit_memo.move_to_end(k)
        while len(self._fit_memo) > 8:
            self._fit_memo.popitem(last=False)

    def _region_box(self, region, f: int):
        """(box rendered with its margin, slice of the region in it) for a region (x0, x1, y0, y1) at the factor f;
        region None: the whole grid."""
        Hf, Wf = self.H // f * f, self.W // f * f            # the part of the grid a reduced rendering covers
        if region is None:
            x0, x1, y0, y1 = 0, Wf, 0, Hf
            box = (0, self.W, 0, self.H)
        else:
            x0, x1, y0, y1 = (int(round(float(v))) for v in region)
            x0, x1 = sorted((min(max(x0, 0), Wf), min(max(x1, 0), Wf)))
            y0, y1 = sorted((min(max(y0, 0), Hf), min(max(y1, 0), Hf)))
            x0, y0 = x0 // f * f, y0 // f * f                  # aligned on the reduced pixels
            x1, y1 = min(-(-x1 // f) * f, Wf), min(-(-y1 // f) * f, Hf)
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"empty region {list(region)} on a {self.W} x {self.H} grid")
            pad = -(-REGION_MARGIN // f) * f
            box = (max(0, x0 - pad), min(Wf, x1 + pad), max(0, y0 - pad), min(Hf, y1 + pad))
        bx0, _, by0, _ = box
        sl = (slice(None), slice((y0 - by0) // f, (y1 - by0) // f), slice((x0 - bx0) // f, (x1 - bx0) // f))
        return box, sl, [x0, x1, y0, y1]

    def _render_region(self, s: RenderSettings, scale: float, region, fit: dict | None) -> Layers:
        f = max(1, int(round(1 / scale)))
        box, sl, aligned = self._region_box(region, f)
        if fit is None or fit.get("key") != settings_key(s):
            fit = self.fit_of(s)
        view = self._region_view(box, fit, f)
        view._stop = self._stop
        try:
            lay = view._render(s, scale)
        finally:
            view._stop = None
        info = {**lay.info, "region": aligned, "fit_scale": 1.0 / fit["f"]}
        return Layers(lay.final[sl], lay.starless[sl], lay.stars[sl], 1.0 / f, info)

    def _region_view(self, box: tuple[int, int, int, int], fit: dict, f: int) -> RenderEngine:
        """This engine cut to a box (x0, x1, y0, y1) of the grid: its layers, stars and masks cut to the box, its own
        caches, and the statistics of ``fit``. Kept for the next rendering of the same box and fit, so that moving a
        star slider at 100 % only redraws the stars."""
        key = (box, fit["key"], fit["f"], f)
        if self._view_cache is not None and self._view_cache[0] == key:
            return self._view_cache[1]
        self._view_cache = None
        v = self._make_view(box, fit, f)
        self._view_cache = (key, v)
        return v

    def _make_view(self, box: tuple[int, int, int, int], fit: dict, f: int) -> RenderEngine:
        x0, x1, y0, y1 = box
        v = copy.copy(self)
        v._host = {n: t[..., y0:y1, x0:x1] for n, t in self._host.items()}
        for n in ("Sreal", "Sshape", "Sall", "Ssp", "Ssp_raw"):
            t = getattr(self, n, None)
            if t is not None:
                setattr(v, n, t[..., y0:y1, x0:x1])
        v.valid = self.valid[y0:y1, x0:x1]
        v.H, v.W = y1 - y0, x1 - x0
        cx0, cx1, cy0, cy1 = self.crop
        v.crop = [min(max(cx0 - x0, 0), v.W), min(max(cx1 - x0, 0), v.W),
                  min(max(cy0 - y0, 0), v.H), min(max(cy1 - y0, 0), v.H)]
        # the stars near the box, in its coordinates (the glow and halo stamps of the others stay in the margin)
        m = (self.X > x0 - 200) & (self.X < x1 + 200) & (self.Y > y0 - 200) & (self.Y < y1 + 200)
        v.X, v.Y, v.G, v.real = self.X[m] - x0, self.Y[m] - y0, self.G[m], self.real[m]
        v.Fm, v.Fpred, v.Fglow = self.Fm[:, m], self.Fpred[:, m], self.Fglow[:, m]
        # the star display curve is fitted on every star of the field, as in the whole rendering
        v._star_fit = lambda s, Gd, Lum: self._field_star_fit(s)
        v._proxy, v._cache, v._scratch, v.last_info, v._fits = {}, {}, {}, {}, {}
        v._fit_memo, v._view_cache, v.last_fit = OrderedDict(), None, None
        v._fit_in, v._fit_f = fit["stats"], fit["f"]
        v._local_ok = int(v._shown(f).sum()) >= REGION_MIN_SHOWN
        v._stop = None
        return v

    def _field_star_fit(self, s: RenderSettings) -> dict:
        Gd, _, Lum = self._star_light(s, self._star_draw(s))
        return self._star_fit(s, Gd, Lum)

    def _render(self, s: RenderSettings, scale: float = 1.0, draft: bool = False) -> Layers:
        f = max(1, int(round(1 / scale)))
        self._trim(f)
        self._scratch = {}
        try:
            return self._render_f(s, f, draft)
        finally:
            self._scratch = {}

    def _render_f(self, s: RenderSettings, f: int, draft: bool) -> Layers:
        info: dict = {}
        neb = self._cached("neb", NEB_KEYS, s, f, lambda sub: self._nebula(s, f, sub), info)
        if s.stars and draft and f > 1:
            st = self._cached("stars_draft", STAR_KEYS, s, f, lambda sub: self._stars_draft(s, f, sub), info)
        elif s.stars:
            st = self._cached("stars", STAR_KEYS, s, f, lambda sub: self._stars(s, f, sub), info)
        else:
            st = torch.zeros_like(neb)
        final = ops.screen(neb, st)
        valid = self._valid(f)
        final[:, ~valid] = 0
        # the parts taken from the cache keep the information of their last computation; the fresh parts' wins
        info = {**self.last_info, **info}
        self.last_info = info
        # the statistics behind this image (the cached parts' from their computation, made with the same settings)
        self.last_fit = {"key": settings_key(s), "f": f, "stats": dict(self._fits.get(f, {}))}
        self._remember_fit(self.last_fit)
        return Layers(final, neb, st, 1.0 / f, info)

    # ------------------------------------------------------------------ nebula
    def _nebula_astro(self, s: RenderSettings, f: int):
        """The calibrated nebula with the diffuse glow taken out, the H-alpha boost and the glow of the faint stars:
        what the star sliders do not move (cached as neb_astro, never modified in place)."""
        astro = self._astro(s, f)
        if not self.layered:                               # cached / host tensor: never modified in place
            astro = astro.clone()
        if s.diffuse_cut > 0 and "Dglow" in self._host:   # the diffuse glow of the field taken as background
            astro = astro - s.diffuse_cut * self._layer("Dglow", f)
        if s.ha_boost > 0:
            Hex = ops.gaussian_blur(astro[0] - self.alpha * astro[1], s.ha_sigma / f)
            astro = astro + s.ha_boost * Hex.clamp(min=0)[None] * self.vh[:, None, None]
        if s.glow_k > 0:
            astro = astro + s.glow_k * self._glow(s, f)
        return astro

    def _nebula_linear(self, s: RenderSettings, f: int):
        valid = self._valid(f)
        astro = self._cached("neb_astro", NEB_ASTRO_KEYS, s, f, lambda: self._nebula_astro(s, f))
        own = False                                        # astro is the cached tensor until an operation makes a new one
        if s.star_glow > 0 and "Shalo" in self._host:
            # the glare of the optics around the stars, as photographed: under the nebula curve, like the residual
            # rays of the brightest stars that stay in the nebula (with the stars, the rays stood out of a dark sky)
            # it is star light: the saturation of the stars, not of the nebula (a saturated blue disc around Alnitak)
            g = self._glare(s, f)
            if self.layered:
                yg = torch.einsum("c,chw->hw", self.lum, g)[None] * self.w_cam[:, None, None]
                g = yg + s.star_sat * (g - yg)
            astro = astro + s.star_glow * s.glare_nebula * g
            own = True
        if s.speck_k > 0 and f == 1:
            if not own:
                astro = astro.clone()
            for c in range(3):
                a_ = astro[c]
                sg_ = ops.robust_noise(a_, sub=7)
                astro[c] = torch.maximum(a_, ops.median_filter(a_, 5) - s.speck_k * sg_)
        lin = torch.einsum("ij,jhw->ihw", self.M_eff(s), astro)
        if s.sky_neutral > 0:
            # background neutralisation: the typical sky (sky_pct, smoothed) made grey at the same brightness, by an
            # offset per channel - the faint emission that fills a field stops tinting it, the bright stays coloured
            sky = torch.tensor(self._stat("sky_neutral", f, lambda: [
                float(ops.percentile(ops.gaussian_blur(lin[c], 8 / f), s.sky_pct, self._shown(f))) for c in range(3)]),
                device=lin.device)
            y = 0.2126 * sky[0] + 0.7152 * sky[1] + 0.0722 * sky[2]
            lin = lin - s.sky_neutral * (sky - y)[:, None, None]
        lin[:, ~valid] = 0
        return lin

    def _glare(self, s: RenderSettings, f: int) -> torch.Tensor:
        """The glare of the stars (camera space), without the part taken as background with the diffuse glow."""
        g = self._layer("Shalo", f)
        if s.diffuse_cut > 0 and "ShaloL" in self._host:
            g = g - s.diffuse_cut * self._layer("ShaloL", f)
        return g.clamp(min=0)

    def _oiii(self, s: RenderSettings, f: int, valid: torch.Tensor) -> torch.Tensor:
        """OIII map at the chosen smoothing, minus its sky level, soft-thresholded at the measured sky noise."""
        O = self._layer("O", f)
        extra = max(s.oiii_smooth ** 2 - float(self.prep.get("oiii_scale", 6.0)) ** 2, 0.0) ** 0.5
        if extra > 0:
            O = ops.gaussian_blur(O, extra / f)
        Eg0 = self._layer("Eg0", f)
        sky = valid & (Eg0 < self._stat("oiii_eg0", f, lambda: ops.percentile(Eg0, 50, valid)))
        o0 = self._stat("oiii_o0", f, lambda: ops.percentile(O, 50, sky))
        sig = self._stat("oiii_sig", f, lambda: 1.4826 * ops.percentile((O - o0).abs(), 50, sky))
        return (O - o0 - s.oiii_threshold * sig).clamp(min=0)

    def _glow(self, s: RenderSettings, f: int) -> torch.Tensor:
        h, w = self._shape(f)
        m = self.G >= s.g_glow
        xi = np.clip(np.round(self.X[m] / f - 0.5 * (f - 1) / f).astype(np.int64), 0, w - 1)
        yi = np.clip(np.round(self.Y[m] / f - 0.5 * (f - 1) / f).astype(np.int64), 0, h - 1)
        idx = torch.from_numpy(yi * w + xi).to(self.dev)
        glow = torch.zeros(3, h * w, device=self.dev)
        for c in range(3):
            glow[c].index_add_(0, idx, torch.from_numpy(self.Fglow[c, m].astype(np.float32)).to(self.dev))
        glow = glow.view(3, h, w) / (f * f)
        return ops.gaussian_blur(glow, s.glow_sigma / f)

    def _nebula(self, s: RenderSettings, f: int, info: dict) -> torch.Tensor:
        """The nebula layer in display sRGB: the stages below, then the saturation (chroma) and the halo
        desaturation."""
        neb_d = self._cached("neb_disp", NEB_DISP_KEYS, s, f, lambda sub: self._neb_disp(s, f, sub), info)
        neb_d = ops.chroma_ok(neb_d, s.chroma, s.chroma_protect)
        if s.bright_halo_desat > 0:
            neb_d = self._halo_desat(neb_d, s, f)
        neb_d[:, ~self._valid(f)] = 0
        return neb_d

    def _neb_pre(self, s: RenderSettings, f: int):
        """What the tone curve stands on: the luminance, the noise, the colour ratio, the percentiles of the curve."""
        lin = self._cached("neb_lin", NEB_LIN_KEYS, s, f, lambda: self._nebula_linear(s, f))
        shown = self._shown(f)
        Yl_full = ops.luminance(lin)
        noise = self._stat("noise", f, lambda: ops.robust_noise(Yl_full, shown, sub=max(1, 7 // f)), scale_free=False)
        Ys = ops.gaussian_blur(Yl_full, 2 / f)
        cb = ops.gaussian_blur(lin, s.chroma_sigma / f)
        cs = ops.gaussian_blur(lin, 1.0 / f)
        w = ((Ys / (noise / 2 + 1e-6) - s.snr_lo) / max(s.snr_hi - s.snr_lo, 1e-6)).clamp(0, 1)
        ratio = (w[None] * cs / ops.luminance(cs).clamp(min=1e-3 * noise)[None]
                 + (1 - w[None]) * cb / ops.luminance(cb).clamp(min=1e-3 * noise)[None]).clamp(-0.5, 4)
        del cb, cs, Ys
        # the tone curve is set on the nebula: the glare of the stars drawn back into it (star_glow) is light added
        # on top, it does not move the sky, the dark areas nor the white of the nebula
        Yn = Yl_full
        if s.star_glow > 0 and "Shalo" in self._host:
            Yn = Yl_full - s.star_glow * s.glare_nebula * ops.luminance(
                torch.einsum("ij,jhw->ihw", self.M_eff(s), self._glare(s, f)))
        p_dark, p_sky = self._stat("tone_p", f, lambda y=Yn: ops.percentile(ops.gaussian_blur(y, 6 / f),
                                                                            [s.dark_pct, s.sky_pct], shown))
        hi = self._stat("tone_hi", f, lambda y=Yn: ops.percentile(y, 100.0 - s.white_ignore_pct, shown))
        return Yl_full, ratio, p_dark, p_sky, hi

    def _neb_tone(self, s: RenderSettings, f: int, info: dict) -> torch.Tensor:
        """The luminance through the tone curve and the large-scale compression (the Sky slider starts here)."""
        Yl_full, _ratio, p_dark, p_sky, hi = self._cached("neb_pre", NEB_PRE_KEYS, s, f, lambda: self._neb_pre(s, f))
        lo, b, white = fit_tone_curve(p_dark, p_sky, hi * s.white_mult, s.dark_level / 255, s.sky_level / 255, s.toe)
        info["tone_curve"] = {"black": lo, "b": b, "white": white}
        L = tone_curve(Yl_full, lo, b, white, s.toe)
        if s.hdr_strength > 0:
            base = ops.guided(L, L, max(1, round(s.hdr_r / f)), s.hdr_eps)
            over = (base - s.hdr_knee).clamp(min=0)
            L = L - s.hdr_strength * over * (L / base.clamp(min=1e-4)).clamp(0, 2)
        return L

    def _neb_sharp_w(self, s: RenderSettings, f: int) -> torch.Tensor:
        """Where the nebula is sharpened, from its brightness (the part that does not depend on the tone curve)."""
        Yl_full = self._cached("neb_pre", NEB_PRE_KEYS, s, f, lambda: self._neb_pre(s, f))[0]
        shown = self._shown(f)
        Eg1 = ops.gaussian_blur(Yl_full, 8 / f)
        q1, q2 = self._stat("sharp_q", f, lambda: ops.percentile(Eg1, [85, 97], shown))
        return ops.gaussian_blur(((Eg1 - q1) / max(q2 - q1, 1e-6)).clamp(0, 1), 6 / f)

    def _neb_disp(self, s: RenderSettings, f: int, info: dict) -> torch.Tensor:
        """Local contrast, sharpening, highlights and the colour, before the saturation (the Contrast slider starts
        here)."""
        L = self._cached("neb_tone", NEB_TONE_KEYS, s, f, lambda sub: self._neb_tone(s, f, sub), info)
        ratio = self._cached("neb_pre", NEB_PRE_KEYS, s, f, lambda: self._neb_pre(s, f))[1]
        shown = self._shown(f)
        out, prev = L.clone(), L
        for sig, gain in zip((2, 8, 32), (s.lc_fine, s.lc_medium, s.lc_large)):
            bl = ops.gaussian_blur(L, sig / f)
            out += (gain - 1) * (prev - bl)
            prev = bl
        L = out.clamp(min=0)
        if s.neb_sharp > 0:
            wsh = self._cached("neb_sharp_w", NEB_PRE_KEYS, s, f, lambda: self._neb_sharp_w(s, f))
            wsh = wsh * (1 - ((ops.gaussian_blur(L, 3 / f) - s.sharp_white) / 0.15).clamp(0, 1))
            b1, b2, b3 = ops.gaussian_blur(L, 1.2 / f), ops.gaussian_blur(L, 3.0 / f), ops.gaussian_blur(L, 8.0 / f)
            L = (L + s.neb_sharp * wsh * ((L - b1) * 0.5 + (b1 - b2) * 1.0 + (b2 - b3) * 0.6)).clamp(min=0)
        L = self._highlights(L, s, f, shown, info)
        neb_raw = ratio * ops.srgb_eotf(L.clamp(0, 1))[None]
        if s.hl_keep_chroma > 0:
            mx = neb_raw.max(0, keepdim=True).values
            scaled = neb_raw / mx.clamp(min=1.0)
            desat = ops.gamut_map_oklab(neb_raw)
            neb = torch.where(mx > 1.0, s.hl_keep_chroma * scaled + (1 - s.hl_keep_chroma) * desat, desat)
        else:
            neb = ops.gamut_map_oklab(neb_raw)
        neb_d = ops.srgb_oetf(neb)
        if s.hue_shift != 0:
            lab = ops.lin_to_oklab(ops.srgb_eotf(neb_d))
            hh = torch.rad2deg(torch.atan2(lab[2], lab[1]))
            C = torch.hypot(lab[1], lab[2])
            wg = ((hh + 60) / 30).clamp(0, 1) * ((40 - hh) / 20).clamp(0, 1) * (C / 0.04).clamp(0, 1)
            h2 = torch.deg2rad(hh + s.hue_shift * wg * (s.hue_target - hh))
            lab = torch.stack([lab[0], C * torch.cos(h2), C * torch.sin(h2)])
            neb_d = ops.srgb_oetf(ops.gamut_map_oklab(ops.oklab_to_lin(lab)))
        return neb_d

    def _highlights(self, L, s, f, shown, info):
        L = L.clamp(min=0)
        knee, top = s.hl_knee, s.hl_max
        B = 0.5 * ops.guided(L, L, max(1, round(s.hl_r1 / f)), s.hl_eps) + \
            0.5 * ops.guided(L, L, max(1, round(s.hl_r2 / f)), s.hl_eps)
        D = L - B
        Bmax = self._stat("hl_bmax", f, lambda: ops.percentile(B, 99.995, shown))
        span = max(Bmax - knee, 1e-3)
        lo_c, hi_c = 1e-4, 1e3
        c_ = 1.0
        for _ in range(80):
            c_ = float(np.sqrt(lo_c * hi_c))
            if c_ * np.log1p(span / c_) > (top - knee):
                hi_c = c_
            else:
                lo_c = c_
        over = (B - knee).clamp(min=0)
        Bc = torch.where(B > knee, knee + c_ * torch.log1p(over / c_), B)
        slope = torch.where(B > knee, 1.0 / (1 + over / c_), torch.ones_like(B))
        gD = slope.clamp(s.hl_dmin, 1.0) * (1 + s.hl_detail * (over / max(Bmax - knee, 1e-3)).clamp(0, 1))
        Lt = Bc + gD * D
        sh = s.hl_shoulder
        L = torch.where(Lt > sh, sh + (1 - sh) * torch.tanh((Lt - sh) / (1 - sh)), Lt)
        info["highlights"] = {"base_max": Bmax, "above_knee_pct": float(100 * (B > knee)[shown].float().mean())}
        return L.clamp(0, 1)

    def _halo_desat(self, neb_d, s, f):
        import cv2
        h, w = neb_d.shape[1:]
        hm = np.zeros((h, w), np.float32)
        for i in np.where((self.G < 9.0) & (self.X > 0) & (self.X < self.W) & (self.Y > 0) & (self.Y < self.H))[0]:
            cv2.circle(hm, (int(round(self.X[i] / f)), int(round(self.Y[i] / f))),
                       int(np.clip(18 + 9 * (9 - self.G[i]), 18, 80) / f), 1.0, -1)
        hm_t = ops.gaussian_blur(torch.from_numpy(hm).to(self.dev), 8 / f)
        lab = ops.lin_to_oklab(ops.srgb_eotf(neb_d))
        lab[1:] = lab[1:] * (1 - s.bright_halo_desat * hm_t.clamp(0, 1))[None]
        return ops.srgb_oetf(ops.gamut_map_oklab(ops.oklab_to_lin(lab)))

    # ------------------------------------------------------------------ stars
    def _star_draw(self, s: RenderSettings) -> np.ndarray:
        """The stars drawn: brighter than the glow limit, near the cropped frame."""
        x0, x1, y0, y1 = self.crop
        return (self.G < s.g_glow) & (self.X > x0 - 60) & (self.X < x1 + 60) & (self.Y > y0 - 60) & (self.Y < y1 + 60)

    def _star_light(self, s: RenderSettings, draw: np.ndarray):
        """(G, linear display flux (3, n), luminance) of the drawn stars."""
        Fd, Gd, pr = self.Fm[:, draw], self.G[draw], self.Fpred[:, draw]
        wch = (np.clip((15.5 - Gd) / 3.0, 0, 1) * np.clip((Gd - 8.5) / 1.5, 0, 1))[None]
        chrom = wch * Fd / np.maximum(Fd.sum(0), 1e-6)[None] + (1 - wch) * pr / np.maximum(pr.sum(0), 1e-6)[None]
        Fs = self.M_eff(s).cpu().numpy().astype(np.float64) @ (chrom * np.clip(Fd.sum(0), 0, None)[None])
        Lum = 0.2126 * Fs[0] + 0.7152 * Fs[1] + 0.0722 * Fs[2]
        return Gd, Fs, Lum

    def _star_fit(self, s: RenderSettings, Gd: np.ndarray, Lum: np.ndarray) -> dict:
        """The star profile and display curve, fitted on every drawn star (a region uses the whole field's)."""
        return fit_star_curve(self.r_all, self.prof_raw, Gd, Lum, s, self.SC)

    def _stars(self, s: RenderSettings, f: int, info: dict, full: str = "stars_full") -> torch.Tensor:
        """The star layer at the factor f: drawn and displayed at full resolution (kept: the same for every factor),
        then reduced."""
        st_d = self._cached(full, STAR_KEYS, s, _FULL, lambda sub: self._star_layer(s, sub, 1), info)
        if f > 1:
            h, w = self.H // f * f, self.W // f * f
            st_d = F.avg_pool2d(st_d[None, :, :h, :w], f)[0]
        return st_d

    def _stars_draft(self, s: RenderSettings, f: int, info: dict) -> torch.Tensor:
        """A quick look of the star layer at the factor f (a draft, while a slider moves): drawn and displayed on a grid
        reduced by f/2 on a GPU, by f on the CPU (the star profile averaged over its pixels), then reduced to f. The
        stars look a little larger than in the exact rendering, which draws them at full resolution."""
        g = max(1, f // 2) if self.dev.type == "cuda" else f
        st_d = self._star_layer(s, info, g)
        if f > g:
            k = f // g
            h, w = st_d.shape[1] // k * k, st_d.shape[2] // k * k
            st_d = F.avg_pool2d(st_d[None, :, :h, :w], k)[0]
        return st_d[:, :self.H // f, :self.W // f]

    def _attr_at(self, name: str, g: int) -> torch.Tensor:
        """A star layer kept as an attribute (Sreal, Sshape, Sall, Ssp, Ssp_raw) on the device, reduced by g."""
        t = getattr(self, name)
        if g == 1:
            return t.to(self.dev).float()
        if (name, g) not in self._proxy:
            h, w = self.H // g * g, self.W // g * g
            x = t.to(self.dev).float()[..., :h, :w]
            r = F.avg_pool2d(x.reshape(-1, 1, h, w), g)
            self._proxy[name, g] = r.reshape(*x.shape[:-2], h // g, w // g)
        return self._proxy[name, g]

    def _star_layer(self, s: RenderSettings, info: dict, g: int = 1) -> torch.Tensor:
        """The star layer displayed, on the grid reduced by g (1: the exact full resolution)."""
        draw = self._star_draw(s)
        Gd, Fs, Lum = self._star_light(s, draw)
        s_g = s.star_sat * np.clip((s.g_lim + 1.0 - Gd) / 3.0, s.sat_min, 1.0)
        Fs = Lum[None] + s_g[None] * (Fs - Lum[None])
        fit = self._star_fit(s, Gd, Lum)
        # the display curve is fitted on every star; the stars shown by their real light are not drawn again
        use_real = self.Sreal is not None and s.real_stars
        dr = ~self.real[draw] if use_real else np.ones(int(draw.sum()), bool)
        info["stars"] = {k: v for k, v in fit.items() if k != "prof"}
        info["stars"]["drawn"] = int(dr.sum())
        if g == 1:
            st_lin = draw_stars(self.X[draw][dr], self.Y[draw][dr], Gd[dr], Fs[:, dr], fit["prof"], self.r_all, self.H,
                                self.W, self.dev)
        else:                                # reduced pixel j covers full pixels j*g ... j*g + g - 1
            off = 0.5 * (g - 1)
            st_lin = draw_stars((self.X[draw][dr] - off) / g, (self.Y[draw][dr] - off) / g, Gd[dr], Fs[:, dr],
                                box_profile(fit["prof"], self.r_all, g), self.r_all, self.H // g, self.W // g, self.dev,
                                scale=g)
        if use_real:
            sr = torch.tensordot(self.M_eff(s), self._attr_at("Sreal", g), 1)
            wsp = self._spike_w(s, g)
            if wsp is not None:             # each star's own spikes, as photographed
                raw = self._attr_at("Ssp_raw", g)
                sp = raw + s.spike_smooth * (self._attr_at("Ssp", g) - raw)
                sr += torch.tensordot(self.M_eff(s), wsp[None] * sp, 1)
                del raw, sp
            if self.Sshape is not None:
                # a drawn star's measured shape is part of its light beyond the core: same attenuation as its wings
                # (off where its spikes are shown as photographed)
                sh_ = self._attr_at("Sshape", g)
                if wsp is not None:
                    sh_ = sh_ * (1 - wsp * self._layer("Wsl", g))[None]
                sr += fit["wing_scale"] * torch.tensordot(self.M_eff(s), sh_, 1)
                del sh_
            lr = ops.luminance(sr)[None]
            st_lin += (lr + s.star_sat * (sr - lr)).clamp(min=0)
            info["stars"]["real_light"] = int((~dr).sum())
            del sr, lr
        if self.Sall is not None and s.stars_photographed > 0:
            sa = torch.tensordot(self.M_eff(s), self._attr_at("Sall", g), 1)
            la = ops.luminance(sa)[None]
            st_lin = (1 - s.stars_photographed) * st_lin + s.stars_photographed * (la + s.star_sat * (sa - la)).clamp(min=0)
            del sa, la
        if s.star_glow > 0 and s.glare_nebula < 1 and "Shalo" in self._host:
            # the rest of the glare with the stars, under their gentler curve (fades out close to the star)
            sh = torch.einsum("ij,jhw->ihw", self.M_eff(s), self._glare(s, g))
            lh = ops.luminance(sh)[None]
            st_lin += s.star_glow * (1 - s.glare_nebula) * (lh + s.star_sat * (sh - lh))
            del sh, lh
        Ys_ = ops.luminance(st_lin)
        Ls = torch.asinh(fit["b"] * Ys_.clamp(min=0) / fit["w"]) / np.arcsinh(fit["b"])
        st_d = ops.color_preserving_stretch_ok(st_lin, Ls.clamp(0, 1))
        del st_lin
        lab = ops.lin_to_oklab(ops.srgb_eotf(st_d))
        lab[1:] = lab[1:] * (1 - s.core_white * ((lab[0] - 0.7) / 0.3).clamp(0, 1))[None]
        st_d = ops.srgb_oetf(ops.gamut_map_oklab(ops.oklab_to_lin(lab)))
        st_d[:, ~(self.valid if g == 1 else self._valid(g))] = 0
        return st_d


class StarRemovalEngines:
    """The rendering engines of the two star removals: the star model (render_prep) and the learned one
    (render_prep_learned, when prepared). The setting 'Learned star removal' chooses at each rendering; the learned
    layers are loaded on first use. Attributes are those of the engine of the last rendering (the model one before)."""

    def __init__(self, prep_dir: str | Path, device: str = "cuda:0", calibrated_white: float = 5778.0):
        self.dirs = {False: Path(prep_dir), True: Path(prep_dir).parent / "render_prep_learned"}
        self._args = (device, calibrated_white)
        self._eng = {False: RenderEngine(prep_dir, device, calibrated_white)}
        self._cur = self._eng[False]

    @property
    def has_learned(self) -> bool:
        return (self.dirs[True] / "N.npy").exists() and (self.dirs[True] / "prep.json").exists()

    def select(self, s: RenderSettings) -> RenderEngine:
        key = bool(s.learned_stars) and self.has_learned
        if key not in self._eng:
            self._eng[key] = RenderEngine(self.dirs[key], *self._args)
        self._cur = self._eng[key]
        return self._cur

    def render(self, s: RenderSettings, scale: float = 1.0, region=None, fit: dict | None = None, **kw) -> Layers:
        return self.select(s).render(s, scale, region, fit, **kw)

    def stars_only(self, s: RenderSettings, scale: float = 1.0, region=None) -> torch.Tensor:
        return self.select(s).stars_only(s, scale, region)

    def fit_of(self, s: RenderSettings, scale: float = FIT_SCALE) -> dict:
        return self.select(s).fit_of(s, scale)

    def __getattr__(self, name):
        return getattr(self._cur, name)


def _fill_across(L: torch.Tensor, w, f: int, sigma: float = 6.0) -> torch.Tensor:
    """A layer interpolated across the strips of weight w (1 = inside) from their surroundings (normalised Gaussian
    of the layer outside them); unchanged where w = 0. w None = no strips."""
    if w is None:
        return L
    m = 1 - w
    num = ops.gaussian_blur(L * m, sigma / f)
    den = ops.gaussian_blur(m[None] if L.dim() == 3 else m, sigma / f).clamp(min=1e-3)
    return L * m + w * (num / den)


# ---------------------------------------------------------------------- tone curve
def tone_curve(y, lo, b, white, toe):
    if toe <= 0:
        if isinstance(y, torch.Tensor):
            return torch.asinh(b * (y - lo).clamp(min=0) / white) / np.arcsinh(b)
        return np.arcsinh(b * np.clip(y - lo, 0, None) / white) / np.arcsinh(b)
    if isinstance(y, torch.Tensor):
        z = torch.asinh(b * (y - lo) / white) / np.arcsinh(b)
        return toe * torch.logaddexp(torch.zeros_like(z), z / toe)
    z = np.arcsinh(b * (y - lo) / white) / np.arcsinh(b)
    return toe * np.logaddexp(0.0, z / toe)


def fit_tone_curve(p_dark, p_sky, white, t_dark, t_sky, toe):
    """Black point lo and strength b such that the sky percentile maps to t_sky and the dark one to t_dark.

    For each black point the strength that puts the sky at its level is found by bisection; the black point is then
    the one that puts the dark reference at its level (the dark error falls steadily with the black point): bracketed
    on a grid, then refined by bisection, so that the result moves smoothly with its inputs (on the grid alone it moved
    by steps of 2 % of the sky - dark interval). When no black point reaches the dark level, the closest grid point."""
    def strength(lo):
        a_, b_ = 1.0, 1e7
        bm = 1.0
        for _ in range(60):
            bm = np.sqrt(a_ * b_)
            if tone_curve(p_sky, lo, bm, white, toe) < t_sky:
                a_ = bm
            else:
                b_ = bm
        return bm

    def dark_err(lo):
        bm = strength(lo)
        return tone_curve(p_dark, lo, bm, white, toe) - t_dark, bm

    los = np.linspace(p_dark - 3 * (p_sky - p_dark), p_dark - 0.01 * (p_sky - p_dark), 200)
    errs = [dark_err(lo) for lo in los]
    e = np.array([v[0] for v in errs])
    i = int(np.argmin(np.abs(e)))
    cross = np.where(np.sign(e[:-1]) != np.sign(e[1:]))[0]
    if len(cross):
        j = int(cross[np.argmin(np.abs(cross - i))])            # the sign change next to the best grid point
        a_, b_ = float(los[j]), float(los[j + 1])
        ea = e[j]
        for _ in range(50):
            m = 0.5 * (a_ + b_)
            em, _ = dark_err(m)
            if np.sign(em) == np.sign(ea):
                a_, ea = m, em
            else:
                b_ = m
        lo = 0.5 * (a_ + b_)
        return float(lo), float(strength(lo)), float(white)
    return float(los[i]), float(errs[i][1]), float(white)


# ---------------------------------------------------------------------- stars
_STAR_FIT_CACHE: dict = {}


def fit_star_curve(r_all, prof_raw, Gd, Lum, s, SC):
    """Star profile (core contraction, wing attenuation) and display curve hitting the size / brightness targets."""
    def lum_at(gm):
        m_ = (Gd > gm - 0.25) & (Gd < gm + 0.25)
        return float(np.median(Lum[m_])) if m_.any() else float(np.interp(gm, np.sort(Gd), Lum[np.argsort(Gd)]))
    F_lim, F9, F10, F7 = lum_at(s.g_lim), lum_at(9.0), lum_at(10.0), lum_at(7.0)
    key = (round(F_lim, 6), round(F9, 6), round(F10, 6), round(F7, 6), s.t_lim, s.d10, s.d7, s.t9, SC)
    if key in _STAR_FIT_CACHE:
        return _STAR_FIT_CACHE[key]
    thr = 0.15
    RR = np.hypot(*np.mgrid[-60:61, -60:61])

    def make_prof(pr_raw, ws):
        wgt = 1 - (1 - ws) * np.clip((r_all - 2.5) / 2.0, 0, 1)
        p_ = pr_raw * wgt
        return p_ / np.interp(RR, r_all, p_).sum()

    def disp(Fv, pv, bb, ww):
        return np.arcsinh(bb * Fv * pv / ww) / np.arcsinh(bb)

    def fit_curve(pr_):
        p0 = pr_[0]
        lo_b, hi_b = 1e-3, 1e9
        bb = ww = 1.0
        for _ in range(100):
            bb = np.sqrt(lo_b * hi_b)
            ww = bb * F_lim * p0 / np.sinh(s.t_lim * np.arcsinh(bb))
            if disp(F9, p0, bb, ww) > s.t9:
                lo_b = bb
            else:
                hi_b = bb
        return bb, ww

    def diam(pr_, bb, ww, Fv):
        dd = disp(Fv, pr_, bb, ww)
        return 2 * r_all[np.where(dd >= thr)[0].max()] / SC if (dd >= thr).any() else 0.0

    best = None
    for cs2 in np.linspace(0.55, 1.0, 10):
        tt = np.clip((r_all - 1.5) / 4.0, 0, 1)
        mr = np.maximum.accumulate(r_all / cs2 * (1 - tt) + r_all * tt)
        pr_raw = np.interp(mr, r_all, prof_raw)
        for ws_ in np.linspace(0.02, 1.0, 25):
            pr_ = make_prof(pr_raw, ws_)
            bb, ww = fit_curve(pr_)
            cost = (diam(pr_, bb, ww, F10) - s.d10) ** 2 + 0.5 * (diam(pr_, bb, ww, F7) - s.d7) ** 2
            if best is None or cost < best[0]:
                best = (cost, cs2, ws_, bb, ww, pr_)
    _, cs2, ws_, bb, ww, pr_ = best
    out = {"core_scale": float(cs2), "wing_scale": float(ws_), "b": float(bb), "w": float(ww), "prof": pr_,
           "d10": float(diam(pr_, bb, ww, F10)), "d7": float(diam(pr_, bb, ww, F7))}
    _STAR_FIT_CACHE[key] = out
    return out


def draw_stars(X, Y, G, Fs, prof, r_all, H, W, dev, scale: int = 1):
    """Linear star layer (3, H, W) on the GPU: radial profile stamps, radius by magnitude group. ``scale``: the grid is
    the full resolution reduced by this factor (a draft): the stamps' radii and the profile's steps follow."""
    st = torch.zeros(3, H * W, device=dev)
    prof_t = torch.from_numpy(prof.astype(np.float32)).to(dev)
    n = len(prof)
    step = 0.25 if scale == 1 else 0.25 / scale
    for gmin, gmax, RS in ((-5, 11.0, 60), (11.0, 13.5, 24), (13.5, 30, 9)):
        if scale != 1:
            RS = max(2, int(np.ceil(RS / scale)))
        sel = np.where((G >= gmin) & (G < gmax))[0]
        if not len(sel):
            continue
        oy, ox = np.mgrid[-RS:RS + 1, -RS:RS + 1]
        oyt = torch.from_numpy(oy).to(dev)
        oxt = torch.from_numpy(ox).to(dev)
        chunk = max(1, 4_000_000 // (2 * RS + 1) ** 2)
        for k in range(0, len(sel), chunk):
            ss = sel[k:k + chunk]
            xi = torch.from_numpy(np.round(X[ss]).astype(np.int64)).to(dev)
            yi = torch.from_numpy(np.round(Y[ss]).astype(np.int64)).to(dev)
            fx = torch.from_numpy(X[ss].astype(np.float32)).to(dev)
            fy = torch.from_numpy(Y[ss].astype(np.float32)).to(dev)
            px = xi[:, None, None] + oxt[None]
            py = yi[:, None, None] + oyt[None]
            rr = torch.sqrt((px - fx[:, None, None]) ** 2 + (py - fy[:, None, None]) ** 2)
            stamp = prof_t[(rr / step).long().clamp(max=n - 1)] * (rr < RS).float()
            ok = ((py >= 0) & (py < H) & (px >= 0) & (px < W)).float()
            idx = (py.clamp(0, H - 1) * W + px.clamp(0, W - 1)).reshape(-1)
            for c in range(3):
                fc = torch.from_numpy(Fs[c, ss].astype(np.float32)).to(dev)
                st[c].index_add_(0, idx, (fc[:, None, None] * stamp * ok).reshape(-1))
    return st.view(3, H, W)


def box_profile(prof: np.ndarray, r_all: np.ndarray, g: int) -> np.ndarray:
    """A star's radial profile averaged over a pixel of a grid reduced by g (the drafts' stars): at each radius of
    r_all, the mean of the profile over a g x g square centred there, so that a reduced pixel holds the mean light of
    the full pixels it covers."""
    m = 2 * g
    u = ((np.arange(m) + 0.5) / m - 0.5) * g
    d = np.hypot(r_all[:, None, None] + u[None, :, None], u[None, None, :])
    return np.interp(d, r_all, prof, right=0.0).mean((1, 2))
