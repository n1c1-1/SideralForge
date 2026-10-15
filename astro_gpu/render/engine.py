"""Interactive rendering engine (GPU).

Works from the outputs of the ``render_prep`` stage. ``render(settings, scale)`` returns the nebula layer, the
star layer and their screen blend as display-referred sRGB tensors in [0, 1].

* ``scale = 1``: full resolution (exports, 100 % view) - identical to the reference implementation.
* ``scale < 1``: fast preview; the nebula is processed on a reduced image with every spatial parameter scaled,
  the stars are always drawn at full resolution and then reduced, so star sizes and brightness match the export.

The nebula is the calibrated image at full resolution (star footprints guarded), mixed live with the optional layers
of ``render_prep`` (fine detail, fine colour, OIII, reflection tones), so these are ordinary sliders. Intermediate
results are cached by the settings they depend on: moving a star slider only redraws the stars, changing the
saturation only redoes the last colour step.
"""
from __future__ import annotations

import json
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
                           "sky_level", "dark_level", "toe", "sky_pct", "dark_pct", "white_mult", "hdr_strength",
                           "hdr_knee", "hdr_r", "hdr_eps", "lc_fine", "lc_medium", "lc_large", "neb_sharp",
                           "sharp_white", "hl_knee", "hl_max", "hl_detail", "hl_dmin", "hl_r1", "hl_r2", "hl_eps",
                           "hl_shoulder", "hl_keep_chroma", "bright_halo_desat")
STAR_KEYS = ("stars", "g_glow", "g_lim", "t_lim", "d10", "d7", "star_sat", "sat_min", "core_white", "t9",
             "white_temp", "tint", "real_stars", "star_glow", "glare_nebula", "diffuse_cut", "spike_photo",
             "spike_g_max", "spike_smooth", "stars_photographed")


@dataclass
class Layers:
    final: torch.Tensor
    starless: torch.Tensor
    stars: torch.Tensor
    scale: float
    info: dict


class RenderEngine:
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
        self._cache: dict[str, tuple] = {}
        self.last_info: dict = {}

    # ------------------------------------------------------------------ helpers
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
            q1, q2 = ops.percentile(Eg0, [s.fine_lo_pct, s.fine_hi_pct], shown)
            wf = ops.gaussian_blur(((Eg0 - q1) / max(q2 - q1, 1e-6)).clamp(0, 1), 6 / f)
            del Eg0
            N = N + kf * (keep * wf * self._layer("dY", f))[None] * self.w_cam[:, None, None]
            del wf
        if sharpen != 0:                     # measured-shape sharpening (brightness), above a brightness of the signal
            Eg0 = self._layer("Eg0", f)
            q1, q2 = ops.percentile(Eg0, [s.sharpen_lo_pct, s.fine_hi_pct], shown)
            ws = ops.gaussian_blur(((Eg0 - q1) / max(q2 - q1, 1e-6)).clamp(0, 1), 6 / f)
            del Eg0
            N = N + sharpen * (keep * ws * self._layer("dS", f))[None] * self.w_cam[:, None, None]
            del ws
        if s.faint_smooth > 0:               # faint areas: only the scales confirmed by the data
            Eg = ops.gaussian_blur(torch.einsum("c,chw->hw", self.lum, N), 8 / f)
            q1, q2 = ops.percentile(Eg, [s.faint_lo_pct, s.faint_hi_pct], shown)
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

    def _cached(self, name, keys, s, extra, fn):
        k = (tuple(getattr(s, n) for n in keys), extra)
        if name in self._cache and self._cache[name][0] == k:
            return self._cache[name][1]
        v = fn()
        self._cache[name] = (k, v)
        return v

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def render(self, s: RenderSettings, scale: float = 1.0) -> Layers:
        f = max(1, int(round(1 / scale)))
        info: dict = {}
        neb = self._cached("neb", NEB_KEYS, s, f, lambda: self._nebula(s, f, info))
        if s.stars:
            st = self._cached("stars", STAR_KEYS, s, f, lambda: self._stars(s, f, info))
        else:
            st = torch.zeros_like(neb)
        final = ops.screen(neb, st)
        valid = self._valid(f)
        final[:, ~valid] = 0
        info.update(self.last_info)
        self.last_info = info
        return Layers(final, neb, st, 1.0 / f, info)

    # ------------------------------------------------------------------ nebula
    def _nebula_linear(self, s: RenderSettings, f: int):
        valid = self._valid(f)
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
        if s.star_glow > 0 and "Shalo" in self._host:
            # the glare of the optics around the stars, as photographed: under the nebula curve, like the residual
            # rays of the brightest stars that stay in the nebula (with the stars, the rays stood out of a dark sky)
            # it is star light: the saturation of the stars, not of the nebula (a saturated blue disc around Alnitak)
            g = self._glare(s, f)
            if self.layered:
                yg = torch.einsum("c,chw->hw", self.lum, g)[None] * self.w_cam[:, None, None]
                g = yg + s.star_sat * (g - yg)
            astro = astro + s.star_glow * s.glare_nebula * g
        if s.speck_k > 0 and f == 1:
            for c in range(3):
                a_ = astro[c]
                sg_ = ops.robust_noise(a_, sub=7)
                astro[c] = torch.maximum(a_, ops.median_filter(a_, 5) - s.speck_k * sg_)
        lin = torch.einsum("ij,jhw->ihw", self.M_eff(s), astro)
        if s.sky_neutral > 0:
            # background neutralisation: the typical sky (sky_pct, smoothed) made grey at the same brightness, by an
            # offset per channel - the faint emission that fills a field stops tinting it, the bright stays coloured
            sky = torch.tensor([float(ops.percentile(ops.gaussian_blur(lin[c], 8 / f), s.sky_pct, self._shown(f))) for c in range(3)],
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
        sky = valid & (Eg0 < ops.percentile(Eg0, 50, valid))
        o0 = ops.percentile(O, 50, sky)
        sig = 1.4826 * ops.percentile((O - o0).abs(), 50, sky)
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
        lin = self._cached("neb_lin", NEB_LIN_KEYS, s, f, lambda: self._nebula_linear(s, f))
        valid, shown = self._valid(f), self._shown(f)
        Yl_full = ops.luminance(lin)
        noise = ops.robust_noise(Yl_full, shown, sub=max(1, 7 // f))
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
        Yl = ops.gaussian_blur(Yn, 6 / f)
        p_dark, p_sky = ops.percentile(Yl, [s.dark_pct, s.sky_pct], shown)
        hi = ops.percentile(Yn, 99.995, shown)
        del Yn
        lo, b, white = fit_tone_curve(p_dark, p_sky, hi * s.white_mult, s.dark_level / 255, s.sky_level / 255, s.toe)
        info["tone_curve"] = {"black": lo, "b": b, "white": white}
        L = tone_curve(Yl_full, lo, b, white, s.toe)
        if s.hdr_strength > 0:
            base = ops.guided(L, L, max(1, round(s.hdr_r / f)), s.hdr_eps)
            over = (base - s.hdr_knee).clamp(min=0)
            L = L - s.hdr_strength * over * (L / base.clamp(min=1e-4)).clamp(0, 2)
        out, prev = L.clone(), L
        for sig, gain in zip((2, 8, 32), (s.lc_fine, s.lc_medium, s.lc_large)):
            bl = ops.gaussian_blur(L, sig / f)
            out += (gain - 1) * (prev - bl)
            prev = bl
        L = out.clamp(min=0)
        if s.neb_sharp > 0:
            Eg1 = ops.gaussian_blur(Yl_full, 8 / f)
            q1, q2 = ops.percentile(Eg1, [85, 97], shown)
            wsh = ops.gaussian_blur(((Eg1 - q1) / max(q2 - q1, 1e-6)).clamp(0, 1), 6 / f)
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
        neb_d = ops.chroma_ok(neb_d, s.chroma, s.chroma_protect)
        if s.bright_halo_desat > 0:
            neb_d = self._halo_desat(neb_d, s, f)
        neb_d[:, ~valid] = 0
        return neb_d

    def _highlights(self, L, s, f, shown, info):
        L = L.clamp(min=0)
        knee, top = s.hl_knee, s.hl_max
        B = 0.5 * ops.guided(L, L, max(1, round(s.hl_r1 / f)), s.hl_eps) + \
            0.5 * ops.guided(L, L, max(1, round(s.hl_r2 / f)), s.hl_eps)
        D = L - B
        Bmax = ops.percentile(B, 99.995, shown)
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
    def _stars(self, s: RenderSettings, f: int, info: dict) -> torch.Tensor:
        x0, x1, y0, y1 = self.crop
        G = self.G
        draw = (G < s.g_glow) & (self.X > x0 - 60) & (self.X < x1 + 60) & (self.Y > y0 - 60) & (self.Y < y1 + 60)
        Fd, Gd, pr = self.Fm[:, draw], G[draw], self.Fpred[:, draw]
        wch = (np.clip((15.5 - Gd) / 3.0, 0, 1) * np.clip((Gd - 8.5) / 1.5, 0, 1))[None]
        chrom = wch * Fd / np.maximum(Fd.sum(0), 1e-6)[None] + (1 - wch) * pr / np.maximum(pr.sum(0), 1e-6)[None]
        Fs = self.M_eff(s).cpu().numpy().astype(np.float64) @ (chrom * np.clip(Fd.sum(0), 0, None)[None])
        Lum = 0.2126 * Fs[0] + 0.7152 * Fs[1] + 0.0722 * Fs[2]
        s_g = s.star_sat * np.clip((s.g_lim + 1.0 - Gd) / 3.0, s.sat_min, 1.0)
        Fs = Lum[None] + s_g[None] * (Fs - Lum[None])
        fit = fit_star_curve(self.r_all, self.prof_raw, Gd, Lum, s, self.SC)
        # the display curve is fitted on every star; the stars shown by their real light are not drawn again
        use_real = self.Sreal is not None and s.real_stars
        dr = ~self.real[draw] if use_real else np.ones(int(draw.sum()), bool)
        info["stars"] = {k: v for k, v in fit.items() if k != "prof"}
        info["stars"]["drawn"] = int(dr.sum())
        st_lin = draw_stars(self.X[draw][dr], self.Y[draw][dr], Gd[dr], Fs[:, dr], fit["prof"], self.r_all, self.H,
                            self.W, self.dev)
        if use_real:
            sr = torch.tensordot(self.M_eff(s), self.Sreal.to(self.dev).float(), 1)
            wsp = self._spike_w(s, 1)
            if wsp is not None:             # each star's own spikes, as photographed
                raw = self.Ssp_raw.to(self.dev).float()
                sp = raw + s.spike_smooth * (self.Ssp.to(self.dev).float() - raw)
                sr += torch.tensordot(self.M_eff(s), wsp[None] * sp, 1)
                del raw, sp
            if self.Sshape is not None:
                # a drawn star's measured shape is part of its light beyond the core: same attenuation as its wings
                # (off where its spikes are shown as photographed)
                sh_ = self.Sshape.to(self.dev).float()
                if wsp is not None:
                    sh_ = sh_ * (1 - wsp * self._layer("Wsl", 1))[None]
                sr += fit["wing_scale"] * torch.tensordot(self.M_eff(s), sh_, 1)
                del sh_
            lr = ops.luminance(sr)[None]
            st_lin += (lr + s.star_sat * (sr - lr)).clamp(min=0)
            info["stars"]["real_light"] = int((~dr).sum())
            del sr, lr
        if self.Sall is not None and s.stars_photographed > 0:
            sa = torch.tensordot(self.M_eff(s), self.Sall.to(self.dev).float(), 1)
            la = ops.luminance(sa)[None]
            st_lin = (1 - s.stars_photographed) * st_lin + s.stars_photographed * (la + s.star_sat * (sa - la)).clamp(min=0)
            del sa, la
        if s.star_glow > 0 and s.glare_nebula < 1 and "Shalo" in self._host:
            # the rest of the glare with the stars, under their gentler curve (fades out close to the star)
            sh = torch.einsum("ij,jhw->ihw", self.M_eff(s), self._glare(s, 1))
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
        st_d[:, ~self.valid] = 0
        if f > 1:
            h, w = self.H // f * f, self.W // f * f
            st_d = F.avg_pool2d(st_d[None, :, :h, :w], f)[0]
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

    def render(self, s: RenderSettings, scale: float = 1.0) -> Layers:
        return self.select(s).render(s, scale)

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
    """Black point lo and strength b such that the sky percentile maps to t_sky and the dark one to t_dark."""
    best = None
    for lo in np.linspace(p_dark - 3 * (p_sky - p_dark), p_dark - 0.01 * (p_sky - p_dark), 200):
        a_, b_ = 1.0, 1e7
        bm = 1.0
        for _ in range(60):
            bm = np.sqrt(a_ * b_)
            if tone_curve(p_sky, lo, bm, white, toe) < t_sky:
                a_ = bm
            else:
                b_ = bm
        err = abs(tone_curve(p_dark, lo, bm, white, toe) - t_dark)
        if best is None or err < best[0]:
            best = (err, float(lo), float(bm))
    return best[1], best[2], float(white)


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


def draw_stars(X, Y, G, Fs, prof, r_all, H, W, dev):
    """Linear star layer (3, H, W) on the GPU: radial profile stamps, radius by magnitude group."""
    st = torch.zeros(3, H * W, device=dev)
    prof_t = torch.from_numpy(prof.astype(np.float32)).to(dev)
    n = len(prof)
    for gmin, gmax, RS in ((-5, 11.0, 60), (11.0, 13.5, 24), (13.5, 30, 9)):
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
            stamp = prof_t[(rr / 0.25).long().clamp(max=n - 1)] * (rr < RS).float()
            ok = ((py >= 0) & (py < H) & (px >= 0) & (px < W)).float()
            idx = (py.clamp(0, H - 1) * W + px.clamp(0, W - 1)).reshape(-1)
            for c in range(3):
                fc = torch.from_numpy(Fs[c, ss].astype(np.float32)).to(dev)
                st[c].index_add_(0, idx, (fc[:, None, None] * stamp * ok).reshape(-1))
    return st.view(3, H, W)
