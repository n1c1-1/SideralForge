"""The Classic stacking method: debayer every frame, put it on the reference
grid, normalise it, reject outliers pixel by pixel, average what is kept. It is how Siril (its OSC_Preprocessing
script) and PixInsight (WBPP without drizzle) stack colour frames, and the method core/coverage.py advises when the
frames are too few or too little dithered for a Bayer drizzle (the weight map varies by more than 20 %, DrizzlePac's
rule).

The steps, each a published method (sources in the modules named):

1. **Debayering by RCD** (core/debayer.py: librtprocess's rcd_demosaic as Siril runs it; bilinear as an option),
   after Siril's cosmetic fill of the bad samples and with the saturated samples at the saturation level; the frame
   white-balanced before RCD and the balance taken off after, as raw converters do (``cfa_balance``; without it RCD
   moved green flux into the weak blue channel of the stars: +46 % on heart, an orange cast after the colour stage).
2. **Resampling onto the reference grid by Lanczos-4 with Siril's clamping** (core/resample.py; PixInsight's clamping
   or none as options), through the registration polynomial the drizzle uses. Output at sensor resolution: pixel
   (j, i) is the reference sensor position (j, i), the drizzle's grid at scale 1.
3. **Normalisation** (the drizzle's: stages/stack.py ``frame_norm``, an additive sky difference to the reference, a
   quadratic per colour, and the frame's transparency as its scale): every colour c of a frame becomes
   (value - Q_c(X, Y)) / scale_c at its reference position (X, Y) = the output pixel.
4. **Rejection by the number of frames, PixInsight's WBPP rule** (core/rejection_classic.py): percentile 0.2 / 0.1
   for 3 to 5 frames, winsorized sigma clipping 3 / 3 for 6 to 15, linear fit clipping 5 / 3.5 from 16, the
   generalized ESD test as an option above 50; per frame, colour and pixel.
5. **The weighted mean** of the kept values, with the frame weights of the drizzle (one per frame), as Siril's
   ``mean_and_reject``; and the two half-stacks A and B (frames dealt by blocks of ``half_block``, as the drizzle),
   with the same rejection, for the noise measure.

Outputs (``ClassicResult``), the drizzle's files and keys: ``full``, ``A``, ``B`` (3, H, W) float32, 0 where no frame
covers; ``weight`` (3, H, W), the summed weight of the kept values (drz_weight.npy); the rejection maps (low, high
counts of rejected values, the three colours together, on the grid reduced by a whole factor until its longer side
is at most 4096: rejection.npz as stack.RejectionMap writes it); the rejection rate in % of the values tested (next to
Siril's 0.1-0.5 % mark [S12]) and per colour; per frame its rejected values; the noise of the result from the two
halves (stack.halves_noise's formula); the saturation shares (saturation.npy, as stack.py). With ``masks=True``, also
what was rejected in each frame, per pixel and colour (``masks``): the rejection data of PixInsight's drizzle
integration, which stages/stack.py applies to a Bayer drizzle (``drizzle_rejection = integration``).

How stages/stack.py calls it (for the integration agent; ``mm``, ``fpn``, ``dev``, ``reg``, ``info``, ``satval``,
``ch_of_k``, ``H``, ``W``, ``ctx``, ``p`` are stack.py's names; the frame selection, the normalisation and the frame
weights are computed as for the drizzle, before)::

    from ..core import stack_classic as sc

    def load(i):                         # the calibrated CFA frame (H, W): NaN = bad pixel, > VALID_MAX = saturated
        fr = torch.from_numpy(np.asarray(mm[i], np.float32)).to(dev)
        return fr if fpn is None else torch.where(fr > VALID_MAX, fr, fr - fpn)

    res = sc.stack_classic(
        load, reg["coef"],
        bad=None,                        # the bad pixels are the NaN of the cache; or calibrate's badmask (H, W) bool
        norm=sc.Norm.from_drizzle(norm, transp),   # stack.py's frame_norm dict {i: (Q (3, 6), sky)} and transp (n,)
        weights=weight,                  # the frame weights (n,)
        params=classic_params(p),        # stack.py's: the Classic settings of the stack (white balance included)
        device=dev, used=used, ch_of_k=ch_of_k, H=H, W=W, sat_level=satval,
        progress=lambda f, msg: ctx.progress(f, msg))
    for name, a in res.arrays().items():  # drz_full, drz_A, drz_B, drz_weight, saturation (.npy)
        ctx.save(name, a)
    res.save_rejection(ctx.out(REJECTION))
    ctx.metric("rejected_pct", res.rejected_pct); ctx.metric("grid", [W, H])
    ctx.save_json("drz_meta.json", res.meta())
    diagnostics.json: per frame ``res.frames`` (index, half, rejected_low, rejected_high, tested, bad_filled), at the
    top ``"noise": res.noise``, ``"half_weights": res.half_weights``, ``"rejection": {**res.choice.as_dict(),
    "file": REJECTION, "factor": res.factor, "grid": [W, H], "shape": list(res.rejection_low.shape),
    "rate_rgb": res.rejected_pct_rgb}``; ctx.log(res.choice.reason).

``used`` are the frame indices to stack (weight > 0); ``weights`` and ``norm`` are indexed by frame (all n frames).
Memory: the resampled frames are kept on the CPU (``store_dtype`` float32: 12 bytes per pixel and frame, 4.4 GB for
30 frames of 12 Mpx; float16 halves it; their saturation flags 1 byte more, ``masks`` 1 more), the rejection runs on
the device by bands of ``band_pixels``, holding about REJECTION_BYTES per value (frame x band pixel): 0.3 GB per
frame for a band of a million pixels. Given ``memory`` (core/devices.memory_budget), the stack stays within it, its
result unchanged to the bit (issue #125): the resampled frames go to a file in ``spill`` when they do not fit (out of
core: a memory-mapped array, its file deleted when closed), and on the CPU each band is cut into pieces of a multiple
of PIECE_ALIGN pixels that do (``_parts``).

With Siril's own recipe (global IKSS normalisation, no weights, winsorized 3 / 3) it gives Siril's stack of the same
frames: the same star FWHM, roundness and background noise, within a few hundredths. Time: about 1 s per 12 Mpx
frame on a mid-range GPU for debayering and resampling, then the rejection and the means.

Sources: see core/debayer.py, core/resample.py, core/rejection_classic.py; [S7] Siril's OSC_Preprocessing.ssf;
[S12] https://siril.org/tutorials/tuto-manual/ ; [P15] WBPP.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import debayer as db
from . import rejection_classic as rc
from . import resample as rs

#: values above this in the calibrated cache are saturated (stages/calibrate.VALID_MAX; not imported: a core module)
VALID_MAX = 60000.0
#: stack.py's NEAR_SATURATION, REJECTION_SIDE
NEAR_SATURATION = 0.85
REJECTION_SIDE = 4096
#: RCD reads up to 4 samples away: a near-saturated sample marks the demosaiced pixels this far
SAT_SPREAD = 2
#: the memory the rejection of a band holds on the CPU per value (a frame's sample of a band pixel): the values, their
#: float64 copies, their sorted order and the masks of the rounds; measured 270-280 bytes with the linear fit clipping
#: of 16 to 64 frames, the most of the four methods
REJECTION_BYTES = 320
#: per pixel, what pass 1 holds besides the resampled frames (the frame being debayered by RCD, resampled by
#: Lanczos-4 and normalised: measured 410 bytes on the CPU) and what pass 2 adds (the means, halves, weights,
#: saturation shares and counts: 64 bytes)
FRAME_BYTES = 420
OUTPUT_BYTES = 64
#: the pieces a band is cut into hold a multiple of this many pixels: the CPU sums a pixel's frames in an order set by
#: its place in blocks of 32 (AVX2) or 64 (AVX-512) pixels of the band, which the pieces keep, so that the result
#: stays the same to the bit (tests/test_stack_memory.py)
PIECE_ALIGN = 256
#: what the frames' file leaves free on its disk: without that much room, the frames stay in memory, as before
DISK_MARGIN = 2 * 2 ** 30


@dataclass
class ClassicParams:
    """The settings of the Classic method, each with its published default (the modules give the sources)."""
    debayer: str = "rcd"                 # debayer.METHODS
    cfa_balance: bool = True             # white balance before RCD, as raw converters do (debayer.py); False: as Siril
    interpolation: str = "lanczos4"      # resample.METHODS
    clamping: str = "siril"              # resample.CLAMPINGS
    clamp_threshold: float = rs.PCL_CLAMPING_THRESHOLD
    rejection: str = "auto"              # "auto" (WBPP's rule) or rejection_classic.METHODS
    low: float | None = None             # None: the method's published thresholds
    high: float | None = None
    gesd_above_50: bool = False
    half_block: int = 20
    store_dtype: str = "float32"         # "float16" halves the memory of the resampled frames
    band_pixels: int = 1 << 20           # pixels per rejection band on the device


#: ``masks``: bit c (0-2) = rejected low in colour c (R, G, B), bit HIGH_SHIFT + c = rejected high (PixInsight's
#: rejection map keeps the low and high rejections apart too; any bit excludes the sample from the drizzle)
HIGH_SHIFT = 3


@dataclass
class Norm:
    """What a frame's colour c is mapped by: normalised = (value - Q[i, c] . terms(X, Y)) / scale[i, c], the terms of
    stack.py's ``quad_terms`` (``_quad``)."""
    scale: np.ndarray                    # (n, 3)
    Q: np.ndarray                        # (n, 3, 6), the frame's ADU

    @classmethod
    def of(cls, norm: Norm | None, n: int) -> Norm:
        """``norm``, or no normalisation (None) for n frames."""
        return cls(np.ones((n, 3)), np.zeros((n, 3, 6))) if norm is None else norm

    @classmethod
    def from_drizzle(cls, frame_norm: dict, transp, n: int | None = None) -> Norm:
        """From stack.py's version-1 normalisation: ``frame_norm`` {i: (Q (3, 6), sky (3,))} and ``transp`` (n,)."""
        t = np.asarray(transp, float)
        n = len(t) if n is None else n
        Q = np.zeros((n, 3, 6))
        for i, (q, _) in frame_norm.items():
            Q[int(i)] = q
        return cls(np.repeat(t[:, None], 3, 1), Q)


@dataclass
class ClassicResult:
    full: np.ndarray
    A: np.ndarray
    B: np.ndarray
    weight: np.ndarray
    saturation: np.ndarray
    rejection_low: np.ndarray
    rejection_high: np.ndarray
    factor: int
    rejected_pct: float
    rejected_pct_rgb: list
    tested: int
    choice: rc.Choice
    frames: list
    noise: dict
    half_weights: list
    params: ClassicParams
    seconds: dict = field(default_factory=dict)
    #: with ``masks=True``: (frames used, H, W) uint8, what was rejected in each frame (bits: HIGH_SHIFT)
    masks: np.ndarray | None = None

    def arrays(self) -> dict[str, np.ndarray]:
        """The drizzle's file names -> arrays."""
        return {"drz_full.npy": self.full, "drz_A.npy": self.A, "drz_B.npy": self.B, "drz_weight.npy": self.weight,
                "saturation.npy": self.saturation}

    def save_rejection(self, path) -> None:
        """rejection.npz, as stack.RejectionMap.save writes it."""
        H, W = self.full.shape[1:]
        np.savez_compressed(path, low=self.rejection_low, high=self.rejection_high, factor=self.factor,
                            grid=np.array([W, H]))

    def meta(self) -> dict:
        """drz_meta.json's content (scale 1: the sensor grid)."""
        H, W = self.full.shape[1:]
        p = self.params
        return {"method": "classic", "scale": 1.0, "pixfrac": None, "n": len(self.frames),
                "reject_pct": self.rejected_pct, "width": W, "height": H, "sensor": [W, H],
                "debayer": p.debayer, "cfa_balance": p.cfa_balance, "interpolation": p.interpolation,
                "clamping": p.clamping,
                "rejection": self.choice.as_dict()}

    def diagnostics(self) -> dict:
        H, W = self.full.shape[1:]
        return {"frames": self.frames, "noise": self.noise, "half_weights": self.half_weights,
                "rejection": {**self.choice.as_dict(), "file": "rejection.npz", "factor": self.factor,
                              "grid": [W, H], "shape": list(self.rejection_low.shape),
                              "rate_pct": self.rejected_pct, "rate_rgb_pct": self.rejected_pct_rgb,
                              "tested": self.tested},
                "seconds": self.seconds}

    def save(self, folder) -> None:
        """Every output in ``folder`` (the drizzle's names, plus classic.json: meta and diagnostics)."""
        d = Path(folder)
        d.mkdir(parents=True, exist_ok=True)
        for name, a in self.arrays().items():
            np.save(d / name, a)
        self.save_rejection(d / "rejection.npz")
        (d / "classic.json").write_text(json.dumps({"meta": self.meta(), "diagnostics": self.diagnostics()},
                                                   indent=1, default=float), encoding="utf-8")


def _quad(Q: np.ndarray, X: torch.Tensor, Y: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """Q (6,) . the terms of stack.py's ``quad_terms`` at the grid X (Wo,) x Y (Ho,) -> (Ho, Wo)."""
    a = (X / (W / 2) - 1.0)[None, :]
    b = (Y / (H / 2) - 1.0)[:, None]
    q = [float(v) for v in Q]
    return q[0] + q[1] * a + q[2] * b + q[3] * a * a + q[4] * a * b + q[5] * b * b


def halves_noise(A: np.ndarray, B: np.ndarray, w_a: float, w_b: float) -> dict[str, float | None]:
    """stack.halves_noise (not imported: a core module): robust sigma of A - B x sqrt(w_a w_b) / (w_a + w_b) per
    colour; None where it cannot be measured."""
    out: dict[str, float | None] = {}
    wa, wb = float(w_a), float(w_b)
    f = math.sqrt(wa * wb) / (wa + wb) if wa > 0 and wb > 0 else float("nan")
    for c, name in enumerate("RGB"):
        a, b = np.asarray(A[c], np.float64), np.asarray(B[c], np.float64)
        d = (a - b)[(a != 0) & (b != 0) & np.isfinite(a) & np.isfinite(b)]
        v = float(1.4826 * np.median(np.abs(d - np.median(d))) * f) if d.size else float("nan")
        out[name] = v if np.isfinite(v) else None
    return out


def _reduce(count: np.ndarray, f: int) -> np.ndarray:
    H, W = count.shape
    Hr, Wr = -(-H // f), -(-W // f)
    pad = np.zeros((Hr * f, Wr * f), np.int64)
    pad[:H, :W] = count
    r = pad.reshape(Hr, f, Wr, f).sum((1, 3))
    return r.astype(np.uint16 if r.max(initial=0) < 65536 else np.uint32)


@dataclass
class MemoryPlan:
    """How the stack stays within the memory it is given (issue #125): the resampled frames on disk (``on_disk``),
    the bands of the rejection cut into pieces of ``piece`` pixels (the band's size when they need not be), and the
    plain line that says so ('' when nothing changes)."""
    on_disk: bool
    piece: int
    note: str = ""

    @classmethod
    def of(cls, memory: tuple[int, int] | None, nu: int, H: int, W: int, itemsize: int, masks: bool, rows: int, dev,
           disk: int) -> MemoryPlan:
        """``disk``: the bytes free where the frames would go (0: nowhere; they then stay in memory, as before)."""
        band = rows * W
        if memory is None:
            return cls(False, band)
        budget, available = memory
        cpu = torch.device(dev).type == "cpu"            # on a card, pass 1's frame and the rejection are in its memory
        frames = nu * H * W * (3 * itemsize + 1 + (1 if masks else 0))     # the resampled frames, flags and masks
        can_spill = frames + DISK_MARGIN <= disk
        on_disk = can_spill and frames + ((FRAME_BYTES if cpu else 0) + OUTPUT_BYTES) * H * W > budget
        room = budget - OUTPUT_BYTES * H * W - (0 if on_disk else frames)
        piece = band
        if cpu and nu * band * REJECTION_BYTES > room:
            # what the pieces may hold; the bound only cuts on the CPU, whose sums it is checked against (a card's
            # reductions are not)
            piece = max(PIECE_ALIGN, room // (nu * REJECTION_BYTES) // PIECE_ALIGN * PIECE_ALIGN)
            if not on_disk and can_spill and room < nu * PIECE_ALIGN * REJECTION_BYTES:
                on_disk = True                                       # no room left for pieces: the frames go too
                room = budget - OUTPUT_BYTES * H * W
                piece = max(PIECE_ALIGN, room // (nu * REJECTION_BYTES) // PIECE_ALIGN * PIECE_ALIGN)
            piece = min(piece, band)
        said = []
        if on_disk:
            said.append(f"the {nu} registered frames ({frames / 2 ** 30:.1f} GB) are kept on disk")
        if piece < band:
            said.append(f"the rejection is computed in {len(_parts(H, W, rows, piece))} parts")
        note = f"memory: {available / 2 ** 30:.1f} GB available, " + " and ".join(said) if said else ""
        return cls(on_disk, piece, note)


def _parts(H: int, W: int, rows: int, piece: int) -> list[tuple[int, int]]:
    """The runs of pixels (first, end), flat in the (H, W) frame, the rejection works on: the bands of ``rows`` rows,
    each cut from its start into pieces of ``piece`` pixels, the last one of a band taking what is left; with ``piece``
    a multiple of PIECE_ALIGN, every pixel keeps its place in the blocks the CPU sums by, and the result its bits."""
    out = []
    for r0 in range(0, H, rows):
        a, b = r0 * W, min(H, r0 + rows) * W
        out.extend((o, min(b, o + piece)) for o in range(a, b, piece))
    return out


def _array(shape: tuple, dtype, on_disk: bool, folder, zero: bool = True) -> np.ndarray:
    """An array of the stack: in memory (zeroed or not), or memory-mapped on a file of ``folder`` that the system
    deletes once the array and its views are gone (a temporary file: deleted on close on Windows, unlinked at once
    elsewhere), the pages written out to it as the memory needs them."""
    if not on_disk:
        return np.zeros(shape, dtype) if zero else np.empty(shape, dtype)
    import tempfile
    return np.memmap(tempfile.TemporaryFile(dir=folder, prefix="classic_", suffix=".tmp"), dtype, mode="w+",
                     shape=shape)


def stack_classic(load: Callable, coef, bad=None, norm: Norm | None = None, weights=None,
                  params: ClassicParams | None = None, device="cpu", *, used, ch_of_k, H: int, W: int, sat_level=None,
                  progress: Callable | None = None, masks: bool = False, raw_template: str | None = None,
                  colour: bool = False, memory: tuple[int, int] | None = None, spill=None,
                  log: Callable | None = None) -> ClassicResult:
    """Stack the frames ``used`` by the Classic method (see the module's docstring for what stack.py passes).

    ``load(i)``: frame i, the calibrated CFA (H, W) (torch or numpy; NaN = bad, > VALID_MAX = saturated); ``coef``
    (n, 2, 10) the registration polynomials; ``bad``: an extra bad-pixel mask (H, W) or None; ``norm``: the
    normalisation (``Norm``; None: none); ``weights``: (n,) frame weights (None: equal); ``sat_level``: the
    saturation level (a float or a map (H, W)), put on the saturated samples (None: VALID_MAX); ``masks``: keep what
    was rejected in each frame (``ClassicResult.masks``); ``raw_template``: a raw file of the session, through which
    an X-Trans frame is demosaiced (core/debayer.markesteijn; None: bilinear); ``colour``: frames already in colour
    (issue #57), ``load(i)`` gives their three planes (3, H, W): no debayering, the bad samples filled plane by
    plane; everything from the resampling on is the same (the three channels of a debayered frame). ``memory``:
    (bytes the stack may fill, bytes available), core/devices.memory_budget; None: no bound. ``spill``: the folder
    where the resampled frames go when they do not fit in it (None, or a disk without the room: kept in memory
    whatever their size); ``log``:
    where the plain line saying so goes (issue #125)."""
    p = params or ClassicParams()
    dev = torch.device(device)
    progress = progress or (lambda f, msg: None)
    t0 = time.time()
    used = [int(i) for i in np.asarray(used).ravel()]
    nu = len(used)
    if nu < 1:
        raise ValueError("no frame to stack")
    n_all = max(used) + 1 if weights is None else len(np.asarray(weights))
    n_all = max(n_all, len(np.asarray(coef)))
    Wt = np.ones(n_all) if weights is None else np.asarray(weights, float)
    if Wt.ndim != 1:
        raise ValueError("weights: one per frame (n,)")
    nm = Norm.of(norm, n_all)
    unmeasured = [i for i in used if not (np.isfinite(nm.scale[i]).all() and np.isfinite(nm.Q[i]).all()
                                          and (np.asarray(nm.scale[i]) > 0).all())]
    if unmeasured:
        raise ValueError(f"no normalisation for frame(s) {unmeasured}: leave them out of ``used``")
    choice = rc.choose(nu, p.rejection, p.low, p.high, p.gesd_above_50)
    in_a = np.array([(i // p.half_block) % 2 == 0 for i in used])
    wu = torch.as_tensor(Wt[used], dtype=torch.float32, device=dev)            # (nu,)
    sat_t = None
    if sat_level is not None:
        sat_t = sat_level.to(dev, torch.float32) if isinstance(sat_level, torch.Tensor) else \
            torch.as_tensor(np.asarray(sat_level, np.float32), device=dev)
    dtype = np.float16 if p.store_dtype == "float16" else np.float32
    rows = max(1, p.band_pixels // W)
    disk = shutil.disk_usage(spill).free if spill is not None and memory is not None else 0
    plan = MemoryPlan.of(memory, nu, H, W, np.dtype(dtype).itemsize, masks, rows, dev, disk)
    if plan.note and log is not None:
        log(plan.note)
    store = _array((nu, 3, H, W), dtype, plan.on_disk, spill, zero=False)
    flags = _array((nu, H, W), np.uint8, plan.on_disk, spill)     # bit 0: near saturation, bit 1: saturated
    Xg = torch.arange(W, device=dev, dtype=torch.float32)
    Yg = torch.arange(H, device=dev, dtype=torch.float32)
    frames = []
    k_spread = torch.ones(1, 1, 2 * SAT_SPREAD + 1, 2 * SAT_SPREAD + 1, device=dev)
    bad_t = None if bad is None else torch.as_tensor(np.asarray(bad), device=dev).bool()
    t_load = t_deb = t_res = 0.0

    # ---------------- pass 1: every frame debayered, on the reference grid, normalised
    for k, i in enumerate(used):
        ta = time.time()
        fr = load(i)
        fr = fr.to(dev, torch.float32) if isinstance(fr, torch.Tensor) else \
            torch.from_numpy(np.asarray(fr, np.float32)).to(dev)
        tb = time.time()
        sat = fr > VALID_MAX
        satv = sat_t if sat_t is not None else torch.tensor(VALID_MAX, device=dev)
        near = sat | (fr > NEAR_SATURATION * satv)
        fr = torch.where(sat, satv.expand_as(fr) if satv.dim() else satv, fr)
        badm = torch.isnan(fr)
        if bad_t is not None:
            badm = badm | bad_t
        bal = None
        if colour:                                       # already in colour (issue #57): no demosaicing
            planes = [db.fill_bad(fr[c], badm[c]) for c in range(3)]
            rgb = torch.stack([a for a, _ in planes])
            filled = torch.stack([m for _, m in planes])
            near, sat = near.any(0), sat.any(0)          # one flag per pixel, as for a mosaic
        else:
            fr, filled = db.fill_bad(fr, badm, ch_of_k=ch_of_k)
            bal = db.cfa_balance(fr, ch_of_k) if p.cfa_balance and p.debayer == "rcd" else None
            rgb = db.debayer(fr, ch_of_k, p.debayer, balance=bal, template=raw_template)
        del fr
        tc = time.time()
        x, y = rs.source_positions(coef[i], H, W, W, H, dev)
        out = rs.resample(rgb, x, y, method=p.interpolation, clamping=p.clamping, threshold=p.clamp_threshold)
        del rgb
        fl = (F.conv2d(near.float()[None, None], k_spread, padding=SAT_SPREAD)[0, 0] > 0).to(torch.uint8) | \
            ((F.conv2d(sat.float()[None, None], k_spread, padding=SAT_SPREAD)[0, 0] > 0).to(torch.uint8) << 1)
        xi, yi = torch.round(x).long(), torch.round(y).long()
        inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        fo = fl[yi.clamp(0, H - 1), xi.clamp(0, W - 1)]
        fo = torch.where(inside, fo, torch.zeros_like(fo))
        del x, y, xi, yi, inside, fl, near, sat
        for c in range(3):
            bg = _quad(nm.Q[i, c], Xg, Yg, W, H)
            out[c] = (out[c] - bg) / float(nm.scale[i, c])
        store[k] = out.cpu().numpy().astype(dtype, copy=False)
        flags[k] = fo.cpu().numpy()
        del out, fo
        td = time.time()
        t_load += tb - ta
        t_deb += tc - tb
        t_res += td - tc
        frames.append({"index": i, "half": "A" if in_a[k] else "B", "scale": [float(v) for v in nm.scale[i]],
                       "offset": [float(v) for v in nm.Q[i, :, 0]], "bad_filled": int(filled.sum()),
                       "cfa_balance": None if bal is None else [round(float(v), 5) for v in bal]})
        progress(0.6 * (k + 1) / nu, f"classic: {k + 1}/{nu} frames debayered and resampled")
    t1 = time.time()

    # ---------------- pass 2: rejection and the weighted means, by bands of rows
    full = np.zeros((3, H, W), np.float32)
    A = np.zeros_like(full)
    B = np.zeros_like(full)
    wsum = np.zeros_like(full)
    sat_share = np.zeros((2, H, W), np.float32)
    low_cnt = np.zeros((H, W), np.int32)
    rej_masks = _array((nu, H, W), np.uint8, plan.on_disk, spill) if masks else None
    high_cnt = np.zeros((H, W), np.int32)
    rej_lo_f = np.zeros(nu, np.int64)
    rej_hi_f = np.zeros(nu, np.int64)
    tested_f = np.zeros(nu, np.int64)
    rej_c = np.zeros(3, np.int64)
    tested_c = np.zeros(3, np.int64)
    ina = torch.as_tensor(in_a, device=dev)
    # every array seen flat per frame and colour: a band of rows, or a piece of it, is a run of pixels (o0, o1)
    full_f, A_f, B_f, wsum_f = (a.reshape(3, H * W) for a in (full, A, B, wsum))
    sat_f, low_f, high_f = sat_share.reshape(2, H * W), low_cnt.reshape(H * W), high_cnt.reshape(H * W)
    store_f, flags_f = store.reshape(nu, 3, H * W), flags.reshape(nu, H * W)
    masks_f = None if rej_masks is None else rej_masks.reshape(nu, H * W)
    for o0, o1 in _parts(H, W, rows, plan.piece):
        P = o1 - o0
        fb = torch.from_numpy(np.ascontiguousarray(flags_f[:, o0:o1])).to(dev)
        near_b, clip_b = (fb & 1) > 0, (fb & 2) > 0
        shares = torch.zeros((2, P), device=dev)
        lo_b = torch.zeros(P, dtype=torch.int32, device=dev)
        hi_b = torch.zeros_like(lo_b)
        for c in range(3):
            v = torch.from_numpy(np.ascontiguousarray(store_f[:, c, o0:o1])).to(dev).float()
            valid = torch.isfinite(v)
            lo, hi = rc.reject(v, choice, valid)
            keep = valid & ~lo & ~hi
            w = wu
            mean, den, kept = rc.combine(v, keep, w)
            mean = torch.where(valid.any(0), mean, torch.zeros_like(mean))
            wk = torch.where(keep, w[:, None].expand_as(v), torch.zeros_like(v))
            x0 = torch.where(keep, v, torch.zeros_like(v))
            half = []
            for sel in (ina, ~ina):
                ws = wk * sel[:, None]
                num, d = (ws * x0).sum(0), ws.sum(0)
                half.append(torch.where(d > 0, num / torch.where(d > 0, d, torch.ones_like(d)), torch.zeros_like(d)))
            for j, m in enumerate((near_b, clip_b)):
                sh = (wk * m).sum(0) / torch.where(den > 0, den, torch.ones_like(den))
                shares[j] = torch.maximum(shares[j], torch.where(den > 0, sh, torch.zeros_like(sh)))
            full_f[c, o0:o1] = mean.cpu().numpy()
            A_f[c, o0:o1] = half[0].cpu().numpy()
            B_f[c, o0:o1] = half[1].cpu().numpy()
            wsum_f[c, o0:o1] = den.cpu().numpy()
            if masks_f is not None:
                bits = (lo.to(torch.uint8) << c) | (hi.to(torch.uint8) << (HIGH_SHIFT + c))
                masks_f[:, o0:o1] |= bits.cpu().numpy()
            lo_b += lo.sum(0, dtype=torch.int32)
            hi_b += hi.sum(0, dtype=torch.int32)
            rej_lo_f += lo.sum(1).cpu().numpy()
            rej_hi_f += hi.sum(1).cpu().numpy()
            nv = valid.sum(1).cpu().numpy()
            tested_f += nv
            rej_c[c] += int((lo | hi).sum())
            tested_c[c] += int(nv.sum())
            del v, valid, lo, hi, keep, mean, den, kept, wk, x0, half
        low_f[o0:o1] = lo_b.cpu().numpy()
        high_f[o0:o1] = hi_b.cpu().numpy()
        sat_f[:, o0:o1] = shares.cpu().numpy()
        progress(0.6 + 0.4 * o1 / (H * W), f"classic: rejection ({choice.method}) {o1 // W}/{H} rows")
    del store, flags, store_f, flags_f, masks_f              # the frames' file, if any, is deleted with its last view
    t2 = time.time()

    f = max(1, -(-max(H, W) // REJECTION_SIDE))
    tested = int(tested_c.sum())
    rate = 100.0 * float(rej_c.sum()) / max(tested, 1)
    rate_rgb = [round(100.0 * float(rej_c[c]) / max(int(tested_c[c]), 1), 4) for c in range(3)]
    for k in range(nu):
        frames[k].update({"rejected_low": int(rej_lo_f[k]), "rejected_high": int(rej_hi_f[k]),
                          "tested": int(tested_f[k])})
    w_a = float(sum(Wt[i] for k, i in enumerate(used) if in_a[k]))
    w_b = float(sum(Wt[i] for k, i in enumerate(used) if not in_a[k]))
    step = max(1, int(np.sqrt(H * W / 4e6)))                 # stack.noise_step
    noise = halves_noise(A[:, ::step, ::step], B[:, ::step, ::step], w_a, w_b)
    sat_u8 = np.ceil(255 * np.clip(sat_share, 0, 1)).astype(np.uint8)
    return ClassicResult(full, A, B, wsum, sat_u8, _reduce(low_cnt, f), _reduce(high_cnt, f), f, round(rate, 4),
                         rate_rgb, tested, choice, frames, noise, [w_a, w_b], p,
                         {"load": round(t_load, 2), "debayer": round(t_deb, 2), "resample_normalise": round(t_res, 2),
                          "pass1": round(t1 - t0, 2), "rejection_mean": round(t2 - t1, 2),
                          "total": round(time.time() - t0, 2)}, rej_masks)
