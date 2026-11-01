"""The Hubble rejection (astro_gpu/core/rejection_hubble.py) on synthetic Bayer-drizzle stacks: known cosmic rays,
satellite trails, hot and dead pixels, stars that must survive a sub-pixel misregistration and a varying seeing, and
clean data whose rejection rate must match the false-positive rate of the noise model."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from astro_gpu.core import rejection_hubble as rh

#: RGGB: the colour of each sensor pixel by (y % 2, x % 2)
CFA = torch.tensor([[0, 1], [1, 2]])
GAIN, RN, SKY = 0.5, 3.0, 400.0          # e-/ADU, ADU, ADU: sigma of the sky about 28 ADU


@dataclass
class Frame:
    val: torch.Tensor        # normalised samples (ADU of the reference)
    Xo: torch.Tensor         # where the pipeline believes each sample lands (with its registration error)
    Yo: torch.Tensor
    grad: torch.Tensor       # the frame's additive sky difference (frame ADU)
    t: float                 # its photometric scale
    excess: torch.Tensor     # what the defects added (normalised units)
    sigma: torch.Tensor      # the true noise of each sample (normalised units)
    star: torch.Tensor       # the stars' signal (normalised units)


def _pix(d: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """A Gaussian of sigma ``s`` integrated over a pixel at offset ``d``."""
    k = 1 / (s * math.sqrt(2))
    return 0.5 * (torch.erf((d + 0.5) * k) - torch.erf((d - 0.5) * k))


def make_stack(n=40, H=96, W=96, n_stars=25, seed=0, misreg=0.0, fwhm_jitter=0.0, rot_deg=0.0, dither=3.0,
               flat=False, defects=None):
    """``n`` frames of a Bayer sensor H x W looking at a sky of stars (FWHM 2.5 px) on a coloured background, each
    dithered (continuous offsets up to ``dither`` px), rotated up to ``rot_deg``, with its own transparency and sky
    gradient, Gaussian noise of the sensor model; the registration handed to the rejection is off by ``misreg`` px rms
    per frame and the seeing varies by ``fwhm_jitter`` (relative rms). ``defects(i, frame_dict)`` adds defects."""
    g = torch.Generator().manual_seed(seed)

    def U(a, b, *shape):
        return a + (b - a) * torch.rand(*shape, generator=g, dtype=torch.float64)

    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    chmap = CFA[yy.long() % 2, xx.long() % 2]
    bkg = torch.tensor([1.0, 1.0, 0.8], dtype=torch.float64) * SKY
    sx, sy = U(6, W - 6, n_stars), U(6, H - 6, n_stars)
    flux = torch.exp(U(math.log(500), math.log(60000), n_stars))
    col = U(0.5, 1.5, n_stars, 3)
    s0 = 2.5 / 2.355
    frames = []
    for i in range(n):
        dx, dy = U(-dither, dither, 1).item(), U(-dither, dither, 1).item()
        th = math.radians(rot_deg) * i / max(n - 1, 1)
        cx, cy = W / 2, H / 2
        X = math.cos(th) * (xx - cx) - math.sin(th) * (yy - cy) + cx + dx
        Y = math.sin(th) * (xx - cx) + math.cos(th) * (yy - cy) + cy + dy
        s = s0 * (1 + fwhm_jitter * torch.randn(1, generator=g, dtype=torch.float64).item())
        star = torch.zeros(H, W, dtype=torch.float64)
        if not flat:
            for k in range(n_stars):
                near = ((X - sx[k]).abs() < 8) & ((Y - sy[k]).abs() < 8)
                star[near] += flux[k] * col[k][chmap[near]] * _pix(X[near] - sx[k], s) * _pix(Y[near] - sy[k], s)
        t = U(0.85, 1.0, 1).item()
        grad = U(-30, 30, 1).item() + U(-20, 20, 1).item() * xx / W
        model = bkg[chmap] + star
        raw = t * model + grad
        sig_raw = (raw / GAIN + RN ** 2).sqrt()
        obs = raw + sig_raw * torch.randn(H, W, generator=g, dtype=torch.float64)
        add = torch.zeros(H, W, dtype=torch.float64)
        if defects is not None:
            defects(i, {"add": add, "sig_raw": sig_raw, "gen": g, "obs": obs, "H": H, "W": W})
        obs = obs + add
        ex, ey = (misreg * torch.randn(2, generator=g, dtype=torch.float64)).tolist()
        frames.append(Frame(val=((obs - grad) / t).float(), Xo=(X + ex).float(), Yo=(Y + ey).float(),
                            grad=grad.float(), t=t, excess=(add / t).float(), sigma=(sig_raw / t).float(),
                            star=star.float()))
    return frames, chmap


def run(frames, chmap, params=rh.HUBBLE, diagnostics=False, H=96, W=96):
    """The integration as the module's docstring says it: store, prediction, driz_cr on every frame, tally."""
    store = rh.new_store(len(frames), H, W)
    for k, f in enumerate(frames):
        rh.add_to_store(store, k, f.val, f.Xo, f.Yo, chmap, torch.ones_like(f.val, dtype=torch.bool), 1.0, 1.0)
    pred = rh.build_prediction(store, [f.t for f in frames], None, GAIN, RN, params=params)
    tally = rh.RejectionTally(H, W)
    out = []
    for f in frames:
        r = rh.driz_cr(f.val, f.Xo, f.Yo, chmap, pred, f.t, f.grad, GAIN, RN, diagnostics=diagnostics)
        tally.add(f.Xo, f.Yo, r)
        out.append(r)
    return pred, out, tally


def _phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


def expected_fp(spread: float, snr=(3.5, 3.0)) -> float:
    """The false-positive rate of driz_cr on pure noise without the slope term, for residuals of rms ``spread``
    (noise units): two-sided beyond SNR1, plus a neighbour (8 of them) of such a sample beyond SNR2."""
    p1 = 2 * _phi(-snr[0] / spread)
    p2 = 2 * _phi(-snr[1] / spread)
    return p1 + 8 * p1 * p2


# ------------------------------------------------------------------------------------------------ the store, the median
def test_one_sample_per_point_per_frame_and_colour():
    """Scale 1, drop 1, integer offset: a frame covers exactly 1/4 of the red points, 1/2 of the green, 1/4 of the
    blue, each with its own sample; rotated, never two samples of one colour on one point (no collision lost)."""
    H = W = 32
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    chmap = CFA[yy.long() % 2, xx.long() % 2]
    val = torch.arange(H * W, dtype=torch.float32).view(H, W)
    keep = torch.ones(H, W, dtype=torch.bool)
    st = rh.new_store(2, H, W)
    rh.add_to_store(st, 0, val, xx + 3, yy + 2, chmap, keep, 1.0, 1.0)
    cov = torch.isfinite(st[0].float()).float().mean((1, 2))
    inner = torch.isfinite(st[0, :, 2:32, 3:31].float()).float().mean((1, 2))      # 30 x 28: whole 2 x 2 cells
    assert torch.allclose(inner, torch.tensor([0.25, 0.5, 0.25]))
    assert float(cov.sum()) < 1.0
    assert float(st[0, 0, 2 + 4, 3 + 6]) == float(val[4, 6])          # the red sample at (4, 6), moved by (3, 2)
    th = math.radians(30)                                              # rotated: as many points as samples per colour
    Xr = math.cos(th) * (xx - 16) - math.sin(th) * (yy - 16) + 64
    Yr = math.sin(th) * (xx - 16) + math.cos(th) * (yy - 16) + 64
    st = rh.new_store(1, 128, 128)
    rh.add_to_store(st, 0, val, Xr, Yr, chmap, keep, 1.0, 1.0)
    per_colour = torch.isfinite(st[0].float()).sum((1, 2))
    expected = torch.bincount(chmap.view(-1), minlength=3)
    assert ((per_colour - expected).abs() <= 0.03 * expected).all()   # a drop covers one centre on average


def test_median_minmed_and_nhigh_follow_astrodrizzle():
    """6+ samples: the median (two middle values averaged when even); 3 to 5: minmed, whose median leaves out the
    highest and which takes the minimum when the median is more than 4 sigma of the summed exposures above it."""
    K, H, W = 6, 4, 4
    st = rh.new_store(K, H, W)
    st[:, :, :, :] = torch.tensor([10, 11, 12, 13, 14, 100.0]).view(K, 1, 1, 1)
    pr = rh.build_prediction(st, np.ones(K), None, gain=1.0, read_noise=0.0)
    assert torch.allclose(pr.pred, torch.full((3, H, W), 12.5))              # median of 6, outlier ignored
    assert int(pr.count[0, 0, 0]) == 6 and pr.summary["red"]["prediction"] == "median"
    pr = rh.build_prediction(st, np.ones(K), None, 1.0, 0.0, params=rh.HubbleParams(nhigh=1))
    assert torch.allclose(pr.pred, torch.full((3, H, W), 12.0))              # nhigh = 1: median of the 5 lowest
    # 3 samples, two of them hit (a coincidence minmed is made for): the median leaves out the highest, (40 + 500) / 2
    # = 270; sigma of the sum sqrt(3 x 270 + 3 x 25) ~ 30; (270 - 40) x 3 >> 4 x 30: the minimum
    st = rh.new_store(3, H, W)
    st[:, :, :, :] = torch.tensor([40.0, 500.0, 540.0]).view(3, 1, 1, 1)
    pr = rh.build_prediction(st, np.ones(3), None, gain=1.0, read_noise=5.0)
    assert torch.allclose(pr.pred, torch.full((3, H, W), 40.0)) and pr.summary["red"]["prediction"] == "minmed"
    assert pr.summary["red"]["minimum_used_pct"] == 100.0
    st[:, :, :, :] = torch.tensor([40.0, 44.0, 900.0]).view(3, 1, 1, 1)       # one hit: median of the 2 lowest
    pr = rh.build_prediction(st, np.ones(3), None, gain=1.0, read_noise=5.0)
    assert torch.allclose(pr.pred, torch.full((3, H, W), 42.0))
    # combine_grow: next to a point where the minimum won at 4 sigma, it wins from 3 sigma on
    # samples (64, 100, 101): median 82, (82 - 64) x 3 / sqrt(3 x 82) = 3.4 sigma; (40, 100, 101): 6.2 sigma
    st = rh.new_store(3, 1, 5)
    v = torch.tensor([64.0, 100.0, 101.0]).view(3, 1, 1, 1).expand(3, 3, 1, 5).clone()
    v[0, :, 0, 2] = 40.0
    st[:] = v
    pr = rh.build_prediction(st, np.ones(3), None, gain=1.0, read_noise=0.0)
    assert pr.pred[0, 0].tolist() == [82.0, 64.0, 40.0, 64.0, 82.0]


# ------------------------------------------------------------------------------------------------ clean data
def test_clean_sky_rejects_at_the_false_positive_rate_of_the_noise_model():
    """Flat sky, 40 frames. Against the exact sky the residuals have unit spread in the sensor model's sigmas (the model
    is right, with the frames' scales and sky differences) and the rate is the Gaussian tail beyond 3.5 sigma, plus the
    neighbours beyond 3.0. Against the median, with the published slope term, the rate is the model's expectation for
    the thresholds it applied (SNR1 + scale1 x deriv / noise: the slope term also absorbs the median's own noise)."""
    frames, chmap = make_stack(n=40, flat=True, seed=1)
    H = W = 96
    sky = (torch.tensor([1.0, 1.0, 0.8]) * SKY).view(3, 1, 1).expand(3, H, W).clone()
    exact = rh.Prediction(sky, torch.full((3, H, W), 40, dtype=torch.uint8), rh.HUBBLE)
    tally = rh.RejectionTally(H, W)
    zs = []
    for f in frames:
        r = rh.driz_cr(f.val, f.Xo, f.Yo, chmap, exact, f.t, f.grad, GAIN, RN, diagnostics=True)
        tally.add(f.Xo, f.Yo, r)
        zs.append(r.z[r.tested])
    z = torch.cat(zs)
    rate, exp = tally.summary()["rejected_pct"] / 100, expected_fp(1.0)
    print(f"\nclean flat sky against the exact sky: residual std {float(z.std()):.4f}, rate {100 * rate:.4f} % for "
          f"{100 * exp:.4f} % expected ({tally.summary()['samples_on_stack']} samples)")
    assert abs(float(z.std()) - 1) < 0.02
    assert 0.75 * exp < rate < 1.3 * exp
    _, res, tally = run(frames, chmap, diagnostics=True)
    s = tally.summary()
    lim = torch.cat([r.limit[r.tested] for r in res]).double()
    exp_m = float((2 * 0.5 * torch.erfc(lim / math.sqrt(2))).mean())
    print(f"clean flat sky against the median, published defaults: rate {s['rejected_pct']:.4f} % (low "
          f"{s['low_pct']:.4f}, high {s['high_pct']:.4f}) for {100 * exp_m:.4f} % expected with the thresholds "
          f"applied (median threshold {float(lim.median()):.2f} sigma); z spread {s['z_spread_median']:.3f}")
    assert s["rejected_pct"] / 100 < exp
    assert 0.5 * exp_m < s["rejected_pct"] / 100 < 3 * exp_m
    _, _, t0 = run(frames, chmap, params=rh.HubbleParams(scale=(0.0, 0.0)))
    print(f"clean flat sky against the median, no slope term: rate {t0.summary()['rejected_pct']:.4f} % (minmed points "
          f"biased low: AstroDrizzle's warning)")


def test_stars_survive_misregistration_and_seeing():
    """Stars with a 0.15 px rms registration error per frame, a 10 % seeing jitter and a 2 degree field rotation over
    the night: the samples on the stars are rejected no more often than the sky (the slope term at work), and the
    whole clean stack stays near the noise model's false-positive rate."""
    frames, chmap = make_stack(n=40, seed=2, misreg=0.15, fwhm_jitter=0.10, rot_deg=2.0)
    pred, res, tally = run(frames, chmap)
    on_star = [(f.star > 5 * f.sigma) & r.tested for f, r in zip(frames, res)]
    n_star = sum(int(m.sum()) for m in on_star)
    rej_star = sum(int(((r.high | r.low) & m).sum()) for r, m in zip(res, on_star))
    s = tally.summary()
    exp = expected_fp(1.0)
    print(f"\nstars, misregistration 0.15 px, seeing 10 %: {rej_star} of {n_star} star samples rejected "
          f"({100 * rej_star / n_star:.4f} %); whole stack {s['rejected_pct']:.4f} % (expected FP {100 * exp:.4f} %)")
    print(pred.describe())
    assert n_star > 3000
    assert rej_star / n_star < 2 * exp
    assert s["rejected_pct"] / 100 < 1.5 * exp
    # without the slope term the same stars are eaten: the term is what protects them
    _, res0, _ = run(frames, chmap, params=rh.HubbleParams(scale=(0.0, 0.0)))
    rej0 = sum(int(((r.high | r.low) & m).sum()) for r, m in zip(res0, on_star))
    print(f"without the slope term: {rej0} star samples rejected ({100 * rej0 / n_star:.3f} %)")
    assert rej0 > 10 * max(rej_star, 1)


# ------------------------------------------------------------------------------------------------ defects
def _defects(seed=3, n_cr=30, trail_frame=5, hot=None, dead=None):
    """Cosmic rays (1 to 4 pixels, 8 to 100 sigma) in every frame, a satellite trail (Gaussian profile, sigma
    0.8 px, peak 15 sigma) across one frame, fixed hot pixels (30 sigma) and dead pixels (zero) in all frames."""
    def add(i, d):
        g, H, W, a, sig = d["gen"], d["H"], d["W"], d["add"], d["sig_raw"]
        for _ in range(n_cr):
            y, x = int(torch.randint(2, H - 3, (1,), generator=g)), int(torch.randint(2, W - 3, (1,), generator=g))
            amp = float(8 + 92 * torch.rand(1, generator=g))
            for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)][: int(torch.randint(1, 5, (1,), generator=g))]:
                a[y + dy, x + dx] += amp * sig[y + dy, x + dx]
        if i == trail_frame:
            yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64),
                                    indexing="ij")
            dist = (0.35 * xx - yy + 30) / math.hypot(0.35, 1)
            a += 15 * sig * torch.exp(-0.5 * (dist / 0.8) ** 2)
        if hot is not None:
            a[hot] += 30 * sig[hot]
        if dead is not None:
            a[dead] -= d["obs"][dead]
    return add


def point_count(pred, f, chmap):
    """The number of samples behind the point each sample of frame ``f`` lands on (its colour's), 0 outside."""
    C, H, W = pred.count.shape
    jx, iy = torch.floor(f.Xo + 0.5).long(), torch.floor(f.Yo + 0.5).long()
    ok = (jx >= 0) & (jx < W) & (iy >= 0) & (iy < H)
    out = torch.zeros(f.val.shape, dtype=torch.long)
    out[ok] = pred.count.view(-1)[chmap[ok] * H * W + iy[ok] * W + jx[ok]].long()
    return out


def test_cosmic_rays_trails_hot_and_dead_pixels_are_rejected():
    """40 frames with cosmic rays in each, a satellite trail across one, and hot and dead pixels left unmasked by the
    calibration. Off the stars every outlier well above the threshold goes, the trail's wings with the 3.0-sigma
    neighbour test; a trail crossing a bright star is kept there (driz_cr's slope term protects star cores). A dead
    pixel is a low outlier: where minmed predicts (3 to 5 samples) and the dead value is the minimum, minmed takes it as
    the prediction, and the hole's slope hides the dead samples around it; with the median everywhere all go.
    AstroDrizzle relies on the bad-pixel masks for those (DQ); our calibration flags cold pixels (stack.py: ``bad``)."""
    g = torch.Generator().manual_seed(9)
    hot = (torch.randint(4, 92, (20,), generator=g), torch.randint(4, 92, (20,), generator=g))
    dead = (torch.randint(4, 92, (10,), generator=g), torch.randint(4, 92, (10,), generator=g))
    frames, chmap = make_stack(n=40, seed=4, misreg=0.1, fwhm_jitter=0.05,
                               defects=_defects(hot=hot, dead=dead))
    pred, res, tally = run(frames, chmap)
    hotm = torch.zeros(96, 96, dtype=torch.bool)
    hotm[hot] = True
    deadm = torch.zeros_like(hotm)
    deadm[dead] = True
    clear = ~hotm & ~deadm
    regime = [point_count(pred, f, chmap) for f in frames]

    def share(sel):
        """(rejected, tested) among the samples ``sel`` (a mask per frame, None: not counted)."""
        n = sum(int((m & r.tested).sum()) for m, r in zip(sel, res) if m is not None)
        k = sum(int((m & r.tested & (r.high | r.low)).sum()) for m, r in zip(sel, res) if m is not None)
        return k, n

    def frame5(m):
        return [None] * 5 + [m] + [None] * (len(frames) - 6)

    f5 = frames[5]
    off_star = [f.star < f.sigma for f in frames]
    k_cr, n_cr = share([(f.excess > 6 * f.sigma) & clear & o if i != 5 else None
                        for i, (f, o) in enumerate(zip(frames, off_star))])
    k_crs, n_crs = share([(f.excess > 6 * f.sigma) & clear & ~o if i != 5 else None
                          for i, (f, o) in enumerate(zip(frames, off_star))])
    k_tr, n_tr = share(frame5((f5.excess > 6 * f5.sigma) & clear & off_star[5]))
    k_trs, n_trs = share(frame5((f5.excess > 6 * f5.sigma) & clear & ~off_star[5]))
    k_wi, n_wi = share(frame5((f5.excess > 3 * f5.sigma) & (f5.excess <= 5 * f5.sigma) & clear & off_star[5]))
    flux = f5.excess.clamp(min=0) * res[5].tested
    removed = float((flux * res[5].high).sum() / flux.sum())
    k_hot, n_hot = share([hotm] * len(frames))
    k_dm, n_dm = share([deadm & (c > rh.HUBBLE.minmed_max) for c in regime])
    k_dmm, n_dmm = share([deadm & (c <= rh.HUBBLE.minmed_max) for c in regime])
    low_dead = sum(int((deadm & r.low).sum()) for r in res)
    k_st, n_st = share([(f.star > 5 * f.sigma) & (f.excess.abs() < 0.5 * f.sigma) & clear for f in frames])
    s = tally.summary()
    print(f"\ncosmic rays > 6 sigma: {k_cr}/{n_cr} off stars, {k_crs}/{n_crs} on stars; trail > 6 sigma: {k_tr}/{n_tr} "
          f"off stars, {k_trs}/{n_trs} on a star; its 3-5 sigma edges {k_wi}/{n_wi}; {100 * removed:.1f} % of its flux "
          f"removed; hot pixels {k_hot}/{n_hot}; dead pixels {k_dm}/{n_dm} at median points, {k_dmm}/{n_dmm} at minmed "
          f"points ({low_dead} low in all); clean star samples rejected {k_st}/{n_st}; stack {s['rejected_pct']:.3f} % "
          f"(low {s['low_pct']:.3f}, high {s['high_pct']:.3f})")
    assert k_cr / n_cr > 0.998
    assert k_tr == n_tr and removed > 0.85
    assert k_wi / n_wi > 0.4                                    # the 3.0-sigma neighbour test takes the edges
    assert k_hot / n_hot > 0.98
    assert (k_dm + k_dmm) / (n_dm + n_dmm) > 0.75
    assert k_st / n_st < 0.002
    _, res_m, _ = run(frames, chmap, params=rh.HubbleParams(minmed_max=2))      # the median everywhere
    k_m = sum(int((deadm & r.tested & (r.high | r.low)).sum()) for r in res_m)
    n_m = sum(int((deadm & r.tested).sum()) for r in res_m)
    print(f"dead pixels with the median everywhere: {k_m}/{n_m}")
    assert k_m == n_m


# ------------------------------------------------------------------------------------------------ few samples
def test_few_frames_use_minmed_and_too_few_are_reported_not_rejected():
    """14 frames: about 3.5 red and blue samples per point (minmed), 7 green (median); a cosmic ray is still caught.
    6 frames: 1.5 red samples per point: those samples are not tested, and it is said."""
    frames, chmap = make_stack(n=14, seed=5, defects=_defects(n_cr=10, trail_frame=-1))
    pred, res, tally = run(frames, chmap)
    sm = pred.summary
    assert sm["red"]["prediction"] == "minmed" and sm["blue"]["prediction"] == "minmed"
    assert sm["green"]["prediction"] == "median"
    cr = [(f.excess > 8 * f.sigma) & r.tested for f, r in zip(frames, res)]
    k = sum(int((m & r.high).sum()) for m, r in zip(cr, res))
    n = sum(int(m.sum()) for m in cr)
    print(f"\n14 frames: {pred.describe()}; cosmic rays > 8 sigma {k}/{n}; stack {tally.summary()}")
    assert k / n > 0.98
    frames, chmap = make_stack(n=6, seed=6, defects=_defects(n_cr=10, trail_frame=-1))
    pred, res, tally = run(frames, chmap)
    s = tally.summary()
    print(f"6 frames: {pred.describe()}; untested {s['untested_pct']:.1f} % of the samples")
    assert pred.summary["red"]["too_few_pct"] > 20 and "not tested" in pred.describe()
    assert s["untested_pct"] > 20
    for r in res:
        assert not bool(((r.high | r.low) & r.untested).any()) and not bool((r.tested & r.untested).any())


def test_tally_file_has_the_keys_of_the_stack_rejection_map(tmp_path):
    frames, chmap = make_stack(n=12, seed=7, defects=_defects(n_cr=5, trail_frame=-1))
    _, res, tally = run(frames, chmap)
    tally.save(tmp_path / "rejection.npz")
    d = np.load(tmp_path / "rejection.npz")
    assert {"low", "high", "untested", "factor", "grid"} <= set(d.files)
    assert int(d["high"].sum()) == tally.totals["high"] and int(d["low"].sum()) == tally.totals["low"]
    assert list(d["grid"]) == [96, 96] and int(d["factor"]) == 1


def test_dilation_matches_the_box_filters_of_driz_cr():
    """The sparse dilation equals a 3 x 3 max filter (driz_cr's neighbour test, nothing outside the image), and the
    grow box of driz_cr_grow = k sits where scipy's convolve2d(mode='same') puts it."""
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(0)
    m = torch.rand(40, 50, generator=g) > 0.97
    m[0, 0] = m[-1, -1] = True
    ref = F.max_pool2d(m.float()[None, None], 3, 1, 1)[0, 0] > 0
    assert torch.equal(rh._dilate(m, 1), ref)
    one = torch.zeros(9, 9, dtype=torch.bool)
    one[4, 4] = True
    got = rh._dilate(one, 1, 0)                         # k = 2: the window [i - 1, i] holds the pixel for i = 4, 5
    assert torch.nonzero(got).tolist() == [[4, 4], [4, 5], [5, 4], [5, 5]]
