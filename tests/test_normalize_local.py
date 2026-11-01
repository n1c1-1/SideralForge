"""Local normalisation per colour (astro_gpu/core/normalize_local.py; stacking v2, docs/spec/STACKING.md point 5).

Synthetic CFA frames (RGGB) of one sky: a red nebula, stars of every colour, each frame with its own registration
(shifts, small turns), seeing, additive background and transparency per colour:
- the transparency follows Bouguer's law with the airmass rising through the night, blue absorbed most
  (k = 0.10, 0.17, 0.30 mag per airmass), and one frame under a grey cloud;
- the background: an offset per frame and colour, a light-pollution gradient common to all, and a moon gradient that
  grows and turns through the night (bluish), up to about 9 sigma from edge to edge.
After normalisation every frame must match the reference within its noise; the truth is known analytically, so the
comparisons need no interpolation.
"""
import math

import numpy as np
import pytest
import torch

from astro_gpu.core import normalize_local as nl

H, W = 512, 768
CH = [0, 1, 1, 2]                                   # RGGB: colour of k = 2 * row + column
GAIN, RN = 2.0, 5.0
SKY = np.array([420.0, 610.0, 530.0])
NEB = np.array([60.0, 18.0, 12.0])                  # a red (H-alpha) nebula
K_EXT = np.array([0.10, 0.17, 0.30])                # extinction, mag per airmass: blue absorbed most
MOON = np.array([0.55, 0.75, 1.0])                  # moonlight per colour
LP = np.array([12.0, 15.0, 10.0])                   # light pollution gradient common to all frames
N = 12
PAIR = (1, 10)                                      # two frames shifted by even pixels, same seeing, no turn


def coef_of(dx, dy, th):
    """Registration polynomial (register.py's form): a turn by th about the centre and a shift (dx, dy)."""
    c = np.zeros((2, 10))
    c[0, :3] = [W / 2 + dx, math.cos(th) * W / 2, -math.sin(th) * H / 2]
    c[1, :3] = [H / 2 + dy, math.sin(th) * W / 2, math.cos(th) * H / 2]
    return c


def smooth_truth(X, Y):
    neb = np.exp(-((X - 0.4 * W) ** 2 + (Y - 0.45 * H) ** 2) / (2 * (0.18 * W) ** 2))
    return SKY.reshape(3, *([1] * np.ndim(X))) + NEB.reshape(3, *([1] * np.ndim(X))) * neb


def bkg_truth(p, i, X, Y):
    """Q (3, ...) of frame i at reference positions: offset + light pollution + moon (frame ADU)."""
    a, b = X / (W / 2) - 1, Y / (H / 2) - 1
    sh = (3, *([1] * np.ndim(X)))
    A, ph = p["amp"][i], p["phi"][i]
    moon = A * (math.cos(ph) * a + math.sin(ph) * b) + 0.4 * A * a * b
    return p["off"][i].reshape(sh) + LP.reshape(sh) * a + MOON.reshape(sh) * moon


def make_set(moon: bool, seed: int):
    rng = np.random.default_rng(seed)
    ns = 260
    stars = np.c_[rng.uniform(8, W - 8, ns), rng.uniform(8, H - 8, ns)]
    flux = 10 ** rng.uniform(4.3, 5.7, ns)
    col = rng.uniform(0.5, 1.5, (ns, 3))
    stars = np.r_[stars, [[0.6 * W, 0.3 * H]]]          # one star far above saturation
    flux = np.r_[flux, 5e6]
    col = np.r_[col, [[1.0, 1.0, 1.0]]]
    airmass = np.linspace(1.15, 2.3, N)
    cloud = np.ones(N)
    cloud[5] = 0.8
    s = 10 ** (-0.4 * K_EXT[None] * (airmass[:, None] - 1)) * cloud[:, None]          # (N, 3) truth
    fwhm = rng.uniform(3.4, 4.8, N)
    fwhm[list(PAIR)] = 4.0
    dx = 2 * rng.integers(-8, 9, N).astype(float)
    dy = 2 * rng.integers(-8, 9, N).astype(float)
    th = np.where(rng.random(N) < 0.5, 0.002, 0.0)
    sub = np.arange(N) % 3 == 0                          # sub-pixel shifts for a third of them
    dx[sub] += 0.37
    dy[sub] -= 0.61
    th[list(PAIR)] = 0.0
    dx[0] = dy[0] = th[0] = 0.0
    p = {"s": s, "fwhm": fwhm, "coef": np.array([coef_of(dx[i], dy[i], th[i]) for i in range(N)]),
         "off": rng.uniform(-40, 40, (N, 3)), "amp": np.linspace(0, 50, N) if moon else np.zeros(N),
         "phi": np.linspace(0.3, 1.3, N), "dx": dx, "dy": dy, "th": th}
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    cmap = np.array(CH)[2 * (yy.astype(int) % 2) + xx.astype(int) % 2]
    frames = []
    for i in range(N):
        X, Y = p["coef"][i, 0, 0] + p["coef"][i, 0, 1] * (xx / (W / 2) - 1) + p["coef"][i, 0, 2] * (yy / (H / 2) - 1), \
            p["coef"][i, 1, 0] + p["coef"][i, 1, 1] * (xx / (W / 2) - 1) + p["coef"][i, 1, 2] * (yy / (H / 2) - 1)
        img = smooth_truth(X, Y)
        sg = fwhm[i] / 2.3548
        rb = int(np.ceil(6 * sg)) + 2
        # the stars' positions in the frame (the inverse of the turn and shift), a box of pixels around each
        ct, st = math.cos(th[i]), math.sin(th[i])
        ux, vy = stars[:, 0] - W / 2 - dx[i], stars[:, 1] - H / 2 - dy[i]
        ix = np.round(ct * ux + st * vy + W / 2).astype(int)
        iy = np.round(-st * ux + ct * vy + H / 2).astype(int)
        o = np.arange(-rb, rb + 1)
        YY = np.broadcast_to(iy[:, None, None] + o[None, :, None], (len(ix), len(o), len(o)))
        XX = np.broadcast_to(ix[:, None, None] + o[None, None, :], YY.shape)
        inside = (YY >= 0) & (YY < H) & (XX >= 0) & (XX < W)
        sid = np.broadcast_to(np.arange(len(ix))[:, None, None], YY.shape)[inside]
        YY, XX = YY[inside], XX[inside]
        g = np.exp(-((X[YY, XX] - stars[sid, 0]) ** 2 + (Y[YY, XX] - stars[sid, 1]) ** 2) / (2 * sg ** 2)) \
            / (2 * math.pi * sg ** 2)
        star = np.bincount(YY * W + XX, flux[sid] * col[sid, cmap[YY, XX]] * g, H * W).reshape(H, W)
        val = s[i][:, None, None] * img + bkg_truth(p, i, X, Y)
        v = np.take_along_axis(val, cmap[None], 0)[0] + s[i][cmap] * star
        fr = v + rng.normal(0, 1, v.shape) * np.sqrt(np.maximum(v, 0) / GAIN + RN ** 2)
        fr[fr > nl.VALID_MAX] = 65000.0                    # the cache's mark of a saturated sample
        frames.append(fr.astype(np.float32))
    p["weight"] = s[:, 1] ** 2 / fwhm ** 2                   # (transparency / noise)^2 x sharpness, as select_frames
    p["frames"] = frames
    return p


@pytest.fixture(scope="module")
def moon():
    p = make_set(True, 1)
    p["res"] = nl.normalize(lambda i: p["frames"][i], p["coef"], p["weight"], CH, H, W,
                            fwhm=float(np.percentile(p["fwhm"], 90)))
    return p


@pytest.fixture(scope="module")
def flat():
    p = make_set(False, 2)
    p["res"] = nl.normalize(lambda i: p["frames"][i].astype(np.float16), p["coef"], p["weight"], CH, H, W,
                            fwhm=float(np.percentile(p["fwhm"], 90)), sat_level=65535.0)
    return p


def test_polynomial_is_registers_and_inverts():
    from astro_gpu.stages.register import apply_poly
    rng = np.random.default_rng(0)
    c = coef_of(13.2, -7.7, 0.01)
    c[:, 3:] = rng.normal(0, 0.5, (2, 7))                    # a cubic part of a few pixels
    x, y = rng.uniform(0, W, 500), rng.uniform(0, H, 500)
    X, Y = nl.poly_apply(c, torch.from_numpy(x), torch.from_numpy(y), W, H)
    Xr, Yr = apply_poly(c, x, y, W, H)
    assert np.allclose(X.numpy(), Xr, atol=1e-9) and np.allclose(Y.numpy(), Yr, atol=1e-9)
    xi, yi = nl.poly_inverse(c, X, Y, W, H)
    assert np.abs(xi.numpy() - x).max() < 1e-6 and np.abs(yi.numpy() - y).max() < 1e-6


def test_ikss_is_sirils():
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(1000, 20, 200_000), rng.uniform(3000, 30000, 6000), np.zeros(1000)]   # sky, stars, zeros
    loc, sc = nl.ikss(x)
    assert abs(loc - 1000) < 0.5                         # the stars clipped, the zeros ignored (Siril's good pixels)
    assert abs(sc / 20 - 1) < 0.02                       # 0.991 x sqrt(BWMV): the sigma of a normal distribution
    assert nl.ikss(np.zeros(10)) == (0.0, 0.0)


def test_scale_is_measured_per_colour_on_the_stars(moon):
    res, s = moon["res"], moon["s"]
    assert res.ok.all()
    kappa = res.scale / s                                # the reference's own scale: the same for every frame
    err = np.array([fn.scale_err for fn in res.frames.values()])
    for c in range(3):
        dev = np.abs(kappa[:, c] / np.median(kappa[:, c]) - 1)
        assert dev.max() < 0.01, (c, dev)
        assert (dev < 4 * err[:, c] / s[:, c] / np.median(kappa[:, c]) + 1e-3).all(), (c, dev, err[:, c])
    # the colour shift through the night (blue absorbed most) is followed: one scale for the three would leave
    # 23 % between blue and red
    true_br = s[:, 2] / s[:, 0]
    got_br = res.scale[:, 2] / res.scale[:, 0]
    assert true_br.max() / true_br.min() > 1.2
    assert np.abs(got_br / true_br / np.median(got_br / true_br) - 1).max() < 0.01
    assert (np.array([fn.stars for fn in res.frames.values()]) >= nl.MIN_STARS).all()


def _grid():
    gu, gv = np.meshgrid(np.linspace(0, W - 1, 9), np.linspace(0, H - 1, 9))
    return gu.ravel(), gv.ravel()


def test_normalised_backgrounds_match_the_reference_within_noise(moon):
    """At every point of the field, every frame's normalised sky (the truth through its normalisation) is the same
    within a fraction of its pixel noise; before normalisation they differ by several sigmas (moon, transparency)."""
    res, p = moon["res"], moon
    X, Y = _grid()
    T = torch.from_numpy(np.stack([X, Y]))
    terms = nl.quad_terms(T[0], T[1], W, H).numpy()                      # (6, 81)
    norm, raw = [], []
    for i in range(N):
        sky = p["s"][i][:, None] * smooth_truth(X, Y) + bkg_truth(p, i, X, Y)          # (3, 81) frame ADU
        norm.append((sky - res.Q[i] @ terms) / res.scale[i][:, None])
        raw.append(sky / np.median(p["s"], 0)[:, None])
    sig = (res.noise / res.scale)[:, :, None]                           # pixel noise, reference units
    z = np.abs(np.array(norm) - np.median(norm, 0)) / sig
    assert z.max() < 0.25, z.max(axis=(1, 2))
    z = np.abs(np.array(raw) - np.median(raw, 0)) / sig
    assert z.max() > 3


def test_two_frames_normalised_as_stack_py_does_differ_by_their_noise_only(moon):
    res, p = moon["res"], moon
    i, j = PAIR
    dev = torch.device("cpu")
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    chmap = torch.tensor(CH)[2 * (yy.long() % 2) + (xx.long() % 2)]
    out = []
    for k in (i, j):
        fr = torch.from_numpy(p["frames"][k]).to(dev)
        X, Y = nl.poly_apply(p["coef"][k], xx, yy, W, H)
        grad = nl.background_at(res.Q[k], X, Y, chmap, W, H)
        t = nl.scale_at(res.scale[k], chmap)
        out.append(((fr - grad) / t).numpy())
    ddx, ddy = int(p["dx"][i] - p["dx"][j]), int(p["dy"][i] - p["dy"][j])
    assert ddx % 2 == 0 and ddy % 2 == 0
    ys1 = slice(max(0, -ddy), H - max(0, ddy))
    xs1 = slice(max(0, -ddx), W - max(0, ddx))
    ys2 = slice(max(0, ddy), H - max(0, -ddy))
    xs2 = slice(max(0, ddx), W - max(0, -ddx))
    D = out[0][ys1, xs1] - out[1][ys2, xs2]
    Draw = p["frames"][i][ys1, xs1] - p["frames"][j][ys2, xs2]
    cm = chmap.numpy()[ys1, xs1]
    for c in range(3):
        expect = math.hypot(res.noise[i, c] / res.scale[i, c], res.noise[j, c] / res.scale[j, c])
        d = D[cm == c]
        robust = 1.4826 * np.median(np.abs(d - np.median(d)))
        assert 0.9 < robust / expect < 1.1, (c, robust, expect)
        # 64 x 64 blocks: their medians are the noise's only
        h, w = (D.shape[0] // 64) * 64, (D.shape[1] // 64) * 64
        for img, lim in ((D, 0.25), (Draw, None)):
            b = np.where(cm[:h, :w] == c, img[:h, :w], np.nan).reshape(h // 64, 64, w // 64, 64)
            med = np.nanmedian(b.transpose(0, 2, 1, 3).reshape(h // 64, w // 64, -1), -1) / expect
            if lim:
                assert np.abs(med).max() < lim, (c, np.abs(med).max())
            else:
                assert np.abs(med).max() > 2


def test_the_reference_integrates_the_best_frames(moon):
    p = moon
    ref = nl.build_reference(lambda i: p["frames"][i], p["coef"], p["weight"], CH, H, W,
                             fwhm=float(np.percentile(p["fwhm"], 90)), n_best=6)
    assert sorted(ref.best) == sorted(np.argsort(-p["weight"])[:6])        # WBPP: the best by weight, not one frame
    one = np.median(p["res"].noise / p["res"].scale, 0)
    assert (ref.noise < one / math.sqrt(6)).all()
    assert ref.image.shape == (3, H // 2, W // 2)
    # the star far above saturation is never measured
    d = np.hypot(ref.stars[:, 0] - 0.6 * W, ref.stars[:, 1] - 0.3 * H)
    assert d.min() > 2 * ref.radius
    assert (np.isfinite(ref.flux).sum(0) >= 30).all()
    w = p["weight"].copy()
    w[np.argmax(w)] = 0.0                                                  # a frame left out is never in it
    with pytest.raises(ValueError):
        nl.build_reference(lambda i: p["frames"][i], p["coef"], np.zeros(N), CH, H, W, fwhm=4.0)
    ref2 = nl.build_reference(lambda i: p["frames"][i], p["coef"], w, CH, H, W, fwhm=4.0, n_best=3)
    assert np.argmax(p["weight"]) not in ref2.best and len(ref2.best) == 3


def test_diagnostic_measures_the_spread_of_the_backgrounds(moon, flat):
    for p, has_moon in ((moon, True), (flat, False)):
        res = p["res"]
        X, Y = _grid()
        g = np.array([bkg_truth(p, i, X, Y) / p["s"][i][:, None] for i in range(N)])        # (N, 3, 81)
        true = np.ptp(g - np.median(g, 0), axis=2) / (res.noise / p["s"])
        got = np.array([res.diagnostic["spread_sigma"][i] for i in range(N)])
        assert np.all(np.abs(got - true) < 0.1 * true + 0.3), np.abs(got - true).max()
        d = res.diagnostic
        if has_moon:
            assert d["max_sigma"] > 4 and d["params"]["colour"] == "blue"
            assert d["reason"] == nl.REASON_TEMPLATES["spread"]
            assert abs(d["params"]["colour_pct"] - 100 * (10 ** (0.4 * 0.13 * 1.15) - 1)) < 1.5   # blue / green: 15 %
            text = d["reason"].format(**d["params"])
            assert "σ" in text and "blue" in text
        else:
            assert d["max_sigma"] < 0.6 and d["reason"] == nl.REASON_TEMPLATES["flat"]
        fd = res.frames[3].as_dict()
        assert len(fd["scale"]) == 3 and np.shape(fd["gradient"]) == (3, 6) and "spread_sigma" in fd


def test_flat_set_from_float16_frames(flat):
    """The frames read as the cache stores them (float16), a saturation level given: the same quality."""
    res, s = flat["res"], flat["s"]
    assert res.ok.all()
    kappa = res.scale / s
    assert (np.abs(kappa / np.median(kappa, 0) - 1) < 0.01).all()


def test_global_mode_is_sirils_additive_with_scaling(moon):
    p = moon
    rf = 4
    res = nl.normalize(lambda i: p["frames"][i], p["coef"], p["weight"], CH, H, W,
                       fwhm=float(np.percentile(p["fwhm"], 90)), mode="global", ref_frame=rf, n_best=4)
    assert res.reference.ref_frame == rf and res.reference.mode == "global"
    Q, sc = res.Q, res.scale
    assert np.all(Q[:, :, 1:] == 0)                                        # no gradient: global
    assert np.allclose(sc[rf], 1.0) and np.allclose(Q[rf, :, 0], 0.0, atol=1e-6)
    def planes(i):
        return [p["frames"][i][ky::2, kx::2] for ky, kx in nl.KS]

    for i in (0, 7, 11):
        for c in range(3):
            v = np.concatenate([pl[pl < nl.VALID_MAX].ravel() for k, pl in enumerate(planes(i)) if CH[k] == c])
            vr = np.concatenate([pl[pl < nl.VALID_MAX].ravel() for k, pl in enumerate(planes(rf)) if CH[k] == c])
            li, si = nl.ikss(v)
            lr, sr = nl.ikss(vr)
            assert abs(sc[i, c] - si / sr) < 1e-9
            # Siril: normalised = (x - location_i) x scale_ref / scale_i + location_ref
            ln, sn = nl.ikss((v - Q[i, c, 0]) / sc[i, c])
            assert abs(ln - lr) < 1e-3 * lr and abs(sn / sr - 1) < 1e-3
    # the per-colour transparency and the diagnostic are still measured
    assert np.isfinite(res.photometric).all() and res.diagnostic["max_sigma"] > 4
    with pytest.raises(ValueError):
        nl.build_reference(lambda i: p["frames"][i], p["coef"], p["weight"], CH, H, W, fwhm=4.0, mode="adaptive")
