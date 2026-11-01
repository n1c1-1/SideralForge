"""Colour channel alignment that follows the night (astro_gpu/core/channel_align.py): synthetic raw CFA frames whose
red and blue star images are shifted from the green ones by the atmospheric dispersion of a 6-hour night (it changes
with the altitude and turns with the parallactic angle) plus a radial lateral chromatic aberration fixed in the
sensor. The per-block correction recovers both; one correction for the whole night cannot."""
from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.special import erf

from astro_gpu.core import channel_align as ca

W, H = 768, 512
CFA = [0, 1, 1, 2]                                   # RGGB: channel of k = 2 (y % 2) + (x % 2)
FWHM = 3.2
SITE = dict(lat_deg=45.0, lon_deg=2.0)
DEC = -15.0
CD = np.array([[-1.0, 0.0], [0.0, -1.0]]) / 3600.0  # 1 arcsec per pixel, north up, east left
AXIS = np.array([W / 2 + 25.0, H / 2 - 15.0])        # the optical axis, off the sensor centre
RC = float(np.hypot(W / 2, H / 2))
LATERAL = {0: 0.7, 2: -0.35}                         # red, blue radial lateral colour (px at r = RC, before r^3)


def lateral(c, x, y):
    """Radial lateral chromatic aberration (Brown-Conrady: r (k1 + k3 r^2)) of colour c at sensor (x, y)."""
    if c == 1:
        return np.zeros(np.shape(x)), np.zeros(np.shape(x))
    rx, ry = (np.asarray(x) - AXIS[0]) / RC, (np.asarray(y) - AXIS[1]) / RC
    f = LATERAL[c] * (1.0 + 0.6 * (rx * rx + ry * ry))
    return f * rx, f * ry


def night_times(n=40, cadence=540.0):
    """UTC seconds of a 6-hour run centred on the meridian crossing of the target (RA = local sidereal time)."""
    t0 = ca.times_s(["2026-08-01T22:00:00"])[0]
    t = t0 + cadence * np.arange(n)
    jd = 2440587.5 + t[n // 2] / 86400.0
    gmst = (280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360.0
    return t, (gmst + SITE["lon_deg"]) % 360.0


def truth(i, x, y, disp):
    """The true offset (3, m, 2) of each colour from green at sensor points of frame i."""
    out = np.zeros((3, len(x), 2))
    for c in (0, 2):
        lx, ly = lateral(c, x, y)
        out[c, :, 0], out[c, :, 1] = lx + disp[i, c, 0], ly + disp[i, c, 1]
    return out


def render(i, sky_xy, gflux, col, shift, disp, rng, transp=1.0):
    """A calibrated raw CFA frame: each colour's image of each star (pixel-integrated gaussian) sampled at that
    colour's pixels only; sky, shot and read noise. Returns the frame and its star list (register's format)."""
    sig = FWHM / 2.3548
    xy = sky_xy - shift                                       # where the stars fall on this frame's sensor
    inside = (xy[:, 0] > 12) & (xy[:, 0] < W - 12) & (xy[:, 1] > 12) & (xy[:, 1] < H - 12)
    xy, f, cl = xy[inside], gflux[inside] * transp, col[inside]
    R = 6                                                     # 4.4 sigma: the rest of the star is below the noise
    oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
    ox, oy = ox.ravel(), oy.ravel()
    cx, cy = np.rint(xy[:, 0]).astype(int), np.rint(xy[:, 1]).astype(int)
    px, py = cx[:, None] + ox[None], cy[:, None] + oy[None]
    ch = np.asarray(CFA)[2 * (py % 2) + px % 2]
    img = np.zeros((H, W))
    val = np.zeros(px.shape)
    for c, s_c in ((0, 0.97 * sig), (1, sig), (2, 1.04 * sig)):     # seeing a little finer in red
        lx, ly = lateral(c, xy[:, 0], xy[:, 1])
        mx = xy[:, 0] + lx + (disp[i, c, 0] if c != 1 else 0.0)
        my = xy[:, 1] + ly + (disp[i, c, 1] if c != 1 else 0.0)
        k = 1.0 / (np.sqrt(2) * s_c)
        gx = 0.5 * (erf((px + 0.5 - mx[:, None]) * k) - erf((px - 0.5 - mx[:, None]) * k))
        gy = 0.5 * (erf((py + 0.5 - my[:, None]) * k) - erf((py - 0.5 - my[:, None]) * k))
        val = np.where(ch == c, (f * cl[:, c])[:, None] * gx * gy, val)
    np.add.at(img, (py.ravel(), px.ravel()), val.ravel())
    sky, gain, rn = 300.0, 1.0, 5.0
    img = img + sky + rng.normal(0, 1, img.shape) * np.sqrt((img + sky) / gain + rn ** 2)
    cat = np.c_[xy + rng.normal(0, 0.15, xy.shape), f, np.full(len(f), FWHM / 2)]
    return img.astype(np.float16), cat                       # calibrated frames keep their sky


def scenario(n=40, n_stars=450, moving=True, seed=3):
    rng = np.random.default_rng(seed)
    t, ra = night_times(n)
    disp = ca.dispersion_offsets(t, SITE["lat_deg"], SITE["lon_deg"], ra, DEC, CD)["offsets"]
    if not moving:
        disp = np.repeat(disp[n // 2:n // 2 + 1], n, 0)
    sky_xy = rng.uniform([-30, -30], [W + 30, H + 30], (n_stars, 2))
    gflux = np.sort(rng.pareto(1.3, n_stars) + 1)[::-1] * 4000.0
    gflux = np.minimum(gflux, 3e5)
    col = np.c_[rng.uniform(0.5, 1.5, n_stars), np.ones(n_stars), rng.uniform(0.4, 1.3, n_stars)]
    shifts = rng.uniform(-20, 20, (n, 2)) + np.linspace(0, 1, n)[:, None] * [15.0, -10.0]     # dither + drift
    coef = np.zeros((n, 2, 10))
    coef[:, 0, 0], coef[:, 0, 1] = W / 2 + shifts[:, 0], W / 2              # frame -> reference: a translation
    coef[:, 1, 0], coef[:, 1, 2] = H / 2 + shifts[:, 1], H / 2
    transp = np.ones(n)
    transp[[k for k in (7, 8, 40) if k < n]] = 0.25                           # passing clouds
    meas, dt = [], 0.0
    for i in range(n):
        fr, cat = render(i, sky_xy, gflux, col, shifts[i], disp, rng, transp[i])
        t0 = time.perf_counter()
        meas.append(ca.measure_frame(fr, cat, CFA, FWHM))
        dt += time.perf_counter() - t0
    return dict(t=t, ra=ra, disp=disp, coef=coef, meas=meas, ms_per_frame=1000 * dt / n)


@pytest.fixture(scope="module")
def night():
    return scenario()


POINTS = np.array([[W / 2, H / 2], [0, 0], [W, 0], [0, H], [W, H]], float)   # centre, then the four corners


def errors(align, sc):
    """rms error (px) of the correction against the truth over the frames, (colour R/B, centre/corners)."""
    e = np.zeros((2, 2))
    n = len(sc["meas"])
    for ci, c in enumerate((0, 2)):
        d = np.array([align.at(i, POINTS[:, 0], POINTS[:, 1])[c] - truth(i, POINTS[:, 0], POINTS[:, 1], sc["disp"])[c]
                      for i in range(n)])                                       # (n, 5, 2)
        r2 = np.sum(d ** 2, -1)
        e[ci] = np.sqrt(r2[:, 0].mean()), np.sqrt(r2[:, 1:].mean())
    return e


def test_measure_is_fast_and_finds_the_stars(night):
    sizes = [len(m) for m in night["meas"]]
    assert np.median(sizes) > 150                       # isolated, unsaturated stars of each frame
    assert night["ms_per_frame"] < 250
    print(f"\nmeasure_frame: {night['ms_per_frame']:.0f} ms a frame, {np.median(sizes):.0f} stars a frame")


def test_blocks_follow_the_night_and_beat_one_correction(night):
    sc = night
    kw = dict(times=sc["t"], coef=sc["coef"])
    blocks = ca.fit_night(sc["meas"], W, H, **kw)
    whole = ca.fit_night(sc["meas"], W, H, block_s=np.inf, **kw)
    assert blocks.applied and whole.applied
    eb, ew = errors(blocks, sc), errors(whole, sc)
    s = blocks.summary()
    print(f"\n{blocks.describe()}")
    print(f"blocks: {s['blocks']} of ~{s['block_minutes']} min, >= {s['stars_per_block_min']} stars each")
    print("rms error vs truth (px)        red centre / corners    blue centre / corners")
    print(f"  per block                     {eb[0, 0]:.3f} / {eb[0, 1]:.3f}           {eb[1, 0]:.3f} / {eb[1, 1]:.3f}")
    print(f"  one for the whole night       {ew[0, 0]:.3f} / {ew[0, 1]:.3f}           {ew[1, 0]:.3f} / {ew[1, 1]:.3f}")
    true_move = np.hypot(*(sc["disp"][:, 2] - sc["disp"][:, 2].mean(0)).T).max()
    print(f"  blue dispersion moved by up to {true_move:.2f} px around its night mean")
    for b in blocks.blocks:
        tr = sc["disp"][b["first"]:b["last"] + 1].mean(0)
        print(f"  block frames {b['first']:2d}-{b['last']:2d} ({b['stars']:5d} stars): blue shift at the "
              f"sensor centre measured {b['shift_px'][2]:+.3f} {b['shift_px'][3]:+.3f}, dispersion "
              f"{tr[2, 0]:+.3f} {tr[2, 1]:+.3f} (+ lateral at the centre)")
    assert s["blocks"] >= 4
    assert eb.max() < 0.06                              # centre and corners, red and blue
    assert ew.min() > 3 * eb.max()                      # one correction for the night is several times worse
    assert s["stars_per_block_min"] >= ca.MIN_STARS


def test_the_measure_follows_the_dispersion_model(night):
    sc = night
    align = ca.fit_night(sc["meas"], W, H, times=sc["t"], coef=sc["coef"])
    model = ca.dispersion_offsets(sc["t"], SITE["lat_deg"], SITE["lon_deg"], sc["ra"], DEC, CD, coef=sc["coef"],
                                  W=W, H=H)
    chk = ca.check_dispersion(align, model)
    print(f"\ndispersion check: {chk}")
    assert abs(chk["ratio"] - 1) < 0.05 and chk["explained"] > 0.9
    z = model["zenith_deg"]
    assert 58 < z.min() < 62 and 70 < z.max() < 75     # dec -15 from latitude 45, +-3 h around the meridian


def test_a_still_night_gets_one_correction():
    sc = scenario(n=20, moving=False, seed=5)
    align = ca.fit_night(sc["meas"], W, H, times=sc["t"], coef=sc["coef"])
    whole = ca.fit_night(sc["meas"], W, H, times=sc["t"], coef=sc["coef"], block_s=np.inf)
    eb, ew = errors(align, sc), errors(whole, sc)
    print(f"\nstill night: {align.summary()['blocks']} block(s); error {eb.max():.3f} px vs whole night {ew.max():.3f}")
    assert align.summary()["blocks"] <= 2
    assert eb.max() < 0.05 and eb.max() < 1.5 * ew.max()


def test_block_field_mode_also_recovers(night):
    sc = night
    align = ca.fit_night(sc["meas"], W, H, times=sc["t"], coef=sc["coef"], field="block")
    assert errors(align, sc).max() < 0.12


def test_too_few_stars_is_not_applied():
    meas = [np.zeros((10, ca.NCOL), np.float32) + [100, 100, 0.1, 0, 0, 0.1, 1e-3, 1e-3] for _ in range(4)]
    align = ca.fit_night(meas, W, H)
    assert not align.applied and "not registered" in align.describe()
    assert ca.ChannelAlignment.from_npz(_roundtrip(align)) is None


def _roundtrip(align):
    import io
    buf = io.BytesIO()
    np.savez(buf, **align.to_npz())
    buf.seek(0)
    return np.load(buf)


def test_saved_and_applied_on_the_drizzle_grid(night):
    """What stack.py does: load the saved correction, shift the raw samples of a frame by channel (numpy and torch
    agree), green untouched; the red star images land on the green ones."""
    sc = night
    align = ca.ChannelAlignment.from_npz(_roundtrip(ca.fit_night(sc["meas"], W, H, times=sc["t"], coef=sc["coef"])))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    chmap = np.asarray(CFA)[2 * (yy.astype(int) % 2) + xx.astype(int) % 2]
    i = 13
    dx, dy = align.shift(i, xx, yy, chmap)
    assert np.all(dx[chmap == 1] == 0) and np.all(dy[chmap == 1] == 0)
    assert align.shift(i, xx, yy, chmap)[0] is dx                   # cached within a block
    m = sc["meas"][i]
    rx, ry = m[:, ca.X] + m[:, ca.RX], m[:, ca.Y] + m[:, ca.RY]          # the red image of each star
    off = align.at(i, rx, ry)[0]
    res = np.c_[rx - off[:, 0] - m[:, ca.X], ry - off[:, 1] - m[:, ca.Y]]
    assert np.abs(res.mean(0)).max() < 0.02                         # on average exactly on green
    assert np.median(np.hypot(*res.T)) < 0.2                        # each star: its own noise (~0.1 px per axis)
    torch = pytest.importorskip("torch")
    tx, ty = ca.poly_shift(align.coefs[i], torch.from_numpy(xx), torch.from_numpy(yy), torch.from_numpy(chmap), W, H)
    assert np.allclose(tx.numpy(), dx, atol=1e-4) and np.allclose(ty.numpy(), dy, atol=1e-4)


def test_a_meridian_flip_starts_a_new_block():
    n = 10
    coef = np.zeros((n, 2, 10))
    coef[:, 0, 1], coef[:, 1, 2] = W / 2, H / 2
    coef[5:, 0, 1], coef[5:, 1, 2] = -W / 2, -H / 2                 # turned by 180 degrees
    seg = ca._segments(np.arange(n), coef, np.ones(n, bool), W, H)
    assert list(seg) == [0] * 5 + [1] * 5


def test_refraction_of_air():
    """Edlen's refractivity: n - 1 = 2.78e-4 for standard air at 550 nm; refraction near 57" at 45 degrees; the
    blue-green dispersion of a Bayer camera about 0.4" x tan z at sea level."""
    n1 = ca.air_refractivity(0.55)
    assert abs(n1 - 2.778e-4) < 0.003e-4
    assert 56 < 206265 * n1 < 59
    d = 206265 * (ca.air_refractivity(0.465) - ca.air_refractivity(0.53))
    assert 0.35 < d < 0.45
