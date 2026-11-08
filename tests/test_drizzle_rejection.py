"""The drizzle with the integration's rejection (stages/stack.py ``drizzle_rejection = integration``): PixInsight's
method for a CFA drizzle. The rejection runs on the debayered, registered frames (core/stack_classic ``masks``), and
``drizzle_deposit`` leaves a frame's raw samples out of the output pixels whose reference pixel was rejected in their
colour, and out of no other (PCL DrizzleIntegrationInstance.cpp: ``Reject(q, c)``, q the reference pixel holding the
output pixel's top-left corner)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from astro_gpu.core import debayer as db
from astro_gpu.core import poly
from astro_gpu.core import stack_classic as sc
from astro_gpu.stages import stack
from astro_gpu.stages.stack import HIGH_SHIFT, drizzle_deposit

CH = [0, 1, 1, 2]                                   # RGGB


def chmap_of(H: int, W: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    return torch.tensor(CH)[2 * (yy % 2) + (xx % 2)]


def drizzle(val, X, Y, chmap, S, Ho, Wo, rej=None, w=1.0, h=None, tally=None):
    """One frame dropped at scale S with drops of 1 (h = S / 2): (num, den) (3, Ho, Wo)."""
    num = torch.zeros(3 * Ho * Wo, dtype=torch.float64)
    den = torch.zeros(3 * Ho * Wo, dtype=torch.float64)
    Xo, Yo = (X + 0.5) * S - 0.5, (Y + 0.5) * S - 0.5
    keep = torch.isfinite(val)
    drizzle_deposit(num, den, val, Xo, Yo, w, keep, chmap, S / 2 if h is None else h, S, Ho, Wo, rej, tally)
    return num.view(3, Ho, Wo), den.view(3, Ho, Wo)


def test_a_drop_reaches_every_output_pixel_it_covers_at_any_scale():
    """A drop of side 3 output pixels (scale 3) covers up to 4 rows: all of them are reached, its whole area deposited."""
    H, W = 12, 12
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    _num, den = drizzle(torch.ones((H, W), dtype=torch.float64), xx + 0.3, yy + 0.2, chmap_of(H, W), 3.0, 40, 40)
    assert float(den.sum()) == pytest.approx(H * W * 9.0)


def test_on_all_the_cores_the_drops_land_where_they_do_on_one(monkeypatch):
    """The frame cut into bands of rows, the same sums as on one core (float rounding apart), with or without rejection
    data; a frame turned a quarter of a turn, whose bands would write to the same rows, is dropped on one core."""
    H, W = 240, 320
    rng = np.random.default_rng(1)
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    X, Y = xx + 0.37 + 1e-3 * yy, yy - 0.21 + 1e-3 * xx
    val = torch.from_numpy(rng.uniform(100, 200, (H, W)).astype(np.float32))
    chmap = chmap_of(H, W)
    keep = torch.ones((H, W), dtype=torch.bool)
    rej = torch.from_numpy((rng.random((H, W)) < 0.01).astype(np.uint8))

    def run(parallel, reject):
        monkeypatch.setattr(stack, "PARALLEL_SAMPLES", 1 if parallel else 10 ** 12)
        num, den = torch.zeros(3 * H * W), torch.zeros(3 * H * W)
        drizzle_deposit(num, den, val, X, Y, 1.0, keep, chmap, 0.35, 1.0, H, W, rej if reject else None)
        return num, den

    before = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        for reject in (False, True):
            a, b = run(False, reject), run(True, reject)
            assert torch.get_num_threads() == 4
            assert torch.allclose(a[0], b[0], rtol=1e-5) and torch.allclose(a[1], b[1], rtol=1e-5)
        assert stack.deposit_bands(Y.ravel(), 0.35, 2, 4) is not None
        assert stack.deposit_bands(X.ravel(), 0.35, 2, 4) is None        # the rows of a frame turned by 90 degrees
    finally:
        torch.set_num_threads(before)


# ------------------------------------------------------------------------------------------------ the mapping
def test_a_rejected_reference_pixel_leaves_that_frame_out_of_its_output_pixel_in_that_colour_only():
    H, W = 16, 20
    rng = np.random.default_rng(3)
    chmap = chmap_of(H, W)
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    X, Y = xx + 0.3, yy - 0.4                        # the frame lands 0.3 px right, 0.4 px up of the reference
    val = torch.from_numpy(rng.uniform(100, 200, (H, W)))
    for S in (1.0, 2.0):
        Ho, Wo = int(round(H * S)), int(round(W * S))
        num0, den0 = drizzle(val, X, Y, chmap, S, Ho, Wo)
        rej = torch.zeros((H, W), dtype=torch.uint8)
        rej[7, 5] = 1 << 0                           # red, rejected low
        rej[3, 9] = 1 << (HIGH_SHIFT + 1)            # green, rejected high
        rej[10, 12] = (1 << 2) | (1 << HIGH_SHIFT)   # blue low, red high
        tally = [0.0, 0.0]
        num1, den1 = drizzle(val, X, Y, chmap, S, Ho, Wo, rej, tally=tally)
        gone = torch.zeros((3, Ho, Wo), dtype=torch.bool)
        n = int(S)
        for (qy, qx), cs in (((7, 5), (0,)), ((3, 9), (1,)), ((10, 12), (2, 0))):
            for c in cs:                             # the output pixels whose top-left corner is in reference pixel q
                gone[c, qy * n:(qy + 1) * n, qx * n:(qx + 1) * n] = True
        assert float(den0[gone].sum()) > 1.0         # the frame did reach some of them (its colour's sites only)
        assert bool((den1[gone] == 0).all()) and bool((num1[gone] == 0).all())
        torch.testing.assert_close(num1[~gone], num0[~gone], rtol=1e-12, atol=1e-9)
        torch.testing.assert_close(den1[~gone], den0[~gone], rtol=1e-12, atol=1e-12)
        assert math.isclose(tally[0], float(den0[gone].sum()), rel_tol=1e-9)
        assert math.isclose(tally[1], float(den0.sum()), rel_tol=1e-9)


def test_no_rejection_data_is_the_drizzle_unchanged_and_a_map_edge_rejects_nothing_outside():
    H, W = 12, 14
    chmap = chmap_of(H, W)
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    val = 100 + xx + 2 * yy
    num0, den0 = drizzle(val, xx - 0.6, yy + 0.7, chmap, 1.0, H, W)
    num1, den1 = drizzle(val, xx - 0.6, yy + 0.7, chmap, 1.0, H, W, torch.zeros((H, W), dtype=torch.uint8))
    assert torch.equal(num0, num1) and torch.equal(den0, den1)


# ------------------------------------------------------------------------------------------------ the whole path
H, W = 56, 64
NF = 10                                             # winsorized sigma clipping (6 to 15 frames)
SIGMA = 4.0
HOT_FRAME, HOT_AT = 2, (15, 50)                     # one raw sample of one frame, far from the stars and the trail
TRAIL_FRAME = 5


def sky(X, Y):
    base = torch.stack([300 + 0.8 * X, 400 + 0.5 * Y, 250 + 0.3 * (X + Y)])
    for x0, y0, a, col in ((20.3, 18.6, 3000, (0.8, 1.0, 0.6)), (45.7, 40.2, 2000, (1.0, 0.9, 1.1))):
        p = a * torch.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * 1.6 ** 2))
        base = base + torch.stack([p * col[0], p * col[1], p * col[2]])
    return base


def trail_of(x, y):
    return np.abs(y - (0.4 * x + 30)) < 1.2


def night():
    """NF dithered CFA frames of ``sky`` (scale and offset per frame and colour, noise SIGMA), a hot pixel in one,
    a satellite trail in another; their registration and normalisation."""
    rng = np.random.default_rng(5)
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    col = db.colour_of("RGGB", H, W)
    scale = rng.uniform(0.85, 1.1, (NF, 3))
    Q = np.zeros((NF, 3, 6))
    Q[:, :, 0] = rng.uniform(-30, 30, (NF, 3))
    coefs, frames, pos = [], [], []
    for i in range(NF):
        dx, dy = rng.uniform(-4, 4, 2)
        c = np.zeros((2, 10))
        c[0, :3] = [W / 2 + dx, W / 2, 0]
        c[1, :3] = [H / 2 + dy, 0, H / 2]
        X, Y = poly.poly_apply(c, xx, yy, W, H)
        cfa = torch.gather(sky(X, Y), 0, col[None])[0].numpy()
        cfa = cfa * scale[i][col.numpy()] + Q[i, :, 0][col.numpy()] + rng.normal(0, SIGMA, (H, W))
        if i == HOT_FRAME:
            cfa[HOT_AT] += 5000.0
        if i == TRAIL_FRAME:
            cfa += 1500.0 * trail_of(xx.numpy(), yy.numpy())
        coefs.append(c)
        frames.append(cfa.astype(np.float32))
        pos.append((X, Y))
    return np.array(coefs), frames, sc.Norm(scale, Q), pos, col


def test_the_integrations_rejection_takes_a_hot_pixel_and_a_trail_out_of_the_drizzle_and_nothing_else():
    coef, frames, norm, pos, col = night()
    res = sc.stack_classic(lambda i: frames[i], coef, None, norm, None, sc.ClassicParams(), "cpu",
                           used=range(NF), ch_of_k=CH, H=H, W=W, masks=True)
    assert res.choice.method == "winsorized" and res.masks.shape == (NF, H, W) and res.masks.dtype == np.uint8
    # the masks hold what the classic stack counted
    lo = sum(int(((res.masks >> c) & 1).sum()) for c in range(3))
    hi = sum(int(((res.masks >> (HIGH_SHIFT + c)) & 1).sum()) for c in range(3))
    assert lo == sum(f["rejected_low"] for f in res.frames) and hi == sum(f["rejected_high"] for f in res.frames)
    chmap = col.long()
    acc = {k: [torch.zeros(3, H, W, dtype=torch.float64) for _ in range(2)] for k in ("plain", "integration")}
    changed = torch.zeros(3, H, W, dtype=torch.bool)
    for i in range(NF):
        v = torch.from_numpy(frames[i]).double()
        v = (v - torch.from_numpy(norm.Q[i, :, 0])[chmap]) / torch.from_numpy(norm.scale[i])[chmap]
        X, Y = pos[i]
        n0, d0 = drizzle(v, X, Y, chmap, 1.0, H, W)
        rej = torch.from_numpy(res.masks[i])
        n1, d1 = drizzle(v, X, Y, chmap, 1.0, H, W, rej)
        for k, (n, d) in (("plain", (n0, d0)), ("integration", (n1, d1))):
            acc[k][0] += n
            acc[k][1] += d
        # nothing else: a frame's drops change only on the output pixels and colours its mask rejected
        lost = (d0 - d1) > 0
        bits = torch.stack([((rej >> c) | (rej >> (HIGH_SHIFT + c))) & 1 for c in range(3)]).bool()
        assert bool((~lost | bits).all())
        changed |= lost
    img = {k: (a[0] / a[1].clamp(min=1e-12)).numpy() for k, a in acc.items()}
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    truth = sky(torch.from_numpy(xx), torch.from_numpy(yy)).numpy()
    noise = SIGMA / math.sqrt(NF)
    m = 8
    inner = np.zeros((H, W), bool)
    inner[m:-m, m:-m] = True
    # the hot pixel: where it lands in the reference, in its own colour
    Xh, Yh = float(pos[HOT_FRAME][0][HOT_AT]), float(pos[HOT_FRAME][1][HOT_AT])
    ch = int(col[HOT_AT])
    near = (np.abs(xx - Xh) < 2) & (np.abs(yy - Yh) < 2)
    assert (img["plain"][ch] - truth[ch])[near].max() > 40 * noise            # it shows without the rejection
    assert np.abs(img["integration"][ch] - truth[ch])[near].max() < 6 * noise
    # the trail: on the reference grid, where frame TRAIL_FRAME's trail lands
    Xt, Yt = pos[TRAIL_FRAME]
    on = np.zeros((H, W), bool)
    hit = trail_of(xx, yy)                           # the trail's raw samples (sensor x, y)
    tx, ty = np.round(Xt.numpy()[hit]).astype(int), np.round(Yt.numpy()[hit]).astype(int)
    ok = (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H)
    on[ty[ok], tx[ok]] = True
    on &= inner
    assert (img["plain"] - truth)[:, on].mean() > 40 * noise
    assert np.abs((img["integration"] - truth)[:, on]).mean() < 3 * noise
    # elsewhere the drizzle hardly moves: a few output pixels lose a frame, the rejection's false alarms on noise (at
    # scale 1 one output pixel per rejected value: about NF x the rate of the clean frames). The debayering and the
    # Lanczos-4 registration of the integration spread the trail and the hot pixel by up to 3 px: those go too
    from scipy.ndimage import binary_dilation
    away = inner & ~binary_dilation(on | near, np.ones((7, 7), bool))
    clean = [f for f in res.frames if f["index"] not in (HOT_FRAME, TRAIL_FRAME)]
    rate = sum(f["rejected_low"] + f["rejected_high"] for f in clean) / sum(f["tested"] for f in clean)
    assert rate < 0.03 and float(changed[:, away].float().mean()) < 1.5 * NF * rate + 0.005
    d = np.abs(img["integration"] - img["plain"])[:, away]
    assert float(np.median(d)) == 0.0 and float(np.percentile(d, 99)) < 4 * noise
