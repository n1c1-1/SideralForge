"""GPU render primitives must match their OpenCV / numpy reference implementations."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from astro_gpu.render import ops
from astro_gpu.render.engine import fit_tone_curve, tone_curve

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Guided filter (He et al. 2013): edge-preserving smoothing of p guided by I."""
    def box(a):
        return cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    a = cv2.GaussianBlur(rng.random((300, 420)).astype(np.float32), (0, 0), 3) + 0.1 * rng.random((300, 420)).astype(np.float32)
    return a


@pytest.mark.parametrize("sigma", [1.0, 3.0, 8.0])
def test_gaussian_matches_opencv(img, sigma):
    ref = cv2.GaussianBlur(img, (0, 0), sigma)
    got = ops.gaussian_blur(torch.from_numpy(img).to(DEV), sigma).cpu().numpy()
    inner = np.s_[int(4 * sigma) + 2:-int(4 * sigma) - 2, int(4 * sigma) + 2:-int(4 * sigma) - 2]
    assert np.abs(got[inner] - ref[inner]).max() < 1e-4


def test_large_gaussian_close(img):
    ref = cv2.GaussianBlur(img, (0, 0), 30)
    got = ops.gaussian_blur(torch.from_numpy(img).to(DEV), 30).cpu().numpy()
    assert np.abs(got[100:200, 150:270] - ref[100:200, 150:270]).max() < 5e-3


def test_box_and_guided(img):
    ref = cv2.boxFilter(img, -1, (9, 9), borderType=cv2.BORDER_REFLECT)
    got = ops.box_filter(torch.from_numpy(img).to(DEV), 4).cpu().numpy()
    assert np.abs(got[10:-10, 10:-10] - ref[10:-10, 10:-10]).max() < 1e-5
    rg = guided_filter(img, img, 6, 0.01)
    gg = ops.guided(torch.from_numpy(img).to(DEV), torch.from_numpy(img).to(DEV), 6, 0.01).cpu().numpy()
    assert np.abs(gg[20:-20, 20:-20] - rg[20:-20, 20:-20]).max() < 1e-4


def test_percentile_like_numpy(img):
    t = torch.from_numpy(img).to(DEV)
    for q in (1, 30, 99.995):
        assert abs(ops.percentile(t, q) - np.percentile(img, q)) < 1e-6


def test_oklab_roundtrip():
    rgb = torch.rand(3, 50, 50, device=DEV)
    back = ops.oklab_to_lin(ops.lin_to_oklab(rgb))
    assert (back - rgb).abs().max() < 1e-4


def test_gamut_map_in_range():
    rgb = torch.randn(3, 40, 40, device=DEV) * 0.6 + 0.5
    out = ops.gamut_map_oklab(rgb)
    assert out.min() >= 0 and out.max() <= 1


def test_tone_curve_hits_targets():
    lo, b, w = fit_tone_curve(10.0, 20.0, 400.0, 10 / 255, 24 / 255, 0.015)
    assert abs(tone_curve(20.0, lo, b, w, 0.015) - 24 / 255) < 2e-3
    assert abs(tone_curve(10.0, lo, b, w, 0.015) - 10 / 255) < 5e-3
    assert tone_curve(lo - 50.0, lo, b, w, 0.015) > 0          # soft toe: no hard clip
