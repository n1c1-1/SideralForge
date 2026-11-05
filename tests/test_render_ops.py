"""GPU render primitives must match their OpenCV / numpy reference implementations."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from astro_gpu.render import ops

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


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


def test_percentile_like_numpy(img):
    t = torch.from_numpy(img).to(DEV)
    for q in (1, 30, 99.995):
        assert abs(ops.percentile(t, q) - np.percentile(img, q)) < 1e-6
