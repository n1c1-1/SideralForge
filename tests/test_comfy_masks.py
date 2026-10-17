"""The ComfyUI mask nodes (comfyui_astro/masks.py): blending views by masks, brightness masks, painted masks.
Needs ComfyUI's code (COMFYUI_ROOT, default ../ComfyUI next to this repository); skipped without it."""
import base64
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(os.environ.get("COMFYUI_ROOT", Path(__file__).resolve().parents[2] / "ComfyUI"))
if (ROOT / "comfy_api").is_dir() and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
pytest.importorskip("comfy_api")
masks = pytest.importorskip("comfyui_astro.masks")


def _img(value, h=40, w=60):
    return torch.full((1, h, w, 3), float(value))


def test_blend_layers_and_coverage():
    m2 = torch.zeros(1, 40, 60)
    m2[:, :, :30] = 1                                      # left half: view 2
    m3 = torch.zeros(1, 20, 30)                            # another size: resized to the image
    m3[:, :10] = 1                                         # top half: view 3, over view 2
    out = masks.AstroBlendViews.execute(_img(0.1), view_2=_img(0.5), mask_2=m2, view_3=_img(0.9), mask_3=m3)
    img, cov = out.result if hasattr(out, "result") else out.args
    img = img[0, ..., 0].numpy()
    assert img[30, 5] == pytest.approx(0.5) and img[30, 50] == pytest.approx(0.1)
    assert img[5, 5] == pytest.approx(0.9) and img[5, 50] == pytest.approx(0.9)
    assert cov.shape == (1, 40, 60, 3) and float(cov.max()) <= 1.0


def test_blend_needs_view_and_mask_together():
    with pytest.raises(ValueError, match="go together"):
        masks.AstroBlendViews.execute(_img(0.1), view_2=_img(0.5))


def test_blend_rejects_other_sizes():
    with pytest.raises(ValueError, match="same scale"):
        masks.AstroBlendViews.execute(_img(0.1), view_2=_img(0.5, 20, 30), mask_2=torch.ones(1, 20, 30))


def test_brightness_mask_percentiles():
    ramp = torch.linspace(0, 1, 100).repeat(50, 1)[None, ..., None].repeat(1, 1, 1, 3)
    out = masks.AstroBrightnessMask.execute(ramp, "percentile", 40.0, 60.0, 0.0, 0.0, 0.0, False)
    m = (out.result if hasattr(out, "result") else out.args)[0][0].numpy()
    assert m[25, 10] == 0 and m[25, 90] == 1 and 0 < m[25, 50] < 1
    out = masks.AstroBrightnessMask.execute(ramp, "level", 40.0, 60.0, 0.0, 0.0, 0.0, True)
    m = (out.result if hasattr(out, "result") else out.args)[0][0].numpy()
    assert m[25, 10] == 1 and m[25, 90] == 0


def test_painted_mask_decoded_and_resized():
    a = np.zeros((20, 30, 4), np.uint8)
    a[:, 15:, 3] = 255                                     # right half painted (alpha)
    ok, buf = cv2.imencode(".png", a)
    url = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
    m = masks._fit(masks._decode_painted(url), 40, 60)
    assert m.shape == (40, 60) and m[20, 5] == 0 and m[20, 55] == 1
    assert masks._decode_painted("") is None
