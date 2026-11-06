"""Issue #23: the Colour mixer (hue, saturation, luminance of eight ranges in CIE L*c*h*) and the background
protection of the Saturation filter."""
from __future__ import annotations

import numpy as np

from astro_gpu.client.postengine import FILTERS
from astro_gpu.client.postfilters.colour import to_lab
from astro_gpu.client.postfilters.mixer import keys, mix


def _chroma(rgb):
    lab = to_lab(rgb[None, None].astype(np.float32))[0, 0]
    return float(np.hypot(lab[1], lab[2])), float(np.degrees(np.arctan2(lab[2], lab[1])) % 360)


def _img():
    teal, red, grey = [0.2, 0.45, 0.45], [0.6, 0.15, 0.15], [0.3, 0.3, 0.3]
    return np.array([[teal, red, grey]], np.float32)


def test_zero_is_the_identity_and_eight_bands():
    x = _img()
    assert mix(x, {k: 0.0 for k in keys()}) is x and len(keys()) == 24


def test_the_aqua_saturation_raises_the_teal_only_and_leaves_grey():
    x = _img()
    out = mix(x, {"sat_aqua": 80.0})
    assert _chroma(out[0, 0])[0] > 1.3 * _chroma(x[0, 0])[0]                    # the OIII teal stronger
    assert abs(_chroma(out[0, 1])[0] - _chroma(x[0, 1])[0]) < 1.0              # the red left
    assert np.allclose(out[0, 2], x[0, 2], atol=1e-3)                           # the grey left


def test_a_hue_shift_turns_the_red_and_luminance_lightens_it():
    x = _img()
    h0 = _chroma(x[0, 1])[1]
    out = mix(x, {"hue_red": 20.0})
    assert 8 < (_chroma(out[0, 1])[1] - h0) % 360 < 25                         # towards orange
    lighter = mix(x, {"lum_red": 60.0})
    assert lighter[0, 1].mean() > x[0, 1].mean()


def test_the_background_protection_spares_the_sky_and_saturates_the_nebula():
    rng = np.random.default_rng(1)
    sky = np.clip(0.1 + rng.normal(0, 0.01, (64, 64, 3)), 0, 1).astype(np.float32)
    sky[20:40, 20:40] = [0.7, 0.3, 0.3]                                          # a red nebula
    f = FILTERS["saturation"]["fn"]
    out = f(sky, {"strength": 80.0, "hues": "All", "protect": 1.5}, None)
    assert np.allclose(out[:10, :10], sky[:10, :10], atol=1e-6)                 # the background's noise not coloured
    assert (out[30, 30, 0] - out[30, 30, 1]) > (sky[30, 30, 0] - sky[30, 30, 1])
    plain = f(sky, {"strength": 80.0, "hues": "All", "protect": 0.0}, None)
    assert not np.allclose(plain[:10, :10], sky[:10, :10], atol=1e-4)            # without it, the noise is
