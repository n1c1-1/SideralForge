"""The registration polynomial in torch (astro_gpu/core/poly.py), which core/resample.py inverts."""
import math

import numpy as np
import torch

from astro_gpu.core import poly as nl

H, W = 512, 768


def coef_of(dx, dy, th):
    """Registration polynomial (register.py's form): a turn by th about the centre and a shift (dx, dy)."""
    c = np.zeros((2, 10))
    c[0, :3] = [W / 2 + dx, math.cos(th) * W / 2, -math.sin(th) * H / 2]
    c[1, :3] = [H / 2 + dy, math.sin(th) * W / 2, math.cos(th) * H / 2]
    return c


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



def test_the_local_normalisation_module_is_gone():
    """Removed on 2 November (no measurable effect on the crossed bench); its polynomial helpers live in core/poly.py."""
    import importlib.util
    assert importlib.util.find_spec("astro_gpu.core.normalize_local") is None
