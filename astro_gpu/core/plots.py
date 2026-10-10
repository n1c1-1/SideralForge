"""Matplotlib helpers producing preview images that match the dark interface."""
from __future__ import annotations

import numpy as np

BG = "#1b1d21"
PANEL = "#23262b"
FG = "#e4e2dd"
MUTED = "#a7abb3"
GRID = "#343840"
AMBER = "#d9a441"
BLUE = "#5aa0d8"


def figure(nrows=1, ncols=1, size=(9, 4.2), **kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows, ncols, figsize=size, dpi=110, **kw)
    fig.patch.set_facecolor(BG)
    for a in np.atleast_1d(ax).ravel():
        a.set_facecolor(PANEL)
        a.tick_params(colors=MUTED, labelsize=9)
        a.xaxis.label.set_color(FG)
        a.yaxis.label.set_color(FG)
        a.title.set_color(FG)
        for s in a.spines.values():
            s.set_color(GRID)
        a.grid(color=GRID, lw=0.6)
    return fig, ax


def to_rgb(fig) -> np.ndarray:
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img
