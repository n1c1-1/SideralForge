"""post_images 7: which part of the composition makes the export differ from the screen. The base layer alone (the
others hidden, the filters deleted), exported in TIFF 16 by the dialog and compared; then a Saturation filter at
+100 %; then the masked denoise layer over it."""
import sys
import time

import numpy as np
from explore5_post_images_common import COPY, PI, WORK, R, Explorer, start, stats

from astro_gpu.client.exportdlg import SUFFIX
from astro_gpu.client.i18n import _
from astro_gpu.client.post3 import ContSlider

start(3000)
ex = Explorer("post_images", lang="fr")
pi = PI(ex)
pi.open(COPY)
v = pi.v
pi.idle(300)
OUT = WORK / "exports2"
OUT.mkdir(exist_ok=True)
steps = sys.argv[1:] or ["base", "sat", "mask"]
base = next(lay for lay in reversed(v.layers) if lay["kind"] == "workflow")
masked = next((x for x in v.layers if x.get("mask")), None)
for f in [x for x in v.layers if x["kind"] == "filter"]:
    v.select(f["id"], "layer")
    ex.settle(3)
    ex.click(v.b_del)
for x in v.layers:
    if x["kind"] == "workflow" and x["id"] != base["id"] and x["visible"]:
        v.set_visible(x["id"], False)
v.select(base["id"], "layer")
pi.idle(200)


def export(name):
    info = {}

    def handle(d):
        d.dest.setText(f"{(OUT / name).as_posix()}/{name}{SUFFIX['tiff16']}")
        t0 = time.monotonic()
        d.go.click()
        while d.running and time.monotonic() - t0 < 1500:
            ex.settle(4)
        info["files"] = [str(f) for f in d.files]
        info["s"] = round(time.monotonic() - t0, 1)
        d.go.click()
    ex.dialog_handler = handle
    ex.click(v.b_export)
    t0 = time.monotonic()
    while "s" not in info and time.monotonic() - t0 < 1600:
        ex.settle(4)
    return info


def compare(tag):
    import cv2
    import tifffile
    v.viewer.fit()
    pi.idle(120)
    screen = pi.screen_array()
    seqkind = [(e[2], e[3]) for e in pi.events if e[1] == "res"][-3:]
    info = export(tag)
    f = next(x for x in info["files"] if x.endswith(".tif"))
    a = tifffile.imread(f).astype(np.float32) / 65535
    half = cv2.resize(a, (screen.shape[1], screen.shape[0]), interpolation=cv2.INTER_AREA) * 255
    d = half - screen.astype(np.float32)
    R("X", tag, "export s", info["s"], "screen from", seqkind, "signed mean per channel",
      [round(float(d[..., c].mean()), 2) for c in range(3)], "mean |d|", round(float(np.abs(d).mean()), 2),
      "p99 |d|", round(float(np.percentile(np.abs(d), 99)), 1), "export", stats(half), "screen", stats(screen))
    # the sky alone (the darkest half of the pixels of the screen)
    lum = screen.mean(-1)
    sky = lum < np.percentile(lum, 50)
    R("X", tag, "sky: export", [round(float(half[..., c][sky].mean()), 1) for c in range(3)], "screen",
      [round(float(screen[..., c][sky].mean()), 1) for c in range(3)])
    pi.save_png(screen, f"x_{tag}_screen")
    pi.save_png(np.clip(half + 0.5, 0, 255).astype(np.uint8), f"x_{tag}_export_reduced")


if "base" in steps:
    compare("base_alone")
if "sat" in steps:
    ex.menu_pick = _("Saturation")
    ex.click(v.b_filter)
    pi.idle(60)
    s = next(x for x in v.right.findChildren(ContSlider) if x.isVisible() and x.name.text() != _("Opacity"))
    pi.drag_slider(s, 0.8)
    pi.idle(120)
    R("X sat value", s.val.text())
    compare("base_saturation")
if "mask" in steps and masked is not None:
    v.set_visible(masked["id"], True)
    pi.idle(120)
    compare("base_saturation_masked_layer")
R("errors", pi.slot_errors(), "popups", ex.popups)
ex.close()
