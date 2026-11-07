"""post_images 6: the rechecks - Invert of a mask clicked on its box as the user does; the 1 px sliders of the multiscale
filters at the fitted view and at 100 %; the full-resolution region against the half image around its edges."""
import sys
import time

import numpy as np
from explore5_post_images_common import COPY, PI, R, Explorer, start, stats
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QCheckBox

from astro_gpu.client.i18n import _
from astro_gpu.client.post3 import ContSlider

start(2400)
ex = Explorer("post_images", lang="fr")
pi = PI(ex)
pi.open(COPY)
v = pi.v
pi.idle(300)
which = sys.argv[1:] or ["invert", "onepx", "region"]
base = next(lay for lay in reversed(v.layers) if lay["kind"] == "workflow")


def diff(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    return round(float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()), 3)


if "invert" in which:
    lay = next((x for x in v.layers if x.get("mask")), None)
    R("I layer with mask", lay and lay["name"], lay and lay["mask"]["invert"])
    if lay is not None:
        v.select(lay["id"], "mask")
        pi.idle(60)
        m0 = v.mask1.copy()
        inv = next(c for c in v.right.findChildren(QCheckBox) if c.isVisible() and c.text() == _("Invert"))
        R("I box", "geometry", inv.geometry().getRect(), "visible region", inv.visibleRegion().boundingRect().getRect(),
          "checked", inv.isChecked())
        for k in range(2):
            i0 = pi.mark()
            t = time.monotonic()
            ex.click(inv, QPoint(9, inv.height() // 2))       # on its box
            r = pi.time_to_image(i0, t, 120)
            pi.idle(60)
            m1 = v.mask1
            R(f"I click #{k + 1}", "checked", inv.isChecked() if inv is not None else None, "mask invert",
              lay["mask"]["invert"], r, "m0+m1-1", round(float(np.abs(m0 + m1 - 1).mean()), 4), "mean", round(
                  float(m1.mean()), 4))
            if k == 0:
                pi.save_png(pi.screen_array(), "recheck_mask_inverted_screen")
                ex.shot("recheck_mask_inverted_window")
            inv = next(c for c in v.right.findChildren(QCheckBox) if c.isVisible() and c.text() == _("Invert"))

if "onepx" in which:
    # hide the masked layer: the base alone
    for x in v.layers:
        if x["id"] != base["id"] and x["visible"] and x["kind"] == "workflow":
            v.set_visible(x["id"], False)
    v.select(base["id"], "layer")
    pi.idle(120)
    for key, name in (("ml_sharpen", "Multiscale sharpening"), ("ml_denoise", "Multiscale noise reduction")):
        for zoom in ("fit", "100"):
            ex.click(v.b_fit if zoom == "fit" else v.b_100)
            pi.idle(120)
            v.select(base["id"], "layer")
            pi.idle(30)
            ex.menu_pick = _(name)
            ex.click(v.b_filter)
            pi.idle(120)
            s0 = pi.screen_array()
            reg0 = v.src.full[1].copy() if v.src.full is not None else None
            sl = next(s for s in v.right.findChildren(ContSlider) if s.isVisible() and s.name.text() == "1 px")
            i0 = pi.mark()
            t = pi.drag_slider(sl, 0.9)
            r = pi.time_to_image(i0, t, 200)
            if zoom == "100":
                pi.wait(lambda: any(e[1] == "res" and e[2] == "region" and e[0] > t for e in pi.events[i0:]), 200)
            pi.idle(120)
            s1 = pi.screen_array()
            reg1 = v.src.full[1] if v.src.full is not None else None
            R("1px", key, zoom, sl.val.text(), r, "screen half diff", diff(s0, s1), "region diff",
              diff(reg0, reg1) if reg0 is not None and reg1 is not None and reg0.shape == reg1.shape else "n/a")
            pi.shot_viewer(f"recheck_{key}_{zoom}")
            ex.click(v.b_del)
            pi.idle(60)

if "region" in which:
    v.select(base["id"], "layer")
    ex.click(v.b_100)
    pi.wait(lambda: v.src.full is not None, 200)
    pi.idle(120)
    (bx0, by0, bx1, by1), arr = v.src.full
    half = v.src.img1
    # the full-resolution region reduced to half vs the half image: should agree (same rendering, two scales)
    import cv2
    red = cv2.resize(arr, ((bx1 - bx0) // 2, (by1 - by0) // 2), interpolation=cv2.INTER_AREA)
    hh = half[by0 // 2: by0 // 2 + red.shape[0], bx0 // 2: bx0 // 2 + red.shape[1]]
    d = np.abs(red.astype(np.int16) - hh.astype(np.int16))
    R("G region vs half", "box", (bx0, by0, bx1, by1), "mean", round(float(d.mean()), 2), "p99",
      float(np.percentile(d, 99)), "per channel mean signed", [round(float((red[..., c].astype(float) -
                                                                             hh[..., c]).mean()), 2) for c in range(3)])
    # sharpness: the region vs the half enlarged (Laplacian variance)
    lap = lambda a: float(cv2.Laplacian(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())  # noqa: E731
    R("G sharpness", "region", round(lap(arr), 1), "half", round(lap(hh), 1))
    # pan by a little: the frame shows half-enlarged outside the region: the step at the region's edge
    vw = v.viewer
    c = QPoint(vw.width() // 2, vw.height() // 2)
    pi.viewer_drag(c, QPoint(c.x() - 700, c.y()), steps=6)
    ex.settle(1)
    g = vw.grab().toImage()
    pi.shot_viewer("recheck_region_edge_just_released")
    R("G region box now", v.src.full and v.src.full[0])
    # where the old region ends on screen: its right edge
    ex_box = v.src.full[0] if v.src.full else None
    if ex_box is not None:
        p = vw.img_to_w(ex_box[2], 0)
        R("G old region right edge on screen at x", round(p.x()), "viewer w", vw.width())
    pi.wait(lambda: v.src.full is not None and v.src.full[0] != ex_box, 200)
    pi.idle(60)
    pi.shot_viewer("recheck_region_edge_after")
R("errors", pi.slot_errors(), "popups", ex.popups)
ex.close()
