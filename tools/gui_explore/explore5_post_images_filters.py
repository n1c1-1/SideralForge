"""post_images 2: the workflow layer's sliders, then each filter added from "+ Filtre", its first slider dragged with
the mouse twice: the time from the release to the updated image, whether the image changed, how it looks."""
import sys
import time

import numpy as np
from explore5_post_images_common import COPY, PI, R, Explorer, dump_json, start, stats
from PySide6.QtWidgets import QApplication

from astro_gpu.client.i18n import _
from astro_gpu.client.post3 import ContSlider
from astro_gpu.client.postengine import FILTERS

start(3000)
ex = Explorer("post_images", lang="fr")
pi = PI(ex)
pi.open(COPY)
v = pi.v
pi.idle(300)
VUES = ex.out / "vues"
VUES.mkdir(exist_ok=True)
which = sys.argv[1:] or ["sliders", "filters"]
table = {}


def small(a, name):
    import cv2
    h, w = a.shape[:2]
    b = cv2.resize(a, (900, int(900 * h / w)), interpolation=cv2.INTER_AREA)
    from PIL import Image
    Image.fromarray(b).save(VUES / f"{name}.jpg", quality=90)


def sliders():
    return [s for s in v.right.findChildren(ContSlider) if s.isVisible()]


def drag_measure(cs, frac, tag):
    before = pi.screen_array()
    i0 = pi.mark()
    t_rel = pi.drag_slider(cs, frac)
    res = pi.time_to_image(i0, t_rel, 300)
    after = pi.screen_array()
    drafts = sum(1 for e in pi.events[i0:] if e[1] == "res" and e[2] == "draft")
    diff = None if before is None or after is None or before.shape != after.shape else \
        float(np.abs(before.astype(np.int16) - after.astype(np.int16)).mean())
    R(tag, f"value {cs.val.text()!r}", "first", res["first"], "final", res["final"], "drafts while dragging", drafts,
      "mean |diff|", None if diff is None else round(diff, 2), "banner", v.viewer.banner)
    return res, diff, after


if "sliders" in which:
    base = next(lay for lay in reversed(v.layers) if lay["visible"] and lay["kind"] == "workflow")
    v.select(base["id"], "layer")
    pi.idle(120)
    R("base", base["name"], [(s.name.text(), s.val.text(), s.s.isEnabled(), s.why.text()) for s in sliders()])
    for s in sliders():
        if not s.s.isEnabled():
            continue
        name = s.name.text()
        for k, frac in enumerate((0.8, 0.3)):
            res, diff, after = drag_measure(s, frac, f"slider {name} #{k + 1}")
            table[f"layer:{name}#{k + 1}"] = {"first": res["first"], "final": res["final"], "diff": diff}
            if after is not None:
                small(after, f"slider_{len(table):02d}_{name}_{k + 1}")
        s.reset()                                        # double-click: back to the suggestion
        pi.idle(120)
        # the sliders are rebuilt with the panel: take them again
    ex.shot("sliders_done")

if "filters" in which:
    keys = [k for k, f in FILTERS.items() if not f.get("hidden") and f["params"] and
            f["params"][0]["type"] == "float"]
    only = [a for a in sys.argv[1:] if a in FILTERS]
    keys = only or keys
    base = next(lay for lay in reversed(v.layers) if lay["visible"] and lay["kind"] == "workflow")
    for key in keys:
        v.select(base["id"], "layer")
        pi.idle(60)
        ref = pi.screen_array()
        name = _(FILTERS[key]["name"])
        ex.menu_pick = name
        n_before = len(v.layers)
        ex.click(v.b_filter)
        ex.settle(6)
        if len(v.layers) == n_before:
            R("filter", key, "NOT ADDED by the menu", ex.menus[-1:] and [d["text"] for d in ex.menus[-1]])
            continue
        fl = v.layer(v.sel[0])
        pi.idle(120)
        added = pi.screen_array()
        same = ref is not None and added is not None and ref.shape == added.shape and \
            float(np.abs(ref.astype(np.int16) - added.astype(np.int16)).mean())
        sl = [s for s in sliders() if s.name.text() not in (_("Opacity"),)]
        R("filter", key, name, "added", fl["name"], "|diff| at defaults", same, "sliders",
          [(s.name.text(), s.val.text()) for s in sl])
        if not sl:
            continue
        s0 = sl[0]
        rows = {}
        for k, frac in enumerate((0.6, 0.3)):
            s0 = [s for s in sliders() if s.name.text() == s0.name.text()][0]
            res, diff, after = drag_measure(s0, frac, f"{key} {s0.name.text()} #{k + 1}")
            rows[k + 1] = {"first": res["first"], "final": res["final"], "diff": diff, "value": s0.val.text()}
            if after is not None:
                small(after, f"filter_{key}_{k + 1}")
        if ref is not None:
            small(ref, f"filter_{key}_0_ref")
        table[key] = rows
        # Delete (the button), then back
        ex.click(v.b_del)
        pi.idle(120)
        R("filter", key, "deleted; layers", len(v.layers), "errors", pi.slot_errors())
    dump_json(ex.out / "timings_filters.json", table)
R("table", table)
R("errors", pi.slot_errors(), "popups", ex.popups, "modals", ex.modals)
ex.close()
