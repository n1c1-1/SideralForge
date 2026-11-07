"""post_images 1: the copy opened, Post shown: its layers, the first image, its look (screen vs the engine)."""
import time

from explore5_post_images_common import COPY, PI, R, Explorer, start, stats

start(1200)
ex = Explorer("post_images", lang="fr")
pi = PI(ex)
ok = pi.open(COPY)
v = pi.v
R("layers", pi.layers_desc())
R("rows", pi.rows_desc())
pi.idle(300)
R("events", pi.after(0, pi.events[0][0] if pi.events else time.monotonic()))
R("zoom", v.zoom_lab.text(), "scale", round(v.viewer.scale, 3), "hint", v.hint.text())
a = pi.screen_array()
if a is not None:
    R("screen", a.shape, stats(a))
    pi.save_png(a, "open_screen_half")
ex.shot("open_window")
pi.shot_viewer("open_viewer")
R("errors", pi.slot_errors(), "popups", ex.popups, "modals", ex.modals)
ex.close()
