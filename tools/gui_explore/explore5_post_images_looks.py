"""post_images 3: each workflow layer shown alone (its eye ticked, the others unticked, with the mouse): its look on
screen against its thumbnail; the colours of each (sky mean per channel)."""
import numpy as np
from explore5_post_images_common import COPY, PI, R, Explorer, start, stats
from PySide6.QtWidgets import QCheckBox

start(1500)
ex = Explorer("post_images", lang="fr")
pi = PI(ex)
pi.open(COPY)
v = pi.v
pi.idle(300)


def eye(lid):
    row = next(r for r in v.rows if r.lay["id"] == lid)
    return row.findChildren(QCheckBox)[0]


ids = [lay["id"] for lay in v.layers if lay["kind"] == "workflow"]
names = {lay["id"]: lay["name"] for lay in v.layers}
for lid in ids:
    for other in ids:                                    # the eyes clicked as the user would
        want = other == lid
        if eye(other).isChecked() != want:
            ex.click(eye(other))
            ex.settle(2)
    pi.idle(300)
    a = pi.screen_array()
    th = v._lthumbs.get(lid)
    tq = None
    if th is not None:
        from PySide6.QtGui import QImage
        img = th[1].toImage().convertToFormat(QImage.Format_RGB32)
        w, h = img.width(), img.height()
        buf = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())[:, :w * 4].reshape(h, w, 4)
        tq = buf[..., [2, 1, 0]]
    R("alone", names[lid], "visible", [(lay["name"][:30], lay["visible"]) for lay in v.layers],
      "screen", stats(a) if a is not None else None, "thumb", stats(tq) if tq is not None else None)
    if a is not None:
        pi.save_png(a[::2, ::2], f"alone_{lid}")
    ex.shot(f"alone_{lid}_window")
R("undo stack", len(v.undo), "errors", pi.slot_errors(), "popups", ex.popups)
ex.close()
