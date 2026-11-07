"""post_images 5: layers by hand (show, reorder by dragging the row, rename by double-click, duplicate, delete, Ctrl+Z),
a mask painted with the brush (invert, feather, show mask), zoom 100 % and pan (the full-resolution region, its edges),
before/after, the histogram of Curves; then Save, the project opened again: the layers and their settings kept."""
import json
import sys
import time

import numpy as np
from explore5_post_images_common import COPY, PI, R, Explorer, start, stats
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QCheckBox, QPushButton

from astro_gpu.client.i18n import _
from astro_gpu.client.post3 import ContSlider

start(3000)
ex = Explorer("post_images", lang="fr")
pi = PI(ex)
pi.open(COPY)
v = pi.v
pi.idle(300)
which = sys.argv[1:] or ["layers", "mask", "zoom", "compare", "hist", "reopen"]


def row_of(lid):
    return next(r for r in v.rows if r.lay["id"] == lid)


def eye(lid):
    return row_of(lid).findChildren(QCheckBox)[0]


def diff(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    return round(float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()), 3)


def row_drag(src_row, dst_row, where="below"):
    """Press on a row's name, move with the button held, release over the lower quarter of another row (between it
    and the next) - the panel's own drag (no QDrag)."""
    a = src_row.n.geometry().center()
    g = dst_row.geometry()
    holder = v.lbox.parentWidget()
    dst = QPoint(g.center().x(), g.top() + int(g.height() * (0.9 if where == "below" else 0.1)))
    gdst = holder.mapToGlobal(dst)

    def send(kind, local, glob, btn, btns):
        QApplication.sendEvent(src_row, QMouseEvent(kind, QPointF(local), QPointF(glob), btn, btns, Qt.NoModifier))
    send(QEvent.MouseButtonPress, a, src_row.mapToGlobal(a), Qt.LeftButton, Qt.LeftButton)
    ga = src_row.mapToGlobal(a)
    for k in range(1, 9):
        gp = QPoint(int(ga.x() + (gdst.x() - ga.x()) * k / 8), int(ga.y() + (gdst.y() - ga.y()) * k / 8))
        send(QEvent.MouseMove, src_row.mapFromGlobal(gp), gp, Qt.NoButton, Qt.LeftButton)
        ex.settle(1)
    shot = ex.shot("layers_drag_hint")
    send(QEvent.MouseButtonRelease, src_row.mapFromGlobal(gdst), gdst, Qt.LeftButton, Qt.NoButton)
    ex.settle(6)
    return shot


def rename_by_dblclick(lid, new):
    r = row_of(lid)

    def h(d):
        from PySide6.QtWidgets import QLineEdit
        e = d.findChildren(QLineEdit)[0]
        e.setText(new)
        d.accept()
    ex.dialog_handler = h
    QApplication.sendEvent(r, QMouseEvent(QEvent.MouseButtonDblClick, QPointF(r.n.geometry().center()),
                                          QPointF(r.mapToGlobal(r.n.geometry().center())), Qt.LeftButton,
                                          Qt.LeftButton, Qt.NoModifier))
    ex.settle(8)


def status():
    sb = ex.w.statusBar() if hasattr(ex.w, "statusBar") else None
    return sb.currentMessage() if sb is not None else ""


wl = [lay for lay in v.layers if lay["kind"] == "workflow"]
base = wl[-1]
den = next(lay for lay in wl if lay["name"].endswith("denoise"))

if "layers" in which:
    s0 = pi.screen_array()
    # show the denoise layer on top (its eye)
    ex.click(eye(den["id"]))
    pi.idle(120)
    s1 = pi.screen_array()
    R("L show", "diff", diff(s0, s1), "visible", [(x["name"][:24], x["visible"]) for x in v.layers])
    # reorder: drag the denoise row below the base
    order0 = [x["name"] for x in v.layers]
    pi.idle(30)
    row_drag(row_of(den["id"]), row_of(base["id"]), "below")
    pi.idle(120)
    s2 = pi.screen_array()
    order1 = [x["name"] for x in v.layers]
    R("L reorder", order0, "->", order1, "diff vs before", diff(s1, s2), "undo", len(v.undo))
    ex.shot("layers_reordered")
    ex.undo()
    pi.idle(120)
    s3 = pi.screen_array()
    R("L reorder undo", [x["name"] for x in v.layers], "diff vs s1", diff(s1, s3))
    # duplicate (Ctrl+J), rename it by double-click, delete it, Ctrl+Z, Ctrl+Y
    v.select(den["id"], "layer")
    ex.settle(4)
    ex.key(Qt.Key_J, v, Qt.ControlModifier)
    pi.idle(120)
    dup = v.layer(v.sel[0])
    R("L dup", dup["name"], [x["name"] for x in v.layers])
    rename_by_dblclick(dup["id"], "Ma copie")
    R("L rename copy", [x["name"] for x in v.layers], "rows", pi.rows_desc()[:3], "status", status())
    # rename a workflow layer: the workflow follows
    rename_by_dblclick(base["id"], "Base iris")
    pi.idle(30)
    R("L rename wf", [x["name"] for x in v.layers], "paths", [q.name for q in ex.s.paths], "status", status())
    ex.shot("layers_renamed")
    ex.undo()
    pi.idle(30)
    R("L rename wf undo", [x["name"] for x in v.layers], "paths", [q.name for q in ex.s.paths])
    s4 = pi.screen_array()
    v.select(dup["id"], "layer")
    ex.settle(4)
    ex.click(v.b_del)
    pi.idle(120)
    s5 = pi.screen_array()
    R("L delete", [x["name"] for x in v.layers], "diff", diff(s4, s5))
    ex.undo()
    pi.idle(120)
    s6 = pi.screen_array()
    R("L delete undo", [x["name"] for x in v.layers], "diff vs before delete", diff(s4, s6))
    ex.redo()
    pi.idle(120)
    R("L delete redo", [x["name"] for x in v.layers], "diff vs after delete", diff(s5, pi.screen_array()))
    # blend mode and opacity of the top layer
    v.select(den["id"], "layer")
    pi.idle(30)
    op = next(s for s in v.right.findChildren(ContSlider) if s.name.text() == _("Opacity"))
    i0 = pi.mark()
    t = pi.drag_slider(op, 0.5)
    R("L opacity 50", pi.time_to_image(i0, t), "row", pi.rows_desc()[:2])
    i0 = pi.mark()
    t = pi.drag_slider(op, 0.2)
    R("L opacity 20", pi.time_to_image(i0, t))
    ex.shot("layers_opacity")

if "mask" in which:
    # the denoise layer visible on top at 100 %, a brush mask: it shows only where painted
    lay = v.layer(den["id"])
    if not lay["visible"]:
        ex.click(eye(den["id"]))
    v.select(den["id"], "layer")
    pi.idle(60)
    if lay["opacity"] != 100:
        op = next(s for s in v.right.findChildren(ContSlider) if s.name.text() == _("Opacity"))
        op.reset()
        pi.idle(60)
    s_top = pi.screen_array()
    ex.click(v.b_maskadd)
    pi.idle(60)
    add = next(b for b in v.right.findChildren(QPushButton) if b.isVisible() and b.text() == _("+ Add to mask"))
    ex.menu_pick = _("Brush")
    ex.click(add)
    pi.idle(60)
    R("M parts", [p["src"] for p in lay["mask"]["parts"]], "viewer mode", v.viewer.mode, "hint", v.hint.text())
    vw = v.viewer
    W, H = v.src.canvas
    a = vw.img_to_w(W * 0.30, H * 0.45).toPoint()
    b = vw.img_to_w(W * 0.65, H * 0.55).toPoint()
    i0 = pi.mark()
    t = pi.viewer_drag(a, b, steps=14)
    r1 = pi.time_to_image(i0, t)
    pi.idle(60)
    m = v.mask1
    R("M stroke", r1, "mask1", None if m is None else (m.shape, round(float(m.mean()), 4), float(m.max())),
      "strokes", len(lay["mask"]["parts"][0]["strokes"]))
    s_m = pi.screen_array()
    pi.save_png(s_m, "mask_brush_screen")
    ex.shot("mask_brush_window")
    # show mask (M), then off
    ex.click(v.b_mask)
    pi.idle(30)
    pi.shot_viewer("mask_shown")
    ex.click(v.b_mask)
    pi.idle(30)
    # invert
    inv = next(c for c in v.right.findChildren(QCheckBox) if c.isVisible() and c.text() == _("Invert"))
    i0 = pi.mark()
    t = time.monotonic()
    ex.click(inv)
    r2 = pi.time_to_image(i0, t)
    pi.idle(60)
    m2 = v.mask1
    R("M invert", r2, "mask1 mean", None if m2 is None else round(float(m2.mean()), 4),
      "sum check", None if m is None or m2 is None else round(float(np.abs(m + m2 - 1).mean()), 4))
    pi.save_png(pi.screen_array(), "mask_inverted_screen")
    # feather
    fe = next(s for s in v.right.findChildren(ContSlider) if s.isVisible() and s.name.text() == _("Feather"))
    for k, frac in enumerate((0.7, 0.35)):
        i0 = pi.mark()
        t = pi.drag_slider(fe, frac)
        r3 = pi.time_to_image(i0, t)
        pi.idle(60)
        m3 = v.mask1
        R(f"M feather #{k + 1}", fe.val.text(), r3, "mask1 mean", None if m3 is None else round(float(m3.mean()), 4),
          "soft pixels", None if m3 is None else round(float(((m3 > 0.02) & (m3 < 0.98)).mean()), 4))
    ex.click(v.b_mask)
    pi.idle(30)
    pi.shot_viewer("mask_feathered_shown")
    ex.click(v.b_mask)
    pi.idle(30)
    # H-alpha and luminance parts
    for src in ("H-alpha", "Luminance range"):
        add = next(b for b in v.right.findChildren(QPushButton) if b.isVisible() and b.text() == _("+ Add to mask"))
        ex.menu_pick = _(src)
        i0 = pi.mark()
        t = time.monotonic()
        ex.click(add)
        r4 = pi.time_to_image(i0, t)
        pi.idle(60)
        m4 = v.mask1
        R("M part", src, r4, "parts", [p["src"] for p in lay["mask"]["parts"]], "mask1 mean",
          None if m4 is None else round(float(m4.mean()), 4), "banner", v.viewer.banner)
    ex.click(v.b_mask)
    pi.idle(30)
    pi.shot_viewer("mask_parts_shown")
    ex.click(v.b_mask)
    ex.shot("mask_parts_window")

if "zoom" in which:
    v.viewer.fit()
    pi.idle(60)
    full_before = pi.screen_array()
    i0 = pi.mark()
    t = time.monotonic()
    ex.click(v.b_100)
    ok = pi.wait(lambda: v.src.full is not None, 300)
    R("Z 100%", "region arrived", ok, round(time.monotonic() - t, 2), "box", v.src.full and v.src.full[0],
      "events", pi.after(i0, t))
    pi.idle(60)
    pi.shot_viewer("zoom100")
    # pan: drag the image by 600 px left and 300 px up; the frame right after the release, then after the region
    vw = v.viewer
    c = QPoint(vw.width() // 2, vw.height() // 2)
    i0 = pi.mark()
    t = pi.viewer_drag(c, QPoint(c.x() - 600, c.y() - 300), steps=10)
    ex.settle(1)
    pi.shot_viewer("zoom100_pan_just_released")
    box0 = v.src.full and v.src.full[0]
    ok = pi.wait(lambda: v.src.full is not None and v.src.full[0] != box0, 300)
    R("Z pan", "new region", ok, round(time.monotonic() - t, 2), "box", v.src.full and v.src.full[0])
    pi.idle(60)
    pi.shot_viewer("zoom100_pan_region")
    # wheel zoom in twice at the centre
    for k in range(3):
        QApplication.sendEvent(vw, QWheelEvent(QPointF(c), QPointF(vw.mapToGlobal(c)), QPoint(0, 0), QPoint(0, 120),
                                               Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
        ex.settle(2)
    pi.idle(60)
    R("Z wheel", v.zoom_lab.text())
    pi.shot_viewer("zoom_wheel")
    # a slider dragged while zoomed in at 100 %: the time to the half image and to the full-resolution region
    ex.click(v.b_100)
    pi.idle(120)
    v.select(base["id"], "layer")
    pi.idle(60)
    sky = next(s for s in v.right.findChildren(ContSlider) if s.isVisible() and s.name.text() == _("Sky"))
    for k, frac in enumerate((0.75, 0.35)):
        box0 = v.src.full
        i0 = pi.mark()
        t = pi.drag_slider(sky, frac)
        r = pi.time_to_image(i0, t, 300)
        okr = pi.wait(lambda: any(e[1] == "res" and e[2] == "region" and e[0] > t for e in pi.events[i0:]), 300)
        treg = next((round(e[0] - t, 2) for e in pi.events[i0:] if e[1] == "res" and e[2] == "region" and e[0] > t),
                    None)
        R(f"Z sky at 100% #{k + 1}", sky.val.text(), r, "region after release (s)", treg, "events", pi.after(i0, t))
        pi.idle(60)
    pi.shot_viewer("zoom100_sky")
    sky.reset()
    pi.idle(60)
    ex.key(Qt.Key_F, v)
    pi.idle(60)
    R("Z fit (F)", v.zoom_lab.text())

if "compare" in which:
    v.viewer.fit()
    pi.idle(30)
    i0 = pi.mark()
    t = time.monotonic()
    ex.click(v.b_cmp)
    ok = pi.wait(lambda: v.before.img1 is not None, 300)
    R("C before", ok, round(time.monotonic() - t, 2), "hint", v.hint.text())
    pi.idle(60)
    pi.shot_viewer("compare_on")
    bef, aft = v.before.img1, pi.screen_array()
    R("C before vs after", "shapes", None if bef is None else bef.shape, aft.shape, "diff", diff(bef, aft),
      "before stats", None if bef is None else stats(bef), "after", stats(aft))
    if bef is not None:
        pi.save_png(bef, "compare_before_img")
        pi.save_png(aft, "compare_after_img")
    vw = v.viewer
    xs = int(vw.width() * vw.curtain)
    pi.viewer_drag(QPoint(xs, vw.height() // 2), QPoint(int(vw.width() * 0.8), vw.height() // 2), steps=6)
    ex.settle(4)
    R("C curtain", round(vw.curtain, 2))
    pi.shot_viewer("compare_curtain")
    ex.click(v.b_cmp)
    pi.idle(30)

if "hist" in which:
    v.select(base["id"], "layer")
    pi.idle(30)
    for key, name in (("curves", "Curves"), ("ghs", "Generalised hyperbolic stretch"), ("histogram",
                                                                                         "Histogram transformation")):
        ex.menu_pick = _(name)
        ex.click(v.b_filter)
        pi.idle(120)
        fl = v.layer(v.sel[0])
        h = v._hists.get(fl["id"])
        ed = v.curves_editor or v.function_graph
        R("H", key, fl["name"], "hist", None if h is None else {k: (len(c), int(np.sum(c))) for k, c in h.items()},
          "editor", type(ed).__name__ if ed else None, "editor hist", ed is not None and getattr(ed, "hist", None)
          is not None)
        ex.shot(f"hist_{key}_panel", v.right)
        ex.click(v.b_del)
        pi.idle(60)

if "reopen" in which:
    v.select(base["id"], "layer")
    ex.menu_pick = _("Saturation")
    ex.click(v.b_filter)
    pi.idle(60)
    sat = v.layer(v.sel[0])
    s = next(x for x in v.right.findChildren(ContSlider) if x.isVisible() and x.name.text() != _("Opacity"))
    pi.drag_slider(s, 0.8)
    pi.idle(120)
    before = json.loads(json.dumps(v.layers, default=str))
    scr = pi.screen_array()
    R("O before save", [(x["name"], x["visible"], x["mode"], x["opacity"], x.get("params"), bool(x.get("mask")))
                        for x in before])
    ex.key(Qt.Key_S, ex.w, Qt.ControlModifier)
    ex.settle(10)
    R("O saved", "popups", ex.popups[-2:], "dirty", getattr(ex.s, "dirty", None))
    ex.w.open_project(COPY)
    ex.settle(10)
    ex.click(ex.w.mode_btns[2])
    pi.hook()
    v = pi.v
    pi.wait(lambda: v.src.img1 is not None, 300)
    pi.idle(300)
    after = json.loads(json.dumps(v.layers, default=str))
    scr2 = pi.screen_array()

    def key(x):
        return {k: x.get(k) for k in ("name", "kind", "visible", "mode", "opacity", "params", "mask", "target",
                                      "sliders", "pid")}
    same = [key(a) == key(b) for a, b in zip(before, after)]
    R("O reopened", "n", len(before), len(after), "same", same, "screen diff", diff(scr, scr2))
    for a, b in zip(before, after):
        if key(a) != key(b):
            R("O differs", key(a), "!=", key(b))
    ex.shot("reopened")
R("errors", pi.slot_errors(), "popups", ex.popups, "modals", ex.modals, "menus", len(ex.menus))
ex.close()
