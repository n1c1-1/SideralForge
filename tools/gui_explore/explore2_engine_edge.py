"""Engine shapes, round 2: graphs the window accepts on unusual data — are they run, or refused before the run with
their reason? (CPU, tiny data.)

    python explore2_engine_edge.py <scenario> [--en]
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore2_engine_window as W  # noqa: E402
from explore2_engine_window import (P, TMP, blk, click_run, cpu, dump_dialog, ex, image_project, panel_rows,  # noqa: E402
                                    panel_texts, place, set_setting, show_state, stage_runs, wait_run)
from explore_engine_common import bayer_frames  # noqa: E402

import numpy as np  # noqa: E402

from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402
from astro_gpu.core.project2 import Session  # noqa: E402


def _proj(name, lights, **kw):
    root = ex.ws / name
    p = project2.create(root, name, Target(name="M 31"))
    p.sessions.append(Session(1, "Session 1", lights=[str(lights)], **kw))
    w = p.add_workflow("Workflow", wfl.TEMPLATES["Classic stack"])
    p.set_blocks(w.id, [])
    project2.save(p)
    return root


def _run_and_tell(root, tag, timeout=200):
    show_state(tag)
    ok = click_run()
    if not ok:
        ex.shot(f"edge_{W.ARGS[0]}_refused")
        return "refused"
    ex.settle(3)
    st = wait_run(timeout)
    dump_dialog("end")
    if st != "done":
        ex.shot(f"edge_{W.ARGS[0]}_{st}", W.run_dialog())
    P("  stage_end:", stage_runs(root))
    P("  log:", W.log_rows()[:4])
    return st


def sc_two_frames():
    d = bayer_frames(TMP / "data" / "two" / "lights", n=2, seed=4)
    root = _proj("two", d)
    ex.open(root)
    cpu()
    place("registration", "stacking", "gradient")
    return _run_and_tell(root, "2 frames")


def sc_one_frame():
    d = bayer_frames(TMP / "data" / "one" / "lights", n=1, seed=4)
    root = _proj("one", d)
    ex.open(root)
    cpu()
    place("registration", "stacking")
    return _run_and_tell(root, "1 frame")


def sc_unreadable():
    """A folder of .fits files that are not FITS (text inside)."""
    d = TMP / "data" / "bad" / "lights"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        (d / f"light_{i:03d}.fits").write_text("not a fits file\n" * 50)
    root = _proj("bad", d)
    ex.open(root)
    cpu()
    ex.click_block(blk("sources") or blk("session"))
    P("  session panel:", panel_texts()[:20])
    place("registration", "stacking")
    return _run_and_tell(root, "unreadable")


def sc_dark_size():
    """Darks of another size than the lights."""
    lights = bayer_frames(TMP / "data" / "dsz" / "lights", n=5, seed=5)
    darks = bayer_frames(TMP / "data" / "dsz" / "darks", n=3, kind="dark", h=100, w=150, seed=6)
    root = _proj("dsz", lights, darks=[str(darks)])
    ex.open(root)
    cpu()
    place("calibration", "registration", "stacking")
    return _run_and_tell(root, "darks of another size")


def sc_mono_frames():
    """Monochrome frames (no Bayer pattern, a FILTER): Registration, Stacking, Gradient; Color greyed?"""
    from astropy.io import fits
    d = TMP / "data" / "mono" / "lights"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    h, w = 200, 300
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sy, sx = rng.uniform(8, h - 8, 80), rng.uniform(8, w - 8, 80)
    fl = rng.uniform(800, 20000, 80)
    for i in range(6):
        dy, dx = rng.uniform(-3, 3, 2)
        img = 1000.0 + 0.5 * xx + rng.normal(0, 15, (h, w))
        for y0, x0, a in zip(sy + dy, sx + dx, fl):
            img += a * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * 1.4 ** 2))
        hd = fits.Header()
        hd["FILTER"] = "Ha"
        hd["EXPTIME"] = 300.0
        hd["DATE-OBS"] = f"2026-10-02T22:{i:02d}:00"
        fits.writeto(d / f"Ha_{i:03d}.fits", np.clip(img, 0, 65535).astype(np.uint16), hd, overwrite=True)
    root = _proj("mono", d, filter="Ha")
    ex.open(root)
    cpu()
    place("registration", "stacking", "gradient")
    c = ex.card("color")
    P("  color card reason:", getattr(c, "reason", None))
    return _run_and_tell(root, "mono")


def sc_selection_rejects_all():
    """Selection set to keep almost nothing (its strongest setting), then Stacking."""
    d = bayer_frames(TMP / "data" / "sel" / "lights", n=5, seed=7)
    root = _proj("sel", d)
    ex.open(root)
    cpu()
    place("registration", "selection", "stacking")
    ex.click_block(blk("selection"))
    rows = [(r.r.get("name"), r.r.get("type"), r.r.get("value"), r.r.get("min"), r.r.get("max"), r.r.get("choices"))
            for r in panel_rows()]
    P("  selection rows:", rows)
    P("  selection panel:", panel_texts()[:40])
    # the keep fraction / threshold: the highest value allowed
    for r in panel_rows():
        n = r.r.get("name") or ""
        if r.r.get("type") in ("float", "int") and r.editor.isEnabled() and r.r.get("max") is not None and \
                any(k in n for k in ("min", "thresh", "weight", "keep", "fwhm", "reject")):
            set_setting(n, r.r["max"] if "min" in n or "thresh" in n or "weight" in n else r.r["min"])
            break
    P("  params:", ex.s.p.workflows[0].params)
    ex.shot("edge_selection_setting")
    return _run_and_tell(root, "selection strict")


def sc_stretched_split():
    """A stretched PNG: Split stars (Stars/background) alone, then Post."""
    root = image_project("strsplit", stretched=True)
    ex.open(root)
    cpu()
    for b in ("gradient", "denoise", "stars", "astrometry", "color"):
        P(f"  card {b}: {getattr(ex.card(b), 'reason', None)!r}")
    place("stars")
    st = _run_and_tell(root, "stretched split")
    ex.close_dialogs()
    W.go_post()
    ex.shot("edge_stretched_split_post")
    return st


def sc_stretched_split2():
    """Second look: stretched PNG, Split stars; the graph's Post block and the Post page after the run; then the starless
    image sent to Post from the block's menu."""
    root = image_project("strsplit2", stretched=True)
    ex.open(root)
    cpu()
    P("  cards at opening:", {b: getattr(ex.card(b), "reason", None) for b in ("gradient", "stars", "session")})
    ex.shot("edge_cards_at_opening")
    place("stars")
    st = _run_and_tell(root, "stretched split 2")
    ex.close_dialogs()
    show_state("after run")
    ex.shot("edge_stretched_split2_graph")
    page = W.go_post()
    ex.shot("edge_stretched_split2_post")
    ex.click(ex.w.mode_btns[0])
    ex.settle(6)
    m = ex.right_click_block(blk("etoiles"), "Sans étoiles vers Post")
    P("  menu:", [(a.get("text"), a.get("enabled")) for a in m])
    ex.settle(6)
    show_state("starless sent")
    W.go_post()
    ex.shot("edge_stretched_split2_post_after_send")
    return st


def sc_stretched_alone():
    """Stretched PNG with no block: Run, then Post (does an image of a stretched source reach Post at all?)."""
    root = image_project("stralone", stretched=True)
    ex.open(root)
    cpu()
    show_state("opened")
    st = _run_and_tell(root, "stretched alone")
    ex.close_dialogs()
    W.go_post()
    ex.shot("edge_stretched_alone_post")
    P("  saved post:", getattr(ex.s.p, "post", None))
    return st


def sc_stretched_split_layers():
    """Stretched PNG + Split: what the Post page's model holds (its layers, and why none is shown)."""
    root = image_project("strlay", stretched=True)
    ex.open(root)
    cpu()
    place("stars")
    _run_and_tell(root, "split")
    ex.close_dialogs()
    page = W.go_post()
    P("  saved post:", getattr(ex.s.p, "post", None))
    for attr in ("layers", "rows", "comp", "composition"):
        v = getattr(page, attr, None)
        if v is not None:
            P(f"  page.{attr}:", v if not callable(v) else "fn")
    P("  graph post_layers:", list(getattr(ex.view.graph, "post_layers", []) or []))


SC = {k[3:]: v for k, v in dict(globals()).items() if k.startswith("sc_")}

if __name__ == "__main__":
    name = W.ARGS[0]
    P(f"=== {name}")
    try:
        P("=== result", SC[name]())
    except Exception as e:  # noqa: BLE001
        P("!! SCENARIO EXCEPTION", type(e).__name__, e, traceback.format_exc()[-2500:])
    P("=== slot errors:", len(W.SLOT_ERRORS))
    for x in W.SLOT_ERRORS:
        P(x[-1500:])
    os._exit(0)
