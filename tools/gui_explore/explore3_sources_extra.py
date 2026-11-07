"""sources_files, round 3: the few round-2 points no explore2 script logs directly: the panel of a template project
with nothing selected (still « Images » ?), two Image blocks' names, the block chosen after a template."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore3_sources_common import TMP, Explorer, panel, ptexts, start, visible, wait  # noqa: E402
from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtWidgets import QLabel  # noqa: E402

start(400)
LOG = []


def log(*a):
    t = " ".join(str(x) for x in a)
    LOG.append(t)
    print(t, flush=True)


ex = Explorer("sources_files", lang="fr")
OUT = ex.out


def new_from(name, template):
    ex.open()
    got = {}

    def h(gal):
        got["cls"] = gal.__class__.__name__
        if gal.__class__.__name__ != "TemplateGallery":
            gal.reject()
            return
        from astro_gpu.client.gallery import GALLERY
        tab = next((x["tab"] for x in GALLERY if x["key"] == template), None)
        keys = getattr(gal, "tab_keys", [])
        if tab in keys:
            gal.tabs.setCurrentIndex(keys.index(tab))
            ex.settle(2)
        t = next((c.t for c in gal.cards if c.t.get("key") == template), None)
        got["found"] = t is not None
        if t is not None:
            gal.pick(t)
        gal.ok.click()
    ex.dialog_handler = h
    root = ex.ws / name
    ex.w.create_project(str(root), name, "M 31")
    ex.settle(15)
    ex.dialog_handler = None
    wait(ex, 1.0)
    return root, got


def empty_click():
    v = ex.view
    vp = v.viewport()
    for fy in (0.95, 0.05, 0.5):
        for fx in (0.95, 0.05, 0.5):
            p = QPoint(int(vp.width() * fx), int(vp.height() * fy))
            if not v.items(p):
                ex.click(vp, p)
                wait(ex, 0.4)
                return p
    return None


# 1. template « Complet » / « Plusieurs nuits »: block chosen at creation, then the panel with nothing selected
from astro_gpu.client.gallery import GALLERY  # noqa: E402
log("gallery keys", [(x["key"], x["tab"]) for x in GALLERY])
for key in ("full", "nights"):
    if not any(x["key"] == key for x in GALLERY):
        continue
    root, got = new_from(f"tpl_{key}", key)
    log(key, got, "sel", ex.s.sel_step, "panel", ptexts(ex)[:8])
    ex.shot(f"x_tpl_{key}_created")
    p = empty_click()
    log(key, "after empty click", p, "sel", ex.s.sel_step, "panel", ptexts(ex)[:8])
    ex.shot(f"x_tpl_{key}_nothing_selected")
    ex.w.close_project()
    wait(ex, 0.4)

# 2. two Image blocks: their names
root = ex.project("image", "two_images")
ex.open(root)
wait(ex, 0.8)
z = ex.new_flow_zone()
rec = ex.drag_palette("image", ex.ghost_center(z)) if z is not None else None
wait(ex, 1.0)
log("second image drop", rec, [(q.name, q.source) for q in ex.s.paths])
log("image blocks", [(b.title, b.summary) for b in ex.blocks() if b.node.key == "image"])
ex.shot("x_two_images")
ex.w.close_project()
wait(ex, 0.3)

(OUT / "logs" / "extra_log.txt").write_text("\n".join(LOG), encoding="utf-8")
print("DONE", TMP)
ex.close()
