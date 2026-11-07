"""sources_files, round 3: the Fusion chosen, saved, reopened; the « Image déjà étirée » template just created; Ctrl+Z
right after a template (twice); a stretched 8-bit TIFF chosen in an Image."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore3_sources_common import TMP, Explorer, _, button, panel, ptexts, start, undo, visible, wait  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton  # noqa: E402

start(500)
LOG = []


def log(*a):
    t = " ".join(str(x) for x in a)
    LOG.append(t)
    print(t, flush=True)


ex = Explorer("sources_files", lang="fr")
ONLY = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None


def want(k):
    return ONLY is None or k in ONLY


def where():
    s = ex.s
    p = s.path
    title = next((w.text() for w in visible(panel(ex), QLabel) if w.text()), "")
    ed = next((w.text() for w in visible(panel(ex), QLineEdit)), "")
    return {"flow": p.name if p else None, "block": s.sel_step, "panel": ed or title}


def new_from(name, template, image=None):
    ex.open()

    def h(gal):
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
        if t is not None:
            gal.pick(t)
            if image is not None and hasattr(gal, "file"):
                gal.file.setText(str(image))
                ex.settle(2)
        gal.ok.click()
    ex.dialog_handler = h
    root = ex.ws / name
    ex.w.create_project(str(root), name, "M 31")
    ex.settle(15)
    ex.dialog_handler = None
    wait(ex, 1.0)
    return root


# 1. the Fusion chosen, a change, Ctrl+S, close, reopen
if want("fusion"):
    from astro_gpu.core import project2, workflow as wfl  # noqa: E401
    from astro_gpu.core.project import Target
    for rep in (1, 2):
        name = f"fusReopen{rep}"
        root = ex.ws / name
        p = project2.create(root, name, Target(name="M 31"))
        p.stack_mode = "per_session"
        for i, f in enumerate(("Ha", "OIII")):
            p.sessions.append(project2.Session(i + 1, f, lights=[str(ex.fits_frames(f"{name}/{f}", 4, f))], filter=f))
        project2.save(p)
        for i, f in enumerate(("Ha", "OIII")):
            w = p.add_workflow(f, wfl.TEMPLATES["Classic stack"])
            w.sessions = [i + 1]
            p.set_blocks(w.id, [wfl.node(b) for b in ("calibration", "registration", "stacking", "astrometry")])
        project2.save(p)
        ex.open(root)
        wait(ex, 1.0)
        ex.double_click_card("fusion")
        wait(ex, 1.0)
        fb = ex.block("fusion")
        if fb is None:
            log("fusion not placed", [b.node.key for b in ex.blocks()])
            continue
        ex.click_block(fb)
        wait(ex, 0.4)
        chosen = where()
        ex.key(Qt.Key_S, ex.w, Qt.ControlModifier)
        wait(ex, 0.5)
        ex.w.close_project()
        wait(ex, 0.5)
        ex.w.open_project(root)
        wait(ex, 1.5)
        back = where()
        log("fusion reopen", rep, chosen, "->", back)
        ex.shot(f"x3_fusion_reopened_{rep}")
        ex.w.close_project()
        wait(ex, 0.3)

# 2. the « Image déjà étirée » template, a PNG
if want("stretched"):
    from PIL import Image as PI
    import numpy as np
    for rep in (1, 2):
        png = TMP / "data" / f"pic{rep}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        PI.fromarray((np.random.default_rng(rep).random((90, 120, 3)) * 255).astype(np.uint8)).save(png)
        new_from(f"etiree{rep}", "image_stretched", png)
        log("stretched", rep, [(q.name, q.source) for q in ex.s.paths], [b.node.key for b in ex.blocks()],
            "run", ex.w.bt_run.isEnabled(), ex.w.bt_run.text(), ex.w.bt_run.toolTip(), "status", ex.w.status.text(),
            "post", [(b.title, b.summary) for b in ex.blocks() if b.node.key == "final"])
        ex.shot(f"x3_stretched_template_{rep}")
        ex.w.close_project()
        wait(ex, 0.3)

# 3. Ctrl+Z right after a template
if want("tplundo"):
    for rep, key in ((1, "classic"), (2, "full")):
        new_from(f"tplundo{rep}", key)
        b0 = sorted(b.node.key for b in ex.blocks())
        st = (len(ex.s._undo), len(ex.s._redo))
        how = undo(ex)
        b1 = sorted(b.node.key for b in ex.blocks())
        log("tplundo", rep, key, st, how, b0, "->", b1, "status", ex.w.status.text())
        ex.shot(f"x3_template_ctrl_z_{rep}")
        ex.w.close_project()
        wait(ex, 0.3)

# 4. a stretched 8-bit TIFF chosen in an Image (twice)
if want("tiff"):
    from PIL import Image as PI
    import numpy as np
    for rep in (1, 2):
        t8 = TMP / "data" / f"photo8_{rep}.tif"
        PI.fromarray((np.random.default_rng(rep).random((80, 100, 3)) * 255).astype(np.uint8)).save(t8)
        root = ex.project("image", f"tiff{rep}")
        ex.open(root)
        wait(ex, 0.8)
        ex.click_block(ex.block("image"))
        wait(ex, 0.3)
        b = button(panel(ex), _("Change image…"), QPushButton) or button(panel(ex), _("Choose image…"), QPushButton)
        ex.file_queue = [str(t8)]
        ex.click(b)
        wait(ex, 0.6)
        log("tiff8", rep, [(q.name, q.source) for q in ex.s.paths], [t for t in ptexts(ex)][:14])
        ex.shot(f"x3_tiff8_{rep}")
        ex.w.close_project()
        wait(ex, 0.3)

(ex.out / "logs" / f"extra3_log_{'_'.join(sorted(ONLY or ['all']))}.txt").write_text("\n".join(LOG), encoding="utf-8")
print("DONE", TMP)
ex.close()
