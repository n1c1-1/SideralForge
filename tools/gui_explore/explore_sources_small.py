"""sources_and_files, part 5: the same screens at 1366×768 (the session panel full of files and warnings, the Image
panel, a LRGB Fusion's panel, the template gallery, the import dialog), and in English (texts left in French, cut
texts)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_sources_common import (  # noqa: E402
    TMP, Dispatch, Explorer, Notes, _, cut_texts, graph_texts, look, menu_button, overlaps, panel, ptexts, start,
    texts, visible, wait,
)
from PySide6.QtWidgets import QPushButton  # noqa: E402

from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

LANG = sys.argv[1] if len(sys.argv) > 1 else "fr"
SIZE = (1366, 768) if LANG == "fr" else (1600, 950)
start(1200)
ex = Explorer("sources_files", lang=LANG, size=SIZE)
N = Notes(ex)
TAG = f"{LANG}_{SIZE[0]}"
WHERE = "1366×768" if LANG == "fr" else "en anglais"


def log(*a):
    print(" ".join(str(x) for x in a), flush=True)


def lrgb_project():
    root = ex.ws / "small_lrgb"
    p = project2.create(root, "small_lrgb", Target(name="M 31"))
    p.stack_mode = "per_session"
    for i, f in enumerate(("L", "R", "G", "B")):
        p.sessions.append(project2.Session(i + 1, f"Nuit du 12 mars — filtre {f}", lights=[
            str(ex.fits_frames(f"small/{f}", 3, f))], filter=f))
    project2.save(p)
    for i, f in enumerate(("L", "R", "G", "B")):
        w = p.add_workflow(f"Workflow très long du filtre {f}", wfl.TEMPLATES["Classic stack"])
        w.sessions = [i + 1]
        p.set_blocks(w.id, [wfl.node(b) for b in ("calibration", "registration", "stacking", "astrometry")])
    project2.save(p)
    return root


# ---- a session full of files and warnings
root = ex.project("jpg_mistake", "small_jpg")
ex.open(root)
wait(ex, 1.0)
ex.w.activateWindow()
it = ex.block("session") or ex.block("sources")
ex.click_block(it)
wait(ex, 0.5)
for k in range(4):                       # the panel is built again after each click
    rows = [x for x in visible(panel(ex), QPushButton) if "📁" in x.text() and x.text().startswith("▸")]
    if not rows:
        break
    ex.click(rows[0])
    wait(ex, 0.4)
    it = ex.block("session") or ex.block("sources")
    ex.click_block(it)
    wait(ex, 0.3)
it = ex.block("session") or ex.block("sources")
ex.click_block(it)
wait(ex, 0.5)
ex.shot(f"small_{TAG}_session_panel")
log("session panel", ptexts(ex)[:30])
look(ex, N, f"panneau Session ({WHERE})", steps=[f"fenêtre {SIZE[0]}×{SIZE[1]}", "projet de 14 ARW et un JPEG",
                                                   "clic sur la Session, dossiers ouverts"])
look(ex, N, f"fenêtre entière, Graphe ({WHERE})", ex.w, steps=[f"fenêtre {SIZE[0]}×{SIZE[1]}", "Session choisie"])

# ---- the Image panel
ex.double_click_card("image")
wait(ex, 1.0)
ex.shot(f"small_{TAG}_image_panel")
look(ex, N, f"panneau Image ({WHERE})", steps=[f"fenêtre {SIZE[0]}×{SIZE[1]}", "Image placée"])

# ---- a LRGB fusion with long names
ex.w.close_project()
wait(ex, 0.4)
ex.open(lrgb_project())
wait(ex, 1.0)
ex.double_click_card("fusion")
wait(ex, 1.0)
it = ex.block("fusion")
if it is not None:
    ex.click_block(it)
    wait(ex, 0.5)
ex.shot(f"small_{TAG}_fusion_panel")
log("fusion panel", ptexts(ex)[:40])
look(ex, N, f"panneau Fusion LRVB, noms longs ({WHERE})", steps=[f"fenêtre {SIZE[0]}×{SIZE[1]}",
                                                                  "4 filtres L, R, G, B aux noms longs", "Fusion placée"])
ex.shot(f"small_{TAG}_fusion_graph")
look(ex, N, f"graphe d'une Fusion LRVB ({WHERE})", ex.w, steps=[f"fenêtre {SIZE[0]}×{SIZE[1]}", "Fusion LRVB placée"])

# ---- the gallery and the import dialog
ex.w.close_project()
wait(ex, 0.4)
seen = {}


def gal(d):
    for i in range(d.tabs.count()):
        d.tabs.setCurrentIndex(i)
        ex.settle(3)
        ex.shot(f"small_{TAG}_gallery_{d.tab_keys[i]}", d)
        seen.setdefault("cut", []).extend(cut_texts(d))
        seen.setdefault("ov", []).extend(overlaps(d))
        seen.setdefault("size", (d.width(), d.height()))
    d.reject()


def npd(d):
    d.name.setText("Petit écran")
    ex.shot(f"small_{TAG}_new_project", d)
    seen["npd_cut"] = cut_texts(d)
    d.ok.click()


disp = Dispatch(ex, NewProjectDialog=npd, TemplateGallery=gal)
ex.click(next(x for x in visible(ex.w.home, QPushButton) if x.text() == _("New project…")))
wait(ex, 1.5)
disp.stop()
log("gallery", seen)
if seen.get("cut"):
    N.note("mineur", f"Galerie des modèles : texte coupé ({WHERE})", "; ".join(sorted(set(seen["cut"])))[:800],
           steps=[f"fenêtre {SIZE[0]}×{SIZE[1]}", "Nouveau projet…", "galerie"])
if seen.get("ov"):
    N.note("mineur", f"Galerie des modèles : textes superposés ({WHERE})", str(sorted(set(seen["ov"])))[:800],
           steps=[f"fenêtre {SIZE[0]}×{SIZE[1]}", "Nouveau projet…", "galerie"])
if seen.get("size") and (seen["size"][0] > SIZE[0] or seen["size"][1] > SIZE[1]):
    N.note("gênant", f"Galerie des modèles plus grande que l'écran ({WHERE})", f"taille {seen['size']}",
           steps=[f"écran {SIZE[0]}×{SIZE[1]}", "Nouveau projet…"])
if ex.w.project is not None:
    # the import dialog inside the project (File > Import folder…)
    lights = ex.fits_frames("small_import/lights", 3)
    imp = {}

    def idlg(d):
        d.finish_now()
        ex.settle(8)
        ex.shot(f"small_{TAG}_import_dialog", d)
        imp["cut"] = cut_texts(d)
        imp["ov"] = overlaps(d)
        imp["size"] = (d.width(), d.height())
        imp["texts"] = texts(d)
        d.reject()

    disp = Dispatch(ex, ImportDialog=idlg)
    ex.file_queue = [str(lights)]
    ex.w.add_sources()
    wait(ex, 1.0)
    disp.stop()
    log("import", {k: v for k, v in imp.items() if k != "texts"})
    if imp.get("cut"):
        N.note("mineur", f"Fenêtre d'import : texte coupé ({WHERE})", "; ".join(imp["cut"])[:800],
               steps=["Fichier > Importer un dossier…"])
    if LANG == "en":
        fr = [t for t in imp.get("texts", []) if any(w in t for w in (" les ", " des ", "Aucune", "brutes"))]
        if fr:
            N.note("mineur", "Interface en anglais : textes français dans la fenêtre d'import", str(fr)[:600],
                   steps=["langue anglaise", "Fichier > Importer un dossier…"])
N.exceptions(["voir le journal du script explore_sources_small.py"])
ex.close()
print("DONE", TMP)
