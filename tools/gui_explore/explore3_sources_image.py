"""sources_and_files, part 3: the Image source. Placed empty, "Choose the image…" (FITS, PNG, JPG, TIFF, a missing file,
a file that is not an image), its kind (Linear / Stretched) and what may follow it, the file changed after blocks were
added, deleted (panel, menu, cross, Del, dragged to the palette), Ctrl+Z / Ctrl+Y, a parallel flow."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from explore3_sources_common import (  # noqa: E402
    TMP, Explorer, Notes, _, button, graph_texts, look, missing_fr, panel, ptexts, start, texts, undo, visible, wait,
)
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QFileDialog, QPushButton  # noqa: E402

start(1500)
ex = Explorer("sources_files")
N = Notes(ex)
LOG: list[str] = []
ASKS: list[tuple] = []
_orig_open = QFileDialog.getOpenFileName


def _rec_open(*a, **k):
    ASKS.append(tuple(str(x) for x in a[1:4]))
    return _orig_open(*a, **k)


QFileDialog.getOpenFileName = staticmethod(_rec_open)


def log(*a):
    t = " ".join(str(x) for x in a)
    LOG.append(t)
    print(t, flush=True)


fits_img = ex.image_file("stack.fits")
png = ex.image_file("final.png")
jpg = ex.image_file("final.jpg")
d = TMP / "data" / "images"
from PIL import Image  # noqa: E402

Image.fromarray((np.random.default_rng(1).random((120, 160, 3)) * 255).astype(np.uint8)).save(d / "final.tif")
tif = d / "final.tif"
txt = d / "notes.txt"
txt.write_text("not an image", encoding="utf-8")
missing = d / "gone.fits"
fits2 = ex.image_file("other.fits")


def img_block():
    return ex.block("image")


def sel_image():
    it = img_block()
    if it is not None:
        ex.click_block(it)
        wait(ex, 0.3)
    return it


def choose(path):
    sel_image()
    b = button(panel(ex), _("Choose image…"), QPushButton) or button(panel(ex), _("Change image…"), QPushButton)
    if b is None:
        return None
    ex.file_queue = [str(path)]
    ex.click(b)
    wait(ex, 0.8)
    sel_image()
    return b


def kind_buttons():
    return [b for b in visible(panel(ex), QPushButton) if b.text() in (_("Linear"), _("Stretched"))]


def state():
    q = next((q for q in ex.s.paths if q.source is not None and "fusion" not in q.source), None)
    if q is None:
        return None
    return {"source": q.source, "blocks": [n["block"] for n in (ex.s.p.workflow(q.pid).blocks or [])],
            "summary": img_block().summary if img_block() else None, "to_post": q.pid in (ex.view.graph.to_post
                                                                                           if ex.view.graph else [])}


def run_state():
    return ex.w.bt_run.text(), ex.w.bt_run.isEnabled(), ex.w.bt_run.toolTip(), ex.w.status.text()


# ================================================================== A. placed empty
before_missing = missing_fr()
ex.new_project("Image A", None)
ex.close_dialogs()
wait(ex, 0.8)
ex.w.activateWindow()
toasts = []
ex.page.toasted.connect(lambda t: toasts.append(t))
ex.double_click_card("image")
wait(ex, 1.0)
log("A placed", graph_texts(ex), state(), ptexts(ex), run_state())
ex.shot("img_A_placed_empty")
steps = ["Projet Vierge", "double-clic sur Image dans la palette"]
look(ex, N, "panneau Image vide", steps=steps)
N.check(any(_("No file yet: choose the image this block reads.") == t for t in ptexts(ex)), "gênant",
        "Image placée vide : son panneau ne dit pas qu'il manque le fichier", f"{ptexts(ex)}", steps=steps)
rs = run_state()
if rs[1]:
    N.note("gênant", "Image sans fichier : Exécuter est actif", str(rs), steps=steps)
elif not (rs[2] or rs[3]):
    N.note("mineur", "Image sans fichier : Exécuter grisé sans dire pourquoi", str(rs), steps=steps)
# the palette with an empty Image: what may follow
pal = {k: c.reason for k, c in ex.page.cards.items()}
log("A palette", pal)
# the context menu of an empty image
items = ex.right_click_block(img_block())
log("A menu", items)

# ================================================================== B. choose files of every kind
b = choose(fits_img)
log("B fits", state(), ASKS[-1:], ptexts(ex))
ex.shot("img_B_fits")
filt = ASKS[-1][2] if ASKS else ""
if ASKS and ".png" not in filt.lower():
    N.note("gênant", "« Choisir l'image… » ne propose pas les PNG ni les JPEG (ni « Tous les fichiers »)",
           f"Filtre de la fenêtre de fichiers : {filt!r}. Le type « Étirée » et le modèle « Image déjà étirée » "
           "parlent pourtant de « TIFF, PNG ou JPEG ».", steps=["Image placée", "Choisir l'image…"])
for f, label in ((png, "PNG"), (jpg, "JPG"), (tif, "TIFF")):
    choose(f)
    stt = state()
    log("B", label, stt, [t for t in ptexts(ex) if "⚠" in t or "warn" in t])
    ex.shot(f"img_B_{label}")
    if stt and label in ("PNG", "JPG") and not stt["source"].get("stretched"):
        N.note("question", f"Image {label} choisie : la nature reste « Linéaire »",
               f"Un {label} est presque toujours une image déjà étirée ; la nature n'est ni changée ni signalée. "
               f"État : {stt}", steps=["Image placée", f"Choisir l'image… > final.{label.lower()}"])
choose(missing)
stt = state()
log("B missing", stt, ptexts(ex), run_state())
ex.shot("img_B_missing_file")
st = ["Image placée", "Choisir l'image… > un fichier qui n'existe pas (gone.fits)"]
if stt and stt["source"].get("path", "").endswith("gone.fits"):
    if not any("⚠" in t or "introuvable" in t.lower() or "n'existe" in t for t in ptexts(ex)):
        N.note("gênant", "Image : un fichier absent est accepté sans avertissement",
               f"Le panneau montre le fichier comme s'il existait ; Exécuter : {run_state()}", steps=st)
choose(txt)
stt = state()
log("B txt", stt, ptexts(ex), run_state())
ex.shot("img_B_text_file")
if stt and stt["source"].get("path", "").endswith("notes.txt"):
    N.note("gênant", "Image : un fichier texte est accepté comme image, sans avertissement",
           f"Panneau : {ptexts(ex)}; Exécuter : {run_state()}",
           steps=["Image placée", "Choisir l'image… > notes.txt"])
# Ctrl+Z after a file chosen: the previous file back?
before = state()["source"].get("path")
undo(ex)
wait(ex, 0.6)
after = state()["source"].get("path") if state() else None
log("B undo", before, "->", after)
N.check(after != before, "gênant", "Ctrl+Z ne rend pas le fichier d'Image précédent", f"{before} → {after}",
        steps=["Image : choisir un fichier puis un autre", "Ctrl+Z"])
choose(fits_img)

# ================================================================== C. the kind, and what may follow it
sel_image()
kb = kind_buttons()
log("C kind buttons", [(b.text(), b.isEnabled(), b.isChecked(), b.toolTip()) for b in kb])
st_b = next((b for b in kb if b.text() == _("Stretched")), None)
if st_b is not None:
    ex.click(st_b)
    wait(ex, 0.8)
    sel_image()
    log("C stretched", state(), {k: (not c.reason) for k, c in ex.page.cards.items()}, ptexts(ex))
    ex.shot("img_C_stretched_alone")
    look(ex, N, "panneau Image étirée", steps=["Image (FITS)", "Nature : Étirée"])
    # what the palette offers after a stretched image: Split stars only
    allowed = [k for k, c in ex.page.cards.items() if not c.reason]
    log("C allowed after stretched", allowed)
    ex.double_click_card("stars")
    wait(ex, 1.0)
    log("C stars after stretched", state(), toasts[-1:])
    ex.shot("img_C_stretched_stars")
    sel_image()
    kb = kind_buttons()
    log("C kind with stars", [(b.text(), b.isEnabled(), b.toolTip()) for b in kb])
    lin = next((b for b in kb if b.text() == _("Linear")), None)
    if lin is not None and lin.isEnabled():
        ex.click(lin)
        wait(ex, 0.8)
        log("C linear while stars follow", state())
    undo(ex)
    wait(ex, 0.6)
    undo(ex)
    wait(ex, 0.6)
    log("C after undos", state())
sel_image()
kb = kind_buttons()
lin = next((b for b in kb if b.text() == _("Linear")), None)
if lin is not None and not lin.isChecked() and lin.isEnabled():
    ex.click(lin)
    wait(ex, 0.6)
# linear: gradient, then the kind is fixed
ex.double_click_card("gradient")
wait(ex, 1.0)
sel_image()
kb = kind_buttons()
log("C kind with gradient", state(), [(b.text(), b.isEnabled(), b.isChecked(), b.toolTip()) for b in kb])
ex.shot("img_C_linear_gradient")
st_b = next((b for b in kb if b.text() == _("Stretched")), None)
if st_b is not None and st_b.isEnabled():
    N.note("gênant", "Image suivie de Gradient : « Étirée » reste cliquable", str(state()),
           steps=["Image linéaire", "Gradient placé", "panneau Image"])
if st_b is not None and not st_b.isEnabled() and not st_b.toolTip():
    N.note("mineur", "Image : « Étirée » grisé sans dire pourquoi", "", steps=["Image linéaire", "Gradient placé"])

# ================================================================== D. the file changed after blocks were added
ex.double_click_card("color")
wait(ex, 0.8)
ex.double_click_card("astrometry")
wait(ex, 0.8)
s0 = state()
log("D before change", s0, graph_texts(ex))
ex.shot("img_D_before_change")
choose(fits2)
s1 = state()
log("D after change", s1, graph_texts(ex))
ex.shot("img_D_after_change")
st = ["Image (stack.fits), Gradient, Couleur, Astrométrie placés", "Changer d'image… > other.fits"]
if s0 and s1 and [b for b in s0["blocks"] if b != "output"] != [b for b in s1["blocks"] if b != "output"]:
    N.note("gênant", "Changer le fichier d'une Image change ses blocs", f"Avant {s0['blocks']}, après {s1['blocks']}",
           steps=st)
if s0 and s1 and s0["to_post"] and not s1["to_post"]:
    N.note("gênant", "Changer le fichier d'une Image coupe son lien vers le Post", f"Avant {s0}, après {s1}", steps=st)
undo(ex)
wait(ex, 0.6)
log("D undo change", state())

# ================================================================== E. delete: panel, menu, cross, Del, palette; Ctrl+Z / Ctrl+Y
n0 = len(ex.s.paths)


def back():
    undo(ex)
    wait(ex, 1.0)
    return len(ex.s.paths) == n0 and img_block() is not None


sel_image()
b = button(panel(ex), _("Delete"), QPushButton)
n_pop = len(ex.popups)
if b is not None:
    ex.click(b)
    wait(ex, 0.8)
log("E panel delete", len(ex.s.paths), graph_texts(ex), ex.popups[n_pop:], toasts[-1:])
ex.shot("img_E_deleted_panel")
N.check(img_block() is None, "gênant", "« Supprimer le bloc » du panneau Image : sans effet", "", steps=["Supprimer le bloc"])
if ex.popups[n_pop:]:
    N.note("gênant", "Supprimer une Image ouvre une fenêtre bloquante", str(ex.popups[n_pop:]),
           steps=["panneau Image", "Supprimer le bloc"])
ok = back()
log("E undo", ok, state())
N.check(ok, "gênant", "Ctrl+Z ne rend pas l'Image supprimée (et ses blocs)", str(graph_texts(ex)),
        steps=["Supprimer l'Image", "Ctrl+Z"])
undo(ex, redo=True)
wait(ex, 0.8)
log("E redo", img_block() is None)
N.check(img_block() is None, "mineur", "Ctrl+Y ne refait pas la suppression de l'Image", "",
        steps=["Supprimer l'Image", "Ctrl+Z", "Ctrl+Y"])
back()
items = ex.right_click_block(img_block(), _("Delete"))
wait(ex, 0.8)
log("E menu delete", [x.get("text") for x in items], img_block() is None)
N.check(img_block() is None, "gênant", "Menu de l'Image > Supprimer le bloc : sans effet", str(items),
        steps=["clic droit sur Image", "Supprimer le bloc"])
back()
ex.click_block(img_block(), "cross")
wait(ex, 0.8)
log("E cross delete", img_block() is None)
N.check(img_block() is None, "gênant", "La croix de l'Image ne la supprime pas", "",
        steps=["survol de l'Image", "clic sur sa croix"])
back()
sel_image()
ex.w.activateWindow()
ex.key(Qt.Key_Delete)
wait(ex, 0.8)
log("E Del delete", img_block() is None)
N.check(img_block() is None, "gênant", "Suppr sur l'Image sélectionnée ne la supprime pas", "",
        steps=["clic sur Image", "Suppr"])
back()
ex.drag_block_to_palette(img_block())
wait(ex, 0.8)
log("E palette delete", img_block() is None, ex.drops[-1:])
N.check(img_block() is None, "mineur", "Image glissée sur la palette : pas supprimée", str(ex.drops[-1:]),
        steps=["glisser le bloc Image sur la palette"])
back()
log("E end", state(), graph_texts(ex))

# ================================================================== F. a second image: a parallel flow; an image beside a session
z = ex.new_flow_zone()
rec = ex.drag_palette("image", ex.ghost_center(z)) if z is not None else None
wait(ex, 1.0)
log("F second image", rec, [(q.name, q.source) for q in ex.s.paths], ex.s.sel_step, [d.__class__.__name__ for d in
                                                                                    ex.dialogs()])
ex.shot("img_F_second_image")
imgs = [b for b in ex.blocks() if b.node.key == "image"]
log("F image blocks", [(b.title, b.summary) for b in imgs])
names = [b.title for b in imgs]
if len(imgs) == 2 and len(set(names)) == 1:
    N.note("mineur", "Deux blocs Image portent le même nom « Image »",
           "Leurs workflows s'appellent « Image » et « Image 2 » mais les deux blocs disent « Image » : on ne sait "
           "lequel est lequel qu'au nom du fichier.", steps=["Image placée", "Image déposée sur « Nouveau flux »"])
# the graph's dialogs: no template window
if any(d.__class__.__name__ == "TemplateGallery" for d in ex.dialogs()):
    N.note("bloquant", "Déposer une Image ouvre la fenêtre des modèles", "", steps=["glisser Image sur le graphe"])
ex.close_dialogs()
# an image beside a session
ex.double_click_card("session")
wait(ex, 1.0)
log("F session beside images", graph_texts(ex), [(q.name, q.source, q.sessions) for q in ex.s.paths])
ex.shot("img_F_session_beside")

missing_t = missing_fr() - before_missing
log("MISSING image", sorted(missing_t))
if missing_t:
    N.note("mineur", "Image : textes restés en anglais", " | ".join(sorted(missing_t))[:1500], shot=False,
           steps=["Image placée, fichiers, nature, suppression"])
(ex.out / "image_log.txt").write_text("\n".join(LOG), encoding="utf-8")
N.exceptions(["voir le journal du script explore_sources_image.py"])
ex.close()
print("DONE", TMP)
