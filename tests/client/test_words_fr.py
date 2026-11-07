"""Words of the interface (GUI exploration 2, words and layout): French agreement and elision with the names of the
blocks, "désactivé" and never "off", the reasons said plainly, Undo / Redo saying what they undid, offscreen.

A scratch project whose frames are empty files (the keys read their names and sizes only): nothing is computed.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.client.blocks import Graph  # noqa: E402
from astro_gpu.client.graph import GraphPage  # noqa: E402
from astro_gpu.client.model import ProjectModel, refused_text  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def world(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    i18n.set_language("en")
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(3):
        (raw / f"a{i}.ARW").write_bytes(b"\0" * (100 + i))
    p = project2.create(tmp_path / "P", "P", Target(name="M 31"))
    p.sessions.append(project2.Session(1, "night 1", lights=[str(raw)]))
    p.add_workflow("Full", wfl.TEMPLATES["Full"])
    project2.save(p)
    w = ApiWorker()
    m = ProjectModel(p, w)
    assert w.wait(120)
    m._sync()
    page = GraphPage(m)
    page.resize(1500, 800)
    yield {"m": m, "w": w, "page": page}
    i18n.set_language("en")
    page.close()
    m.close(save=False)
    w.stop()


def settle(world):
    world["w"].wait(60)
    QApplication.processEvents()
    world["page"].refresh()


def test_a_name_takes_its_gender_and_elision_in_french():
    i18n.set_language("fr")
    try:
        f = i18n.noun_form
        assert (f("Calibration", "e"), f("Gradient 2", "e"), f("Étoiles/fond", "e")) == ("e", "", "")
        assert (f("Astrométrie", "Un"), f("Empilement", "un")) == ("Une", "un")
        assert (f("Alignement", "de"), f("Calibration", "de"), f("Empilement brut", "de")) == \
            ("d'Alignement", "de Calibration", "d'Empilement brut")
        assert (f("Astrométrie", "le"), f("Couleur", "le"), f("Débruitage", "le")) == \
            ("l'Astrométrie", "la Couleur", "le Débruitage")
        # a translation that asks a form is formatted with it; a number keeps "{x:e}"; the English is untouched
        assert i18n._("{block} removed · Ctrl+Z").format(block="Calibration") == "Calibration supprimée · Ctrl+Z"
        assert i18n._("{block} removed · Ctrl+Z").format(block="Gradient 2") == "Gradient 2 supprimé · Ctrl+Z"
        assert i18n._("one {block} per workflow").format(block="Sélection") == "une seule Sélection par workflow"
        assert i18n.Text("{x:e}").format(x=12345.0) == "1.234500e+04"
    finally:
        i18n.set_language("en")
    assert i18n._("{block} removed · Ctrl+Z").format(block="Calibration") == "Calibration removed · Ctrl+Z"


def test_the_palette_reasons_are_french_sentences():
    """Elision ("pas d'Alignement"), agreement ("Une seule Calibration"), and the Stars/background card's reason names
    Stars/background, never the Output (not a block on screen)."""
    i18n.set_language("fr")
    try:
        img = [wfl.node("gradient")]
        e = wfl.palette_problem(img, "image", "registration")
        assert refused_text(e) == "pas d'Alignement à partir d'un fichier image"
        frames = [wfl.node("calibration"), wfl.node("registration")]
        e = wfl.palette_problem(frames, "frames", "stars")
        assert e.fields == {"block": "Stars/background"}
        assert refused_text(e) == "Étoiles/fond : il faut l'Empilement avant"
        e = wfl.palette_problem(frames + [wfl.node("stacking")], "frames", "calibration")
        assert refused_text(e) == "une seule Calibration par workflow"
        off = [wfl.node("calibration"), wfl.node("registration", off=True), wfl.node("stacking")]
        assert refused_text(wfl.palette_problem(off, "frames", "registration")) == \
            "Alignement déjà là, désactivé : réactiver (D)"
    finally:
        i18n.set_language("en")
    assert refused_text(wfl.palette_problem([wfl.node("calibration")], "frames", "stars")) == \
        "Stars/background needs Stacking first"


def test_switch_off_remove_and_keep_say_it_in_french_and_undo_says_what(world):
    m, page = world["m"], world["page"]
    toasts = []
    page.toasted.connect(toasts.append)
    i18n.set_language("fr")
    page.toggle_cut(Graph(m).node_of(m.path, "fond"), None)
    settle(world)
    assert toasts[-1] == "Gradient désactivé · Full · Ctrl+Z"
    assert Graph(m).summary(Graph(m).node_of(m.path, "fond")) == "désactivé : image transmise"
    assert m.undo() and m.restored == "Gradient désactivé · Full"          # what Ctrl+Z took back, in its words
    assert m.redo() and m.restored == "Gradient désactivé · Full"
    QApplication.processEvents()                                          # the gesture is over
    page.remove_tool(Graph(m).node_of(m.path, "astrometrie"), None)       # Color needs it: refused
    assert toasts[-1] == "Astrométrie gardée : Couleur en a besoin"
    page.remove_tool(Graph(m).node_of(m.path, "empilement"), None)
    assert toasts[-1] == "Empilement gardé : les blocs suivants ont besoin de ce qu'il donne (image)"
    page.remove_tool(Graph(m).node_of(m.path, "couleur"), None)
    settle(world)
    assert toasts[-1] == "Couleur supprimée · Ctrl+Z"
    assert any(t == '{block} removed from "{name}"' and a["block"] == "Couleur" for _h, t, a in m.history)
    assert m.undo() and m.restored == "Couleur supprimée"
    i18n.set_language("en")
    assert m.redo() and m.restored == "Couleur supprimée"                 # said as it was shown


def test_a_step_without_words_is_undone_plainly(world):
    m = world["m"]
    QApplication.processEvents()
    m.set_graph(m.path, [wfl.node("calibration"), wfl.node("registration"), wfl.node("stacking")])
    QApplication.processEvents()                                          # its gesture ended without words
    m.log("Run finished in {t}", t="1 min")                               # a later line never names it
    assert m.undo() and m.restored == ""
