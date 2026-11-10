"""The ProjectModel surface: every member the views use exists, and what the model returns has the views' shape.

``SURFACE`` lists what the views (Inspect, Frames, Run, Post, the graph and its panel) use on the model, by group;
``REPLACED`` names what the model deliberately does not serve (the project's raw data: the views go through the API).
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from conftest import HAVE_TEST_PROJECT, NO_TEST_PROJECT, TEST_PROJECT  # noqa: E402
from PySide6.QtCore import QObject  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import model as modelmod  # noqa: E402
from astro_gpu.client.model import ProjectModel, WorkflowView  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

#: what the views use on the model, by group
SURFACE = {
    "signals": ["pathsChanged", "currentChanged", "costChanged", "sourcesChanged", "testChanged", "readyChanged",
                "paramsChanged", "factsChanged", "selectionChanged"],
    "attributes": ["paths", "path", "current", "sel_step", "scope", "pending_change", "sessions", "stack_mode", "gpu",
                   "history", "sources_loaded", "frames_excluded", "test_on", "test_zone", "root", "name", "target",
                   "ready", "trunk_params", "p", "error", "device"],
    "workflows": ["add_path", "remove_path", "rename", "duplicate", "unique_name", "set_opts", "has_tool", "with_tool",
                  "without_tool", "tool_cost", "add_exit", "scope_paths", "select", "index_of", "view", "batch"],
    "changes": ["apply_change", "add_variant", "keep_variant", "drop_variant", "throws_away", "cost_with",
                "cost_of_change", "set_param", "set_trunk_param", "undo", "redo", "can_undo", "can_redo"],
    "stages and states": ["state", "state_info", "previous_key", "pending_upto", "all_pending", "pending", "is_pending",
                          "stage_key", "owner", "sharing", "all_stages", "path_stages", "deps", "upstream",
                          "downstream", "block_keys", "block_key", "title", "mark_stages", "results_changed"],
    "durations and plans": ["seconds", "path_seconds", "opts_seconds", "opts_measured", "cost_of", "cost",
                            "total_seconds", "plan_all", "plan_to_block", "method_seconds", "trunk_pending",
                            "trunk_seconds", "set_machine"],
    "settings": ["params", "param_info", "param_value", "default_of", "reason_for"],
    "selection": ["thresholds_now", "suggested_thresholds", "thresholds_unit", "set_selection", "set_excluded",
                  "kept_frames", "selection_total", "excluded_mask", "request_selection", "fwhm_median"],
    "project facts": ["info", "image_size", "n_lights", "load_sources", "reload_sources", "add_session",
                      "set_stack_mode", "zone_fraction"],
    "Post": ["slider_infos", "slider_state", "available_sliders", "unlock_text"],
    "lifecycle": ["flush", "close", "log", "edited"],
}
#: what the model does not serve on purpose: the name, and what to use instead
REPLACED = {
    "data": "info() and image_size() (the project's facts), frame_table / stage_array of the API (the frames)",
    "stages": "settings.resolve() and the model's params(p, stage) (the engine's describe)",
    "layers": "api.render_layer / the RenderWorker of Post (the layers are the engine's)",
    "sliders": "slider_infos(p) (api.sliders_of)",
}
@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def empty_model(qapp, tmp_path_factory):
    """A model on a project with no frames and no workflow: nothing to read from the disk."""
    root = tmp_path_factory.mktemp("surface") / "P"
    p = project2.create(root, "P", Target(name="M 1"))
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    assert w.wait(120)
    assert m.ready
    yield m
    w.stop()


def test_every_member_of_the_surface_exists(empty_model):
    m = empty_model
    missing = [f"{group}: {n}" for group, names in SURFACE.items() for n in names if not hasattr(m, n)]
    assert not missing, missing
    for n in SURFACE["signals"]:
        assert hasattr(ProjectModel, n), n
    assert isinstance(m, QObject)


def test_the_model_does_not_serve_what_it_replaced(empty_model):
    for name in REPLACED:
        assert not hasattr(empty_model, name), f"{name} is replaced: the model should not have it"


def test_the_panel_the_graph_and_the_dialogs_find_what_they_read(empty_model):
    """The Panel, the GraphView and the dialogs use these."""
    need = ["scope_paths", "pending_change", "paths", "path", "current", "sel_step", "scope", "slider_state",
            "thresholds_now", "kept_frames", "sessions", "frames_excluded", "stack_mode", "gpu", "has_tool",
            "with_tool", "without_tool", "add_variant", "keep_variant", "drop_variant", "cost_with", "apply_change",
            "throws_away", "opts_seconds", "unique_name", "log", "add_exit", "set_opts", "add_path", "remove_path",
            "all_pending", "seconds", "info", "image_size"]
    assert not [n for n in need if not hasattr(empty_model, n)]


def test_the_workflow_view_has_the_fields_of_pathdef():
    fields = {"pid", "name", "opts", "params", "end", "variant_of", "computed", "rendered", "layer", "provisional"}
    v = WorkflowView(1, "x", {})
    assert fields <= set(v.__dataclass_fields__)
    assert modelmod.PathDef is WorkflowView and modelmod.Session is ProjectModel
    assert modelmod.TEMPLATES.keys() == {"Vierge", "Classique + débruitage", "Complet"}
    assert callable(modelmod.hm)
    # what the views import from the model module: TRUNK, PathDef, Session, hm (steps, sources, tune), TOOLS (the
    # window), TEMPLATES (dialogs); no DATA (info() and the API serve the project's facts)
    assert modelmod.TRUNK == ["inventory", "calibrate", "register", "stack"]
    assert {"debruitage", "etoiles", "deconv"} <= set(modelmod.TOOLS) and "appris" not in modelmod.TOOLS
    assert not hasattr(modelmod, "DATA")


def test_empty_project_answers(empty_model):
    m = empty_model
    assert m.paths == [] and m.path is None and m.sessions == [] and not m.sources_loaded
    assert m.info()["n_lights"] == 0 and m.image_size() is None
    assert m.all_pending() == [] and m.total_seconds() == 0
    assert m.state(None, "stack") == "pending" and m.all_stages(None) == []
    assert m.frames_excluded == set() and m.stack_mode == "together"
    assert m.thresholds_now()[0] == 60.0
    m.log("Project opened")
    assert m.history[-1][1] == "Project opened"


def test_options_speak_the_views_vocabulary():
    """The views' option names (debruitage, etoiles, fond...) go to the engine's options and back."""
    for key, opts in modelmod.TEMPLATES.items():
        eng = modelmod.to_engine(opts)
        back = modelmod.from_engine(eng)
        assert modelmod.to_engine(back) == eng, key
        assert back["debruitage"] == opts["debruitage"] and back["fond"] == opts["fond"]
    assert modelmod.to_engine(modelmod.TEMPLATES["Complet"])["stars"] == "neural"
    assert modelmod.from_engine({"denoise": "pretrained", "stars": "psf_neural"})["etoiles"] == "retirer"   # an old value
    # the with / without cases hold, through the engine's rules
    full = modelmod.from_engine(modelmod.to_engine(modelmod.TEMPLATES["Complet"]))
    classic = modelmod.from_engine(modelmod.to_engine(modelmod.TEMPLATES["Vierge"]))
    new, notes = ProjectModel.with_tool(classic, "etoiles")
    assert new["etoiles"] == "retirer" and new["debruitage"] and any("Denoise added" in n for n in notes)
    new, notes = ProjectModel.with_tool(classic, "auto")
    assert new["fond"] == "auto" and new["etoiles"] == "garder" and not notes
    no, why = ProjectModel.without_tool(full, "debruitage")
    assert no is None and "needed by Stars/background" in why[0]
    new, notes = ProjectModel.without_tool(full, "etoiles")
    assert new["etoiles"] == "garder" and new["debruitage"] and new["fond"] == "simple" and not notes
    no, why = ProjectModel.without_tool(full, "simple")
    assert no is None
    new, _n = ProjectModel.without_tool(classic, "debruitage")
    assert new is not None and not new["debruitage"]
    assert ProjectModel.has_tool(full, "simple") and not ProjectModel.has_tool(full, "auto")
    new, notes = ProjectModel.with_tool(full, "deconv")
    assert new["deconv"] and new["deconv_m"] == "psf" and notes == []


# ---------------------------------------------------------------------------------------------- the French
def _shown_texts() -> dict[str, list[str]]:
    """Every English text the graph's modules can show: the ``_("...")`` of the code, the catalogue of settings, the
    block table, the methods and the states."""
    from astro_gpu import api
    from astro_gpu.client import blocks, settings
    root = ROOT / "astro_gpu" / "client"
    found: dict[str, list[str]] = {}
    for name in ("model", "blocks", "graph", "panel", "settings"):
        tree = ast.parse((root / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_" and node.args:
                a = node.args[0]
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    found.setdefault(a.value, []).append(f"{name}.py:{node.lineno}")
    for title, items in [g for groups in settings.CATALOG.values() for g in groups]:
        found.setdefault(title, []).append("CATALOG")
        for it in items:
            for k in ("label", "help", "unit", "reason"):
                if it.get(k):
                    found.setdefault(it[k], []).append("CATALOG")
            for c in it.get("choices") or []:
                found.setdefault(c, []).append("CATALOG")
            if it["kind"] == "new" and it["type"] == "choice":
                found.setdefault(it["default"], []).append("CATALOG")
    for b in blocks.STEPS:
        found.setdefault(b["what"], []).append("STEPS")
        found.setdefault(b["title"], []).append("STEPS")
    for v in blocks.TOOLS.values():
        found.setdefault(v["what"], []).append("TOOLS")
        found.setdefault(v["title"], []).append("TOOLS")
    for t in list(blocks.TITLES.values()) + list(blocks.STATE_WORD.values()) + [p[0] for p in blocks.PHASES]:
        found.setdefault(t, []).append("blocks")
    for labs in blocks.SIDE_INPUTS.values():
        for _k, lab in labs:
            found.setdefault(lab, []).append("SIDE_INPUTS")
    found.setdefault("H-alpha survey", []).append("SIDE_INPUTS")
    for t in settings.TIPS.values():
        found.setdefault(t, []).append("TIPS")
    for block in ("denoise", "gradient", "stars", "deconvolution"):
        for m in api.methods(block):
            for t in (m.name, m.short, m.what):
                found.setdefault(t, []).append(f"methods({block})")
    return found


def test_every_text_of_the_graph_has_its_french():
    """The French catalogue (base, shell, engine, graph) holds every text the graph can show: none is left in English."""
    import json
    cat: dict[str, str] = {}
    for f in sorted((ROOT / "astro_gpu" / "client" / "locale" / "fr").glob("*.json")):
        cat.update(json.loads(f.read_text(encoding="utf-8")))
    missing = {t: where[:2] for t, where in _shown_texts().items() if t and t not in cat}
    assert not missing, missing
    graph = json.loads((ROOT / "astro_gpu" / "client" / "locale" / "fr" / "graph.json").read_text(encoding="utf-8"))
    assert len(graph) > 20 and all(isinstance(v, str) and v for v in graph.values())
    # the engine's own texts are all translated too
    from astro_gpu import texts
    assert not any(texts.coverage("fr").values())


def test_the_graph_modules_import_only_the_client_libraries():
    """No leftover of the old prototype (its imports, its Demo menu, its data files) in the graph's modules."""
    root = ROOT / "astro_gpu" / "client"
    for name in ("model", "blocks", "graph", "panel", "settings"):
        src = (root / f"{name}.py").read_text(encoding="utf-8")
        for bad in ("alt2", "alt3", "data.json", "datatext", "import project", "from project", "from steps", "Demo"):
            assert bad not in src, (name, bad)
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                top = node.module.split(".")[0]
                assert top in {"__future__", "contextlib", "copy", "dataclasses", "math", "pathlib", "time", "typing",
                               "PySide6", "numpy"} or top == "astro_gpu", (name, node.module)


def test_exclusions_and_session_edits_are_saved_like_any_edit(qapp, tmp_path):
    """Frames excluded by hand and the sessions are part of the definition: Save writes them, as it does the
    workflows (issue #1; Undo stays for the workflows: an exclusion is not taken back by it)."""
    import json

    root = tmp_path / "P"
    p = project2.create(root, "P", Target(name="M 1"))
    p.sessions.append(project2.Session(1, "2026-08-16", lights=[str(tmp_path / "raw")]))
    project2.save(p)
    w = ApiWorker()
    m = ProjectModel(p, w)
    try:
        assert w.wait(120)

        def saved():
            assert m.dirty                                  # waiting for Save, then saved by it
            m.flush()
            return json.loads((root / "project.astroproj").read_text(encoding="utf-8"))
        m.set_excluded([(1, "DSC0001.ARW")], True)
        assert m._dirty and not m.can_undo()
        assert saved()["excluded"] == [[1, "DSC0001.ARW"]]
        m.set_excluded([(1, "DSC0001.ARW")], False)
        assert saved()["excluded"] == []
        p.sessions.append(project2.Session(2, "2026-08-17", lights=[str(tmp_path / "raw2")]))
        m.reload_sources()                                  # what Import folder and Add session do after their edit
        assert len(saved()["sessions"]) == 2
        assert w.wait(120)
    finally:
        m.close()
        w.stop()


def test_durations_are_read_by_the_worker_before_the_views_are_told(qapp, tmp_path):
    """The method cards, the toolbar cards and the New workflow dialog quote durations that read manifests: the worker
    fills the model's cache before ``readyChanged``, so that drawing them reads nothing in the GUI thread."""
    from astro_gpu import api
    from astro_gpu.client.model import TEMPLATES
    data = TEST_PROJECT
    if not HAVE_TEST_PROJECT:
        pytest.skip(NO_TEST_PROJECT)
    p = project2.load(data)                                  # read only: the model never writes it
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    told = []
    m.readyChanged.connect(lambda: told.append(len(m._est)))
    try:
        assert w.wait(120) and m.ready
        assert told and told[-1] > 0                         # the cache was full when the views were told
        calls = []
        real = api.estimate
        api.estimate = lambda *a, **k: (calls.append(a), real(*a, **k))[1]
        try:
            v = m.paths[0]
            for stage in m.all_stages(v):
                m.seconds(stage, v)
            for opts in TEMPLATES.values():
                m.opts_seconds(opts)
            for k in ("debruitage", "etoiles"):              # the toolbar asks only for the tools not there yet
                if not m.has_tool(v.opts, k):
                    m.tool_cost(v.opts, k)
            for block in ("debruitage", "etoiles", "fond"):
                for meth in modelmod.methods_of(block):
                    m.method_seconds(v, block, meth["key"])
        finally:
            api.estimate = real
        assert calls == []                                    # all answered from the cache
    finally:
        m.close()
        w.stop()
