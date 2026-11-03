"""The graph, the panel and the model, offscreen, on the test project with four workflows.

The test project (tests/conftest.py, TEST_PROJECT) has Full computed; the tests add Classic stack, Classic + denoise and
Full + neural stars in memory. The model is read-only (nothing is saved into the project), except the autosave test,
which works on a copy of the definition in a temporary folder. ASTRO_GPU_SETTINGS points to a temporary file in every
test; the user's settings file is checked untouched. ASTRO_GPU_SHOTS names the folder of the screenshots (default: a
temporary folder).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")      # the offscreen platform has no fonts else

pytest.importorskip("PySide6")
from conftest import TEST_PROJECT, needs_test_project, scratch_project  # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QFont, QFontMetricsF  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGraphicsPathItem,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from astro_gpu import api  # noqa: E402
from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client.blocks import COLUMNS, Graph, block_state  # noqa: E402
from astro_gpu.client.graph import BH, BW, MIN_SCALE, BlockItem, GraphPage  # noqa: E402
from astro_gpu.client.i18n import hm  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.panel import CompactRow, MethodCard  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.client.widgets import QSS  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

F_PROJECT = TEST_PROJECT
pytestmark = needs_test_project
FOUR = (("Classic stack", "Classic stack", None),
        ("Classic + denoise", "Classic + denoise", {"denoise": {"iterations": 6000}}),
        ("Full + neural stars", "Full + neural stars", {"denoise": {"iterations": 6000}}))


def _user_settings() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "astro_gpu" / "settings.json"


def _fingerprint(p: Path):
    return (p.stat().st_mtime_ns, hashlib.sha1(p.read_bytes()).hexdigest()) if p.exists() else None


def _tree_snapshot(folder: Path):
    """Every file of the test project, its store included (the scratch copies read it through a junction)."""
    return sorted((p.relative_to(folder).as_posix(), p.stat().st_mtime_ns, p.stat().st_size)
                  for p in folder.rglob("*") if p.is_file())


USER_SETTINGS = _user_settings()
USER_BEFORE = _fingerprint(USER_SETTINGS)
PROJECT_BEFORE = _tree_snapshot(F_PROJECT) if F_PROJECT.is_dir() else None


# ---------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def qapp():
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return a


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    appmod.apply_language("en")
    shown = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, _k=kind, **k: shown.append((_k, a[1:3])))
    yield {"tmp": tmp_path, "shown": shown}
    appmod.apply_language("en")
    assert _fingerprint(USER_SETTINGS) == USER_BEFORE, "the user's settings file was touched"


def _open_four(folder: Path | None = None):
    p = api.open_project(folder or F_PROJECT)
    for name, template, params in FOUR:
        api.add_workflow(p, name, template, params)
    return p


@pytest.fixture(scope="module")
def world(qapp, tmp_path_factory):
    folder = scratch_project(tmp_path_factory.mktemp("graph") / "heart")
    p = _open_four(folder)
    w = ApiWorker()
    m = ProjectModel(p, w)
    assert not m.ready                                # the GUI thread did not wait for the disk
    assert w.wait(180) and m.ready, m.error
    yield {"p": p, "w": w, "m": m, "folder": folder}
    m.flush()
    w.stop()


def _events():
    """Process the events, and delete the widgets a rebuild let go of (no event loop runs in the tests, so the
    deferred deletions would otherwise pile up, hidden, under the panel)."""
    QApplication.processEvents()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def settle(w, page=None, n=12):
    """Process events until the worker is idle and the page has nothing left to build."""
    for _i in range(n):
        w.wait(60)
        _events()
        if page is not None and page._pending:
            _events()
        time.sleep(0.01)
    w.wait(60)
    _events()


def _reset(m: ProjectModel):
    while m.can_undo():
        m.undo()
    m._redo.clear()
    m.scope, m.pending_change, m.current, m.sel_step = None, None, 0, "empilement"
    m.p.stack_mode = "together"
    m.flush()


@pytest.fixture()
def model(world):
    m = world["m"]
    _reset(m)
    yield m
    _reset(m)


@pytest.fixture()
def page(world, model, qapp):
    root = QWidget()
    root.setObjectName("root")
    v = QVBoxLayout(root)
    v.setContentsMargins(0, 0, 0, 0)
    pg = GraphPage(model)
    v.addWidget(pg)
    root.resize(1680, 934)
    root.show()
    settle(world["w"], pg)
    pg.root = root
    yield pg
    root.close()
    root.deleteLater()
    qapp.processEvents()


def node(page, key, pid=None):
    g = page.graph.graph
    return g.node_of(page.s.view(pid) if pid else page.s.path, key)


def texts_of(root: QWidget) -> list[str]:
    out = []
    for w in [root] + root.findChildren(QWidget):
        if not w.isVisibleTo(root) and w is not root:
            continue
        if isinstance(w, QLabel):
            out.append(re.sub(r"<[^>]+>", "", w.text()))
        elif isinstance(w, QAbstractButton):
            out.append(w.text())
        elif isinstance(w, QComboBox):
            out += [w.itemText(i) for i in range(w.count())]
        if w.toolTip():
            out.append(w.toolTip())
    return [t.strip() for t in out if t and t.strip()]


NAMES = {"Full", "Classic stack", "Classic + denoise", "Full + neural stars", "Raw stack"}


def untranslated(texts) -> list[str]:
    """Texts shown in English although a French translation exists (the names the user typed are never translated)."""
    cat = i18n._cat
    return [t for t in texts if t in cat and cat[t] != t and t not in NAMES]


def cut_texts(root: QWidget) -> list[str]:
    """Labels, buttons and check boxes narrower than their text, and scroll areas whose content is wider than them
    (their horizontal bar is off: what passes the right edge is cut)."""
    out = []
    for sa in root.findChildren(QScrollArea):
        inner = sa.widget()
        if sa.isVisibleTo(root) and inner is not None and sa.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff:
            if inner.minimumSizeHint().width() > sa.viewport().width() + 1:
                out.append(f"{type(sa.parent()).__name__}: content {inner.minimumSizeHint().width()} px > "
                           f"{sa.viewport().width()} px")
    for w in root.findChildren(QWidget):
        if not w.isVisibleTo(root):
            continue
        if isinstance(w, QLabel) and not w.wordWrap() and w.text() and (w.pixmap() is None or w.pixmap().isNull()):
            if w.width() + 1 < w.sizeHint().width():
                out.append(w.text())
        elif isinstance(w, (QPushButton, QCheckBox)) and w.text():
            if w.width() + 1 < w.sizeHint().width():
                out.append(w.text())
    return out


def graph_faults(page) -> list[str]:
    """What is wrong in the graph as drawn: a block outside the view or drawn smaller than 80 % (off Show all), a
    title that does not fit its block, a wire that runs behind a block it does not join."""
    gv = page.graph
    out = []
    items = [it for it in gv.scene().items() if isinstance(it, BlockItem)]
    if not gv.b_all.isChecked():
        if gv.transform().m11() < MIN_SCALE - 1e-6:
            out.append(f"scale {gv.transform().m11():.3f}")
        vp = QRectF(gv.viewport().rect())
        for it in items:
            r = QRectF(gv.mapFromScene(it.sceneBoundingRect()).boundingRect())
            if not vp.contains(r):
                out.append(f"{it.title} outside the view")
    f = QFont("Segoe UI", 10)
    f.setBold(True)
    for it in items:
        if it.node.key != "post" and QFontMetricsF(f).horizontalAdvance(it.title) > it.w - 18:
            out.append(f"title {it.title!r} wider than its block")
    rects = [QRectF(it.pos().x(), it.pos().y(), it.w, BH).adjusted(1, 1, -1, -1) for it in items]
    for wire in gv.scene().items():
        if isinstance(wire, QGraphicsPathItem) and wire.zValue() == 1:
            path = wire.path()
            for i in range(1, 200):
                pt = path.pointAtPercent(i / 200)
                hit = next((r for r in rects if r.contains(pt)), None)
                if hit is not None:
                    out.append(f"a wire runs behind the block at x={hit.x():.0f}, y={hit.y():.0f}")
                    break
    return out


def panel_faults(page) -> list[str]:
    """The editors of the panel's setting rows do not all start at the same x."""
    panel = page.panel
    host = panel.findChildren(QScrollArea)[0].widget()
    eds = [w for w in host.findChildren(QWidget) if w.isVisibleTo(panel) and (
        isinstance(w, QAbstractSpinBox) or (isinstance(w, QCheckBox) and not w.text()) or
        (isinstance(w, (QLineEdit, QPushButton)) and isinstance(w.parent(), CompactRow) and w.text() != "↺"))]
    xs = sorted({w.mapTo(panel, QPoint(0, 0)).x() for w in eds})
    return [f"editors at x = {xs}"] if len(xs) > 1 else []


def button(root: QWidget, start: str) -> QPushButton:
    return next(b for b in root.findChildren(QPushButton) if b.text().startswith(start) and b.isVisibleTo(root))


# ---------------------------------------------------------------------------------------------- the tree
def test_the_model_does_not_wait_for_the_disk(qapp):
    """Opening a project is instant: the worker warms the API; keys, states and costs come after readyChanged."""
    p = _open_four()
    w = ApiWorker()
    t0 = time.perf_counter()
    m = ProjectModel(p, w, readonly=True)
    assert time.perf_counter() - t0 < 1.0 and not m.ready
    assert [v.name for v in m.paths] == ["Full", "Classic stack", "Classic + denoise", "Full + neural stars"]
    assert m.all_stages() == [] and m.total_seconds() == 0.0 and m.state(None, "stack") == "pending"
    got = []
    m.readyChanged.connect(lambda: got.append(m.ready))
    assert w.wait(180)
    assert m.ready and got and got[-1]
    assert m.all_stages(m.paths[0])[:4] == ["inventory", "calibrate", "register", "stack"]
    w.stop()


def test_the_tree_shares_blocks_as_the_engine_does(world, model, page):
    """Calibration to Color (Gradient included) shared by the 4 workflows, Denoise by 3, a split at Stars, 4
    outputs."""
    g = Graph(model)
    by_key = {}
    for n in g.nodes:
        by_key.setdefault(n.key, []).append(n)
    ids = {v.name: v.pid for v in model.paths}
    for key in ("calibration", "alignement", "selection", "empilement", "astrometrie", "fond", "couleur"):
        assert len(by_key[key]) == 1 and len(by_key[key][0].paths) == 4, key
    den = by_key["debruitage"]
    assert len(den) == 1 and {q.name for q in den[0].paths} == {"Full", "Classic + denoise", "Full + neural stars"}
    stars = by_key["etoiles"]
    assert len(stars) == 2 and all(len(n.paths) == 1 for n in stars)           # the methods differ
    assert {q.name for n in stars for q in n.paths} == {"Full", "Full + neural stars"}
    assert len(by_key["post"]) == 4 and all(len(n.paths) == 1 for n in by_key["post"])
    assert g.final.paths and len(g.final.paths) == 4
    # the leaves go to Post, one per workflow
    assert {g.leaves[pid].key for pid in ids.values()} == {"post"}
    assert by_key["debruitage"][0].shared and not stars[0].shared


def test_the_graph_shows_the_real_engine_order(world, model, page):
    """Frames, Calibration, Registration, Selection, Stacking, Astrometry, Gradient, Color, Denoise, Stars, Output,
    Post (the engine's order); the phase bands; Deconvolution only when something uses it."""
    assert COLUMNS == ["sources", "calibration", "alignement", "selection", "empilement", "astrometrie", "fond",
                       "couleur", "deconv", "debruitage", "etoiles", "post", "final"]
    assert model.block_keys(model.paths[0]) == ["calibration", "alignement", "selection", "empilement", "astrometrie",
                                                "fond", "couleur", "debruitage", "etoiles", "post"]
    items = {it.node.key + str(id(it.node)): it for it in page.graph.scene().items() if isinstance(it, BlockItem)}
    xs = sorted({it.pos().x() for it in items.values()})
    assert len(xs) == 12                                                        # no Deconvolution column
    gradient = [it for it in items.values() if it.node.key == "fond"]
    color = [it for it in items.values() if it.node.key == "couleur"]
    assert max(g.pos().x() for g in gradient) < min(c.pos().x() for c in color)  # Gradient before Color
    bands = [it for it in page.graph.scene().items() if type(it).__name__ == "BandItem"]
    assert sorted(b.text for b in bands) == ["Linear processing", "Pre-processing"]


def test_states_come_from_the_api(world, model, page):
    api_states = api.states(model.p, update=False)
    for (wid, stage), st in api_states.items():
        assert model.state(model.view(wid), stage) == st.state
    g = Graph(model)
    full = model.paths[0]
    assert full.computed and full.rendered and not model.paths[1].computed
    states = {(n.key, n.paths[0].name): g.state(n) for n in g.nodes if n.paths and n.key not in ("sources", "final")}
    assert states[("empilement", "Full")] == "done"
    assert states[("fond", "Classic stack")] == "pending" and states[("post", "Classic stack")] == "pending"
    assert states[("etoiles", "Full + neural stars")] == "pending" and states[("etoiles", "Full")] == "done"
    assert g.state(g.final) == "pending"                                       # not every layer is rendered
    # a block is as bad as its worst stage in any of its workflows
    model._states[(2, "gradient")] = api.StageState("failed", "boom", "k")
    assert block_state(model, model.view(2), "fond") == "failed"
    model._states[(2, "gradient")] = api.StageState("running", "", "k")
    assert block_state(model, model.view(2), "fond") == "running"
    model._sync()
    assert block_state(model, model.view(2), "fond") == "pending"
    assert g.summary(g.root).startswith("1 session · ") and "40 lights" in g.summary(g.root)


def test_a_shared_stage_is_one_key(world, model):
    full, classic, cd, neural = model.paths
    assert [q.name for q in model.sharing(full, "stack")] == [q.name for q in model.paths]
    assert model.owner(neural, "stack") is full
    assert {q.name for q in model.sharing(full, "denoise")} == {"Full", "Classic + denoise", "Full + neural stars"}
    assert model.stage_key(full, "stack") == api.stage_key(model.p, full.pid, "stack")
    assert model.upstream(full, "color") >= {"inventory", "calibrate", "register", "stack", "astrometry"}
    assert "render_prep" in model.downstream(full, "stars")
    assert [x for _p, x in model.pending_upto(classic, ["render_simple"])] == ["gradient", "render_simple"]
    assert model.pending(full) == []
    assert model.seconds("stack", full) > 0 and model.seconds("denoise_cl") == 300.0


# ---------------------------------------------------------------------------------------------- option rules
def test_adding_and_removing_blocks_follows_the_rules_with_toasts(world, model, page):
    classic = model.view(2)
    toasts = []
    page.toasted.connect(toasts.append)
    page.add_tool("etoiles", classic.pid)                                      # Stars needs Denoise
    settle(world["w"], page)
    assert classic.opts["etoiles"] == "mesurer" and classic.opts["debruitage"]
    assert any("Denoise added" in t for t in toasts)
    assert model.sel_step == "etoiles" and model.path is classic
    # Denoise cannot go while Stars is there
    toasts.clear()
    page.remove_tool(node(page, "debruitage", classic.pid), [classic.pid])
    assert classic.opts["debruitage"] and toasts and "remove Stars first" in toasts[0]
    # removing Stars turns the H-alpha gradient into the polynomial one, and the neural removal goes too
    model.set_opts(classic, {**classic.opts, "fond": "halpha", "appris": True})
    toasts.clear()
    page.remove_tool(node(page, "etoiles", classic.pid), [classic.pid])
    assert classic.opts["etoiles"] == "garder" and classic.opts["fond"] == "simple" and not classic.opts["appris"]
    assert len(toasts) == 1 and toasts[0].count(" · ") == 1
    # Gradient cannot be removed, and the card after the output says so
    new, why = model.without_tool(classic.opts, "simple")
    assert new is None and why


def test_the_toolbar_cards_say_what_each_block_costs(world, model, page):
    from astro_gpu.client.graph import ToolCard
    cards = {c.key: c for c in page.findChildren(ToolCard)}
    assert set(cards) == {"debruitage", "etoiles"}                             # Deconvolution is ◇: Spec mode only
    assert all(c.state == "in" for c in cards.values())                        # Full has both
    model.select(index=1)                                                      # Classic stack
    page.refresh()
    cards = {c.key: c for c in page.findChildren(ToolCard)}
    assert cards["debruitage"].state == "add" and cards["etoiles"].state == "add"
    texts = [lb.text() for c in cards.values() for lb in c.findChildren(QLabel)]
    assert any(t.lstrip("≈ ").startswith("+") for t in texts)                 # "+2 min", or "≈ +5 h 08" (an estimate)
    assert hm(model.tool_cost(model.path.opts, "etoiles")) in " ".join(texts) or "same duration" in " ".join(texts)
    assert page.findChildren(QLabel, "h2")[0].text() == "Add step"


# ---------------------------------------------------------------------------------------------- replace / variant
def _edit_kappa(page, value):
    """Edit Stacking's rejection threshold the way the user does: in the row of the panel."""
    settle(page.s.worker, page, 6)
    page.panel.expert = True
    page.panel.refresh(page.panel.node)
    _events()
    row = next(r for r in live_rows(page) if r.r["key"] == "stack.kappa")
    page.panel.set_setting(row.r, value)
    for _i in range(3):
        _events()                                     # the page builds itself again, with the choice box
    return row.r


def live_rows(page) -> list[CompactRow]:
    """The setting rows the panel shows now (not the ones a rebuild hid before deleting them)."""
    return [r for r in page.panel.findChildren(CompactRow)
            if not (r.isHidden() and r.testAttribute(Qt.WA_WState_ExplicitShowHide))]


def method_cards(page):
    return [c for c in page.panel.findChildren(MethodCard) if c.isVisibleTo(page.panel)]


def test_a_change_on_a_shared_done_block_offers_replace_or_variant(world, model, page):
    page.select_node(node(page, "empilement"))
    settle(world["w"], page)
    assert model.scope == [v.pid for v in model.paths]
    r = _edit_kappa(page, 4.0)
    settle(world["w"], page)
    ch = model.pending_change
    assert ch and ch["step"] == "empilement" and ch["value"] == 4.0 and len(ch["pids"]) == 4
    rep = button(page.panel, "Replace")
    var = button(page.panel, "Try as variant")
    paths = [q for q in model.paths if q.pid in ch["pids"]]
    c_all = model.cost_of_change(paths, ch)
    c_one = model.cost_of_change([model.path], ch)
    assert c_all.seconds > c_one.seconds > 0                                     # the scope of 4 costs more than Full
    assert hm(c_all.seconds) in rep.text() and hm(c_one.seconds) in var.text()
    assert c_all.seconds == api.cost(model.p, [api.with_change(model.p, q.pid, {"kind": "param", "stage": "stack",
                                             "name": "kappa", "value": 4.0}) for q in paths]).seconds
    # nothing was changed yet
    assert all("kappa" not in (q.params.get("stack") or {}) for q in model.paths)
    button(page.panel, "Cancel").click()
    assert model.pending_change is None and r["value"] == 5.0
    # Replace: the change goes to the scope, their results downstream are Outdated
    _edit_kappa(page, 4.0)
    button(page.panel, "Replace").click()
    settle(world["w"], page)
    assert all(q.params["stack"]["kappa"] == 4.0 for q in model.paths)
    assert model.state(model.paths[0], "stack") == "stale" and model.state(model.paths[0], "render_prep") == "stale"
    assert model.state(model.paths[0], "register") == "done"
    assert model.paths[0].computed is False and model.paths[0].rendered                       # a previous result stays
    g = Graph(model)
    assert g.state(node(page, "empilement")) == "stale"
    # back to the suggestion: the old results are Up to date again, nothing runs (5.1)
    model.undo()
    settle(world["w"], page)
    assert model.paths[0].computed and model.state(model.paths[0], "stack") == "done"
    assert "kappa" not in (model.paths[0].params.get("stack") or {})


def test_try_as_variant_keeps_the_original_and_shares_the_blocks_before(world, model, page):
    page.select_node(node(page, "empilement", 1))
    _edit_kappa(page, 3.5)
    n_before = len(model.paths)
    button(page.panel, "Try as variant").click()
    settle(world["w"], page)
    assert len(model.paths) == n_before + 1
    q = model.paths[-1]
    full = model.view(1)
    assert q.variant_of == full.pid and q.name.startswith("Full · ") and "3,5" not in q.name
    assert q.params["stack"]["kappa"] == 3.5 and "stack" not in full.params or "kappa" not in full.params.get("stack", {})
    assert model.path is q and model.p.current == q.pid
    g = Graph(model)
    for key in ("calibration", "alignement", "selection"):                      # shared up to the changed block
        assert g.node_of(q, key) is g.node_of(full, key)
    assert g.node_of(q, "empilement") is not g.node_of(full, "empilement")
    # the panel says what it is, with Keep and Discard
    page.select_node(g.node_of(q, "empilement"))
    settle(world["w"], page)
    assert any("Variant of" in t for t in texts_of(page.panel))
    assert button(page.panel, "Keep") and button(page.panel, "Discard")
    button(page.panel, "Discard").click()
    settle(world["w"], page)
    assert len(model.paths) == n_before and model.path.pid == full.pid
    # Keep: the variant replaces its original and takes its name
    page.select_node(node(page, "empilement", 1))
    _edit_kappa(page, 3.5)
    button(page.panel, "Try as variant").click()
    settle(world["w"], page)
    q = model.paths[-1]
    page.select_node(Graph(model).node_of(q, "empilement"))
    settle(world["w"], page)
    button(page.panel, "Keep").click()
    settle(world["w"], page)
    assert len(model.paths) == n_before and model.path.name == "Full" and model.path.params["stack"]["kappa"] == 3.5
    assert model.path.variant_of is None and model.view(1) is None


def test_a_method_card_asks_the_same_question(world, model, page):
    """Gradient's method card follows the rule too (SPEC §5.2): on a computed Gradient it offers Replace / variant."""
    page.select_node(node(page, "fond", 1))
    settle(world["w"], page)
    cards = method_cards(page)
    assert [c.ok for c in cards] == [True, True]
    other = next(c for c in cards if "Polynomial" in c.findChildren(QLabel)[0].text())
    other.clicked.emit()
    settle(world["w"], page)
    assert model.pending_change and model.pending_change["key"] == "fond"
    assert "Method: Polynomial" in model.pending_change["what"]
    var = button(page.panel, "Try as variant")
    var.click()
    settle(world["w"], page)
    q = model.paths[-1]
    assert q.opts["fond"] == "simple" and q.opts["etoiles"] == "mesurer" and q.variant_of == 1
    assert q.name == "Full · polynomial"


def test_a_method_card_says_the_duration_on_each_device_and_which_one_is_used(world, model, page):
    """A card never quotes one figure for an unnamed device: "GPU ≈ 8 min · CPU ≈ 23 h", the device of this machine in
    bold. A machine whose GPUs are hidden still shows what the GPU would take."""
    from astro_gpu.client.panel import durations_text
    w = world["w"]
    page.select_node(node(page, "etoiles", 1))
    settle(w, page)
    try:
        for gpu in (True, False):
            model.set_machine(gpu)
            settle(w, page)
            here = "gpu" if gpu else "cpu"
            cards = method_cards(page)
            assert len(cards) >= 2
            seen = 0
            for c in cards:
                lines = [lb.text() for lb in c.findChildren(QLabel) if "GPU" in lb.text() or "CPU" in lb.text()]
                if not lines:
                    continue
                seen += 1
                plain = re.sub(r"<[^>]+>", "", lines[0])
                assert "CPU" in plain and re.search(r"(GPU|CPU) (≈ )?[\d≤]", plain), plain
                if "GPU" in plain and "CPU" in plain:             # both: the used one is the bold one
                    bold = re.findall(r"<b[^>]*>(.*?)</b>", lines[0])
                    assert len(bold) == 1 and bold[0].startswith(here.upper()), (lines[0], here)
            assert seen >= 2
            # the figures are the engine's for each device, not the machine's twice
            ok = [m for m in model_methods("etoiles") if m["runs"] != "cpu" and not m["new"]]
            assert ok
            t = model.method_times(model.path, "etoiles", ok[0]["key"])
            assert set(t) == {"gpu", "cpu"} and t["gpu"][0] > 0 and t["cpu"][0] > t["gpu"][0] * 3, t
            assert durations_text(t, here).count("<b") == 1
        # the cards as the user sees them, on a machine whose GPUs are hidden (the case that quoted the CPU as the GPU)
        d = Path(os.environ.get("ASTRO_GPU_SHOTS") or world["folder"].parent / "shots")
        d.mkdir(parents=True, exist_ok=True)
        model.set_machine(False)
        for lang in ("en", "fr"):
            appmod.apply_language(lang)
            page.refresh()
            settle(w, page)
            assert not untranslated(texts_of(page.panel))
            page.root.grab().save(str(d / f"graph_method_cards_{lang}.png"))
    finally:
        appmod.apply_language("en")
        model.set_machine(True)
        page.refresh()
        settle(w, page)
    assert durations_text(None, "gpu") == "…"
    assert re.sub(r"<[^>]+>", "", durations_text({"cpu": (90.0, True)}, "gpu")) == "CPU " + hm(90.0)        # CPU only
    assert "CPU ≈ " in durations_text({"cpu": (90.0, False)}, "cpu")


def model_methods(block):
    from astro_gpu.client.model import methods_of
    return methods_of(block)


def test_the_durations_of_the_cards_are_read_by_the_worker_and_the_panel_follows(qapp, monkeypatch):
    """The GUI thread only reads the cache: a duration nobody has estimated yet is asked of the worker, the card shows
    "…", and costChanged says when the figures are in."""
    seen = []
    orig = api.estimate

    def watch(*a, **k):
        seen.append(threading.current_thread() is threading.main_thread())
        return orig(*a, **k)
    monkeypatch.setattr(api, "estimate", watch)
    p = _open_four()
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    assert w.wait(180) and m.ready, m.error
    try:
        v = m.path
        key = next(x["key"] for x in model_methods("debruitage") if not x["new"] and x["runs"] != "cpu")
        m._est.clear()                                              # as after an edit, before the worker has read
        fired = []
        m.costChanged.connect(lambda: fired.append(1))
        seen.clear()
        assert m.method_times(v, "debruitage", key) is None          # asks the worker, estimates nothing here
        assert m.method_times(v, "debruitage", key) is None          # and only once
        assert not any(seen)
        assert w.wait(180)
        assert fired and m.method_times(v, "debruitage", key) is not None
        assert seen and not any(seen), "the estimates were made in the GUI thread"
        # a ◇ method has the spec's figure for the device it is meant for, at once
        new = next(x for x in model_methods("debruitage") if x["new"])
        t = m.method_times(v, "debruitage", new["key"])
        assert len(t) == 1 and not next(iter(t.values()))[1]
    finally:
        w.stop()


def test_undo_and_redo_restore_the_workflows(world, model, page):
    n0 = len(model.paths)
    p = model.add_path("Extra", {"debruitage": False, "etoiles": "garder", "appris": False, "fond": "simple"})
    assert len(model.paths) == n0 + 1 and model.can_undo()
    model.rename(p.pid, "Renamed")
    model.set_param(model.view(1), "stack", "kappa", 4.0)
    settle(world["w"], page)
    assert model.view(1).params["stack"]["kappa"] == 4.0
    assert model.undo() and "stack" not in model.view(1).params
    assert model.undo() and model.view(p.pid).name == "Extra"
    assert model.undo() and len(model.paths) == n0
    assert model.redo() and len(model.paths) == n0 + 1
    assert model.redo() and model.view(p.pid).name == "Renamed"
    assert model.redo() and model.view(1).params["stack"]["kappa"] == 4.0
    assert not model.redo()
    assert model.undo() and model.can_redo()
    # one batch is one undo step
    with model.batch():
        model.rename(p.pid, "A")
        model.rename(p.pid, "B")
    assert model.view(p.pid).name == "B" and not model.can_redo()
    assert model.undo() and model.view(p.pid).name == "Renamed"
    # an edit that changes nothing is not an undo step
    depth = len(model._undo)
    model.rename(p.pid, "Renamed")
    assert len(model._undo) == depth
    # the computed results are never removed: a deleted workflow comes back with its states
    full = model.view(1)
    model.remove_path(model.index_of(1))
    assert model.view(1) is None
    model.undo()
    assert model.view(1) is not None and model.view(1).computed and full.pid == 1


def test_outputs_pulled_to_post_make_a_raw_stack_workflow(world, model, page):
    n0 = len(model.paths)
    stacking = node(page, "empilement")
    item = next(it for it in page.graph.scene().items() if isinstance(it, BlockItem) and it.node is stacking)
    assert item.port and not any(it.port for it in page.graph.scene().items()
                                 if isinstance(it, BlockItem) and it.node.key != "empilement")
    # drag the port onto Post, as the user does
    page.graph.start_wire(item)
    post = page.graph.final_item.sceneBoundingRect().center()
    page.graph.move_wire(post)
    assert page.graph.final_item.target
    page.graph.end_wire(post)
    settle(world["w"], page)
    assert len(model.paths) == n0 + 1
    q = model.paths[-1]
    assert q.name == "Raw stack" and q.end == "empilement" and model.path is q and model.sel_step == "post"
    assert model.block_keys(q) == ["calibration", "alignement", "selection", "empilement", "post"]
    assert model.all_stages(q)[-1] == "render_simple"
    g = Graph(model)
    assert g.summary(g.node_of(q, "post")) == "from Stacking"
    assert g.node_of(q, "empilement") is g.node_of(model.view(1), "empilement")      # shares everything up to Stacking
    assert any("Raw stack" in t for t in texts_of(page.graph))
    # no block can be added after an early output
    from astro_gpu.client.graph import ToolCard
    assert all(c.state == "in" for c in page.findChildren(ToolCard))
    assert [c.text() for c in page.findChildren(QLabel) if c.text() == "after the output"]
    ok, why, need = model.slider_state(q, "stars_number")
    assert ok is False and "before the Stars block" in why and need is None             # nothing to add after the exit
    assert model.slider_state(model.view(1), "stars_number")[0] is True


def test_the_mouse_over_and_on_every_block_works(world, model, page):
    """Hovering a block and pressing it (on its body and on its output port) is what the user does first; a lost helper
    method once broke it in the real client while no test moved the mouse."""
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QGraphicsSceneHoverEvent, QGraphicsSceneMouseEvent
    items = [it for it in page.graph.scene().items() if isinstance(it, BlockItem)]
    assert items
    for it in items:
        for pos in (QPointF(it.w / 2, BH / 2), QPointF(it.w, BH / 2)):           # the body, then the port's place
            ev = QGraphicsSceneHoverEvent(QEvent.GraphicsSceneHoverMove)
            ev.setPos(pos)
            it.hoverMoveEvent(ev)
            assert it.cursor().shape() == (Qt.CrossCursor if it.port and pos.x() == it.w else Qt.PointingHandCursor)
        assert isinstance(it._x_rect(), QRectF)
    stacking = next(it for it in items if it.port)
    press = QGraphicsSceneMouseEvent(QEvent.GraphicsSceneMousePress)
    press.setButton(Qt.LeftButton)
    press.setPos(QPointF(stacking.w, BH / 2))
    stacking.mousePressEvent(press)                                               # starts pulling the output
    assert stacking._wire
    page.graph.end_wire(QPointF(-1e4, -1e4))                                      # dropped on nothing
    stacking._wire = False
    settle(world["w"], page)


def test_new_workflow_duplicate_rename_delete(world, model, page, monkeypatch):
    from astro_gpu.client import graph as graphmod

    class FakeDialog:
        def __init__(self, session, start, parent):
            self.name = type("N", (), {"text": lambda s: "Mine"})()
            self.cur, self.opts, self.params = 0, dict(session.paths[0].opts), {}

        def exec(self):
            return True
    monkeypatch.setattr(graphmod, "NewWorkflowDialog", FakeDialog)
    n0 = len(model.paths)
    page.new_workflow()
    settle(world["w"], page)
    assert len(model.paths) == n0 + 1 and model.path.name == "Mine" and model.sel_step == "post"
    # Duplicate keeps the options, the settings and the exit
    src = model.view(1)
    FakeDialog.__init__ = lambda s, session, start, parent: (setattr(s, "name", type("N", (), {"text": lambda z: "Full copy"})()),
                                                              setattr(s, "cur", 3 + session.paths.index(start)),
                                                              setattr(s, "opts", dict(start.opts)), setattr(s, "params", {}))[0]
    page.workflow_action("duplicate", src.pid)
    settle(world["w"], page)
    cp = model.path
    assert cp.name == "Full copy" and cp.opts == src.opts and cp.params == src.params
    # Rename and Delete (confirmed)
    page.rename(cp.pid, "Other")
    assert model.view(cp.pid).name == "Other"
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    page.workflow_action("delete", cp.pid)
    settle(world["w"], page)
    assert model.view(cp.pid) is None and len(model.paths) == n0 + 1
    assert model.sel_step == "sources"
    assert any("deleted" in h[1] for h in model.history)


def test_clicking_selects_and_sets_the_scope(world, model, page):
    n = node(page, "debruitage")
    page.select_node(n)
    settle(world["w"], page)
    assert model.sel_step == "debruitage" and model.scope == [v.pid for v in n.paths] and len(n.paths) == 3
    assert model.path in n.paths
    n = node(page, "etoiles", 4)
    page.select_node(n)
    assert model.path.pid == 4 and model.scope == [4]
    page.select_node(page.graph.graph.final)
    assert model.sel_step == "final" and model.scope is None
    settle(world["w"], page)
    assert "Layers" in texts_of(page.panel) and "Pending" in " ".join(texts_of(page.panel))
    opened = []
    page.openBlock.connect(lambda: opened.append(1))
    page.graph.blockOpened.emit(n)
    assert opened


# ---------------------------------------------------------------------------------------------- the panel
def test_the_panel_lists_the_settings_of_the_engine(world, model, page):
    page.select_node(node(page, "empilement"))
    settle(world["w"], page)
    rows = {r.r["key"]: r for r in live_rows(page)}
    assert {"stack.scale", "stack.pixfrac"} <= set(rows)
    sc = rows["stack.scale"].r
    assert sc["suggested"] == 1.5 and re.search(r"stars of [\d.]+ px", sc["reason"]), sc["reason"]   # from the star size
    assert sc["min"] == 1.0 and sc["max"] == 3.0
    assert sc["unit"] == "×" and not sc["changed"] and sc["type"] == "float"
    # advanced settings: the others, from the engine, nothing left out
    page.panel.expert = True
    page.panel.refresh(page.panel.node)
    keys = {r.r["key"] for r in live_rows(page)}
    assert {"stack.kappa", "stack.gain", "stack.half_block", "stack.read_noise"} <= keys
    assert "stack.exclude" not in keys and "stack.min_transparency" not in keys    # set by the project / Selection
    # the Details of a clicked setting: range bar, "Suggested: v · reason", help
    rows = {r.r["key"]: r for r in live_rows(page)}
    rows["stack.scale"].focused.emit(rows["stack.scale"].r)                          # a click on the row
    dt = texts_of(page.panel.details)
    assert any(t.startswith("Suggested: 1.5 · " + sc["reason"][:12]) for t in dt), dt
    page.panel.expert = False


def test_the_colour_block_offers_one_factor_per_colour(world, model, page):
    """The colour correction (2 November): a normal setting of the Colour block, full correction by default, its
    choices in words."""
    page.select_node(node(page, "couleur"))
    settle(world["w"], page)
    rows = {r.r["key"]: r for r in live_rows(page)}
    assert "color.correction" in rows
    r = rows["color.correction"].r
    assert r["label"] == "Colour correction" and r["value"] == "full" and not r["changed"]
    assert r["choices"] == [("full", "Full correction"), ("per_colour", "One factor per colour")]
    assert "One factor per colour" in r["help"]


def test_a_settings_row_changes_marks_and_goes_back_to_the_suggestion(world, model, page):
    page.select_node(node(page, "debruitage"))
    settle(world["w"], page)
    rows = {r.r["key"]: r for r in live_rows(page)}
    it = rows["denoise.iterations"]
    assert it.r["changed"] and it.r["value"] == 6000 and it.r["suggested"] == 24000      # Full has 6000 stored
    back = [b for b in it.findChildren(QPushButton) if b.text() == "↺"][0]
    assert not back.isHidden()
    assert "24000" in back.toolTip() or "24 000" in back.toolTip() or "24,000" in back.toolTip()
    name = it.findChildren(QLabel)[0]
    assert "e6a23c" in name.styleSheet().lower()                                          # the name of a changed row
    back.click()                                                                          # ↺: back to the suggestion
    settle(world["w"], page)
    ch = model.pending_change                       # the denoise is done and shared by 3: Replace / variant is asked
    assert ch and ch["kind"] == "param" and ch["value"] is None and len(ch["pids"]) == 3
    button(page.panel, "Replace").click()
    settle(world["w"], page)
    assert "iterations" not in (model.view(1).params.get("denoise") or {})              # the stored value is gone
    rows = {r.r["key"]: r for r in live_rows(page)}
    assert not rows["denoise.iterations"].r["changed"] and rows["denoise.iterations"].r["value"] == 24000
    assert model.state(model.view(1), "denoise") == "stale"                               # 24000 steps: to compute again
    model.undo()
    settle(world["w"], page)
    assert model.state(model.view(1), "denoise") == "done"


def stack_cards(page):
    from astro_gpu.client.panel import StackMethodCard
    return [c for c in page.panel.findChildren(StackMethodCard) if c.isVisibleTo(page.panel)]


def test_the_stacking_method_is_chosen_on_cards_and_shows_only_its_own_settings(world, model, page):
    """Drizzle (the default) and Classic stack, from the engine's list of methods: each shows its own settings only;
    choosing the other on a computed stack asks Replace or Try as variant; the withdrawn switches never show."""
    page.select_node(node(page, "empilement"))
    settle(world["w"], page)
    page.panel.expert = True
    page.panel.refresh(page.panel.node)
    settle(world["w"], page)
    cards = stack_cards(page)
    assert [c.key for c in cards] == ["drizzle", "classic"] and [c.on for c in cards] == [True, False]
    texts = " ".join(texts_of(page.panel))
    assert "Drizzle" in texts and "Classic stack" in texts
    keys = {r.r["key"] for r in live_rows(page)}
    assert {"stack.scale", "stack.pixfrac", "stack.kappa"} <= keys
    assert not any(k.startswith("stack.classic_") for k in keys) and "stack.method" not in keys
    assert "stack.rejection" not in keys                          # Hubble withdrawn: nothing left to choose
    rows = {r.r["key"]: r.r for r in live_rows(page)}
    for key, gone in (("stack.weighting", "noise"), ("stack.normalisation", "local")):
        if key in rows:
            assert gone not in [c[0] for c in rows[key]["choices"]], key
    done = model.state(model.view(1), "stack") == "done"
    cards[1].clicked.emit()                                       # Classic stack
    settle(world["w"], page)
    ch = model.pending_change
    if done:                                                      # a computed stack: Replace or Try as variant
        assert ch and ch["kind"] == "param" and (ch["stage"], ch["name"], ch["value"]) == ("stack", "method",
                                                                                           "classic")
        button(page.panel, "Replace").click()
        settle(world["w"], page)
    else:
        assert ch is None
    assert model.view(1).params["stack"]["method"] == "classic"
    cards = stack_cards(page)
    assert [c.on for c in cards] == [False, True]
    keys = {r.r["key"] for r in live_rows(page)}
    assert {"stack.classic_debayer", "stack.classic_interpolation", "stack.classic_clamping",
            "stack.classic_rejection"} <= keys
    assert not keys & {"stack.scale", "stack.pixfrac", "stack.kappa", "stack.scale_rule", "stack.gain"}
    labels = {r.r["name"]: r.r["label"] for r in live_rows(page)}
    assert labels["classic_debayer"] == "Debayering" and labels["classic_clamping"] == "Clamping"
    # the register's per-block colour alignment is withdrawn too: not in Registration
    page.select_node(node(page, "alignement"))
    settle(world["w"], page)
    assert "register.channel_align" not in {r.r["key"] for r in live_rows(page)}
    page.panel.expert = False
    model.undo()
    settle(world["w"], page)
    assert "method" not in (model.view(1).params.get("stack") or {})


def test_the_engine_advice_is_a_recommended_outline_that_never_changes_the_method(world, model, page, monkeypatch):
    """The engine's advice on "method" (ParamInfo.advice, here a fake one: Classic, with its reason) puts a discreet
    "Recommended" outline and an "i" with the reason on that card; the method stays the user's."""
    import copy

    from astro_gpu.client import i18n
    from astro_gpu.client.widgets import ACCENT
    real = model.param_info
    reason = "{n} frames (drizzle needs 15): Classic"                     # one of the engine's (core/coverage.py)

    def fake(p, stage, name):
        info = real(p, stage, name)
        if (stage, name) == ("stack", "method") and info is not None:
            info = copy.copy(info)
            info.advice, info.advice_reason, info.advice_args = "classic", reason, {"n": 12}
        return info
    monkeypatch.setattr(model, "param_info", fake)
    page.select_node(node(page, "empilement"))
    settle(world["w"], page)
    page.panel.refresh(page.panel.node)
    settle(world["w"], page)
    cards = {c.key: c for c in stack_cards(page)}
    assert cards["drizzle"].on and not cards["drizzle"].recommended and cards["drizzle"].tag is None
    rec = cards["classic"]
    assert rec.recommended and not rec.on and rec.tag.text() == "Recommended"
    assert ACCENT in rec.tag.styleSheet() and "border-radius" in rec.tag.styleSheet()
    assert rec.info.text() == "i" and rec.info.toolTip() == "12 frames (drizzle needs 15): Classic"
    assert model.pending_change is None and "method" not in (model.view(1).params.get("stack") or {})
    # in French: the outline and the reason are translated
    appmod.apply_language("fr")
    try:
        page.panel.refresh(page.panel.node)
        settle(world["w"], page)
        rec = {c.key: c for c in stack_cards(page)}["classic"]
        assert rec.tag.text() == "Conseillé" and rec.info.toolTip() == "12 images (le drizzle en demande 15) : Classique"
        assert "Recommended" not in i18n.missing()
    finally:
        appmod.apply_language("en")
    # the engine's own advice (none: nothing is marked)
    from astro_gpu.client.settings import method_advice
    monkeypatch.setattr(model, "param_info", real)
    page.panel.refresh(page.panel.node)
    settle(world["w"], page)
    adv = method_advice(model, model.path)
    assert [c.key for c in stack_cards(page) if c.recommended] == ([adv[0]] if adv else [])


def test_the_selection_rule_is_said_and_the_thresholds_act_only_with_the_thresholds_rule(world, model, page):
    """The Selection panel says the active rule in plain words (min_weight by default, k × σ, thresholds); the two
    threshold editors are enabled with the thresholds rule only, with a one-line reason otherwise; the chart has no
    grips then. In French too. (The state of Registration does not matter here: the stored rule is set in memory.)"""
    from PySide6.QtWidgets import QDoubleSpinBox, QLabel
    model.scope = None
    page.select_node(node(page, "selection"))
    settle(world["w"], page)
    stored = model.path.params.setdefault("stack", {})
    before = stored.get("selection", "<unset>")

    def thresholds_editors():
        return [e for e in page.panel.findChildren(QDoubleSpinBox)
                if e.isVisibleTo(page.panel) and not isinstance(e.parent(), CompactRow)]

    def reason():
        return [b.text() for b in page.panel.findChildren(QLabel) if b.objectName() == "thresholds_reason"
                and b.isVisibleTo(page.panel)]
    try:
        assert model.selection_rule() == "min_weight" and not model.thresholds_active()      # the default
        for rule, said, live, summary in (
                ("min_weight", "Every aligned frame is kept, except those weighing less than 5 % of the best one.",
                 False, "every frame, weight ≥ 5 % of the best"),
                ("sota", "Frames far worse than the night's median (over 2.5 σ) in star size, roundness or "
                         "transparency are left out, then those weighing less than 5 % of the best one.", False,
                 "k × σ filters · weight ≥ 5 % of the best"),
                ("thresholds", "Frames under the minimum transparency or over the maximum star size below are left out.",
                 True, "transparency ≥ ")):
            stored["selection"] = rule
            model._cache.clear()
            page.panel.refresh(page.panel.node)
            settle(world["w"], page)
            assert model.selection_rule() == rule and model.thresholds_active() is live
            assert said in texts_of(page.panel)
            eds = thresholds_editors()
            assert len(eds) == 2 and all(e.isEnabled() is live for e in eds), rule
            assert reason() == ([] if live else ["These thresholds act only when the frame selection is “Thresholds”."]), rule
            assert summary in Graph(model).summary(node(page, "selection"))
            assert any(r.r["key"] == "stack.selection" for r in live_rows(page))      # the rule is chosen right here
        # in French
        stored["selection"] = "min_weight"
        model._cache.clear()
        appmod.apply_language("fr")
        try:
            page.panel.refresh(page.panel.node)
            settle(world["w"], page)
            assert "Toutes les images alignées sont gardées, sauf celles qui pèsent moins de 5 % de la meilleure."                 in texts_of(page.panel)
            assert reason() == ["Ces seuils n'agissent que si la sélection des images est sur « Seuils »."]
        finally:
            appmod.apply_language("en")
    finally:
        if before == "<unset>":
            stored.pop("selection", None)
        else:
            stored["selection"] = before
        if not stored:
            model.path.params.pop("stack", None)
        model._cache.clear()


def test_selection_thresholds_come_from_the_stack_settings(world, model, page):
    """With the Thresholds selection rule (the only one the thresholds act with) they are the stack's settings and the
    panel's count is the engine's preview. (Needs Registration done at the present settings.)"""
    model.scope = None
    stored = model.path.params.setdefault("stack", {})
    before = stored.get("selection", "<unset>")
    stored["selection"] = "thresholds"
    model._cache.clear()
    try:
        page.select_node(node(page, "selection"))
        settle(world["w"], page)
        assert model.thresholds_active()
        t, f = model.thresholds_now()
        assert t == 60 and abs(f - 2.0 * 3.95) < 1e-6 and model.thresholds_unit() == "px"
        st, sf = model.suggested_thresholds()
        assert st == 60 and abs(sf - f) < 1e-6
        assert model.kept_frames() == 40 and any("40 of 40 frames kept" in t for t in texts_of(page.panel))
        # a change goes through the same choice (the stack is done)
        page.panel.propose_selection(70, f)
        settle(world["w"], page)
        assert model.pending_change and model.pending_change["kind"] == "params"
        assert model.pending_change["items"][0] == ("stack", "min_transparency", 0.7)
        button(page.panel, "Cancel").click()
        model.request_selection(70, f)
        settle(world["w"], page)
        assert model.kept_frames() <= 40 and model.selection_total() == 40
        assert "transparency ≥ 60 % · stars ≤ 7.9 px" in Graph(model).summary(node(page, "selection"))
    finally:
        if before == "<unset>":
            stored.pop("selection", None)
        else:
            stored["selection"] = before
        if not stored:
            model.path.params.pop("stack", None)
        model._cache.clear()


# ---------------------------------------------------------------------------------------------- Spec mode
def test_spec_mode_shows_what_the_engine_does_not_have_disabled(world, model, page):
    from astro_gpu.client.graph import ToolCard
    page.select_node(node(page, "debruitage"))
    settle(world["w"], page)
    plain = texts_of(page.panel)
    assert not any("◇" in t for t in plain)
    assert len(method_cards(page)) == 1                                        # Noise2Noise only
    prefs.set("spec_mode", True)
    page.refresh()
    settle(world["w"], page)
    cards = method_cards(page)
    assert len(cards) == 4 and [c.ok for c in cards] == [True, False, False, False]
    assert any("◇" in "".join(lb.text() for lb in c.findChildren(QLabel)) for c in cards[1:])
    page.panel.expert = False
    rows = live_rows(page)
    new_rows = [r for r in rows if r.r["kind"] == "new"]
    assert new_rows and not [b for r in new_rows for b in r.findChildren(QPushButton)
                             if b.text() == "↺" and b.isVisibleTo(r)]           # never changed: no ↺
    for r in new_rows:
        editors = r.findChildren(QComboBox) + r.findChildren(QCheckBox) + r.findChildren(QDoubleSpinBox) + \
            r.findChildren(QSpinBox)
        assert editors and not any(e.isEnabled() for e in editors), r.r["key"]
    tc = {c.key: c for c in page.findChildren(ToolCard)}
    assert "deconv" in tc and not tc["deconv"].isEnabled() and tc["deconv"].state == "off"
    assert "◇" in " ".join(lb.text() for lb in tc["deconv"].findChildren(QLabel))
    # a click on a ◇ method changes nothing
    cards[1].clicked.emit() if cards[1].ok else None
    assert model.pending_change is None
    # the Gradient block says the reorder is for later
    page.select_node(node(page, "fond"))
    settle(world["w"], page)
    assert any("Gradient before Color" in t for t in texts_of(page.panel))
    # a ◇ value is never stored and never changes a key
    keys_before = {s: model.stage_key(model.paths[0], s) for s in model.all_stages(model.paths[0])}
    page.panel.set_setting(next(r for r in new_rows).r, 1)
    assert keys_before == {s: model.stage_key(model.paths[0], s) for s in model.all_stages(model.paths[0])}
    assert model.pending_change is None and not model.can_undo()


# ---------------------------------------------------------------------------------------------- language
def _visit_everything(page, w):
    g = page.graph.graph
    for n in list(g.nodes):
        page.select_node(n)
        settle(w, page, 6)


def test_the_french_switch_translates_everything_visible(world, model, page):
    prefs.set("spec_mode", True)
    appmod.apply_language("fr")
    i18n.missing()
    page.graph.b_all.setText(i18n._("Show all"))
    page.refresh()
    settle(world["w"], page)
    _visit_everything(page, world["w"])
    page.graph.show_toast(i18n._("+ {t} to run").format(t="3 min"))
    page.select_node(node(page, "empilement"))
    _edit_kappa(page, 4.0)                                                    # the choice box, in French
    settle(world["w"], page)
    t = texts_of(page.root)
    assert not untranslated(t), untranslated(t)
    assert not i18n.missing(), sorted(i18n.missing())
    assert "Remplacer" in " ".join(t) and "Essayer en variante" in " ".join(t)
    assert any(x.startswith("Suggéré : ") for x in t)                        # the Details of the setting
    button(page.panel, "Annuler").click()
    # block names, states, summaries and the toolbar in French
    g = page.graph.graph
    tip = " ".join(it.toolTip() for it in page.graph.scene().items() if isinstance(it, BlockItem))
    assert "Empilement" in tip and "À jour" in tip and "À calculer" in tip
    assert i18n._("Gradient") == "Gradient"
    assert g.summary(next(n for n in g.nodes if n.key == "alignement")) == "sur l'image la plus nette"
    assert "Ajouter une étape" in " ".join(t)
    # back to English at runtime: the page is built again, in English
    appmod.apply_language("en")
    page.refresh()
    settle(world["w"], page)
    assert "Add step" in texts_of(page.root)


def test_decimal_commas_in_french_reasons(world, model, page):
    appmod.apply_language("fr")
    page.select_node(node(page, "empilement"))
    settle(world["w"], page)
    r = next(x for x in live_rows(page) if x.r["key"] == "stack.scale").r
    assert re.search(r"\d,\d+ px", r["reason"]) and not re.search(r"\d\.\d", r["reason"]), r["reason"]   # the star size


# ---------------------------------------------------------------------------------------------- GUI thread
def test_nothing_slow_runs_in_the_gui_thread(qapp, monkeypatch):
    """Disk scans, describe, outputs, previews and warm run in the worker thread; the keys, states and costs the GUI
    asks for afterwards read caches."""
    from astro_gpu.core import store
    calls = []

    def watch(mod, name):
        orig = getattr(mod, name)

        def f(*a, **k):
            calls.append((name, threading.current_thread() is threading.main_thread()))
            return orig(*a, **k)
        monkeypatch.setattr(mod, name, f)
    for name in ("warm", "describe", "stage_outputs", "selection_preview", "frame_table", "scan_folder"):
        watch(api, name)
    for name in ("list_frames", "file_id"):
        watch(store, name)
    p = _open_four()
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    root = QWidget()
    root.setObjectName("root")
    v = QVBoxLayout(root)
    pg = GraphPage(m)
    v.addWidget(pg)
    root.resize(1680, 934)
    root.show()
    assert w.wait(180)
    settle(w, pg)
    assert api._MACHINE and any(c == "warm" and not main for c, main in calls)   # torch was imported by the worker
    calls.clear()                                                            # now: what the GUI thread does alone
    for n in list(pg.graph.graph.nodes):
        pg.select_node(n)
        QApplication.processEvents()
    pg.select_node(node(pg, "empilement"))
    settle(w, pg)
    _edit_kappa(pg, 4.0)
    button(pg.panel, "Try as variant").click()
    m.add_path("X", {"debruitage": True, "etoiles": "mesurer", "appris": True, "fond": "halpha"})
    m.undo()
    m.cost()
    m.all_pending()
    m.states_probe = None
    settle(w, pg)
    main_calls = [c for c, main in calls if main]
    assert not main_calls, main_calls
    assert any(c in ("describe",) for c, main in calls if not main)           # the worker answered the panel
    assert not any(main for _c, main in calls)
    w.stop()
    root.close()


def test_autosave_writes_only_the_definition_and_flushes_on_close(qapp, tmp_path):
    root = tmp_path / "copy"
    root.mkdir()
    shutil.copy(F_PROJECT / "project.astroproj", root / "project.astroproj")
    p = api.open_project(root)
    w = ApiWorker()
    m = ProjectModel(p, w)
    assert w.wait(180)
    before = (root / "project.astroproj").read_bytes()
    m.add_path("Autosaved", {"debruitage": True, "etoiles": "garder", "appris": False, "fond": "simple"})
    assert (root / "project.astroproj").read_bytes() == before                  # debounced: not yet
    time.sleep(0.45)
    qapp.processEvents()
    assert "Autosaved" in (root / "project.astroproj").read_text(encoding="utf-8")
    m.rename(m.paths[-1].pid, "Last one")
    m.flush()                                                                    # Save flushes at once
    assert [x.name for x in api.open_project(root).workflows][-1] == "Last one"
    m.select(index=0)
    m.close()
    names = {f.name for f in root.iterdir()}
    assert names <= {"project.astroproj", "state.json", "logs", "cache", "store", "inputs", "scratch"}, names
    w.stop()


# ---------------------------------------------------------------------------------------------- screenshots
def _shots_dir(tmp_path) -> Path:
    d = Path(os.environ.get("ASTRO_GPU_SHOTS") or tmp_path / "shots")
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_shots(world, model, page, tmp_path):
    """graph, graph_all, panel_shared (Stacking, Denoise), panel_stars, variant and spec, in English and in French:
    no text cut, no English left in French, no overlap of the blocks."""
    out = _shots_dir(tmp_path)
    w = world["w"]
    cut = {}

    def shot(name):
        settle(w, page, 8)
        page.root.grab().save(str(out / f"{name}.png"))
        assert (out / f"{name}.png").stat().st_size > 5000
        c = cut_texts(page.root) + graph_faults(page) + panel_faults(page)
        if c:
            cut[name] = c
        if i18n.language() == "fr":
            assert not untranslated(texts_of(page.root)), (name, untranslated(texts_of(page.root)))

    for lang in ("en", "fr"):
        appmod.apply_language(lang)
        _reset(model)
        prefs.set("spec_mode", False)
        page.graph.toast.hide()                          # a word said in the other language
        page.panel.expert, page.panel.focus_key = False, None
        page.refresh()
        page.graph.b_all.setText(i18n._("Show all"))
        shot(f"graph_{lang}")
        page.graph.b_all.setChecked(True)
        shot(f"graph_all_{lang}")
        page.graph.b_all.setChecked(False)
        for key, name in (("empilement", "panel_shared"), ("debruitage", "panel_denoise"), ("etoiles", "panel_stars"),
                          ("selection", "panel_selection"), ("fond", "panel_gradient")):
            page.select_node(node(page, key))
            shot(f"{name}_{lang}")
        # the variant choice
        page.select_node(node(page, "empilement"))
        settle(w, page)
        _edit_kappa(page, 4.0)
        shot(f"variant_{lang}")
        button(page.panel, i18n._("Try as variant")).click()
        assert model.paths[-1].name == f"Full · {i18n._('Rejection threshold').lower()} 4"   # named in the language
        page.select_node(Graph(model).node_of(model.paths[-1], "empilement"))
        shot(f"variant_made_{lang}")
        _reset(model)
        page.graph.toast.hide()
        page.panel.expert, page.panel.focus_key = False, None
        # Spec mode
        prefs.set("spec_mode", True)
        page.panel.expert = False
        page.select_node(node(page, "debruitage"))
        page.refresh()
        shot(f"spec_denoise_{lang}")
        page.select_node(node(page, "calibration"))
        shot(f"spec_calibration_{lang}")
        prefs.set("spec_mode", False)
        page.refresh()
    assert not cut, cut
    # no block overlaps another one
    items = [it for it in page.graph.scene().items() if isinstance(it, BlockItem)]
    rects = [it.sceneBoundingRect().adjusted(2, 2, -2, -2) for it in items]
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            assert not a.intersects(b)
    assert BW > 0 and BH > 0 and QPointF


# ---------------------------------------------------------------------------------------------- more of the page
def test_the_page_says_it_is_reading_while_the_worker_works(qapp):
    p = _open_four()
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    root = QWidget()
    root.setObjectName("root")
    v = QVBoxLayout(root)
    pg = GraphPage(m)
    v.addWidget(pg)
    root.resize(1680, 934)
    root.show()
    qapp.processEvents()
    assert pg.graph.wait.isVisible() and pg.graph.wait.text() == "Reading the project…"
    assert not any(isinstance(it, BlockItem) and it.node.key == "calibration" for it in pg.graph.scene().items())
    assert w.wait(180)
    settle(w, pg)
    assert not pg.graph.wait.isVisible()
    assert any(isinstance(it, BlockItem) and it.node.key == "calibration" for it in pg.graph.scene().items())
    appmod.apply_language("fr")
    pg.graph.show_wait()
    m.error = "boom"
    pg.graph.show_wait()
    assert pg.graph.wait.isVisible() and "boom" in pg.graph.wait.text() and "ne peut pas" in pg.graph.wait.text()
    m.error = ""
    w.stop()
    root.close()


def test_dropping_a_tool_on_a_lane_adds_it_where_the_rules_say(world, model, page):
    graph = page.graph
    graph.drag_tool = "etoiles"                                  # a card of the toolbar is being dragged
    graph.rebuild()
    assert {g.pid for g in graph.ghosts} == {2, 3}               # the lanes that do not have Stars yet
    graph.drag_tool = None
    graph.rebuild()
    toasts = []
    page.toasted.connect(toasts.append)
    graph.toolDropped.emit("etoiles", 3)                         # on Classic + denoise
    settle(world["w"], page)
    assert model.view(3).opts["etoiles"] == "mesurer" and model.sel_step == "etoiles" and model.path.pid == 3
    assert model.scope == [1, 3]                                 # its Stars are the same as Full's: one block, two workflows
    assert not toasts or "Denoise added" not in toasts[0]        # it had Denoise
    graph.toolDropped.emit("etoiles", 2)                         # on Classic stack: Denoise comes with it
    settle(world["w"], page)
    assert model.view(2).opts["debruitage"] and any("Denoise added" in t for t in toasts)
    graph.toolDropped.emit("deconv", 2)                          # ◇: nothing happens, and it says so
    assert any(t.startswith("◇") for t in toasts)
    assert model.undo() and model.undo()


def test_a_project_without_workflows_has_a_graph_and_an_import_button(qapp, tmp_path):
    p = project2.create(tmp_path / "P", "P", Target(name="M 1"))
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    assert w.wait(120)
    root = QWidget()
    root.setObjectName("root")
    v = QVBoxLayout(root)
    pg = GraphPage(m)
    v.addWidget(pg)
    root.resize(1680, 934)
    root.show()
    settle(w, pg)
    items = [it for it in pg.graph.scene().items() if isinstance(it, BlockItem)]
    assert [it.node.key for it in items] == ["sources", "final"] or {it.node.key for it in items} == {"sources", "final"}
    assert "+  New workflow" in [getattr(g, "text", "") for g in pg.graph.scene().items()]
    got = []
    pg.addSources.connect(lambda: got.append(1))
    pg.select_node(pg.graph.graph.root)
    settle(w, pg)
    button(pg.panel, "Import folder").click()
    assert got
    assert [lb.text() for lb in pg.findChildren(QLabel) if lb.text() == "Add step"]
    assert m.total_seconds() == 0 and m.all_pending() == []
    w.stop()
    root.close()


def test_a_change_of_an_automatic_setting_is_seen_by_the_what_if(world, model, page):
    """The API keys what-if workflows by object and has already asked the keys of the copy it returns (the
    suggestion of an automatic setting needs them): the model copies it again, so that scale, min_transparency, the
    iterations... are seen to change a result (they used to give the old key, and 0 s)."""
    import copy
    full = model.view(1)
    stale = api.with_change(model.p, 1, {"kind": "param", "stage": "stack", "name": "scale", "value": 2.0})
    mine = model._changed(full, {"kind": "param", "stage": "stack", "name": "scale", "value": 2.0})
    assert api.stage_key(model.p, mine, "stack") != model.stage_key(full, "stack")
    assert api.stage_key(model.p, copy.deepcopy(stale), "stack") == api.stage_key(model.p, mine, "stack")
    for name, value in (("scale", 2.0), ("half_block", 20), ("min_transparency", 0.7)):
        ch = {"kind": "param", "stage": "stack", "name": name, "value": value}
        assert model.throws_away(model.paths, ch), name
        assert model.cost_with([full], ch) > 0, name
    ch = {"kind": "param", "stage": "denoise", "name": "iterations", "value": 8000}
    assert model.throws_away([full], ch) and model.cost_with([full], ch) > 0
    # an invalid value is refused with the name of the setting, and nothing changes
    toasts = []
    page.toasted.connect(toasts.append)
    page.select_node(node(page, "empilement"))
    settle(world["w"], page)
    page.panel.propose("stack", "scale", 9.0, 1.5, "Drizzle scale: 1.5 → 9", "scale 9")
    assert toasts and "stack.scale" in toasts[0] and model.pending_change is None and not model.can_undo()
    assert model.view(1).params.get("stack", {}).get("scale") is None


def test_no_setting_of_the_engine_is_out_of_reach(world, model, page):
    """Every setting a stage describes (but the ones the project or the options own) has a row in the panel of its
    block, in some workflow; no row is listed twice in a panel."""
    from astro_gpu.client.settings import HIDDEN, setting_method, stack_method
    want = set()
    for v in model.paths:
        for stage in model.all_stages(v):
            for info in api.describe(model.p, v.pid, stage):
                if info.level == "internal" or (stage, info.name) in HIDDEN:
                    continue
                m = setting_method({"stage": stage, "name": info.name})
                if m is not None and m != stack_method(model, v):
                    continue                                      # the other stacking method's (its own test)
                want.add(f"{stage}.{info.name}")
    got = set()
    page.panel.expert = True
    for v in model.paths:
        for key in model.block_keys(v) + ["demi_piles"]:
            n = Graph(model).node_of(v, key) if key != "demi_piles" else None
            if n is None:
                continue
            page.select_node(n)
            settle(world["w"], page, 5)
            keys = [r.r["key"] for r in live_rows(page)]
            assert len(keys) == len(set(keys)), (v.name, key, keys)
            got |= set(keys)
    page.panel.expert = False
    assert not (want - got), sorted(want - got)


# ---------------------------------------------------------------------------------------------- breakpoints
def block_items(page):
    return [it for it in page.graph.scene().items() if isinstance(it, BlockItem)]


def item_of(page, key, pid=None):
    n = node(page, key, pid)
    return next(it for it in block_items(page) if it.node is n)


def menu_of(page, n, monkeypatch):
    """The context menu of a block, as built (QMenu.exec is not run: it would wait for a click)."""
    from PySide6.QtWidgets import QMenu

    from astro_gpu.client import graph as graphmod
    seen = []

    class Menu(QMenu):
        def exec(self, *a, **k):
            seen.append(self)
    monkeypatch.setattr(graphmod, "QMenu", Menu)
    page.graph.block_menu(n, QPoint(0, 0))
    return seen[-1] if seen else None


def pixel_near(page, it, colour: str, radius: float = 9.0, tol: int = 36) -> bool:
    """Whether the view draws something of this colour around the corner of a block (where its review dot sits)."""
    from PySide6.QtGui import QColor
    QApplication.processEvents()
    img = page.graph.viewport().grab().toImage()
    c = page.graph.mapFromScene(it.mapToScene(2, 2))
    want = QColor(colour)
    for dx in range(-int(radius), int(radius) + 1):
        for dy in range(-int(radius), int(radius) + 1):
            x, y = c.x() + dx, c.y() + dy
            if 0 <= x < img.width() and 0 <= y < img.height() and dx * dx + dy * dy <= radius * radius:
                got = img.pixelColor(x, y)
                if abs(got.red() - want.red()) + abs(got.green() - want.green()) + abs(got.blue() - want.blue()) <= tol:
                    return True
    return False


_API_STATES = api.states


def pin(model, monkeypatch, *done):
    """The engine's states with these blocks (of the views) computed and every other stage not, whatever is on the disk
    (the keys stay the real ones: the settings of a stage are asked by them)."""
    from astro_gpu.client.blocks import ENGINE_BLOCK

    def states(p, update=True):
        out = dict(_API_STATES(p, update=False))
        block_of = {(w.id, st): ENGINE_BLOCK[b.block] for w in p.workflows for b in api.blocks(p, w.id) for st in b.stages}
        for (wid, stage), st in out.items():
            if block_of.get((wid, stage)) in done:
                out[(wid, stage)] = api.StageState("done", "", st.key)
            else:
                out[(wid, stage)] = api.StageState("pending", "never computed", st.key)
        return out
    monkeypatch.setattr(api, "states", states)
    model._sync(prefetch=False)
    model.pathsChanged.emit()


def test_every_block_has_pause_here_for_review_and_the_dot_shows(world, model, page, monkeypatch):
    from astro_gpu.client.blocks import BREAK_COL
    pin(model, monkeypatch)                                                     # nothing computed
    settle(world["w"], page)
    # no menu, no dot, on the Frames and Post blocks
    assert menu_of(page, page.graph.graph.root, monkeypatch) is None
    assert menu_of(page, page.graph.graph.final, monkeypatch) is None
    assert all(it.bp is None for it in block_items(page))
    keys = [n.key for n in page.graph.graph.nodes if n.key not in ("sources", "final")]
    assert {"calibration", "alignement", "selection", "empilement", "astrometrie", "couleur", "debruitage", "etoiles",
            "fond", "post"} <= set(keys)
    for n in page.graph.graph.nodes:
        if n.key in ("sources", "final"):
            continue
        m = menu_of(page, n, monkeypatch)
        act = next(a for a in m.actions() if a.text() == "Pause here for review")
        assert act.isCheckable() and not act.isChecked(), n.key
        assert not any(a.text() == "Mark as reviewed" for a in m.actions())
    # choose it on Calibration: every workflow going through the block gets the breakpoint (the block is shared)
    cal = node(page, "calibration")
    assert len(cal.paths) == len(model.paths) == 4
    act = next(a for a in menu_of(page, cal, monkeypatch).actions() if a.text() == "Pause here for review")
    act.trigger()
    settle(world["w"], page)
    assert [w.breakpoints for w in model.p.workflows] == [["calibration"]] * 4
    it = item_of(page, "calibration")
    assert it.bp == ("pending", 4, 4) and all(i.bp is None for i in block_items(page) if i is not it)
    assert "Review point: the run stops here for your review." in it.toolTip()
    assert pixel_near(page, it, BREAK_COL)                                      # a red dot, on the block's corner
    assert any(h[1] == "Pause set after {block}" for h in model.history)
    assert graph_faults(page) == []                                             # the dot hides nothing and cuts nothing
    # the menu now shows it checked, and choosing it again takes it off
    again = next(a for a in menu_of(page, cal, monkeypatch).actions() if a.text() == "Pause here for review")
    assert again.isChecked()
    again.trigger()
    settle(world["w"], page)
    assert [w.breakpoints for w in model.p.workflows] == [[]] * 4 and item_of(page, "calibration").bp is None
    assert not pixel_near(page, item_of(page, "calibration"), BREAK_COL)
    # on a block of one workflow only
    leaf = node(page, "post", 2)
    next(a for a in menu_of(page, leaf, monkeypatch).actions() if a.text() == "Pause here for review").trigger()
    settle(world["w"], page)
    assert [w.breakpoints for w in model.p.workflows] == [[], ["output"], [], []]
    assert item_of(page, "post", 2).bp == ("pending", 1, 1)


def test_the_dot_says_pending_paused_or_reviewed_and_the_graph_follows(world, model, page, monkeypatch):
    from astro_gpu.client.blocks import BREAK_COL, PAUSE_COL, REVIEWED_COL
    p = model.view(1)
    pin(model, monkeypatch)
    model.set_breakpoint([p], "calibration")
    settle(world["w"], page)
    assert model.review_state(p, "calibration") == "pending" and item_of(page, "calibration").bp[0] == "pending"
    # the block is computed and nobody has checked it: paused (yellow, with a menu entry to say it was checked)
    pin(model, monkeypatch, "calibration")
    settle(world["w"], page)
    assert model.review_state(p, "calibration") == "paused"
    it = item_of(page, "calibration")
    assert it.bp == ("paused", 1, 4)
    assert pixel_near(page, it, PAUSE_COL) and not pixel_near(page, it, BREAK_COL, 5)
    assert "Paused: check this block (Inspect), then Continue from the Run window." in it.toolTip()
    assert "Pause set in 1 workflow of 4." in it.toolTip()                      # only one of the four workflows
    m = menu_of(page, node(page, "calibration"), monkeypatch)
    assert "Mark as reviewed" in [a.text() for a in m.actions()]
    # a second breakpoint, further on, waits (red) until the first is checked
    model.set_breakpoint([p], "empilement")
    pin(model, monkeypatch, "calibration", "empilement")
    settle(world["w"], page)
    assert [model.review_state(p, k) for k in ("calibration", "empilement")] == ["paused", "pending"]
    # Mark as reviewed: green, and the next one is the one the run will wait at
    next(a for a in m.actions() if a.text() == "Mark as reviewed").trigger()
    settle(world["w"], page)
    assert [model.review_state(p, k) for k in ("calibration", "empilement")] == ["reviewed", "paused"]
    assert item_of(page, "calibration").bp[0] == "reviewed" and pixel_near(page, item_of(page, "calibration"), REVIEWED_COL)
    assert item_of(page, "empilement").bp[0] == "paused"
    assert any(h[1] == "{block} marked as reviewed" for h in model.history)
    state = json.loads((Path(world["folder"]) / "state.json").read_text(encoding="utf-8"))
    assert state["reviewed"] == [model.review_key(p, "calibration")]
    # a block that is not computed any more shows as waiting again (the run will go through it, and stop after it)
    pin(model, monkeypatch)
    assert [model.review_state(p, k) for k in ("calibration", "empilement")] == ["pending", "pending"]
    model._reviewed.clear()
    st = project2.read_state(world["folder"])
    st.pop("reviewed", None)
    project2.write_state(world["folder"], st)


def test_the_pause_is_stored_per_workflow_undone_copied_and_leaves_with_its_block(world, model, page):
    p, dn = model.view(1), model.view(3)
    assert "denoise" in model.all_stages(dn)
    model.set_breakpoint([p], "calibration")
    model.set_breakpoint([dn], "debruitage")
    model.set_breakpoint([dn], "empilement")
    assert model.p.workflow(3).breakpoints == ["stacking", "denoise"]            # kept in the graph's order
    assert model.breakpoints(dn) == ["empilement", "debruitage"] and model.breakpoints(model.view(2)) == []
    model.flush()
    saved = project2.load(world["folder"])
    assert [w.breakpoints for w in saved.workflows] == [["calibration"], [], ["stacking", "denoise"], []]
    # Undo takes a breakpoint back, as any edit
    assert model.undo() and model.p.workflow(3).breakpoints == ["denoise"]
    assert model.redo() and model.p.workflow(3).breakpoints == ["stacking", "denoise"]
    # a copy keeps them
    cp = model.duplicate(dn, "Copy of denoise")
    settle(world["w"], page)
    assert model.p.workflow(cp.pid).breakpoints == ["stacking", "denoise"]
    # a block that leaves the workflow takes its breakpoint away; Undo brings both back
    page.remove_tool(node(page, "debruitage", cp.pid), [cp.pid])
    settle(world["w"], page)
    assert "denoise" not in model.all_stages(cp) and model.p.workflow(cp.pid).breakpoints == ["stacking"]
    assert model.p.workflow(3).breakpoints == ["stacking", "denoise"]            # the other workflow keeps its own
    assert model.undo() and model.p.workflow(cp.pid).breakpoints == ["stacking", "denoise"]
    assert "denoise" in model.all_stages(model.view(cp.pid))
    # only blocks of the engine can hold one
    with pytest.raises(ValueError):
        model.set_breakpoint([p], "sources")


def test_breakpoints_change_no_key_and_no_state(world, model, page):
    """The same rule as the engine's: the keys the engine makes, and so the states it reads, are what they were."""
    def snapshot():
        return ({(w.id, s): api.stage_key(model.p, w.id, s) for w in model.p.workflows for s in api.chain(model.p, w.id)},
                {k: (v.state, v.key, v.previous_key) for k, v in api.states(model.p, update=False).items()},
                {w.id: [(b.block, b.key) for b in api.blocks(model.p, w.id)] for w in model.p.workflows},
                [(j.key, j.stage) for j in model.plan_all().jobs])
    before = snapshot()
    model.set_breakpoint(model.paths, "calibration")
    model.set_breakpoint(model.paths, "post")
    model.set_breakpoint([model.view(1)], "selection")
    assert any(w.breakpoints for w in model.p.workflows)
    assert snapshot() == before
    model.undo()
    model.undo()
    model.undo()
    assert snapshot() == before and not any(w.breakpoints for w in model.p.workflows)


def test_selection_is_reviewed_with_its_thresholds_and_runs_to_registration(world, model, page):
    p = model.view(1)
    model.set_breakpoint([p], "selection")
    pause = model.pause_point()
    assert pause.block == "selection" and pause.pids == [1]
    plan, pause2 = model.plan_run()
    assert pause2.block == "selection" and (not plan.jobs or plan.jobs[-1].stage == "register")
    assert all(j.stage != "stack" for j in plan.jobs)                              # before stacking, as asked
    k0 = model.review_key(p, "selection")
    model.apply_change([p], {"kind": "params", "items": [("stack", "min_transparency", 0.73)]})
    assert model.review_key(p, "selection") != k0                                  # other thresholds, other selection
    model.apply_change([p], {"kind": "params", "items": [("stack", "min_transparency", None)]})
    assert model.review_key(p, "selection") == k0
    model.set_excluded([(1, "DSC00001.ARW")], True)
    assert model.review_key(p, "selection") != k0                                  # a frame excluded by hand too
    model.set_excluded([(1, "DSC00001.ARW")], False)
    assert model.review_key(p, "selection") == k0


def test_the_pause_in_french_on_the_graph(world, model, page, monkeypatch):
    p = model.view(1)
    appmod.apply_language("fr")
    pin(model, monkeypatch)
    model.set_breakpoint([p], "calibration")
    page.refresh()
    settle(world["w"], page)
    it = item_of(page, "calibration")
    assert "Point de vérification : l'exécution s'arrête ici pour votre vérification." in it.toolTip()
    assert "Pause définie dans 1 workflow sur 4." in it.toolTip()
    m = menu_of(page, node(page, "calibration"), monkeypatch)
    names = [a.text() for a in m.actions()]
    assert "Pause ici pour vérification" in names and "Pause here for review" not in names
    next(a for a in m.actions() if a.text() == "Pause ici pour vérification").trigger()
    assert page.graph.toast.text() == "L'exécution s'arrêtera après Calibration pour votre vérification dans 4 workflows"
    for text in ("Pause here for review", "Mark as reviewed", "No more pause after {block}",
                 "The run will stop after {block} for your review in {n} workflows", "{block} marked as reviewed",
                 "Pause set after {block}", "Pause removed after {block}",
                 "Paused: check this block (Inspect), then Continue from the Run window.",
                 "Reviewed: the run goes through while these results stay as they are."):
        assert text in i18n._cat and i18n._cat[text] != text, text
    assert untranslated(texts_of(page)) == []
    model.undo()
    appmod.apply_language("en")
    page.refresh()
    settle(world["w"], page)


def whole_chain_or_left(page, where: str):
    """The default view's rule: every block in view and the scale at least MIN_SCALE; when the chain is wider than the view
    at MIN_SCALE, the scale is MIN_SCALE, the left end (Frames) is in view, the view is scrolled to its left edge, and
    only the right end (Post) is cut."""
    gv = page.graph
    k = gv.transform().m11()
    assert k >= MIN_SCALE - 1e-6 and k <= 1.3 + 1e-6, (where, k)
    vp = QRectF(gv.viewport().rect())
    box = lambda it: QRectF(gv.mapFromScene(it.sceneBoundingRect()).boundingRect())          # noqa: E731
    frames = next(it for it in block_items(page) if it.node.key == "sources")
    post = page.graph.final_item
    assert vp.contains(box(frames)), (where, "Frames is cut", box(frames), vp)
    if gv.horizontalScrollBar().maximum() == 0:                                             # it fits: both ends in view
        assert vp.contains(box(post)), (where, "Post is cut", box(post), vp)
        assert all(vp.contains(box(it)) for it in block_items(page)), where
    else:                                                                                   # it does not: left, scrolling right
        bar = gv.horizontalScrollBar()
        assert abs(k - MIN_SCALE) < 1e-6 and bar.value() == bar.minimum(), (where, k, bar.value())
    return k, gv.horizontalScrollBar().maximum() == 0


def test_the_default_view_shows_the_whole_chain_from_frames_to_post_at_any_width_and_in_both_languages(world, model, page):
    from astro_gpu.client.graph import GraphView
    shots = Path(os.environ.get("TEMP", ".")) / "b3_shots"
    shots.mkdir(parents=True, exist_ok=True)
    fits = {}
    try:
        for lang in ("en", "fr"):
            appmod.apply_language(lang)
            page.refresh()
            for width in (1280, 1600, 1920):
                page.root.resize(width, 934)
                settle(world["w"], page)
                fits[(lang, width)] = whole_chain_or_left(page, f"{lang} {width}")
                assert page.graph.grab().save(str(shots / f"fit_{width}_{lang}.png"))
                assert page.root.grab().save(str(shots / f"fit_{width}_{lang}_window.png"))
            # the panel changes width (a wider one): the graph is fitted again to what is left of the window
            page.root.resize(1600, 934)
            settle(world["w"], page)
            before = page.graph.viewport().width()
            page.panel.setFixedWidth(520)
            settle(world["w"], page)
            assert page.graph.viewport().width() < before
            whole_chain_or_left(page, f"{lang} 1600, panel 520")
            page.panel.setFixedWidth(260)
            settle(world["w"], page)
            assert page.graph.viewport().width() > before
            whole_chain_or_left(page, f"{lang} 1600, panel 260")
            page.panel.setFixedWidth(390)
            settle(world["w"], page)
    finally:
        appmod.apply_language("en")
        page.panel.setFixedWidth(390)
        page.refresh()
    assert isinstance(page.graph, GraphView)
    assert fits[("en", 1920)][1] and fits[("fr", 1920)][1]                          # a wide window shows all of it
    # Show all goes below the floor, as before
    page.graph.b_all.setChecked(True)
    page.root.resize(1280, 934)
    settle(world["w"], page)
    assert page.graph.horizontalScrollBar().maximum() == 0 and page.graph.transform().m11() <= MIN_SCALE + 1e-6
    page.graph.b_all.setChecked(False)


# ---------------------------------------------------------------------------------------------- last
def test_the_test_project_is_untouched():
    """Run last (pytest keeps the order of the file): nothing above wrote into the test project or its store."""
    assert _tree_snapshot(F_PROJECT) == PROJECT_BEFORE
