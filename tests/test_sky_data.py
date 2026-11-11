"""What a run needs of the sky is known before it starts (core/sky_data.py, runner2.Planner): a step whose data is not
on this computer is left out of the run with its reason, with the steps that read its result; every other step and
workflow still runs, and nothing is downloaded (tests/test_no_download_in_run.py holds that rule)."""
from __future__ import annotations

from pathlib import Path

import pytest

from astro_gpu.core import project2, runner2, sirilcat, sky_data
from astro_gpu.core.project import Target
from astro_gpu.core.workflow import TEMPLATES, node
from tests import skyfield
from tests.test_no_download_in_run import _events, no_socket, offline_sky  # noqa: F401 - fixtures


def _project(root: Path, sky: dict, blocks=("astrometry", "color")) -> tuple:
    p = project2.create(root, "P", Target(**sky["target"]))
    w = p.add_workflow("Image", TEMPLATES["Classic stack"])
    w.source = {"path": str(sky["image"]), "stretched": False}
    p.set_blocks(w.id, [node(b) for b in blocks])
    other = p.add_workflow("Plain", TEMPLATES["Classic stack"])        # no block that needs the sky
    other.source = {"path": str(sky["image"]), "stretched": False}
    p.set_blocks(other.id, [node("gradient")])
    project2.save(p)
    return p, w, other


def _use(monkeypatch, folder: Path | None) -> None:
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", str(folder) if folder else "")
    sirilcat.reset()
    sky_data.reset()


def test_with_the_extracts_everything_is_planned(offline_sky, monkeypatch):  # noqa: F811
    sky = skyfield.field(offline_sky / "sky")
    _use(monkeypatch, sky["extracts"])
    p, w, other = _project(offline_sky / "P", sky)
    plan = runner2.plan_all(p)
    assert plan.left_out == {}
    assert {"astrometry", "catalogs", "surveys", "color", "render_simple"} <= {j.stage for j in plan.jobs if w.id in j.wf}


def test_without_the_star_catalogue_the_astrometry_and_what_follows_are_left_out_the_rest_runs(offline_sky,  # noqa: F811
                                                                                                  monkeypatch):
    sky = skyfield.field(offline_sky / "sky")
    _use(monkeypatch, None)
    p, w, other = _project(offline_sky / "P", sky)
    plan = runner2.plan_all(p)
    assert plan.left_out == {w.id: {"astrometry": sky_data.NO_STARS}}
    mine = {j.stage for j in plan.jobs if w.id in j.wf}
    assert "image" in mine and not mine & {"astrometry", "catalogs", "surveys", "color", "render_simple"}
    assert {j.stage for j in plan.jobs if other.id in j.wf} >= {"image", "gradient", "render_simple"}
    assert (w.id, "astrometry") not in plan.targets
    again = runner2.Plan.from_dict(plan.to_dict())                   # frozen in the run's plan file as it is
    assert again.left_out == plan.left_out


def test_without_the_spectra_only_the_colour_is_left_out(offline_sky, no_socket, monkeypatch):  # noqa: F811
    """The star extract alone (the spectra are the 10 GB part): the field is solved, its survey read, its gradient
    removed; the Catalogues step, which the Color block needs, is left out with Color and the Output after it; the run
    of what is planned is done, and nothing reached the network."""
    sky = skyfield.field(offline_sky / "sky", spectra=False)
    _use(monkeypatch, sky["extracts"])
    p, w, other = _project(offline_sky / "P", sky, ("astrometry", "gradient", "color"))
    plan = runner2.plan_all(p)
    assert plan.left_out == {w.id: {"catalogs": sky_data.NO_SPECTRA}}
    mine = {j.stage for j in plan.jobs if w.id in j.wf}
    assert {"astrometry", "surveys", "gradient"} <= mine and not mine & {"catalogs", "color", "render_simple"}
    assert sky_data.NEEDED_BY["catalogs"] == "color"
    status = runner2.run_plan(p.root, runner2.prepare_run(p, plan, "cpu").plan_file, echo=False)
    assert status == "done" and no_socket == []
    ends = {e["stage"]: e["status"] for e in _events(p.root) if e.get("type") == "stage_end"}
    assert ends.get("astrometry") == "done" and "catalogs" not in ends


def test_a_field_already_solved_is_checked_where_it_is_and_cached_tiles_count(offline_sky, monkeypatch):  # noqa: F811
    """Without the extracts, the sky tiles an earlier version downloaded serve the field they hold (no catalogue
    asked of the network: they are read where they are), and only that field."""
    from astro_gpu.core import skycache
    _use(monkeypatch, None)
    assert sky_data.missing("astrometry", (38.0, 61.0)) == sky_data.NO_STARS
    assert sky_data.missing("astrometry", None) == sky_data.NO_STARS
    lay = skycache.Layer(["RA_ICRS", "DE_ICRS", "Gmag"], None, 13.5, 5)
    t = int(skycache.tile_of(5, [38.0], [61.0])[0])
    import numpy as np
    lay.save(t, {"RA_ICRS": np.zeros(1), "DE_ICRS": np.zeros(1), "Gmag": np.zeros(1)})
    sky_data.reset()
    assert sky_data.missing("astrometry", (38.0, 61.0)) == ""
    assert sky_data.missing("astrometry", (200.0, -30.0)) == sky_data.NO_STARS
    # their stars are Gaia's (ids): spectra cached by id serve them; none cached: the colour cannot be calibrated
    assert sky_data.missing("catalogs", (38.0, 61.0)) == sky_data.NO_SPECTRA
    skycache.save_npy(skycache.cache_dir() / "xp" / "42.npy", np.zeros(0))
    sky_data.reset()
    assert sky_data.missing("catalogs", (38.0, 61.0)) == ""


@pytest.mark.parametrize("stage", ["surveys", "gradient", "stack", "color"])
def test_the_other_steps_need_nothing_of_the_sky(stage, offline_sky, monkeypatch):  # noqa: F811
    _use(monkeypatch, None)
    assert sky_data.missing(stage, None) == ""


def test_a_state_folder_of_its_own_still_sees_the_installed_catalogue(offline_sky, monkeypatch):  # noqa: F811
    """11 November: a run whose state folder was not the user's saw no star catalogue although it was installed, and
    downloaded. The extracts installed under the default state folder are found whatever ASTRO_GPU_HOME says, and a
    process that gives the engine a state folder of its own (tests, bench, tools: ``sirilcat.private_home``) names the
    installed ones for itself and its run workers, read in place."""
    import os
    sky = skyfield.field(offline_sky / "sky", spectra=False)
    installed = Path.home() / ".astro_gpu" / "cache" / "gaia_extracts"
    installed.mkdir(parents=True)
    (installed / "siril_cat_healpix8_astro.dat").write_bytes((sky["extracts"] / "siril_cat_healpix8_astro.dat")
                                                             .read_bytes())
    _use(monkeypatch, None)
    monkeypatch.setenv("ASTRO_GPU_HOME", str(offline_sky / "a_run_s_own_state"))
    sirilcat.reset()
    assert sirilcat.locate()["astro"].path.parent == installed            # the default state folder's, read
    assert sky_data.missing("astrometry") == ""
    env = dict(os.environ)
    sirilcat.private_home(offline_sky / "a_tool_s_state", env)
    assert env["ASTRO_GPU_HOME"] == str(offline_sky / "a_tool_s_state")
    assert str(installed) in env["ASTRO_GPU_GAIA_EXTRACTS"].split(os.pathsep)
    # a worker of that tool, its user profile moved too (the bench): the catalogue still found, by the name it got
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: offline_sky / "elsewhere"))
    for k in ("ASTRO_GPU_HOME", "ASTRO_GPU_GAIA_EXTRACTS"):
        monkeypatch.setenv(k, env[k])
    sirilcat.reset()
    assert sirilcat.locate()["astro"] is not None and sirilcat.locate()["astro"].path.parent == installed


def test_every_tool_gives_the_engine_its_state_folder_through_the_one_helper():
    """The tools and the tests that run the chain with a state folder of their own go through sirilcat.private_home,
    so that the installed catalogue stays seen (a setting of ASTRO_GPU_HOME by hand lost it)."""
    import re
    root = Path(__file__).resolve().parents[1]
    by_hand = re.compile(r"""environ(\[["']ASTRO_GPU_HOME["']\]\s*=|\.setdefault\(\s*["']ASTRO_GPU_HOME)""")
    bad = [f"{f.relative_to(root)}:{i}" for f in sorted((root / "tools").rglob("*.py")) + [root / "tests" / "conftest.py"]
           for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1) if by_hand.search(line)]
    assert bad == []
