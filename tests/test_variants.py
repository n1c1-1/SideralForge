"""Variants: the same data processed another way, side by side with main."""
from astro_gpu.core.project import Inputs, Project
from astro_gpu.core.runner import Pipeline
from astro_gpu.stages import all_stages


def _project(tmp_path):
    return Project.create(tmp_path / "p", "test", Inputs(lights=[str(tmp_path)]))


def test_variant_stages_are_the_changed_ones_and_after(tmp_path):
    p = _project(tmp_path)
    p.create_variant("Finer solve", {"astrometry": {"sip_order": 4}})
    own = p.variant_stages("Finer solve")
    assert {"astrometry", "catalogs", "surveys", "color"} <= own
    assert not own & {"inventory", "calibrate", "register", "stack", "reference", "denoise"}


def test_active_variant_folders_and_parameters(tmp_path):
    p = _project(tmp_path)
    p.set_override("astrometry", "blind", False)
    p.create_variant("Finer solve", {"astrometry": {"sip_order": 4}})
    assert p.stage_dir("astrometry") == p.work / "astrometry"
    assert "sip_order" not in p.overrides("astrometry")
    p.set_active_variant("Finer solve")
    assert p.stage_dir("astrometry") == p.root / "variants" / "Finer_solve" / "astrometry"
    assert p.stage_dir("stack") == p.work / "stack"
    assert p.overrides("astrometry") == {"blind": False, "sip_order": 4}
    # an edit while the variant is active belongs to the variant
    p.set_override("color", "aperture", 6.0)
    assert p.config.params.get("color") is None
    assert p.config.variants["Finer solve"]["params"]["color"] == {"aperture": 6.0}
    # clearing a value main sets: the variant goes back to the automatic value
    p.clear_override("astrometry", "blind")
    assert "blind" not in p.overrides("astrometry")
    assert p.config.params["astrometry"]["blind"] is False
    # saved and read back
    q = Project(p.root)
    assert q.config.active_variant == "Finer solve"
    assert q.overrides("astrometry") == {"sip_order": 4}


def test_pipeline_uses_the_variant_parameters(tmp_path):
    p = _project(tmp_path)
    p.create_variant("Finer solve", {"astrometry": {"sip_order": 4}})
    pipe = Pipeline(p, all_stages())
    assert pipe.effective_params(pipe.by_name["astrometry"])[0].sip_order == 3
    p.set_active_variant("Finer solve")
    assert pipe.effective_params(pipe.by_name["astrometry"])[0].sip_order == 4


def test_promote_swaps_folders_and_keeps_old_main(tmp_path):
    p = _project(tmp_path)
    (p.work / "astrometry").mkdir(parents=True)
    (p.work / "astrometry" / "old.txt").write_text("main")
    (p.work / "stack").mkdir(parents=True)
    p.create_variant("Finer solve", {"astrometry": {"sip_order": 4}})
    p.set_active_variant("Finer solve")
    p.stage_dir("astrometry").mkdir(parents=True)
    (p.stage_dir("astrometry") / "new.txt").write_text("variant")
    p.promote_variant("Finer solve", keep_main_as="Previous main")
    assert p.config.active_variant == ""
    assert (p.work / "astrometry" / "new.txt").read_text() == "variant"
    assert (p.work / "stack").exists()
    assert p.config.params["astrometry"]["sip_order"] == 4
    assert "Finer solve" not in p.config.variants
    kept = p.config.variants["Previous main"]
    assert kept["params"] == {"astrometry": {"sip_order": None}}
    p.set_active_variant("Previous main")
    assert (p.stage_dir("astrometry") / "old.txt").read_text() == "main"
    assert "sip_order" not in p.overrides("astrometry")
    p.delete_variant("Previous main")
    assert not (p.root / "variants" / "Previous_main").exists()
    assert p.config.active_variant == ""


def test_variant_names(tmp_path):
    p = _project(tmp_path)
    p.create_variant("A b/c", {"astrometry": {"sip_order": 4}})
    p.create_variant("A b c", {"astrometry": {"blind": False}})
    slugs = {v["slug"] for v in p.config.variants.values()}
    assert len(slugs) == 2
    for bad in ("", "main", "A b/c"):
        try:
            p.create_variant(bad, {})
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")
