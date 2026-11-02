"""Variants: the same data processed another way, side by side with main."""
from astro_gpu.core.project import Inputs, Project
from astro_gpu.core.runner import Pipeline
from astro_gpu.stages import all_stages


def _project(tmp_path):
    return Project.create(tmp_path / "p", "test", Inputs(lights=[str(tmp_path)]))


def test_variant_stages_are_the_changed_ones_and_after(tmp_path):
    p = _project(tmp_path)
    p.create_variant("Bounded halo", {"stars": {"halo_bound": True}})
    own = p.variant_stages("Bounded halo")
    assert {"stars", "starless", "background", "render_prep", "render_prep_learned"} <= own
    assert not own & {"inventory", "calibrate", "register", "stack", "optics", "catalogs"}


def test_active_variant_folders_and_parameters(tmp_path):
    p = _project(tmp_path)
    p.set_override("stars", "far_halo", True)
    p.create_variant("Bounded halo", {"stars": {"halo_bound": True}})
    assert p.stage_dir("stars") == p.work / "stars"
    assert "halo_bound" not in p.overrides("stars")
    p.set_active_variant("Bounded halo")
    assert p.stage_dir("stars") == p.root / "variants" / "Bounded_halo" / "stars"
    assert p.stage_dir("stack") == p.work / "stack"
    assert p.overrides("stars") == {"far_halo": True, "halo_bound": True}
    # an edit while the variant is active belongs to the variant
    p.set_override("background", "star_flat", False)
    assert p.config.params.get("background") is None
    assert p.config.variants["Bounded halo"]["params"]["background"] == {"star_flat": False}
    # clearing a value main sets: the variant goes back to the automatic value
    p.clear_override("stars", "far_halo")
    assert "far_halo" not in p.overrides("stars")
    assert p.config.params["stars"]["far_halo"] is True
    # saved and read back
    q = Project(p.root)
    assert q.config.active_variant == "Bounded halo"
    assert q.overrides("stars") == {"halo_bound": True}


def test_pipeline_uses_the_variant_parameters(tmp_path):
    p = _project(tmp_path)
    p.create_variant("Bounded halo", {"stars": {"halo_bound": True}})
    pipe = Pipeline(p, all_stages())
    assert pipe.effective_params(pipe.by_name["stars"])[0].halo_bound is False
    p.set_active_variant("Bounded halo")
    assert pipe.effective_params(pipe.by_name["stars"])[0].halo_bound is True


def test_promote_swaps_folders_and_keeps_old_main(tmp_path):
    p = _project(tmp_path)
    (p.work / "stars").mkdir(parents=True)
    (p.work / "stars" / "old.txt").write_text("main")
    (p.work / "stack").mkdir(parents=True)
    p.create_variant("Bounded halo", {"stars": {"halo_bound": True}})
    p.set_active_variant("Bounded halo")
    p.stage_dir("stars").mkdir(parents=True)
    (p.stage_dir("stars") / "new.txt").write_text("variant")
    p.promote_variant("Bounded halo", keep_main_as="Previous main")
    assert p.config.active_variant == ""
    assert (p.work / "stars" / "new.txt").read_text() == "variant"
    assert (p.work / "stack").exists()
    assert p.config.params["stars"]["halo_bound"] is True
    assert "Bounded halo" not in p.config.variants
    kept = p.config.variants["Previous main"]
    assert kept["params"] == {"stars": {"halo_bound": None}}
    p.set_active_variant("Previous main")
    assert (p.stage_dir("stars") / "old.txt").read_text() == "main"
    assert "halo_bound" not in p.overrides("stars")
    p.delete_variant("Previous main")
    assert not (p.root / "variants" / "Previous_main").exists()
    assert p.config.active_variant == ""


def test_variant_names(tmp_path):
    p = _project(tmp_path)
    p.create_variant("A b/c", {"stars": {"halo_bound": True}})
    p.create_variant("A b c", {"stars": {"far_halo": False}})
    slugs = {v["slug"] for v in p.config.variants.values()}
    assert len(slugs) == 2
    for bad in ("", "main", "A b/c"):
        try:
            p.create_variant(bad, {})
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")
