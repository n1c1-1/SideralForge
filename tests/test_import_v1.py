"""Import of v1 projects (DESIGN §1.6, P6): the write guard, the adoption checks, the copy, the dry run.

Toy stages inventory → calibrate → stack → starless, run by v1's Pipeline into a v1 project in a temporary folder,
then imported. calibrate has a field without effect on results (workers, with an auto rule) and a frame cache in its
scratch folder; stack an auto field (scale) and a frame list; starless an earlier learned star layer.
The real projects and F:/astro_essais/projets/heart_a7s are only read, under the guard, and snapshotted.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

import astro_gpu.core.project as project_mod
from astro_gpu.core import import_v1 as iv
from astro_gpu.core import project2, runner2
from astro_gpu.core import workflow as wfl
from astro_gpu.core.journal import Journal
from astro_gpu.core.params import Auto, P, StageParams
from astro_gpu.core.project import Inputs, Project, list_frames
from astro_gpu.core.runner import Pipeline
from astro_gpu.core.stage import Stage
from astro_gpu.core.store import KeyRules, Keys, Store
from astro_gpu.core.workflow import TEMPLATES, Chains
from astro_gpu.io.frames import FrameInfo, read_info

H, W = 4, 6                       # the toy stack grid

# ---------------------------------------------------------------------------------------------- toy stages


class InvParams(StageParams):
    max_lights: int = P(0, label="max lights", help="", ge=0)


class Inv(Stage):
    name = "inventory"
    title = "Inventory"
    Params = InvParams

    def run(self, ctx, p):
        out = {}
        for kind in project2.KINDS:
            files = list_frames(getattr(ctx.project.config.inputs, kind))
            like = FrameInfo("x.ARW", "raw", W, H, "RGGB", 0.0, 1.0)
            out[kind] = [read_info(f, like=like).to_dict() for f in files]   # no EXIF: the file date
        ctx.save_json("frames.json", out)
        ctx.preview("middle_frame", np.zeros((4, 4)))


class CalParams(StageParams):
    hot_sigma: float = P(0.6, label="hot", help="")
    workers: int = P(4, label="workers", help="", ge=1)


class Cal(Stage):
    name = "calibrate"
    title = "Calibration"
    deps = ("inventory",)
    Params = CalParams
    version = "3"
    cores = 3

    def auto(self, ctx):
        return {"workers": Auto(Cal.cores, "cores")}

    def run(self, ctx, p):
        n = len(ctx.load_json("inventory", "frames.json")["lights"])
        ctx.save("master_dark.npy", np.zeros((H, W), np.float32))
        cache = ctx.scratch / "lights.f16"
        cache.write_bytes(b"\0" * 64)
        ctx.save_json("cache.json", {"path": str(cache), "n": n})


class StackParams(StageParams):
    scale: float = P(1.5, label="scale", help="", ge=1.0, le=3.0)
    kappa: float = P(5.0, label="kappa", help="")
    exclude: list[str] = P([], label="exclude", help="")


class Stack(Stage):
    name = "stack"
    title = "Stack"
    deps = ("calibrate", "inventory")
    Params = StackParams
    version = "3"

    def auto(self, ctx):
        n = len(ctx.load_json("inventory", "frames.json")["lights"])
        return {"scale": Auto(2.0 if n >= 3 else 1.0, f"{n} frames")}

    def run(self, ctx, p):
        meta = ctx.load_json("inventory", "frames.json")["lights"]
        ctx.save_json("frames.json", [{"path": m["path"], "time": m["time"]} for m in meta])
        ctx.save("drz_full.npy", np.ones((3, H, W), np.float32))
        ctx.metric("grid", [W, H])


class SlParams(StageParams):
    learned_stars: str = P("", label="layer", help="")
    amount: float = P(1.0, label="amount", help="")


class Starless(Stage):
    name = "starless"
    title = "Starless"
    deps = ("stack",)
    Params = SlParams
    version = "24"

    def run(self, ctx, p):
        ctx.save_json("sl.json", {"layer": p.learned_stars, "amount": p.amount})


STAGES = [Inv(), Cal(), Stack(), Starless()]
CHAIN = [s.name for s in STAGES]


def toy_rules() -> KeyRules:
    chains = Chains(STAGES, rule=lambda options, end, reference, learned: CHAIN)
    return KeyRules(chains, auto_fields={"calibrate": ("workers",), "stack": ("scale",), "starless": ("learned_stars",)},
                    path_fields={"starless": ("learned_stars",)}, non_key={"calibrate": ("workers",)},
                    implied=lambda options, stage: {})


RULES = toy_rules()


@pytest.fixture(autouse=True)
def _private_state(tmp_path, monkeypatch):
    """The v1 runs of the toy projects never reach the user's timings (ASTRO_GPU_HOME)."""
    monkeypatch.setattr(project_mod, "STATE", tmp_path / "_state")
    Cal.cores = 3


def write_frames(folder: Path, names: list[str]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        (folder / n).write_bytes(b"\0" * (100 + i))
    return folder


def make_v1(tmp_path: Path, params: dict | None = None, reference: str = "") -> Path:
    """A v1 project run to the end by v1's Pipeline, with the leftovers a real one has (a checkpoint folder, a
    temporary file); its calibrated frame cache is in its own scratch folder."""
    lights = write_frames(tmp_path / "raw" / "signal", ["f1.ARW", "f2.ARW", "f3.ARW"])
    darks = write_frames(tmp_path / "raw" / "dark", ["d1.ARW", "d2.ARW"])
    root = tmp_path / "old"
    proj = Project.create(root, "Toy", Inputs(lights=[str(lights)], darks=[str(darks)]),
                          scratch_dir=tmp_path / "old_scratch", reference_image=reference)
    proj.config.params = params or {}
    proj.save()
    assert Pipeline(Project(root), STAGES, Journal(root / "logs", echo=False)).run() == "done"
    (root / "work" / "calibrate" / "_ckpt").mkdir()
    (root / "work" / "calibrate" / "_ckpt" / "state.bin").write_bytes(b"x" * 1000)
    (root / "work" / "stack" / "drz_full.tmp.npy").write_bytes(b"x" * 1000)
    return root


def set_old_param(root: Path, stage: str, name: str, value) -> None:
    """Edit the old project.json behind the guard's back, as the user would in v1 (no re-run)."""
    d = json.loads((root / "project.json").read_text(encoding="utf-8"))
    d["params"].setdefault(stage, {})[name] = value
    (root / "project.json").write_text(json.dumps(d), encoding="utf-8")


def moved_raws(tmp_path: Path) -> dict[str, str]:
    """The raws copied elsewhere (same names, sizes and dates), and the path map to them."""
    new = tmp_path / "raw_new"
    shutil.copytree(tmp_path / "raw", new)
    return {str(tmp_path / "raw"): new.as_posix()}


def run_import(old, new, path_map=None, dry_run=False):
    return iv.import_v1(old, new, path_map, dry_run=dry_run, rules=RULES)


# ---------------------------------------------------------------------------------------------- paths and the guard

def test_map_path(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_frames(b, ["x.ARW"])
    m = {"F:/backup/photos": "G:/Astro", "F:/backup/photos/flat": "D:/flats",
         "F:\\old\\split": [str(a), str(b)]}
    assert iv.map_path("F:/backup/photos/Rosette/signal", m) == ["G:/Astro/Rosette/signal"]
    assert iv.map_path("f:\\BACKUP\\photos\\957MSDCF\\DSC00639.ARW", m) == ["G:/Astro/957MSDCF/DSC00639.ARW"]
    assert iv.map_path("F:/backup/photos/flat", m) == ["D:/flats"]                  # the longest entry wins
    assert iv.map_path("F:/backup/photosX/a", m) == ["F:/backup/photosX/a"]         # a folder, not a prefix
    assert iv.map_path("E:/elsewhere/x.ARW", m) == ["E:/elsewhere/x.ARW"]           # unmapped: as it is
    assert iv.map_path("F:/old/split", m) == [a.as_posix(), b.as_posix()]           # one folder now two
    assert iv.map_path("F:/old/split/x.ARW", m) == [(b / "x.ARW").as_posix()]       # a file: where it is
    assert iv.map_path("F:/x", None) == ["F:/x"]


def test_the_write_guard_refuses_writes_and_lets_reads_through(tmp_path):
    root = tmp_path / "proj"
    (root / "work").mkdir(parents=True)
    (root / "work" / "a.txt").write_text("x", encoding="utf-8")
    out = tmp_path / "elsewhere"
    out.mkdir()
    before = iv.snapshot([root])
    writes = [lambda: (root / "work" / "b.txt").write_text("y", encoding="utf-8"),
              lambda: open(root / "work" / "a.txt", "a").close(),
              lambda: open(root / "work" / "a.txt", "r+b").close(),
              lambda: os.open(root / "c.bin", os.O_CREAT | os.O_WRONLY),
              lambda: (root / "d").mkdir(),
              lambda: (root / "work" / "e.txt").touch(),
              lambda: os.replace(root / "work" / "a.txt", root / "a2.txt"),
              lambda: os.replace(out / "nothing", root / "f.txt"),
              lambda: (root / "work" / "a.txt").unlink(),
              lambda: shutil.copy2(tmp_path / "x.txt", root / "x.txt"),
              lambda: shutil.copyfile(tmp_path / "x.txt", root / "x2.txt"),
              lambda: shutil.rmtree(root / "work"),
              lambda: os.utime(root / "work" / "a.txt"),
              lambda: np.save(root / "n.npy", np.zeros(2))]
    (tmp_path / "x.txt").write_text("z", encoding="utf-8")
    with pytest.raises(iv.WriteRefused):
        with iv.WriteGuard([root]) as g:
            for w in writes:
                with pytest.raises(iv.WriteRefused):
                    w()
            assert (root / "work" / "a.txt").read_text(encoding="utf-8") == "x"       # reads pass
            shutil.copy2(root / "work" / "a.txt", out / "a.txt")                    # copying out passes
            assert (out / "a.txt").read_text(encoding="utf-8") == "x"
            try:
                (root / "g").mkdir()                                                 # a refusal swallowed...
            except OSError:
                pass
            assert len(g.refused) >= len(writes) + 1      # touch() tries utime, swallows it, then open: two
    # ...still fails the block when it ends
    assert iv.snapshot([root]) == before


# ---------------------------------------------------------------------------------------------- adoption and copy

def test_toy_import_adopts_everything_and_copies_only_outputs(tmp_path):
    old = make_v1(tmp_path)
    path_map = moved_raws(tmp_path)
    watched = iv.folders_of(old)
    assert str(tmp_path / "old_scratch") in watched and str(tmp_path / "raw" / "signal") in watched
    before = iv.snapshot(watched)
    new = tmp_path / "v2" / "toy"

    dry = run_import(old, new, path_map, dry_run=True)
    assert not new.exists() and not (tmp_path / "v2").exists()                       # nothing written at all
    assert dry.adopted(1) == CHAIN and dry.recomputed(1) == []
    assert dry.copy_bytes > 0 and dry.free_bytes and dry.enough_space

    rep = run_import(old, new, path_map)
    assert iv.snapshot(watched) == before                                           # the old project untouched
    assert rep.written and [(r.stage, r.adopted) for r in rep.stages] == [(s, True) for s in CHAIN]
    assert [r.key for r in rep.stages] == [r.key for r in dry.stages] and rep.copy_bytes == dry.copy_bytes
    p = project2.load(new)
    assert p.sessions[0].lights == [(tmp_path / "raw_new" / "signal").as_posix()]
    keys = Keys(p, RULES).keys(1)
    store = Store(new)
    old_ids = {s: json.loads((old / "work" / s / "manifest.json").read_text())["run_id"] for s in CHAIN}
    for s in CHAIN:
        assert store.done(s, keys[s])
        m = store.manifest(s, keys[s])
        assert m["key"] == m["run_id"] == keys[s] and m["key_parts"]["s"] == s
        assert m["adopted_from"]["run_id"] == old_ids[s] and m["adopted_from"]["folder"] == str(old / "work" / s)
        assert m["inputs"] == {d: keys[d] for d in RULES.chains.stages[s].deps}
    # only what later stages read: no checkpoint, no temporary file, no frame cache, no .tmp folder left
    files = [f.relative_to(new).as_posix() for f in new.rglob("*") if f.is_file()]
    assert not [f for f in files if "_ckpt" in f or ".tmp" in f or f.endswith(".f16")]
    assert (store.dir("inventory", keys["inventory"]) / "previews" / "middle_frame.jpg").is_file()
    assert (store.dir("stack", keys["stack"]) / "drz_full.npy").is_file()
    # frames.json moved to the new raws; calibrate's cache to the new scratch folder (rebuilt when needed)
    inv = json.loads((store.dir("inventory", keys["inventory"]) / "frames.json").read_text())
    assert [Path(f["path"]).parent for f in inv["lights"]] == [tmp_path / "raw_new" / "signal"] * 3
    st = json.loads((store.dir("stack", keys["stack"]) / "frames.json").read_text())
    assert [f["path"] for f in st] == [f["path"] for f in inv["lights"]]
    cache = json.loads((store.dir("calibrate", keys["calibrate"]) / "cache.json").read_text())
    assert Path(cache["path"]) == Path(os.path.abspath(p.scratch)) / keys["calibrate"] / "calibrate" / "lights.f16"
    assert store.usable_as_input("inventory", keys["inventory"])
    assert not store.usable_as_input("calibrate", keys["calibrate"])
    # nothing to run, and the report is kept
    plan = runner2.Planner(p, keys=Keys(p, RULES), store=store, history={}).plan([(1, s) for s in CHAIN])
    assert not plan
    assert json.loads((new / "logs" / iv.REPORT_FILE).read_text())["status"] == "done"


def test_a_changed_old_parameter_stops_adoption_there(tmp_path, monkeypatch):
    old = make_v1(tmp_path)
    Cal.cores = 7                         # workers (no effect on results) differ: still adopted
    rep = run_import(old, tmp_path / "v2", dry_run=True)
    assert rep.adopted(1) == CHAIN
    set_old_param(old, "stack", "kappa", 7.0)
    rep = run_import(old, tmp_path / "v2", dry_run=True)
    assert rep.adopted(1) == ["inventory", "calibrate"]
    assert rep.recomputed(1) == [("stack", "settings changed: kappa 5.0 -> 7.0"),
                                 ("starless", "'stack' is recomputed")]
    set_old_param(old, "stack", "kappa", 5.0)     # typed back to the value it ran with
    assert run_import(old, tmp_path / "v2", dry_run=True).adopted(1) == CHAIN
    set_old_param(old, "stack", "scale", 1.5)     # the static default, over an automatic 2.0: another result
    assert run_import(old, tmp_path / "v2", dry_run=True).adopted(1) == ["inventory", "calibrate"]
    set_old_param(old, "stack", "scale", 2.0)     # the automatic value it ran with, now stored
    assert run_import(old, tmp_path / "v2", dry_run=True).adopted(1) == CHAIN
    monkeypatch.setattr(Starless, "version", "25")
    assert run_import(old, tmp_path / "v2", dry_run=True).recomputed(1) == [("starless",
                                                                             "algorithm updated (24 -> 25)")]


def test_an_old_result_run_again_breaks_the_chain(tmp_path):
    old = make_v1(tmp_path)
    m = old / "work" / "calibrate" / "manifest.json"
    d = json.loads(m.read_text())
    d["run_id"] = "another"                       # calibrate re-ran after stack was made
    m.write_text(json.dumps(d))
    rep = run_import(old, tmp_path / "v2", dry_run=True)
    assert rep.adopted(1) == ["inventory", "calibrate"]
    assert rep.recomputed(1)[0] == ("stack", "'calibrate' was run again after this result")
    d["status"] = "interrupted"
    m.write_text(json.dumps(d))
    assert run_import(old, tmp_path / "v2", dry_run=True).recomputed(1)[0] == ("calibrate",
                                                                            "the old result is interrupted")


@pytest.mark.parametrize("change,reason", [
    ("rename", "lights #2: f3.ARW now, f2.ARW in the old result (other files or another order)"),
    ("extra", "lights: 4 files now, 3 in the old result"),
    ("missing", "darks: 1 files now, 2 in the old result"),
    ("date", "lights f3.ARW: captured "),
    ("exposure", "lights f1.ARW: exposure 30.0 s now, None s in the old result"),
])
def test_the_frames_must_match_by_name_order_count_time_and_exposure(tmp_path, monkeypatch, change, reason):
    old = make_v1(tmp_path)
    path_map = moved_raws(tmp_path)
    sig = tmp_path / "raw_new" / "signal"
    if change == "rename":
        (sig / "f2.ARW").rename(sig / "g2.ARW")
    elif change == "extra":
        (sig / "f4.ARW").write_bytes(b"\0")
    elif change == "missing":
        (tmp_path / "raw_new" / "dark" / "d2.ARW").unlink()
    elif change == "date":
        st = (sig / "f3.ARW").stat()
        os.utime(sig / "f3.ARW", ns=(st.st_atime_ns, st.st_mtime_ns + 3_600_000_000_000))
    else:
        real = read_info

        def exposed(path, like=None):
            info = real(path, like)
            if Path(path).name == "f1.ARW":
                info.exposure = 30.0
            return info
        monkeypatch.setattr("astro_gpu.io.frames.read_info", exposed)
    rep = run_import(old, tmp_path / "v2", path_map, dry_run=True)
    assert rep.adopted(1) == []
    assert rep.recomputed(1)[0][0] == "inventory" and rep.recomputed(1)[0][1].startswith(reason)
    assert rep.recomputed(1)[1] == ("calibrate", "'inventory' is recomputed")


def test_without_a_map_dead_raws_recompute_everything(tmp_path):
    old = make_v1(tmp_path)
    shutil.move(tmp_path / "raw", tmp_path / "raw_moved")     # the raws are gone from where v1 knew them
    rep = run_import(old, tmp_path / "v2", dry_run=True)
    assert rep.adopted(1) == [] and rep.recomputed(1)[0] == ("inventory", "lights: 0 files now, 3 in the old result")


def test_free_space_is_checked_before_anything_is_copied(tmp_path, monkeypatch):
    old = make_v1(tmp_path)
    new = tmp_path / "v2"
    need = run_import(old, new, dry_run=True).copy_bytes
    monkeypatch.setattr(iv, "disk_free", lambda path: int(need * 1.05))       # short of the 10 % margin
    rep = run_import(old, new, dry_run=True)
    assert not rep.enough_space and "NOT ENOUGH" in rep.text()
    with pytest.raises(iv.NotEnoughSpace, match="not enough free space"):
        run_import(old, new)
    assert not new.exists()
    monkeypatch.setattr(iv, "disk_free", lambda path: int(need * 1.2))
    assert run_import(old, new).written


def test_destinations_that_are_refused(tmp_path):
    old = make_v1(tmp_path)
    for bad in (old / "v2", tmp_path / "raw" / "signal" / "v2", tmp_path / "old_scratch" / "v2"):
        with pytest.raises(ValueError, match="old project's folders"):
            run_import(old, bad, dry_run=True)
    with pytest.raises(ValueError, match="hold or be inside"):
        run_import(old, tmp_path, dry_run=True)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        run_import(old, busy)
    with pytest.raises(ValueError, match="not a v1 project"):
        run_import(busy, tmp_path / "v2")


def test_an_interrupted_import_resumes(tmp_path, monkeypatch):
    old = make_v1(tmp_path)
    new = tmp_path / "v2"
    real = iv._Import._rewrite
    calls = []

    def fail_on_stack(self, s, *a):
        calls.append(s)
        if s == "stack" and calls.count("stack") == 1:
            raise OSError("disk unplugged")
        return real(self, s, *a)
    monkeypatch.setattr(iv._Import, "_rewrite", fail_on_stack)
    with pytest.raises(OSError, match="unplugged"):
        run_import(old, new)
    k = Keys(project2.load(new), RULES).keys(1)
    assert Store(new).done("calibrate", k["calibrate"]) and not Store(new).done("stack", k["stack"])
    assert Store(new).tmp_dir("stack", k["stack"]).is_dir()
    rep = run_import(old, new)                                    # the same import again: it goes on
    assert rep.written and calls.count("inventory") == 1 and calls.count("stack") == 2
    assert all(Store(new).done(s, k[s]) for s in CHAIN) and not Store(new).tmp_dir("stack", k["stack"]).exists()


# ---------------------------------------------------------------------------------------------- inputs/

def make_layer(path: Path, shape=(3, H, W)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.zeros(shape, np.float32))
    return path


def test_an_earlier_star_layer_is_tied_to_its_stack(tmp_path):
    layer = make_layer(tmp_path / "diag" / "runE" / "star_layer_net.npy")
    old = make_v1(tmp_path, {"starless": {"learned_stars": layer.as_posix()}, "star_net": {"enabled": False}})
    new = tmp_path / "v2"
    rep = run_import(old, new)
    assert rep.adopted(1) == CHAIN and rep.workflows == {1: "Full + earlier neural layer"}
    p = project2.load(new)
    w = p.workflow(1)
    assert w.params["starless"]["learned_stars"] == "inputs/star_layer_net.npy"
    assert (new / "inputs" / "star_layer_net.npy").stat().st_mtime_ns == layer.stat().st_mtime_ns
    keys = Keys(p, RULES)
    rec = json.loads((new / "inputs" / "star_layer_net.json").read_text())
    assert rec[runner2.LAYER_STACK_KEY] == keys.key(1, "stack") == rep.layers[layer.as_posix()]
    assert runner2.layer_record(new / "inputs" / "star_layer_net.npy") == new / "inputs" / "star_layer_net.json"
    assert keys.keys(1) == {r.stage: r.key for r in rep.of(1)}         # moving the layer kept every key
    assert runner2.Planner(p, keys=keys, store=Store(new), history={}).blocked(1) is None
    # another stack (a frame excluded): the layer is refused
    p.excluded = [(1, "f2.ARW")]
    keys = Keys(p, RULES)
    assert keys.key(1, "stack") != rec[runner2.LAYER_STACK_KEY]
    assert runner2.Planner(p, keys=keys, store=Store(new), history={}).blocked(1) == runner2.LAYER_OTHER_STACK


def test_a_layer_not_in_the_stack_grid_is_refused(tmp_path):
    layer = make_layer(tmp_path / "diag" / "star_layer_net.npy", shape=(3, H + 1, W))
    old = make_v1(tmp_path, {"starless": {"learned_stars": layer.as_posix()}, "star_net": {"enabled": False}})
    new = tmp_path / "v2"
    rep = run_import(old, new)
    assert rep.recomputed(1) == [("starless", "the earlier layer belongs to another stack")]
    assert any("cannot be used" in n for n in rep.notes)
    rec = json.loads((new / "inputs" / "star_layer_net.json").read_text())
    assert rec[runner2.LAYER_STACK_KEY].startswith(iv.LAYER_FROM_V1)
    p = project2.load(new)
    assert runner2.Planner(p, keys=Keys(p, RULES), store=Store(new), history={}).blocked(1) == \
        runner2.LAYER_OTHER_STACK


def test_the_reference_image_goes_to_inputs_or_is_dropped(tmp_path):
    ref = tmp_path / "pics" / "full.png"
    ref.parent.mkdir()
    ref.write_bytes(b"png" * 10)
    old = make_v1(tmp_path, reference=ref.as_posix())
    new = tmp_path / "v2"
    rep = run_import(old, new)
    assert project2.load(new).reference_image == "inputs/full.png" and (new / "inputs" / "full.png").is_file()
    assert rep.copy_bytes >= 30 and not rep.notes
    ref.unlink()                                                     # gone, as M31's
    rep = run_import(old, tmp_path / "v2b", dry_run=True)
    assert rep.project.reference_image == ""
    assert rep.notes == [f"the reference image {ref.as_posix()} no longer exists: dropped"]


# ---------------------------------------------------------------------------------------------- heart_a7s (a copy)

HEART = Path("F:/astro_essais/projets/heart_a7s")
FULL_CHAIN = ["inventory", "calibrate", "register", "stack", "astrometry", "catalogs", "surveys", "color", "denoise",
              "optics", "stars", "starless", "background", "render_prep"]


def _up_to_date(root: Path, chain: list[str], logs: Path) -> list[str]:
    """The leading stages of ``chain`` that the v1 project at ``root`` has done and up to date (what an import can
    adopt)."""
    from astro_gpu.core.journal import Journal
    from astro_gpu.core.project import Project
    from astro_gpu.core.runner import Pipeline
    from astro_gpu.stages import all_stages
    state = {s.name: s.state for s in Pipeline(Project(root), all_stages(), Journal(logs, echo=False)).status()}
    out = []
    for name in chain:
        if state.get(name) != "done":
            break
        out.append(name)
    return out


def _light_copy(src: Path, dst: Path, limit: int = 2_000_000) -> None:
    """project.json and the small files of each stage (manifests, frame lists, registration, metadata): enough to
    resolve every stage's settings, without the large arrays."""
    dst.mkdir(parents=True)
    shutil.copy2(src / "project.json", dst / "project.json")
    for f in (src / "work").rglob("*"):
        if f.is_file() and f.stat().st_size < limit and "previews" not in f.parts:
            (dst / f.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / f.relative_to(src))


@pytest.mark.skipif(not (HEART / "project.json").is_file(), reason="F:/astro_essais not reachable")
def test_the_guard_on_a_copy_of_heart_a7s(tmp_path, monkeypatch):
    """The dry run on a copy of a real project adopts the stages of the Full chain that the old project has up to date
    (EXIF of the real raws, real settings), and a stage trying to write into the old project fails the import and
    changes nothing."""
    copy = tmp_path / "heart_a7s"
    _light_copy(HEART, copy)
    up_to_date = _up_to_date(copy, FULL_CHAIN, tmp_path / "logs")
    assert up_to_date[:3] == ["inventory", "calibrate", "register"]
    watched = iv.folders_of(copy)
    before = iv.snapshot(watched)
    rep = iv.import_v1(copy, tmp_path / "v2", dry_run=True)
    assert rep.workflows == {1: "Full"} and rep.adopted(1) == up_to_date
    with pytest.raises(iv.WriteRefused, match="1 write"):           # the guarded block fails as a whole
        with iv.WriteGuard(watched):
            with pytest.raises(iv.WriteRefused):                     # a deliberate write
                (copy / "work" / "stack" / "oops.txt").write_text("x", encoding="utf-8")
            (tmp_path / "elsewhere.txt").write_text("x", encoding="utf-8")       # outside: allowed
    register = type(wfl.chains().stages["register"])        # its settings are checked, against the old calibration

    def sneaky(self, ctx):
        (ctx.project.stage_dir("calibrate") / "oops.txt").write_text("x", encoding="utf-8")
        return {}
    monkeypatch.setattr(register, "auto", sneaky)
    with pytest.raises(iv.WriteRefused):
        iv.import_v1(copy, tmp_path / "v2", dry_run=True)
    assert iv.snapshot(watched) == before
    assert not (tmp_path / "v2").exists()


@pytest.mark.slow
@pytest.mark.skipif(not (HEART / "project.json").is_file(), reason="F:/astro_essais not reachable")
def test_heart_a7s_import(tmp_path):
    """The whole import of the test project heart_a7s: the stages of Full it has up to date adopted, Full plans the
    others only, a Classic stack shares what it can, and the old project is unchanged. The copy goes to a temporary
    folder on F: (8 GB), removed at the end."""
    watched = iv.folders_of(HEART)
    before = iv.snapshot(watched)
    new = Path("F:/astro_essais/v2/_p6_test") / uuid.uuid4().hex[:8]
    try:
        dry = iv.import_v1(HEART, new, dry_run=True)
        rep = iv.import_v1(HEART, new)
        assert iv.snapshot(watched) == before
        adopted = _up_to_date(HEART, FULL_CHAIN, tmp_path / "logs")
        assert rep.adopted(1) == dry.adopted(1) == adopted and rep.copy_bytes == dry.copy_bytes
        files = [f for f in new.rglob("*") if f.is_file()]
        assert not [f for f in files if "_ckpt" in f.parts or f.suffix == ".f16" or ".tmp" in f.name]
        p = project2.load(new)
        planned = {j.stage for j in runner2.plan_workflows(p, [1], history={}).jobs}
        # the calibrated frames are not copied: when a stage after it must run, the calibration runs again (its cache)
        assert planned - {"calibrate"} == set(FULL_CHAIN) - set(adopted)
        classic = p.add_workflow("Classic stack", TEMPLATES["Classic stack"])
        plan = runner2.plan_workflows(p, [classic.id], history={})
        assert {"background_poly", "render_simple"} <= {j.stage for j in plan.jobs}
        assert not (set(adopted) - {"calibrate"}) & {j.stage for j in plan.jobs}
    finally:
        if new.exists():
            shutil.rmtree(new)                                       # this test's own folder
        if new.parent.is_dir() and not any(new.parent.iterdir()):
            new.parent.rmdir()


# ---------------------------------------------------------------------------------------------- the real projects

G28 = "G:/Astro/2024-12-28_divers_astro"
DARKS28 = {"F:/backup_pc/photos/28.12.24/957MSDCF": f"{G28}/957MSDCF"}
PREFIX = ["inventory", "calibrate", "register", "stack", "astrometry", "reference", "catalogs", "surveys", "color",
          "denoise", "optics"]
NO_REF = [s for s in PREFIX if s != "reference"]
#: (folder, where its raws are now when they moved, its chain up to optics)
REAL = [
    ("F:/photos/heart_nebula/astro_gpu", {}, PREFIX),
    ("F:/photos/NGC7822/astro_gpu", {}, PREFIX),
    ("C:/Users/Nicolas/astro_gpu_scratch/projects/Iris", {}, PREFIX),
    ("C:/Users/Nicolas/astro_gpu_scratch/projects/M33", {}, PREFIX),
    ("E:/astro_gpu/projects/Rosette",
     {"F:/backup_pc/photos/28.12.24/Rosette/signal": f"{G28}/Rosette/signal", **DARKS28}, PREFIX),
    ("E:/astro_gpu/projects/Horsehead",
     {"F:/backup_pc/photos/28.12.24/HorseHead/signal": f"{G28}/HorseHead/signal", **DARKS28}, PREFIX),
    ("E:/astro_gpu/projects/FlamingStar",
     {"F:/backup_pc/photos/28.12.24/Etoile_flamboyante/signal": ["G:/Astro/2024-12-27_Etoile-Flamboyante/raw",
                                                                 f"{G28}/Etoile_flamboyante/signal"], **DARKS28},
     PREFIX),
    ("E:/astro_gpu/projects/M31", {"F:/backup_pc/photos/24.08.23": "G:/Astro/2023-08-24_M31_top"}, NO_REF),
    ("E:/astro_gpu/projects/Wizard", {}, NO_REF),                                # no reference image
]


@pytest.mark.slow
@pytest.mark.parametrize("folder,path_map,chain", REAL, ids=[Path(r[0]).name if "astro_gpu" != Path(r[0]).name
                                                             else Path(r[0]).parent.name for r in REAL])
def test_dry_run_on_the_real_projects(folder, path_map, chain, tmp_path):
    """The dry run of the import of each real project: it adopts a leading part of its chain (how much depends on
    the engine versions its results were made with) and says why the first other stage is computed again; nothing
    is written into the old project, its scratch folder, its inputs or the raws (snapshots), nor anywhere else."""
    old = Path(folder)
    if not (old / "project.json").is_file():
        pytest.skip(f"not reachable: {old}")
    watched = iv.folders_of(old) + [v for vs in path_map.values() for v in ([vs] if isinstance(vs, str) else vs)]
    before = iv.snapshot(watched)
    new = tmp_path / "v2"
    rep = iv.import_v1(old, new, path_map, dry_run=True)
    assert iv.snapshot(watched) == before
    assert not new.exists()
    got = rep.adopted()
    assert got[:1] == ["inventory"] and got == chain[:len(got)]
    assert rep.recomputed()[0][1].startswith("algorithm updated") or rep.recomputed()[0] == (
        "reference", "no result in the old project")
    assert rep.enough_space and rep.timings
