"""The v2 store (DESIGN §1.3, §1.4): the key rule, sharing, optional and extra inputs, the project parts, the
folders and the shim that runs unchanged stage code through v1's runner.

Toy stages: inventory → stack → b (optional) → c → e, where c declares b as optional and e reads b through an EXTRA
edge (the starless / star_net pattern). stack has a field with an auto rule (scale), one without (kappa), a field
left out of the key (workers) and the injected exclusions; b has a path field (model).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from astro_gpu.core import project2
from astro_gpu.core.journal import Journal
from astro_gpu.core.params import Auto, P, StageParams
from astro_gpu.core.project import Inputs, Project, list_frames
from astro_gpu.core.project2 import Session
from astro_gpu.core.runner import Pipeline
from astro_gpu.core.stage import Context, Stage, StageError
from astro_gpu.core.store import InputScan, KeyRules, Keys, Store, StoreProject, canonical_json, file_id
from astro_gpu.core.workflow import TEMPLATES, Chains

# ---------------------------------------------------------------------------------------------- toy stages


class InvParams(StageParams):
    max_lights: int = P(0, label="max lights", help="", ge=0)


class Inv(Stage):
    name = "inventory"
    title = "Inventory"
    Params = InvParams

    def run(self, ctx, p):
        files = list_frames(ctx.project.config.inputs.lights)
        ctx.save_json("frames.json", {"lights": [{"path": str(f)} for f in files]})


class StackParams(StageParams):
    scale: float = P(1.5, label="scale", help="", ge=1.0, le=3.0)
    kappa: float = P(5.0, label="kappa", help="", ge=2.0)
    workers: int = P(4, label="workers", help="", ge=1)
    exclude: list[str] = P([], label="exclude", help="")


class Stack(Stage):
    name = "stack"
    title = "Stack"
    deps = ("inventory",)
    Params = StackParams
    version = "3"

    def auto(self, ctx):
        n = len(ctx.load_json("inventory", "frames.json")["lights"])
        return {"scale": Auto(2.0 if n >= 3 else 1.0, f"{n} frames")}

    def run(self, ctx, p):
        lights = [f["path"] for f in ctx.load_json("inventory", "frames.json")["lights"]]
        ctx.save_json("stack.json", {"n": len([f for f in lights if f not in p.exclude]), "scale": p.scale})


class BParams(StageParams):
    steps: int = P(10, label="steps", help="", ge=1)
    model: str = P("", label="model", help="")


class B(Stage):
    name = "b"
    title = "B"
    deps = ("stack",)
    Params = BParams

    def run(self, ctx, p):
        ctx.save_json("b.json", {"steps": p.steps})


class KParams(StageParams):
    k: float = P(1.0, label="k", help="")


class C(Stage):
    name = "c"
    title = "C"
    deps = ("stack", "b")                 # b is optional (OPTIONAL below)
    Params = KParams
    seen: dict = {}

    def run(self, ctx, p):
        try:
            ctx.input("b", "b.json")
            b_missing = False
        except StageError:
            b_missing = True
        C.seen = {"has_b": ctx.has("b"), "has_stack": ctx.has("stack"), "b_missing": b_missing}
        steps = ctx.load_json("b", "b.json")["steps"] if ctx.has("b") else None
        ctx.save_json("c.json", {"n": ctx.load_json("stack", "stack.json")["n"], "steps": steps})


class E(Stage):
    name = "e"
    title = "E"
    deps = ("c",)                         # reads b too: an EXTRA edge, optional
    Params = KParams

    def run(self, ctx, p):
        ctx.save_json("e.json", ctx.load_json("c", "c.json"))


def toy_rule(options, end, reference, learned_layer):
    return ["inventory", "stack"] + (["b"] if options.get("denoise") else []) + ["c", "e"]


def toy_implied(options, stage):
    return {"k": 2.0} if stage == "e" and options.get("denoise") else {}


def toy_chains() -> Chains:
    return Chains([Inv(), Stack(), B(), C(), E()], extra={"e": ("b",)}, optional={"c": ("b",), "e": ("b",)},
                  rule=toy_rule)


def toy_rules(auto_fields=None, **kw) -> KeyRules:
    return KeyRules(toy_chains(), auto_fields={"stack": ("scale",)} if auto_fields is None else auto_fields,
                    path_fields={"b": ("model",)}, non_key={"stack": ("workers",)}, implied=toy_implied, **kw)


def K(p) -> Keys:
    """Fresh keys, fresh disk scan."""
    return Keys(p, toy_rules())


def toy_keys(root: str) -> dict:
    """Every key of a saved toy project (also called from a subprocess)."""
    p = project2.load(root)
    k = K(p)
    return {str(w.id): k.keys(w.id) for w in p.workflows}


def write_frames(folder: Path, sizes: dict[str, int]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, n in sizes.items():
        (folder / name).write_bytes(b"\0" * n)
    return folder


CLASSIC, DENOISE = 1, 2


@pytest.fixture
def raws(tmp_path) -> Path:
    return write_frames(tmp_path / "raw" / "s1", {"f1.ARW": 100, "f2.ARW": 200, "f3.ARW": 300})


@pytest.fixture
def proj(tmp_path, raws):
    p = project2.create(tmp_path / "proj", "toy")
    p.sessions.append(Session(1, "night 1", lights=[str(raws)]))
    p.add_workflow("Classic", TEMPLATES["Classic stack"])               # no b
    p.add_workflow("Denoise", TEMPLATES["Classic + denoise"])           # with b
    project2.save(p)
    C.seen = {}
    return p


# ---------------------------------------------------------------------------------------------- the key rule

def test_auto_field_stored_at_default_is_another_key(proj):
    base = K(proj).key(CLASSIC, "stack")
    proj.workflow(CLASSIC).params["stack"] = {"scale": 1.5}     # the static default, on a field with an auto rule
    assert K(proj).key_params(CLASSIC, "stack") == {"scale": 1.5}
    assert K(proj).key(CLASSIC, "stack") != base


def test_field_without_auto_rule_at_default_is_the_same_key(proj):
    base = K(proj).key(CLASSIC, "stack")
    proj.workflow(CLASSIC).params["stack"] = {"kappa": 5.0}
    assert K(proj).key_params(CLASSIC, "stack") == {}
    assert K(proj).key(CLASSIC, "stack") == base
    proj.workflow(CLASSIC).params["stack"] = {"kappa": 6}
    assert K(proj).key(CLASSIC, "stack") != base
    # with no AUTO_FIELDS declared every field counts as automatic: nothing is dropped (the safe side)
    proj.workflow(CLASSIC).params["stack"] = {"kappa": 5.0}
    safe = Keys(proj, KeyRules(toy_chains(), None, {}, {}, toy_implied))
    assert safe.key_params(CLASSIC, "stack") == {"kappa": 5.0}


def test_values_are_validated_before_hashing(proj):
    w = proj.workflow(CLASSIC)
    keys = set()
    for v in (2, 2.0, "2", "2.0"):
        w.params["stack"] = {"scale": v}
        keys.add(K(proj).key(CLASSIC, "stack"))
    assert len(keys) == 1
    steps = set()
    for v in (3, 3.0, "3"):
        proj.workflow(DENOISE).params["b"] = {"steps": v}
        steps.add(K(proj).key(DENOISE, "b"))
    assert len(steps) == 1
    for bad in ("abc", 9.0, None):
        w.params["stack"] = {"scale": bad}
        with pytest.raises(ValueError, match=r"stack\.scale"):
            K(proj).key(CLASSIC, "stack")
    w.params["stack"] = {"nope": 1}
    with pytest.raises(ValueError, match=r"stack\.nope"):
        K(proj).key(CLASSIC, "c")                            # raised for a downstream key too


def test_path_field_is_keyed_by_its_file(proj, tmp_path):
    f = proj.root / "models" / "m.pt"
    f.parent.mkdir()
    f.write_bytes(b"x" * 50)
    w = proj.workflow(DENOISE)
    none = K(proj).key(DENOISE, "b")
    forms = ["models/m.pt", "models\\m.pt", "./models/../models/m.pt", str(f), f.as_posix(),
             str(f).replace("/", "\\"), "models/m.pt (run 1a2b3c4d)"]
    keys = set()
    for v in forms:
        w.params["b"] = {"model": v}
        keys.add(K(proj).key(DENOISE, "b"))
    assert len(keys) == 1 and none not in keys
    one = keys.pop()
    absolute = Path(os.path.abspath(f)).as_posix()
    w.params["b"] = {"model": "models\\m.pt"}
    assert K(proj).overrides(DENOISE, "b") == {"model": absolute}
    w.params["b"] = {"model": "models/m.pt (run 1a2b3c4d)"}
    assert K(proj).overrides(DENOISE, "b") == {"model": absolute + " (run 1a2b3c4d)"}
    # the same name, size and mtime elsewhere keeps the key
    other = tmp_path / "elsewhere" / "m.pt"
    other.parent.mkdir()
    shutil.copy2(f, other)
    w.params["b"] = {"model": str(other)}
    assert K(proj).key(DENOISE, "b") == one
    # a newer mtime changes it
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    w.params["b"] = {"model": "models/m.pt"}
    assert K(proj).key(DENOISE, "b") != one
    assert file_id(tmp_path / "missing.pt") == ["missing.pt", None, None]


def test_file_id_rounds_the_mtime_up_to_the_even_second_as_exfat_does(tmp_path):
    """exFAT (the F: disk) keeps mtimes to 2 s and rounds a copied file's time UP to the next even second: the key
    takes the time rounded up the same way, so an NTFS file and its copy on exFAT have the same id, and a file whose
    time is already on an even second (anything written on exFAT) keeps the id the older rule (whole seconds) gave."""
    f = tmp_path / "m.pt"
    f.write_bytes(b"x" * 7)
    base = 1_700_000_000 * 1_000_000_000                       # an even second, in ns
    S = 1_000_000_000

    def fid(ns: int) -> list:
        os.utime(f, ns=(ns, ns))
        assert f.stat().st_mtime_ns == ns                      # NTFS keeps it to 100 ns
        return file_id(f)

    want = {base: 1_700_000_000, base + 100: 1_700_000_002, base + S // 2: 1_700_000_002, base + S: 1_700_000_002,
            base + 13 * S // 10: 1_700_000_002, base + 2 * S: 1_700_000_002, base + 2 * S + 100: 1_700_000_004}
    for ns, s in want.items():
        assert fid(ns) == ["m.pt", 7, s], ns
    # NTFS time → the exFAT copy's time (rounded up to even): the same id
    for frac in (0, 1, 3, 5, 9):
        ns = base + S + frac * S // 10                         # an odd second plus a fraction
        on_ntfs = fid(ns)
        assert fid(-(-ns // (2 * S)) * 2 * S) == on_ntfs, frac
    # an even-second time (every key computed from a file on exFAT): the same as whole seconds, the rule before
    for k in range(0, 20, 2):
        ns = base + k * S
        assert fid(ns)[2] == ns // S
    # what os.utime then stat gave on F: (exFAT), P8 2026-10-31 (docs: astro_projets/_p8/exfat_mtime_test.txt)
    from astro_gpu.core.store import mtime_key
    for set_ms, got_s in ((1_000_000_001_500, 1_000_000_002), (1_000_000_002_300, 1_000_000_004),
                          (1_000_000_003_000, 1_000_000_004), (1_000_000_004_990, 1_000_000_006)):
        assert mtime_key(set_ms * 1_000_000) == mtime_key(got_s * S) == got_s


def test_excluded_pairs_are_keyed_by_session_and_name(proj, raws, tmp_path):
    none = K(proj).keys(CLASSIC)
    proj.excluded = [(1, "f2.ARW")]
    k1 = K(proj)
    assert k1.key(CLASSIC, "stack") != none["stack"] and k1.key(CLASSIC, "inventory") == none["inventory"]
    assert k1.key_params(CLASSIC, "stack") == {"exclude": [[1, "f2.ARW"]]}
    assert k1.overrides(CLASSIC, "stack") == {"exclude": [str(raws / "f2.ARW")]}
    # the raws move: same names and sizes, so the same keys; the job gets the new files
    moved = tmp_path / "moved" / "night1"
    shutil.copytree(raws, moved)
    proj.sessions[0].lights = [str(moved)]
    k2 = K(proj)
    assert k2.keys(CLASSIC) == k1.keys(CLASSIC)
    assert k2.overrides(CLASSIC, "stack") == {"exclude": [str(moved / "f2.ARW")]}
    # a camera counter restarts: the same name in another session is another frame
    s2 = write_frames(tmp_path / "raw" / "s2", {"f2.ARW": 999})
    proj.sessions.append(Session(2, "night 2", lights=[str(s2)]))
    ka = K(proj).key(CLASSIC, "stack")
    proj.excluded = [(2, "f2.ARW")]
    assert K(proj).key(CLASSIC, "stack") != ka
    assert K(proj).overrides(CLASSIC, "stack") == {"exclude": [str(s2 / "f2.ARW")]}
    proj.excluded = [(2, "f2.ARW"), (1, "f2.ARW")]
    kb = K(proj).key(CLASSIC, "stack")
    proj.excluded = [(1, "f2.ARW"), (2, "f2.ARW")]
    assert K(proj).key(CLASSIC, "stack") == kb
    # a stored 'exclude' never reaches the job or the key: the definition's pairs do
    proj.excluded = []
    proj.workflow(CLASSIC).params["stack"] = {"exclude": [str(moved / "f1.ARW")]}
    assert K(proj).key_params(CLASSIC, "stack") == {} and K(proj).overrides(CLASSIC, "stack") == {}


# ---------------------------------------------------------------------------------------------- sharing

def test_sharing_and_downstream_keys(proj):
    k = K(proj)
    a, b = k.keys(CLASSIC), k.keys(DENOISE)
    assert list(a) == ["inventory", "stack", "c", "e"] and list(b) == ["inventory", "stack", "b", "c", "e"]
    assert a["inventory"] == b["inventory"] and a["stack"] == b["stack"] and a["c"] != b["c"]
    assert k.sharing(CLASSIC, "stack") == [CLASSIC, DENOISE] and k.sharing(CLASSIC, "c") == [CLASSIC]
    w3 = proj.add_workflow("Copy", TEMPLATES["Classic stack"])
    assert K(proj).keys(w3.id) == a and K(proj).sharing(w3.id, "e") == [CLASSIC, w3.id]
    # a parameter changes only the keys downstream of it
    proj.workflow(DENOISE).params["b"] = {"steps": 20}
    b2 = K(proj).keys(DENOISE)
    assert [s for s in b if b2[s] != b[s]] == ["b", "c", "e"]
    proj.workflow(CLASSIC).params["stack"] = {"kappa": 7.5}
    a2 = K(proj).keys(CLASSIC)
    assert [s for s in a if a2[s] != a[s]] == ["stack", "c", "e"]
    # reverting restores the old keys
    del proj.workflow(DENOISE).params["b"]
    proj.workflow(CLASSIC).params["stack"] = {}
    assert K(proj).keys(DENOISE) == b and K(proj).keys(CLASSIC) == a
    # worker counts are not in the key, but the job gets them
    proj.workflow(CLASSIC).params["stack"] = {"workers": 1}
    assert K(proj).keys(CLASSIC) == a and K(proj).overrides(CLASSIC, "stack") == {"workers": 1}


def test_implied_values_win_over_stored_ones(proj):
    base = K(proj).key(DENOISE, "e")
    proj.workflow(DENOISE).params["e"] = {"k": 7.0}
    assert K(proj).key(DENOISE, "e") == base
    assert K(proj).overrides(DENOISE, "e") == {"k": 2.0} and K(proj).key_params(DENOISE, "e") == {"k": 2.0}


def test_a_subprocess_computes_the_same_keys(proj, tmp_path):
    f = proj.root / "m.pt"
    f.write_bytes(b"m")
    proj.excluded = [(1, "f3.ARW")]
    proj.workflow(CLASSIC).params["stack"] = {"scale": "2", "kappa": 6}
    proj.workflow(DENOISE).params.update({"b": {"model": "m.pt", "steps": 7}, "e": {"k": 3}})
    project2.save(proj)
    here = toy_keys(str(proj.root))
    code = (f"import json, sys; sys.path.insert(0, {str(Path(__file__).parent)!r}); import test_store; "
            f"print(json.dumps(test_store.toy_keys({str(proj.root)!r})))")
    for seed in ("0", "4242"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                             env={**os.environ, "PYTHONHASHSEED": seed}, cwd=tmp_path)
        assert out.returncode == 0, out.stderr
        assert json.loads(out.stdout.strip().splitlines()[-1]) == here
    assert canonical_json({"b": [1, 2.0], "a": None}) == '{"a":null,"b":[1,2.0]}'


# ---------------------------------------------------------------------------------------------- dependencies

def test_optional_and_extra_dependencies_only_when_present(proj):
    k = K(proj)
    assert k.deps(CLASSIC, "c") == ("stack",) and k.deps(CLASSIC, "e") == ("c",)
    assert k.deps(DENOISE, "c") == ("stack", "b") and k.deps(DENOISE, "e") == ("c", "b")
    assert set(k.key_parts(DENOISE, "e")["u"]) == {"c", "b"} and set(k.key_parts(CLASSIC, "e")["u"]) == {"c"}
    assert k.ancestors(DENOISE, "e") == ["inventory", "stack", "b", "c"]
    with pytest.raises(ValueError):
        k.key(CLASSIC, "b")
    # b's settings stored in a workflow without b change nothing there
    proj.workflow(CLASSIC).params["b"] = {"steps": 50}
    assert K(proj).keys(CLASSIC) == k.keys(CLASSIC)
    assert K(proj).keys(DENOISE) == k.keys(DENOISE)


# ---------------------------------------------------------------------------------------------- project parts

def test_inventory_part_follows_names_sizes_and_order(proj, raws, tmp_path):
    base = K(proj).key(CLASSIC, "inventory")
    lights = [n for kind, n, _ in InputScan().frames(proj) if kind == "lights"]
    assert lights == [f.name for f in list_frames(proj.inputs().lights)]       # the order the inventory reads
    # copied raws (same names and sizes) keep the key
    copy = tmp_path / "copy" / "s1"
    shutil.copytree(raws, copy)
    proj.sessions[0].lights = [str(copy)]
    assert K(proj).key(CLASSIC, "inventory") == base
    # a changed size or a new dark changes it
    (copy / "f1.ARW").write_bytes(b"\0" * 101)
    assert K(proj).key(CLASSIC, "inventory") != base
    (copy / "f1.ARW").write_bytes(b"\0" * 100)
    proj.sessions[0].darks = [str(write_frames(tmp_path / "dark", {"d1.ARW": 10}))]
    assert K(proj).key(CLASSIC, "inventory") != base
    # reordered frames change it: two folders whose order flips
    one = write_frames(tmp_path / "order" / "1", {"x1.ARW": 100})
    two = write_frames(tmp_path / "order" / "2", {"x2.ARW": 200})
    proj.sessions[0] = Session(1, "night 1", lights=[str(one), str(two)])
    ordered = K(proj).key(CLASSIC, "inventory")
    write_frames(tmp_path / "flipped" / "1", {"x2.ARW": 200})
    write_frames(tmp_path / "flipped" / "2", {"x1.ARW": 100})
    proj.sessions[0] = Session(1, "night 1", lights=[str(tmp_path / "flipped" / "2"), str(tmp_path / "flipped" / "1")])
    assert [n for _, n, _ in InputScan().frames(proj)] == ["x2.ARW", "x1.ARW"]
    assert K(proj).key(CLASSIC, "inventory") != ordered


def test_target_and_reference_parts(proj, tmp_path):
    k = K(proj)
    assert k.project_part("c") is None and k.project_part("reference") is None
    proj.target.ra = 10
    ra = K(proj).project_part("astrometry")
    proj.target.ra = 10.0
    assert K(proj).project_part("astrometry") == ra and canonical_json(ra) == canonical_json(
        K(proj).project_part("astrometry"))
    proj.target.ra = 10.5
    assert K(proj).project_part("astrometry") != ra
    ref = proj.root / "inputs" / "ref.png"
    ref.parent.mkdir()
    ref.write_bytes(b"png")
    proj.reference_image = "inputs/ref.png"
    assert K(proj).project_part("reference") == file_id(ref)


# ---------------------------------------------------------------------------------------------- store and shim

def run_job(keys: Keys, wf: int, stage: Stage) -> StoreProject:
    """One job as the runner will run it: v1's _run_stage on the shim, then <key>.tmp renamed <key>."""
    sp = StoreProject(keys, wf, stage.name)
    assert Pipeline(sp, [stage], Journal(sp.logs, echo=False))._run_stage(stage) == "done"
    tmp = sp.stage_dir(stage.name)
    assert tmp.name == sp.key + ".tmp"
    assert json.loads((tmp / "manifest.json").read_text())["status"] == "done"
    assert not sp.store.done(stage.name, sp.key)                 # a .tmp folder is never done
    os.replace(tmp, sp.store.dir(stage.name, sp.key))
    assert sp.store.done(stage.name, sp.key)
    return sp


def test_jobs_run_through_the_shim(proj, monkeypatch):
    k = K(proj)
    for st in (Inv(), Stack(), C(), E()):
        run_job(k, CLASSIC, st)
    assert C.seen == {"has_b": False, "has_stack": True, "b_missing": True}
    assert not (proj.root / "store" / "_absent").exists()                 # absent stages are never created
    store = Store(proj.root)
    assert json.loads((store.dir("stack", k.key(CLASSIC, "stack")) / "stack.json").read_text()) == {"n": 3,
                                                                                                    "scale": 2.0}
    # the second workflow reuses inventory and stack, and runs b, c, e
    for st in (B(), C(), E()):
        run_job(k, DENOISE, st)
    assert C.seen["has_b"] and not C.seen["b_missing"]
    assert json.loads((store.dir("e", k.key(DENOISE, "e")) / "e.json").read_text()) == {"n": 3, "steps": 10}
    assert sorted(p.name for p in (proj.root / "store" / "stack").iterdir()) == [k.key(CLASSIC, "stack")]
    assert not (proj.root / "store" / "_absent").exists()
    assert all(store.done(s, key) for w in (CLASSIC, DENOISE) for s, key in k.keys(w).items())


def test_shim_maps_ancestors_and_hands_out_absolute_paths(proj, raws, tmp_path):
    k = K(proj)
    store = Store(proj.root)
    sp = StoreProject(k, DENOISE, "e")
    assert sp.stage_dir("e") == store.tmp_dir("e", k.key(DENOISE, "e"))
    for a in ("inventory", "stack", "b", "c"):
        assert sp.stage_dir(a) == store.dir(a, k.key(DENOISE, a))
    assert sp.stage_dir("zz") == store.absent_dir("zz")
    assert StoreProject(k, CLASSIC, "e").stage_dir("b") == store.absent_dir("b")
    assert StoreProject(k, DENOISE, "c").stage_dir("e") == store.absent_dir("e")       # downstream: not readable
    assert sp.scratch == Path(os.path.abspath(proj.scratch)) / sp.key and sp.logs == proj.root / "logs"
    assert sp.results == proj.root / "export"
    assert sp.config.inputs.lights == [raws.as_posix()] and sp.config.params == {} and not sp.config.variants
    assert sp.overrides("zz") == {}
    # a relative reference image and session folder come out absolute
    proj.reference_image = "inputs/ref.png"
    proj.sessions[0].darks = ["darks"]
    sp = StoreProject(K(proj), CLASSIC, "c")
    assert sp.config.reference_image == (proj.root / "inputs" / "ref.png").as_posix()
    assert sp.config.inputs.darks == [(proj.root / "darks").as_posix()]
    assert not (proj.root / "store").exists()                  # building a shim creates nothing


def test_resolve_outside_the_store(proj, monkeypatch, tmp_path):
    """effective_params (describe) with own_dir: the auto values come from the done upstream, the store gets no
    stray folder."""
    k = K(proj)
    run_job(k, CLASSIC, Inv())
    sp = StoreProject(k, CLASSIC, "stack", own_dir=tmp_path / "resolve" / "stack")
    p, origin, _ = Pipeline(sp, [Stack()], Journal(sp.logs, echo=False)).effective_params(Stack())
    assert (p.scale, origin["scale"]) == (2.0, "auto")
    assert not (proj.root / "store" / "stack").exists()
    proj.workflow(CLASSIC).params["stack"] = {"scale": "1.5"}
    k = K(proj)
    sp = StoreProject(k, CLASSIC, "stack", own_dir=tmp_path / "resolve" / "stack")
    p, origin, _ = Pipeline(sp, [Stack()], Journal(sp.logs, echo=False)).effective_params(Stack())
    assert (p.scale, origin["scale"]) == (1.5, "user")


def test_done_and_usable_as_input(proj, raws, tmp_path, monkeypatch):
    k = K(proj)
    inv = run_job(k, CLASSIC, Inv())
    store = Store(proj.root)
    assert store.usable_as_input("inventory", inv.key)
    (raws / "f3.ARW").unlink()                                 # a frame listed in frames.json is gone
    assert store.usable_as_input("inventory", inv.key)         # cached per key
    store.forget()
    assert not store.usable_as_input("inventory", inv.key)
    # calibrate: the calibrated frame cache of cache.json
    key = "0123456789abcdef"
    d = store.dir("calibrate", key)
    d.mkdir(parents=True)
    cache = tmp_path / "scratch" / "lights.f16"
    (d / "cache.json").write_text(json.dumps({"path": str(cache)}))
    (d / "manifest.json").write_text(json.dumps({"status": "done", "key": key}))
    assert store.done("calibrate", key) and not store.usable_as_input("calibrate", key)
    cache.parent.mkdir()
    cache.write_bytes(b"f16")
    assert Store(proj.root).usable_as_input("calibrate", key)
    # never done: a .tmp folder, another key's manifest, a failed result
    t = store.tmp_dir("calibrate", "fedcba9876543210")
    t.mkdir()
    (t / "manifest.json").write_text(json.dumps({"status": "done"}))
    assert not store.done("calibrate", "fedcba9876543210")
    with pytest.raises(ValueError):
        store.done("calibrate", "fedcba9876543210.tmp")
    (d / "manifest.json").write_text(json.dumps({"status": "done", "key": "ffffffffffffffff"}))
    assert not store.done("calibrate", key)
    (d / "manifest.json").write_text(json.dumps({"status": "failed", "key": key}))
    assert not store.done("calibrate", key) and not store.usable_as_input("calibrate", key)


def test_context_has(tmp_path):
    v1 = Project.create(tmp_path / "v1", "v1", Inputs())
    ctx = Context(v1, B(), Journal(v1.logs, echo=False), "cpu")
    assert not ctx.has("stack")
    (v1.work / "stack").mkdir()
    assert ctx.has("stack")


# ---------------------------------------------------------------------------------------------- the v2 stages

def test_default_rules_read_stages_v2(monkeypatch):
    from astro_gpu.core import store
    from astro_gpu.stages import v2
    monkeypatch.setattr(store, "_RULES", [])
    r = store.key_rules()
    assert r.auto_fields("stack") == v2.AUTO_FIELDS["stack"] and r.auto_fields("optics") == frozenset()
    assert r.path_fields("starless") == {"learned_stars"} and r.path_fields("denoise") == {"reuse_model"}
    assert r.non_key("calibrate") == {"workers"} and r.non_key("star_net") == {"device"}
    assert r.retired("stack") == v2.RETIRED["stack"] and r.constants("register") == {"fwhm_method": "moffat"}


def test_real_stage_graph_shares_as_designed(tmp_path, raws):
    p = project2.create(tmp_path / "real", "real")
    p.sessions.append(Session(1, "night 1", lights=[str(raws)]))
    ids = {name: p.add_workflow(name, TEMPLATES[name]).id for name in TEMPLATES}
    classic, cdn, full, neural = (ids[n] for n in ("Classic stack", "Classic + denoise", "Full",
                                                   "Full + neural stars"))
    k = Keys(p)
    for s in ("inventory", "calibrate", "register", "stack", "astrometry", "catalogs", "color"):
        assert len({k.key(i, s) for i in ids.values()}) == 1, s
    assert k.sharing(classic, "color") == sorted(ids.values())
    assert k.key(cdn, "denoise") == k.key(full, "denoise") == k.key(neural, "denoise")
    assert k.key(full, "stars") == k.key(neural, "stars")
    assert k.key(full, "starless") != k.key(neural, "starless")          # star_net is read (EXTRA)
    assert k.block_key(classic, "calibration") == (k.key(classic, "inventory"), k.key(classic, "calibrate"))
    assert k.block_key(full, "astrometry") == (k.key(full, "astrometry"), k.key(full, "catalogs"))  # no surveys
    assert k.block_key(full, "output")[-1] == f"workflow {full}"
    # the implied star_net.enabled wins over a stored value; worker counts are not in the key
    p.workflow(neural).params["star_net"] = {"enabled": False, "device": "aux"}
    p.workflow(full).params["calibrate"] = {"workers": 2}
    k2 = Keys(p)
    assert k2.keys(neural) == k.keys(neural) and k2.keys(full) == k.keys(full)
    assert k2.overrides(neural, "star_net") == {"enabled": True, "device": "aux"}
