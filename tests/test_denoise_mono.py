"""Issue #45: the Denoise network of the monochrome stacks of few frames, its key, and the bench tool of its training."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from astro_gpu.core import project2
from astro_gpu.core.project2 import Session
from astro_gpu.core.store import Keys
from astro_gpu.core.workflow import TEMPLATES
from astro_gpu.stages import denoise as dn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "bench"))
import denoise_dataset  # noqa: E402
import denoise_mono  # noqa: E402

DENOISED = TEMPLATES["Classic + denoise"]
FEW = 2                                                 # light frames of a default test project


class Zero(torch.nn.Module):
    """A network whose correction is nil: the result is its input."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros_like(x)[:, :3] * self.w


def frames(folder: Path, n: int) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"a{i}.fit").write_bytes(b"\0" * (100 + i))
    return folder


def project(root: Path, filt: str, method: str = "pretrained", lights: int = FEW, params=None) -> project2.Project2:
    folder = frames(root.parent / f"raw_{root.name}", lights)
    p = project2.create(root, "p")
    p.sessions.append(Session(1, "n1", lights=[str(folder)], filter=filt))
    p.add_workflow("W", {**DENOISED, "denoise": method}, params)
    project2.save(p)
    return project2.load(root)


@pytest.fixture
def mono_file(tmp_path, monkeypatch):
    f = tmp_path / "pretrained_unet_mono.pt"
    f.write_bytes(b"x")
    monkeypatch.setattr(dn, "MODEL_MONO", f)
    monkeypatch.setattr(dn, "MODEL_MONO_ID", "abc1234567")
    return f


MONO_V = f"{dn.Denoise.version}+mono-abc1234567"


# ---------------------------------------------------------------------------------------------- the network shipped
def test_the_monochrome_network_is_in_the_app_and_in_what_the_installer_copies():
    assert dn.MODEL_MONO.exists()
    tracked = subprocess.run(["git", "ls-files", "astro_gpu/resources/denoise"], capture_output=True, text=True, cwd=ROOT,
                             check=True).stdout.split()                  # installer/build.py copies what git tracks
    assert "astro_gpu/resources/denoise/pretrained_unet_mono.pt" in tracked


def test_the_monochrome_network_loads_and_differs_from_the_colour_one():
    dev = torch.device("cpu")
    a, b = dn.pretrained(dev), dn.pretrained(dev, mono=True)
    assert any(not torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


# ---------------------------------------------------------------------------------------------- the key
def test_only_a_monochrome_stack_of_few_frames_with_the_pretrained_method_changes_key(tmp_path, mono_file):
    colour = Keys(project(tmp_path / "c", ""))
    mono = Keys(project(tmp_path / "m", "Ha"))
    assert colour.key_parts(1, "denoise")["v"] == dn.Denoise.version       # colour: as before the issue
    assert mono.mono_lights(1) == FEW and colour.mono_lights(1) is None
    assert mono.key_parts(1, "denoise")["v"] == MONO_V
    assert mono.key(1, "denoise") != colour.key(1, "denoise")


def test_a_monochrome_stack_of_30_frames_or_more_keeps_its_key(tmp_path, mono_file):
    k = Keys(project(tmp_path / "m", "Ha", lights=dn.MONO_FEW_FRAMES))
    assert k.mono_lights(1) == dn.MONO_FEW_FRAMES and k.key_parts(1, "denoise")["v"] == dn.Denoise.version


def test_the_frame_limit_counts(tmp_path, mono_file):
    k = Keys(project(tmp_path / "m", "Ha", lights=40, params={"inventory": {"max_lights": 20}}))
    assert k.mono_lights(1) == 20 and k.key_parts(1, "denoise")["v"] == MONO_V


def test_the_adapted_method_keeps_its_key_on_a_monochrome_stack(tmp_path, mono_file):
    k = Keys(project(tmp_path / "m", "Ha", "adapted"))
    assert k.key_parts(1, "denoise")["v"] == dn.Denoise.version


def test_without_the_monochrome_file_the_key_is_the_colour_one(tmp_path, monkeypatch):
    monkeypatch.setattr(dn, "MODEL_MONO", tmp_path / "absent.pt")
    assert Keys(project(tmp_path / "m", "Ha")).key_parts(1, "denoise")["v"] == dn.Denoise.version


def test_a_monochrome_workflows_other_stages_have_the_versions_of_a_colour_one(tmp_path, mono_file):
    colour = Keys(project(tmp_path / "c", ""))
    mono = Keys(project(tmp_path / "m", "Ha"))
    shared = [s for s in mono.chain(1) if s in colour.chain(1) and s != "denoise"]
    assert "stack" in shared
    assert all(mono.key_parts(1, s)["v"] == colour.key_parts(1, s)["v"] for s in shared)


def test_the_setting_removed_is_ignored_where_it_is_stored(tmp_path, mono_file):
    plain = Keys(project(tmp_path / "a", "Ha"))
    stored = Keys(project(tmp_path / "b", "Ha", params={"denoise": {"mono_adapt": False}}))
    assert stored.key_parts(1, "denoise") == plain.key_parts(1, "denoise")


# ---------------------------------------------------------------------------------------------- the run
class Ctx:
    def __init__(self, tmp: Path, img: np.ndarray, lights: int | None):
        self.device = torch.device("cpu")
        self.tmp = tmp
        self.img = img
        self.project = SimpleNamespace(keys=SimpleNamespace(mono_lights=lambda w: lights), workflow=object())
        self.metrics: dict = {}
        self.logs: list[str] = []
        self.saved: dict[str, np.ndarray] = {}

    def load(self, stage, name, mmap=False):
        return np.ones_like(self.img) if name == "drz_weight.npy" else self.img.copy()

    def metric(self, k, v):
        self.metrics[k] = v

    def log(self, msg, *_a):
        self.logs.append(msg)

    def out(self, name):
        return self.tmp / name

    def save(self, name, a):
        self.saved[name] = a

    def progress(self, *_a):
        pass

    def preview(self, *_a):
        pass

    def check_stop(self):
        pass


def run_stage(monkeypatch, tmp_path, img, lights=12, **params):
    asked, adapted = [], []
    monkeypatch.setattr(dn, "image_in", lambda ctx, half="full": SimpleNamespace(img=img.copy()))
    monkeypatch.setattr(dn, "pretrained", lambda dev, mono=False: asked.append(mono) or Zero())
    monkeypatch.setattr(dn, "adapt", lambda *a, **k: adapted.append(a[8]))
    ctx = Ctx(tmp_path, img, lights)
    dn.Denoise().run(ctx, dn.DenoiseParams(**params))
    return asked, adapted, ctx


def noisy(grey: bool) -> np.ndarray:
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, (3, 128, 128)).astype(np.float32)
    if grey:
        a[0] = a[1]
        a[2] = a[1]
    return a


@pytest.mark.parametrize("lights", [1, 12, dn.MONO_FEW_FRAMES - 1])
def test_a_grey_stack_of_few_frames_uses_the_monochrome_network_and_is_not_adapted(monkeypatch, tmp_path, mono_file, lights):
    asked, adapted, ctx = run_stage(monkeypatch, tmp_path, noisy(grey=True), lights=lights)
    assert asked == [True] and adapted == [] and ctx.metrics["adaptation_steps"] == 0
    full = ctx.saved["n2n_full.npy"]
    assert np.array_equal(full[0], full[1]) and np.array_equal(full[2], full[1])       # it stays grey


@pytest.mark.parametrize("lights", [dn.MONO_FEW_FRAMES, 200, None])
def test_many_frames_or_a_workflow_not_monochrome_keep_the_colour_network(monkeypatch, tmp_path, mono_file, lights):
    asked, adapted, _ = run_stage(monkeypatch, tmp_path, noisy(grey=True), lights=lights)
    assert asked == [False] and adapted == []


def test_a_colour_stack_keeps_the_colour_network(monkeypatch, tmp_path, mono_file):
    assert run_stage(monkeypatch, tmp_path, noisy(grey=False), lights=None)[0] == [False]


def test_the_adapted_method_stays_a_choice_and_starts_from_the_colour_network(monkeypatch, tmp_path, mono_file):
    asked, adapted, _ = run_stage(monkeypatch, tmp_path, noisy(grey=True), method="adapted")
    assert asked == [False] and adapted == [500]


def test_without_the_monochrome_file_a_grey_stack_uses_the_colour_network(monkeypatch, tmp_path):
    monkeypatch.setattr(dn, "MODEL_MONO", tmp_path / "absent.pt")
    assert run_stage(monkeypatch, tmp_path, noisy(grey=True))[0] == [False]


def test_a_mono_key_and_run_agree_on_the_count(tmp_path, mono_file):
    k = Keys(project(tmp_path / "m", "Ha", lights=dn.MONO_FEW_FRAMES - 1))
    assert dn.uses_mono_network("pretrained", k.mono_lights(1))
    assert not dn.uses_mono_network("adapted", k.mono_lights(1))


def test_is_grey():
    assert dn.is_grey(noisy(grey=True)) and not dn.is_grey(noisy(grey=False))


def test_the_tooltip_says_it_brightens_faint_stars_in_both_languages():
    from astro_gpu import texts
    en, fr = texts.param_tip("denoise", "method"), texts.param_tip("denoise", "method", "fr")
    assert "+8 to +11 %" in en and "not for photometry" in en
    assert "+8 à +11 %" in fr and "pas à la photométrie" in fr


# ---------------------------------------------------------------------------------------------- the bench tool
def pair(base: Path, name: str, role: str, size: int = 400) -> None:
    d = base / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for k in "AB":
        np.save(d / f"{k}.npy", rng.normal(0, 1, (3, size, size)).astype(np.float16))
    np.save(d / "nm.npy", np.ones((size, size), np.float16))
    (d / "pair.json").write_text(json.dumps({"name": name, "role": role, "shape": [3, size, size]}))


def test_a_pair_whose_halves_differ_beyond_their_noise_is_measured_as_such(tmp_path):
    pair(tmp_path, "good", "train")
    pair(tmp_path, "bad", "train")
    b = np.load(tmp_path / "bad" / "B.npy")
    b[:, 100:110, 100:110] += 2000.0                    # one half holds what the other has not
    np.save(tmp_path / "bad" / "B.npy", b)
    assert denoise_mono.mismatch(tmp_path / "good") < 2 < denoise_mono.MAX_MISMATCH < denoise_mono.mismatch(tmp_path / "bad")


def test_training_can_start_from_given_weights(tmp_path):
    torch.manual_seed(123)
    net = dn.UNet()
    init = tmp_path / "init.pt"
    torch.save({"net": {k: v.half() for k, v in net.state_dict().items()}}, init)
    pair(tmp_path, "train_a", "train")
    # no learning: the file written at the end is the starting point
    denoise_dataset.train(2e-4, "cpu", 64, 2, 0.0, False, "_t", base=tmp_path, init=init)
    out = torch.load(tmp_path / "pretrained_unet_t.pt", weights_only=False)["net"]
    assert all(torch.allclose(out[k].float(), v.half().float()) for k, v in net.state_dict().items())


def test_export_writes_under_the_given_folder(tmp_path):
    from astro_gpu.stages.denoise import norm_params
    s = tmp_path / "stack"
    s.mkdir()
    rng = np.random.default_rng(2)
    a = (rng.normal(1.0, 0.1, (3, 64, 64))).astype(np.float32)
    for n in ("drz_A", "drz_B"):
        np.save(s / f"{n}.npy", a)
    np.save(s / "drz_weight.npy", np.ones((3, 64, 64), np.float32))
    assert norm_params(a, np.ones((3, 64, 64), np.float32))[0].shape == (3,)
    m = denoise_dataset.export_one("p", s, "train", tmp_path / "out")
    assert m["role"] == "train" and (tmp_path / "out" / "p" / "pair.json").exists()


def write_json(f: Path, o) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(o))


def test_held_out_cases_are_found_from_the_store(tmp_path, monkeypatch):
    s = tmp_path / "b" / "store"
    for kind, key, folder in (("pretrained", "dp", "stk1"), ("adapted", "da", "stk1")):
        write_json(s / "denoise" / key / "manifest.json", {
            "status": "done", "key_parts": {"p": {"method": "adapted"} if kind == "adapted" else {},
                                            "u": {"stack": folder, "gradient": "gr1"}}})
    write_json(s / "stack" / "stk1" / "manifest.json", {"metrics": {"frames_used": 13}, "key_parts": {"u": {"calibrate": "c1"}}})
    write_json(s / "calibrate" / "c1" / "manifest.json", {"key_parts": {"u": {"inventory": "i1"}}})
    write_json(s / "inventory" / "i1" / "frames.json", {"lights": [{"path": "F:/x/lights/rp/f.fits"}]})
    monkeypatch.setattr(denoise_mono, "HELD_OUT", {"m51": tmp_path / "b"})
    [c] = denoise_mono.held_out_cases()
    assert c["name"] == "m51_rp" and c["frames"] == 13
    assert c["pretrained"].name == "dp" and c["adapted"].name == "da" and c["gradient"].name == "gr1"


def test_the_measures_take_any_network_and_images():
    import mono_models
    img = np.random.default_rng(0).normal(1.0, 0.01, (3, 400, 400)).astype(np.float32)
    img[:, 100:103, 100:103] += 2.0                     # a star, compact
    bk, sg, nm = dn.norm_params(img, np.ones((3, 400, 400), np.float32))
    net = Zero().eval()
    res = mono_models.measure_net(net, img, img, img, img.copy(), nm, bk, sg)
    assert res["noise_reduction"] == 1.0 and "faint_sky" in res["half_consistency"]
