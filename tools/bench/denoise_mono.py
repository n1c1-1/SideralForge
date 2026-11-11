"""Bench of issue #45: a fine-tuning pass of the pretrained denoiser on monochrome half-stack pairs.

    python tools/bench/denoise_mono.py stack [--only m16,m101,...]     the monochrome sets stacked on the processor
    python tools/bench/denoise_mono.py export                          the pairs (A, B, noise map) of every stack
    python tools/bench/denoise_mono.py train [--hours 1] [--lr 1e-4]   the fine-tuning (a GPU: set CUDA_VISIBLE_DEVICES)
    python tools/bench/denoise_mono.py adapt                           the Adapted method on the held-out stacks
    python tools/bench/denoise_mono.py measure [--models pretrained,...]  the held-out stacks, the three networks

Everything under F:/astro_essais/mono_affinage; the data bank and the benches of the issue are only read. ``stack``:
one project per set in projects/<name>, one workflow per filter and per number of lights (the whole set, then fewer:
the noise of few frames is what the pretrained network has not seen), stacked by this chain (up to the stack, on the
processor). ``export``: the pairs, normalised as the denoiser normalises them (tools/bench/denoise_dataset.export_one),
in pairs/<name>; the held-out stacks (M51, Helix: the issue's benches; M45 for a stack of many frames) as validation
pairs, never trained on. ``train``: denoise_dataset.train, from the weights of the bundled pretrained network, at a
low learning rate. ``measure``: the measures of tools/bench/mono_models.py (invented dots, star flux, peak and width,
noise reduction) on the held-out stacks, for the pretrained network, the fine-tuned one and the Adapted method
(denoise.adapt on each stack, or its result in the bench store when the bench ran it), on the images Denoise reads
(the stack minus the sky gradient of the bench's run), and a sheet of crops per stack.

Outcome (8 November 2026): the fine-tuned networks removed most of the invented dots but did not keep the stars within
2 % on the held-out stacks (1 h: faint stars' flux +4 to +11 %; 20 min: bright stars' peak -15 to -28 % on Helix and M45).
The 1 h one ships for the monochrome stacks of few frames, for pictures and not for photometry (resources/denoise,
stages/denoise.py): the Adapted method, the alternative, left a black dot in a bright star of M51.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

HOME = Path("F:/astro_essais/mono_affinage")
PROJECTS = HOME / "projects"
PAIRS = HOME / "pairs"
ADAPTED = HOME / "adapted"
DATA = Path("F:/dataset_astro/data")
TAG = "_mono"
#: the monochrome sets the fine-tuning trains on: (folder of the data bank, filters, the numbers of lights to stack,
#: 0 for all of them). M51 (the sets 06 and 28), Helix (21) and M45 (07) are held out: their fields are not in here
SETS = {
    "m16": ("19_mono_lco04m_sq30_m16_sho", ("H-Alpha", "OIII", "SII"), (20, 10)),
    "m101": ("20_mono_lco04m_sq46_m101_habvr", ("H-Alpha", "rp"), (0,)),        # 61 megapixels: 12 min per stack
    "m42_kb27": ("54_mono_lco04m_kb27_meade16_m42_short", ("B", "V", "rp"), (0, 16, 10)),
    "ngc602": ("46_mono_newton_asi1600mm_ngc602_wing_sho", ("Ha", "OIII", "SII"), (0, 20, 10)),
    "ngc6946": ("51_mono_newton_asi1600mm_ngc6946_rgb", ("Red", "Green", "Blue"), (0, 20, 10)),
    "m42_f16m": ("26_mono_refractor_apogeef16m_toa150_m42_lrgb", ("Blue", "Green", "Red"), (0,)),
}
#: the bench's stacks that train too (read only): M16's whole sets, stacked for the issue
BENCH_TRAIN = Path("F:/astro_essais/bench_mono/m16_sho")
#: the held-out stacks: their bench project
HELD_OUT = {"m51": Path("F:/astro_essais/bench43/m51_rgb"), "helix": Path("F:/astro_essais/bench43/helix_sho"),
            "m45": Path("F:/astro_essais/bench_mono/m45_g")}
#: a training pair whose halves differ by more than this many times their noise (``mismatch``) is left out: the good
#: pairs measure 1 to 5, the frames that do not match (most of M42 kb27, M101 in R) 12 to 90
MAX_MISMATCH = 8.0
#: the star-detection threshold of a second try when the registration finds too few stars
FAINT_SIGMA = 5.0


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


# ---------------------------------------------------------------------------------------------- stack
def make(name: str, folder: str, filters: tuple, counts: tuple):
    from mono_path import FILTER

    from astro_gpu.core import project2
    from astro_gpu.core import workflow as wfl
    from astro_gpu.core.project import Target
    from astro_gpu.core.project2 import Session
    root = PROJECTS / name
    if (root / project2.FILE).exists():
        return project2.load(root)
    data = DATA / folder
    p = project2.create(root, name, Target(name=name))
    p.stack_mode = "per_session"
    p.devices["main"] = "cpu"
    dirs = {k: [str(data / k)] if (data / k).is_dir() else [] for k in ("darks", "bias")}
    for i, f in enumerate(filters, 1):
        flats = data / "flats" / f
        p.sessions.append(Session(i, f, lights=[str(data / "lights" / f)], darks=dirs["darks"], bias=dirs["bias"],
                                  flats=[str(flats)] if flats.is_dir() else [], filter=FILTER.get(f, "L")))
        for n in counts:
            params = {"inventory": {"max_lights": n}} if n else {}
            w = p.add_workflow(f"{FILTER.get(f, f)} {n or 'all'}", wfl.TEMPLATES["Classic stack"], params)
            w.sessions = [i]
    project2.save(p)
    return project2.load(root)


def stack(only: list[str] | None) -> int:
    from astro_gpu.core.sirilcat import private_home
    private_home(HOME / "_state")                                   # the installed catalogue stays seen
    from astro_gpu import api
    from astro_gpu.core import runner2
    report_f = HOME / "stack.json"
    report = json.loads(report_f.read_text()) if report_f.exists() else {}
    for name, (folder, filters, counts) in SETS.items():
        if only and name not in only:
            continue
        p = make(name, folder, filters, counts)
        for w in p.workflows:
            tag = f"{name}/{w.name}"
            if report.get(tag, {}).get("status") == "done":
                continue
            t0 = time.time()
            why, ok = "", False
            for attempt in (1, 2):
                plan = api.plan_to_block(p, w.id, "stacking", "cpu")
                if plan.jobs:
                    log(f"{tag}: {len(plan.jobs)} jobs: {', '.join(j.stage for j in plan.jobs)}")
                    # in this process: a detached worker imports the installed package, not this checkout's code
                    info = runner2.prepare_run(p, plan, "cpu")
                    log(f"{tag}: {runner2.run_plan(p.root, info.plan_file, echo=False)}")
                    api.refresh(p)
                states = api.states(p)
                st, reg = states.get((w.id, "stack")), states.get((w.id, "register"))
                ok = st is not None and st.state == "done"
                why = " ".join(x.reason for x in (reg, st) if x is not None and x.state == "failed")
                if ok or attempt == 2 or not ("too few stars" in why or "usable frames" in why):
                    break
                log(f"{tag}: {why}: again with the star detection threshold at {FAINT_SIGMA}")
                api.set_param(p, [w.id], "register", "detect_sigma", FAINT_SIGMA)
                api.save(p)
            report[tag] = {"status": "done" if ok else "failed", "minutes": round((time.time() - t0) / 60, 1),
                           "why": why}
            log(f"{tag}: {report[tag]}")
            report_f.write_text(json.dumps(report, indent=1))
    return 0


# ---------------------------------------------------------------------------------------------- the benches' stores
def manifest(d: Path) -> dict | None:
    try:
        return json.loads((d / "manifest.json").read_text())
    except (OSError, ValueError):
        return None


def stack_filter(store: Path, stack_dir: Path) -> str:
    """The filter of a stack of a bench project: the name of its first light's folder."""
    cal = manifest(stack_dir)["key_parts"]["u"]["calibrate"]
    inv = manifest(store / "calibrate" / cal)["key_parts"]["u"]["inventory"]
    frames = json.loads((store / "inventory" / inv / "frames.json").read_text())
    return Path(frames["lights"][0]["path"]).parent.name


def held_out_cases() -> list[dict]:
    """The held-out stacks: {name, stack, gradient, pretrained, adapted (a denoise folder or None)}."""
    cases: dict[str, dict] = {}
    for proj, root in HELD_OUT.items():
        store = root / "store"
        for d in sorted((store / "denoise").iterdir()):
            m = manifest(d) if not d.name.endswith(".tmp") else None
            if m is None or m.get("status") != "done" or "stack" not in m.get("key_parts", {}).get("u", {}):
                continue
            u = m["key_parts"]["u"]
            stack_dir = store / "stack" / u["stack"]
            name = f"{proj}_{stack_filter(store, stack_dir)}"
            c = cases.setdefault(name, {"name": name, "root": root, "stack": stack_dir,
                                        "gradient": store / "gradient" / u["gradient"] if "gradient" in u else None,
                                        "frames": (manifest(stack_dir).get("metrics") or {}).get("frames_used"),
                                        "pretrained": None, "adapted": None})
            method = (m["key_parts"].get("p") or {}).get("method", "pretrained")
            c["adapted" if method == "adapted" else "pretrained"] = d
    return [cases[k] for k in sorted(cases)]


# ---------------------------------------------------------------------------------------------- export
def export() -> int:
    from denoise_dataset import export_one

    from astro_gpu.core import project2
    from astro_gpu.core.store import Keys, Store
    pairs: list[tuple[str, Path, str]] = []
    for name in SETS:
        root = PROJECTS / name
        if not (root / project2.FILE).exists():
            log(f"{name}: no project")
            continue
        p = project2.load(root)
        k = Keys(p)
        for w in p.workflows:
            d = Store(root).dir("stack", k.key(w.id, "stack"))
            if (d / "drz_A.npy").is_file():
                pairs.append((f"{name}_{w.name}".replace(" ", "_"), d, "train"))
            else:
                log(f"{name} {w.name}: no stack")
    for d in sorted((BENCH_TRAIN / "store" / "stack").iterdir()):         # M16, the whole sets the issue stacked
        if (d / "drz_A.npy").is_file():
            pairs.append((f"m16_{stack_filter(BENCH_TRAIN / 'store', d)}_all".replace("-", ""), d, "train"))
    for c in held_out_cases():                            # watched during the training, never trained on
        pairs.append((f"heldout_{c['name']}".replace("-", ""), c["stack"], "validation"))
    PAIRS.mkdir(parents=True, exist_ok=True)
    for name, d, role in pairs:
        t = time.time()
        m = export_one(name, d, role, PAIRS)
        r = mismatch(PAIRS / name)
        if role == "train" and r > MAX_MISMATCH:              # the halves differ beyond their noise: not trained on
            m["role"] = "excluded"
            (PAIRS / name / "pair.json").write_text(json.dumps(m, indent=1))
        log(f"{name}: {m['shape']} ({m['role']}, halves' difference {r:.1f} times its noise) in {time.time() - t:.0f} s")
    return 0


def mismatch(d: Path) -> float:
    """How much the two half-stacks of a pair differ beyond their noise: the RMS of A - B over its robust noise (MAD),
    on the covered field. A pair made of frames that do not match (a bad registration, a passing cloud) is far above
    the others: its target is not the input's sky plus independent noise, which Noise2Noise needs."""
    A, B = (np.asarray(np.load(d / f"{k}.npy", mmap_mode="r")[1, ::3, ::3], np.float32) for k in "AB")
    n = np.asarray(np.load(d / "nm.npy", mmap_mode="r")[::3, ::3], np.float32)
    x = (A - B)[(n > 0) & (n < 1.6)]
    return float(x.std() / max(1.4826 * np.median(np.abs(x - np.median(x))), 1e-6))


# ---------------------------------------------------------------------------------------------- train
def train(a) -> int:
    import torch
    from denoise_dataset import train as run

    from astro_gpu.stages.denoise import MODEL
    log(f"card: {torch.cuda.get_device_name(0)}")
    return run(a.hours, a.device, a.patch, a.batch, a.lr, a.resume, a.tag, base=PAIRS, init=MODEL)


# ---------------------------------------------------------------------------------------------- eval
def load_case(c: dict) -> dict:
    """The images Denoise read for this stack in the bench (the half-stacks and the stack minus the Gradient block's
    sky), float32, and the normalisation of the denoiser."""
    from types import SimpleNamespace

    from astro_gpu.stages import denoise as dn
    from astro_gpu.stages.gradient import sky_variation
    ims = {k: np.load(c["stack"] / f"drz_{k}.npy").astype(np.float32) for k in ("A", "B", "full")}
    wt = np.load(c["stack"] / "drz_weight.npy").astype(np.float32)
    if c["gradient"] is not None:
        g = c["gradient"]
        ctx = SimpleNamespace(load_json=lambda _e, n: json.loads((g / n).read_text()),
                              load=lambda _e, n: np.load(g / n))
        sky = sky_variation(ctx, *ims["full"].shape[1:], entry="gradient").astype(np.float32)
        ims = {k: v - sky for k, v in ims.items()}
    bk, sg, nm = dn.norm_params(ims["A"], wt)
    return {**ims, "wt": wt, "bk": bk, "sg": sg, "nm": nm}


def denoised(net, im: dict, dev) -> np.ndarray:
    """A network's result on a held-out stack, as the Denoise block computes it for a grey stack."""
    from astro_gpu.stages import denoise as dn
    out = dn.apply_net(net, im["full"], (im["nm"] / np.sqrt(2)).astype(np.float32), im["bk"], im["sg"], dev)
    return dn.keep_level(np.repeat(out.mean(0, keepdims=True), 3, axis=0), im["full"], im["wt"])


class Quiet:
    """The context denoise.adapt asks for: no project here."""

    def check_stop(self):
        pass

    def progress(self, *_a):
        pass

    def log(self, msg, *_a):
        log(f"    {msg}")


def adapted_net(c: dict, im: dict, dev):
    """The Adapted method's network for this stack: what the bench's run saved, or the adaptation done here (saved in
    adapted/<case>.pt)."""
    import torch

    from astro_gpu.stages import denoise as dn
    ADAPTED.mkdir(parents=True, exist_ok=True)
    f = ADAPTED / f"{c['name']}.pt"
    net = dn.pretrained(dev)
    if c["adapted"] is not None:
        f = c["adapted"] / "n2n_model.pt"
    elif not f.exists():
        steps, lr = dn.ADAPTATIONS["short"]
        log(f"{c['name']}: adapting ({steps} steps)")
        dn.adapt(Quiet(), net, im["A"], im["B"], im["nm"], im["bk"], im["sg"], dev, steps, lr, 0)
        torch.save({"net": net.state_dict()}, f)
    net.load_state_dict(torch.load(f, map_location=dev, weights_only=False)["net"])
    net.eval()
    return net


def adapt(a) -> int:
    """The adaptations the bench did not run (the Adapted method, 500 steps): on the training's card."""
    import torch
    dev = torch.device(a.device)
    for c in held_out_cases():
        if c["adapted"] is None:
            adapted_net(c, load_case(c), dev)
    return 0


def faint_crop(g: np.ndarray, sig: float, size: int = 200) -> tuple[int, int]:
    """A crop corner in the faintest sky, away from the stars."""
    import cv2
    box = cv2.boxFilter((g > np.median(g) + 3 * sig).astype(np.float32), -1, (size, size))
    level = cv2.boxFilter(g, -1, (size, size))
    h = size // 2
    ok = np.zeros_like(g, bool)
    ok[h:-h, h:-h] = True
    score = np.where(ok, level + 30 * sig * box, np.inf)
    y, x = np.unravel_index(np.argmin(score), g.shape)
    return int(y) - h, int(x) - h


def sheet(out: Path, title: str, imgs: dict, sig: float, bkg: float) -> None:
    """The stack and each network's result on a faint-sky crop (stretched to show its dots) and around stars."""
    import cv2
    from mono_models import crops, stretch
    size = 200
    g = imgs["stack"]
    y0, x0 = faint_crop(g, sig, size)
    star_y, star_x = crops(g, sig, bkg, size)[1]                     # the densest star field
    faint_top, top = bkg + 12 * sig, float(np.percentile(g[::5, ::5], 99.8))
    rows = []
    for (y, x), t in (((y0, x0), faint_top), ((star_y, star_x), top)):
        cells = [stretch(a[y:y + size, x:x + size], bkg, t) for a in imgs.values()]
        rows.append(np.hstack([np.pad(c, 3, constant_values=0.25) for c in cells]))
    a = (np.vstack(rows) * 255).astype(np.uint8)
    f = 2
    a = cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    a = cv2.copyMakeBorder(a, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=40)
    for i, lab in enumerate(imgs):
        cv2.putText(a, lab, (i * f * (size + 6) + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 230, 2)
    cv2.putText(a, title, (a.shape[1] - 11 * len(title) - 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)
    cv2.imwrite(str(out), a, [cv2.IMWRITE_JPEG_QUALITY, 92])


def evaluate(a) -> int:
    import torch
    from mono_models import DEV, mad, measure_net

    from astro_gpu.stages import denoise as dn
    out = HOME / "eval"
    out.mkdir(parents=True, exist_ok=True)
    models = [x for x in a.models.split(",") if x]
    res_f = out / "mono_eval.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    only = [x for x in a.only.split(",") if x]
    for c in held_out_cases():
        if only and not any(x in c["name"] for x in only):
            continue
        t0 = time.time()
        log(f"{c['name']}: {c['frames']} frames")
        im = load_case(c)
        g = im["full"][1]
        bkg, sig = mad(g)
        (out / "images").mkdir(exist_ok=True)
        for name in models:
            if name == "pretrained":
                net = dn.pretrained(DEV)
                net.eval()
            elif name == "mono":
                net = dn.UNet().to(DEV)
                state = torch.load(a.model, map_location=DEV, weights_only=False)["net"]
                net.load_state_dict({k: v.float() for k, v in state.items()})
                net.eval()
            else:
                if c["adapted"] is None and not (ADAPTED / f"{c['name']}.pt").exists():
                    log(f"  {name}: not adapted yet (the adapt command, on the card)")
                    continue
                net = adapted_net(c, im, DEV)
            t1 = time.time()
            dF = denoised(net, im, DEV)
            res.setdefault(c["name"], {})[name] = measure_net(net, im["full"], dF, im["A"], im["B"], im["nm"], im["bk"],
                                                              im["sg"])
            res[c["name"]]["frames"] = c["frames"]
            log(f"  {name}: {json.dumps(res[c['name']][name])} ({time.time() - t1:.0f} s)")
            res_f.write_text(json.dumps(res, indent=1))
            np.save(out / "images" / f"{c['name']}_{name}.npy", dF[1].astype(np.float32))
        # the sheet holds every result measured so far, whichever run measured it
        saved = {m: out / "images" / f"{c['name']}_{m}.npy" for m in ("pretrained", "mono", "adapted")}
        imgs = {"stack": g, **{m: np.load(f) for m, f in saved.items() if f.exists()}}
        sheet(out / f"{c['name']}.jpg", f"{c['name']} {c['frames']} frames", imgs, sig, bkg)
        log(f"{c['name']}: done in {(time.time() - t0) / 60:.1f} min")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("stack", "export", "train", "adapt", "measure"))
    ap.add_argument("--only", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--patch", type=int, default=192)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default=TAG, help="suffix of the checkpoint files")
    ap.add_argument("--models", default="pretrained,mono,adapted")
    ap.add_argument("--model", default=str(PAIRS / f"pretrained_unet{TAG}.pt"))
    a = ap.parse_args(argv)
    only = [x.strip() for x in a.only.split(",") if x.strip()] or None
    if a.cmd == "stack":
        return stack(only)
    if a.cmd == "export":
        return export()
    if a.cmd == "train":
        return train(a)
    if a.cmd == "adapt":
        return adapt(a)
    return evaluate(a)


if __name__ == "__main__":
    sys.exit(main())
