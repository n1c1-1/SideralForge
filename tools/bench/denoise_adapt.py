"""Bench: how the denoiser's per-image adaptation should work, against the collapse of bright star cores.

    python tools/bench/denoise_adapt.py run --case m51_rp --tag base --seeds 0,1,2 [--set steps=250 ...] [--mod-b blur=0.8,gain=0.85]
    python tools/bench/denoise_adapt.py curve --case m51_rp --tag adapted [--load net.pt]   synthetic stars through a network
    python tools/bench/denoise_adapt.py table --cases all --cfgs pretrained,adapted,gains,l2sp   the candidates' full measures
    python tools/bench/denoise_adapt.py sheets --dest FOLDER                                the report's crops

Report-only: the product keeps ``astro_gpu.stages.denoise.adapt``; ``adapt_variant`` is a parametrised copy of it with
one switch per suspect (sampling, loss weight, steps, learning rate, weights tuned), so that one thing changes at a time.
Everything is written under F:/astro_essais/adaptation. The held-out stacks come from tools/bench/denoise_mono.py.

Outcome (8 November 2026): the collapse of the star cores comes from half-stacks whose stars disagree (M51: frames of
different nights unevenly dealt between the halves, peak ratio A/B 0.62 to 0.75), which breaks the condition of
Noise2Noise that both images hold the same signal. Dealing M51's red frames one by one (ratio 0.96) removes it with the
same method; the same disagreement put into stacks that do not have it creates it. Tile sampling, the loss weight, the
number of steps and the learning rate do not remove it; tuning only the channel gains (``params="gains"``, Mohan et al.
2021) does, on every stack, at the cost of keeping more of the invented dots.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from denoise_mono import held_out_cases, load_case, manifest, stack_filter  # noqa: E402
from mono_models import DEV, mad, measure_net, phot, stars_of  # noqa: E402

from astro_gpu.stages import denoise as dn  # noqa: E402

HOME = Path("F:/astro_essais/adaptation")
PAIRS = Path("F:/astro_essais/mono_affinage/pairs")
RESTACK = HOME / "restack" / "m51_rgb"
#: the colour stacks of the bench: (project, the stack's frame count, for the log)
COLOUR = {"iris": Path("F:/astro_essais/bench/iris/ours_express"), "m8": Path("F:/astro_essais/bench/m8/ours_express")}
PATCH, BATCH = dn.PATCH, dn.BATCH


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


@dataclass
class Cfg:
    """One way of adapting; the default is the shipped Adapted method."""
    steps: int = 500
    lr: float = 2e-4
    sched: str = "onecycle"  # onecycle | drop (lr, then lr/10 after 20 % of the steps) | none
    opt: str = "adamw"       # adamw | adam
    wd: float = 1e-4
    params: str = "all"      # all | gains | decoder | last
    weight: str = "current"  # current (1/(nm^2 + signal/40)) | noise (1/nm^2) | none
    loss: str = "l2"         # l2 | l1
    sampling: str = "uniform"  # uniform | bright (half the tiles centred on bright pixels)
    l2sp: float = 0.0        # the pull towards the pretrained weights (Li et al. 2018), instead of towards zero
    clip: float = 1.0        # gradient clipping (0: none)
    seed: int = 0
    holdout: float = 0.0     # this right-hand share of the image is never trained on, for a measure without ground truth


def attach_gains(net) -> list[torch.Tensor]:
    """Gain tuning (Mohan et al., NeurIPS 2021): one multiplicative gain per channel of every convolution but the last
    one. A hook multiplies each channel's output, which is the paper's scaling of the channel's weights and bias."""
    gains = []
    for name, m in net.named_modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)) and name != "out":
            g = torch.nn.Parameter(torch.ones(m.out_channels, device=DEV))
            gains.append(g)
            m.register_forward_hook(lambda _m, _i, o, g=g: o * g.view(1, -1, 1, 1))
    return gains


def trainable(net, cfg: Cfg) -> list[torch.nn.Parameter]:
    if cfg.params == "gains":
        for p in net.parameters():
            p.requires_grad_(False)
        return attach_gains(net)
    if cfg.params == "decoder":
        for p in list(net.inc.parameters()) + list(net.downs.parameters()) + list(net.mid.parameters()):
            p.requires_grad_(False)
    elif cfg.params == "last":
        for name, p in net.named_parameters():
            p.requires_grad_(name.startswith("out.") or name.startswith("decs.2."))
    return [p for p in net.parameters() if p.requires_grad]


def bright_centres(A: np.ndarray, bk, sg) -> tuple[np.ndarray, np.ndarray]:
    """Pixels of the half-stack far above the sky (all channels' mean, 1 px smoothing): the bright structures, rare."""
    a = ((A - bk[:, None, None]) / sg[:, None, None]).mean(0)
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.0)
    y, x = np.nonzero(a > 30.0)
    return y, x


def adapt_variant(net, A, B, nm, bk, sg, cfg: Cfg, probes=None, valid_tiles=None):
    """A copy of denoise.adapt with one switch per suspect. ``probes``: (y, x) pixels; how many tiles covered each one
    is counted. Returns the history (the running loss and, every 50 steps, the held-out loss when valid_tiles is set)."""
    tA, tB = torch.from_numpy(A).to(DEV), torch.from_numpy(B).to(DEV)
    tN = torch.from_numpy(nm).to(DEV)[None]
    bk_t, sg_t = torch.from_numpy(bk).to(DEV), torch.from_numpy(sg).to(DEV)
    Hh, Ww = tA.shape[1:]
    valid = (tN[0] > 0) & (tN[0] < 1.6)
    torch.manual_seed(cfg.seed)
    params = trainable(net, cfg)
    ref = [p.detach().clone() for p in params] if cfg.l2sp else None
    if cfg.opt == "adamw":
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    else:
        opt = torch.optim.Adam(params, lr=cfg.lr)
    if cfg.sched == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.05)
    elif cfg.sched == "drop":
        drop = max(1, cfg.steps // 5)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1.0 if i < drop else 0.1)
    else:
        sched = None
    bright_y, bright_x = bright_centres(A, bk, sg) if cfg.sampling == "bright" else (None, None)
    net.train()
    seen = np.zeros(len(probes)) if probes is not None else None
    hist = []
    run = 0.0
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    for it in range(cfg.steps):
        ys = torch.randint(0, Hh - PATCH, (BATCH * 2,), device="cpu").tolist()
        xs = torch.randint(0, int(Ww * (1 - cfg.holdout)) - PATCH, (BATCH * 2,), device="cpu").tolist()
        if bright_y is not None and len(bright_y):                 # half the tiles around a random bright pixel
            for i in range(0, BATCH * 2, 2):
                j = int(rng.integers(len(bright_y)))
                ys[i] = int(np.clip(bright_y[j] - rng.integers(0, PATCH), 0, Hh - PATCH - 1))
                xs[i] = int(np.clip(bright_x[j] - rng.integers(0, PATCH), 0, Ww - PATCH - 1))
        xin, tgt, nmp = [], [], []
        for y, x in zip(ys, xs):
            if not bool(valid[y + PATCH // 2, x + PATCH // 2]):
                continue
            src, dst = (tA, tB) if torch.rand(1).item() < 0.5 else (tB, tA)
            p_in, p_t, p_n = (src[:, y:y + PATCH, x:x + PATCH], dst[:, y:y + PATCH, x:x + PATCH],
                              tN[:, y:y + PATCH, x:x + PATCH])
            if seen is not None:
                for q, (py, px) in enumerate(probes):
                    if y <= py < y + PATCH and x <= px < x + PATCH:
                        seen[q] += 1
            k = int(torch.randint(0, 4, (1,)))
            p_in, p_t, p_n = (torch.rot90(t, k, (1, 2)) for t in (p_in, p_t, p_n))
            xin.append(p_in)
            tgt.append(p_t)
            nmp.append(p_n)
            if len(xin) == BATCH:
                break
        if not xin:
            continue
        xin, tgt, nmp = torch.stack(xin), torch.stack(tgt), torch.stack(nmp)
        with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            out = net(dn.to_input(xin, nmp, bk_t, sg_t)).float()
        a_in = (xin - bk_t.view(1, 3, 1, 1)) / sg_t.view(1, 3, 1, 1)
        b_t = (tgt - bk_t.view(1, 3, 1, 1)) / sg_t.view(1, 3, 1, 1)
        with torch.no_grad():
            if cfg.weight == "current":
                sm = F.avg_pool2d(a_in, 5, 1, 2).clamp(min=0)
                wgt = 1.0 / ((nmp ** 2).clamp(min=0.25) + sm / 40.0)
            elif cfg.weight == "noise":
                wgt = 1.0 / (nmp ** 2).clamp(min=0.25)
            else:
                wgt = torch.ones_like(a_in)
        err = a_in + out - b_t
        loss = (wgt * (err ** 2 if cfg.loss == "l2" else err.abs())).mean()
        if ref is not None:
            loss = loss + cfg.l2sp * sum(((p - r) ** 2).sum() for p, r in zip(params, ref))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.clip:
            torch.nn.utils.clip_grad_norm_(params, cfg.clip)
        opt.step()
        if sched is not None:
            sched.step()
        run = 0.98 * run + 0.02 * loss.item() if it else loss.item()
        if (it + 1) % 50 == 0:
            hist.append({"step": it + 1, "loss": run})
    net.eval()
    return {"seconds": time.time() - t0, "loss": run, "seen": None if seen is None else seen.tolist(), "hist": hist}


# ---------------------------------------------------------------------------------------------- cases
def colour_cases() -> list[dict]:
    cases = []
    for name, root in COLOUR.items():
        store = root / "store"
        for d in sorted((store / "denoise").iterdir()):
            m = manifest(d) if not d.name.endswith(".tmp") else None
            if m is None or m.get("status") != "done" or (m["key_parts"].get("p") or {}).get("method") == "adapted":
                continue
            u = m["key_parts"]["u"]
            sd = store / "stack" / u["stack"]
            cases.append({"name": name, "root": root, "stack": sd, "gradient": store / "gradient" / u["gradient"],
                          "frames": (manifest(sd).get("metrics") or {}).get("frames_used"), "colour": True})
            break
    return cases


def get_cases(which: str) -> list[dict]:
    if which.startswith("pair:"):
        return [{"name": n, "pair": PAIRS / n, "colour": False, "frames": None} for n in which[5:].split(",")]
    if which == "balanced":                                        # M51 stacked again, the frames dealt one by one
        store = RESTACK / "store"
        out = []
        for d in sorted((store / "stack").iterdir()):
            m = manifest(d) if not d.name.endswith(".tmp") else None
            if m and m.get("status") == "done" and (m.get("params") or {}).get("half_block") == 1:
                out.append({"name": f"m51_{stack_filter(store, d)}_bal", "root": RESTACK, "stack": d, "gradient": None,
                            "frames": (m.get("metrics") or {}).get("frames_used"), "colour": False})
        return out
    mono = held_out_cases()
    for c in mono:
        c["colour"] = False
    if which == "mono":
        return mono
    if which == "colour":
        return colour_cases()
    if which == "all":
        return mono + colour_cases()
    wanted = which.split(",")
    return [c for c in mono + colour_cases() if c["name"] in wanted]


def load(c: dict) -> dict:
    """The images of a case: a held-out or colour stack (denoise_mono.load_case), or a pair of the fine-tuning's bank
    (never a held-out stack), turned back to the stack's units."""
    if "pair" not in c:
        return load_case(c)
    meta = json.loads((c["pair"] / "pair.json").read_text())
    bk, sg = np.array(meta["bk"], np.float32), np.array(meta["sg"], np.float32)
    A, B = (np.load(c["pair"] / f"{k}.npy").astype(np.float32) * sg[:, None, None] + bk[:, None, None] for k in "AB")
    nm = np.load(c["pair"] / "nm.npy").astype(np.float32)
    return {"A": A, "B": B, "full": 0.5 * (A + B), "wt": np.repeat((nm > 0).astype(np.float32)[None], 3, axis=0), "bk": bk,
            "sg": sg, "nm": nm}


def modify(x: np.ndarray, bk: np.ndarray, spec: str) -> np.ndarray:
    """A half-stack changed for a causal test: ``blur=<sigma px>,gain=<factor>``, applied to the signal above the sky."""
    o = dict(kv.split("=") for kv in spec.split(","))
    blur, gain = float(o.get("blur", 0)), float(o.get("gain", 1))
    out = x.copy()
    for c in range(3):
        d = x[c] - bk[c]
        if blur:
            d = cv2.GaussianBlur(d, (0, 0), blur)
        out[c] = bk[c] + gain * d
    return out


def finish(c: dict, o: np.ndarray, full: np.ndarray, wt: np.ndarray) -> np.ndarray:
    """The Denoise block's last touches: the grey mean of a monochrome stack, the sky level."""
    if not c.get("colour"):
        o = np.repeat(o.mean(0, keepdims=True), 3, axis=0)
    return dn.keep_level(o, full, wt)


def denoise_with(net, c: dict, im: dict) -> np.ndarray:
    return finish(c, dn.apply_net(net, im["full"], (im["nm"] / np.sqrt(2)).astype(np.float32), im["bk"], im["sg"], DEV),
                  im["full"], im["wt"])


def quick(c: dict, im: dict, dF: np.ndarray) -> dict:
    """The fast measures, on the green channel: noise, stars (the brightest 300 isolated and the faint), the bright
    pixels damaged, and the star by star peak ratio."""
    g = im["full"][1]
    bkg, sig = mad(g)
    d = dF[1]
    r = {"noise_reduction": round(dn._noise(g) / max(dn._noise(d), 1e-12), 2)}
    high = (g - bkg) > 100 * sig
    # (edges excluded, as for the stars)
    m = np.zeros_like(g, bool)
    m[24:-24, 24:-24] = True
    damaged = high & ((d - bkg) < 0.5 * (g - bkg)) & m
    r["damaged"] = int(damaged.sum())
    r["high_pixels"] = int((high & m).sum())
    for name, lo, hi in (("bright", 50, 1e9), ("faint", 5, 15)):
        y, x = stars_of(g, sig, bkg, lo, hi, n=300)
        a, b = phot(g, y, x), phot(d, y, x)
        ok = a[:, 0] > 0
        r[name] = {"n": int(len(y)), "flux": round(float(np.median(b[ok, 0] / a[ok, 0])), 3) if ok.any() else None,
                   "peak": round(float(np.median(b[a[:, 1] > 0, 1] / a[a[:, 1] > 0, 1])), 3) if len(y) else None,
                   "width": round(float(np.median(b[ok, 2] / a[ok, 2])), 3) if ok.any() else None}
    # worst and per-brightness: the peak kept by each bright star
    y, x = stars_of(g, sig, bkg, 50, 1e9, n=300)
    if len(y):
        a, b = phot(g, y, x), phot(d, y, x)
        pa, pb = phot(im["A"][1], y, x), phot(im["B"][1], y, x)                  # the two halves against each other
        r["halves_peak_A_B"] = round(float(np.median(pa[:, 1] / pb[:, 1])), 3)
        r["halves_flux_A_B"] = round(float(np.median(pa[:, 0] / pb[:, 0])), 3)
        ratio = b[:, 1] / np.maximum(a[:, 1], 1e-9)
        h = a[:, 1] / sig
        r["worst_peak"] = round(float(ratio.min()), 3)
        r["by_height"] = [(lo, int(((h >= lo) & (h < hi)).sum()), round(float(np.median(ratio[(h >= lo) & (h < hi)])), 3)
                           if ((h >= lo) & (h < hi)).any() else None)
                          for lo, hi in ((50, 200), (200, 1000), (1000, 3000), (3000, 1e9))]
    return r


# ---------------------------------------------------------------------------------------------- commands
def run_one(c, im, cfg: Cfg, probes=None) -> tuple[dict, object]:
    net = dn.pretrained(DEV)
    h = adapt_variant(net, im["A"], im["B"], im["nm"], im["bk"], im["sg"], cfg, probes=probes)
    return h, net


def cross_error(net, im: dict, frac: float, margin: int = 192) -> dict:
    """The Noise2Noise error on the part of the image never trained on (the right-hand ``frac``, less a margin): the
    denoised half A against the raw half B and the other way round, in noise units. The floor is B's own noise, so a
    lower value means less noise left and nothing of the signal lost; no ground truth needed (the half-map check of
    cryo-EM, here on a strip of the field). 'sky': pixels within 3 noise sigmas of the sky; 'all': every covered pixel."""
    W = im["A"].shape[2]
    x0 = int(W * (1 - frac)) + margin
    nm = im["nm"]
    dA = dn.apply_net(net, im["A"], nm, im["bk"], im["sg"], DEV)
    dB = dn.apply_net(net, im["B"], nm, im["bk"], im["sg"], DEV)
    mono = dn.is_grey(im["full"])
    if mono:
        dA, dB = (np.repeat(d.mean(0, keepdims=True), 3, axis=0) for d in (dA, dB))
    sg = im["sg"][1]
    bk = im["bk"][1]
    covered = (nm > 0)[:, x0:]
    a, b = (im[k][1][:, x0:] for k in "AB")
    sky = covered & (np.abs(0.5 * (a + b) - bk) < 3 * sg)
    def rms(p, q, m):
        return round(float(np.sqrt(np.mean(((p[m] - q[m]) / sg) ** 2))), 4)
    return {"sky": round(0.5 * (rms(dA[1][:, x0:], b, sky) + rms(dB[1][:, x0:], a, sky)), 4),
            "all": round(0.5 * (rms(dA[1][:, x0:], b, covered) + rms(dB[1][:, x0:], a, covered)), 4),
            "raw_sky": round(rms(a, b, sky), 4), "raw_all": round(rms(a, b, covered), 4)}


def pretrained_quick(c, im):
    net = dn.pretrained(DEV).eval()
    return quick(c, im, denoise_with(net, c, im))


def parse_set(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        k, v = p.split("=")
        try:
            v = float(v) if "." in v or "e" in v else int(v)
        except ValueError:
            pass
        out[k] = v
    return out


def cmd_run(a) -> int:
    HOME.mkdir(exist_ok=True, parents=True)
    res_f = HOME / a.out
    cases = get_cases(a.case)
    for c in cases:
        im = load(c)
        log(f"{c['name']}: {im['full'].shape}")
        if a.mod_a:
            im["A"] = modify(im["A"], im["bk"], a.mod_a)
        if a.mod_b:
            im["B"] = modify(im["B"], im["bk"], a.mod_b)
        over = parse_set(a.set)
        if a.baseline:
            r = pretrained_quick(c, im)
            if over.get("holdout"):
                r["cross"] = cross_error(dn.pretrained(DEV).eval(), im, over["holdout"])
            log(f"{c['name']} pretrained: {json.dumps(r)}")
            with res_f.open("a") as fh:
                fh.write(json.dumps({"case": c["name"], "cfg": "pretrained", "res": r}) + "\n")
        for seed in [int(s) for s in a.seeds.split(",")]:
            cfg = dataclasses.replace(Cfg(), **parse_set(a.set), seed=seed)
            t0 = time.time()
            h, net = run_one(c, im, cfg)
            r = quick(c, im, denoise_with(net, c, im))
            if cfg.holdout:
                r["cross"] = cross_error(net, im, cfg.holdout)
            log(f"{c['name']} {a.tag} seed {seed}: {h['seconds']:.0f}s adapt, loss {h['loss']:.3f}: {json.dumps(r)}")
            with res_f.open("a") as fh:
                fh.write(json.dumps({"case": c["name"], "cfg": a.tag, "over": parse_set(a.set), "seed": seed,
                                     "adapt_s": round(h["seconds"], 1), "res": r}) + "\n")
            if a.keep:
                torch.save({"net": net.state_dict()}, HOME / f"net_{c['name']}_{a.tag}_{seed}.pt")
            del net
            torch.cuda.empty_cache()
            log(f"   total {time.time() - t0:.0f}s")
    return 0


#: the candidates of the final comparison. Their settings are published or chosen on stacks outside the held-out ones
#: (the pairs of tools/bench/denoise_mono.py) before the comparison is run: see the report
CANDIDATES = {
    "pretrained": None,
    "adapted": {},                                                   # the shipped method
    # gains only (Mohan et al. 2021): the learning rate is the highest of 2e-4, 1e-3 that kept the bright pixels on the
    # stacks outside the held-out set whose halves disagree (M42 kb27: 1e-3 damages 32 and 84 of them, 2e-4 keeps 2 and 11)
    "gains": {"params": "gains", "opt": "adam", "lr": 2e-4},
    # all the weights, pulled towards the pretrained ones (L2-SP, Li et al. 2018): the smallest of 0.01, 0.1, 1, 10 that
    # kept the bright pixels on the same stacks (10; 1 damages 40 and 192 of them)
    "l2sp": {"l2sp": 10.0, "wd": 0},
}


def crops_of(c: dict, im: dict, imgs: dict[str, np.ndarray], out: Path) -> None:
    """Crops of the green channel for the report: a bright unsaturated star's core, the faint sky: {cfg: (star, sky)}."""
    from denoise_mono import faint_crop
    g = im["full"][1]
    bkg, sig = mad(g)
    y, x = stars_of(g, sig, bkg, 50, 1e9, n=300)
    h = phot(g, y, x)[:, 1] / sig
    i = int(np.argmin(np.abs(np.log(np.maximum(h, 1)) - np.log(2000.0))))           # the star nearest 2000 noise units
    cy, cx = int(y[i]), int(x[i])
    n = 24
    size = 200
    fy, fx = faint_crop(g, sig, size)
    np.savez_compressed(out, star=np.stack([imgs[k][cy - n:cy + n, cx - n:cx + n] for k in imgs]),
                        sky=np.stack([imgs[k][fy:fy + size, fx:fx + size] for k in imgs]), names=np.array(list(imgs)),
                        star_xy=np.array([cy, cx]), height=float(h[i]), bkg=bkg, sig=sig)


def cmd_table(a) -> int:
    """The full measures of the candidates on the cases: table.jsonl, and the crops of the report."""
    out = HOME / "table"
    out.mkdir(exist_ok=True, parents=True)
    res_f = HOME / a.out
    done = set()
    if res_f.exists():
        for ln in res_f.read_text().splitlines():
            r = json.loads(ln)
            done.add((r["case"], r["cfg"], r.get("seed", 0)))
    names = a.cfgs.split(",")
    cases = get_cases(a.cases)
    for c in cases:
        if all((c["name"], n, s) in done for n in names for s in [int(x) for x in a.seeds.split(",")]):
            continue
        im = load(c)
        g = im["full"][1]
        imgs = {"pile": g}
        for name in names:
            spec = CANDIDATES[name]
            seeds = [0] if spec is None else [int(x) for x in a.seeds.split(",")]
            for seed in seeds:
                if (c["name"], name, seed) in done:
                    continue
                net = dn.pretrained(DEV)
                t0 = time.time()
                h = None
                if spec is not None:
                    h = adapt_variant(net, im["A"], im["B"], im["nm"], im["bk"], im["sg"],
                                      dataclasses.replace(Cfg(), **spec, seed=seed))
                dF = denoise_with(net.eval(), c, im)
                t1 = time.time()
                r = quick(c, im, dF)
                m = measure_net(net, im["full"], dF, im["A"], im["B"], im["nm"], im["bk"], im["sg"])
                m.pop("channel_spread", None)
                row = {"case": c["name"], "cfg": name, "seed": seed, "frames": c.get("frames"),
                       "adapt_s": None if h is None else round(h["seconds"], 1), "denoise_s": round(t1 - t0 - (h["seconds"] if h else 0), 1),
                       "quick": r, "full": m}
                with res_f.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                log(f"{c['name']} {name} seed {seed}: adapt {row['adapt_s']} s, noise /{r['noise_reduction']}, damaged {r['damaged']}, "
                    f"bright {r['bright']}, faint {r['faint']}, dots sky {m['half_consistency']['faint_sky']['denoised']}")
                if seed == seeds[0]:
                    imgs[name] = dF[1]
                del net
                torch.cuda.empty_cache()
        crops_of(c, im, imgs, out / f"{c['name']}.npz")
        del im, imgs
    return 0


LABELS = {"pile": "Pile", "pretrained": "Tel quel", "adapted": "Adapte", "gains": "Gains", "l2sp": "Poids retenus"}


def cmd_sheets(a) -> int:
    """The report's crops: for each case's crops (table command), a sheet with the star core (top) and the faint sky."""
    from mono_models import stretch
    out = Path(a.dest)
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted((HOME / "table").glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        names = [str(n) for n in z["names"]]
        star, sky = z["star"], z["sky"]
        bkg, sig = float(z["bkg"]), float(z["sig"])
        top = max(float(star[0].max()), bkg + 20 * sig)
        f3 = 4
        cells_star = [cv2.resize(stretch(s, bkg, top), None, fx=f3, fy=f3, interpolation=cv2.INTER_NEAREST) for s in star]
        cells_sky = [stretch(s, bkg, bkg + 12 * sig) for s in sky]
        size = cells_star[0].shape[0]
        cells_sky = [cv2.resize(c, (size, size), interpolation=cv2.INTER_NEAREST) for c in cells_sky]
        rows = [np.hstack([np.pad(c, 3, constant_values=0.3) for c in row]) for row in (cells_star, cells_sky)]
        a_img = (np.vstack(rows) * 255).astype(np.uint8)
        a_img = cv2.copyMakeBorder(a_img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=40)
        for i, n in enumerate(names):
            cv2.putText(a_img, LABELS.get(n, n), (i * (size + 6) + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 230, 2)
        cv2.putText(a_img, f"{f.stem}: etoile de hauteur {float(z['height']):.0f} bruits", (8, a_img.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)
        cv2.imwrite(str(out / f"{f.stem}.jpg"), a_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        log(f"sheet {f.stem}")
    return 0


def cmd_curve(a) -> int:
    """The output of a network on synthetic Gaussian stars on a noisy flat sky, for several heights and widths: what is
    left of the peak (in noise units of the half-stack). --load: a network saved by the bench; the pretrained otherwise."""
    c = get_cases(a.case)[0]
    im = load(c)
    bk, sg = im["bk"], im["sg"]
    net = dn.pretrained(DEV)
    if a.load:
        net.load_state_dict(torch.load(a.load, map_location=DEV, weights_only=False)["net"])
    net.eval()
    rng = np.random.default_rng(0)
    S, step = 512, 48
    heights = [10, 30, 100, 300, 1000, 3000, 10000]
    res = {}
    for sigma in (1.2, 1.5, 1.9):
        yy, xx = np.mgrid[:S, :S].astype(np.float32)
        stars = np.zeros((S, S), np.float32)
        cents = []
        for i, h in enumerate(heights):
            for j in range(2):
                cy, cx = 32 + i * step, 32 + j * 200
                stars += h * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
                cents.append((cy, cx, h))
        sky = rng.normal(0, 1, (S, S)).astype(np.float32)
        a_in = np.repeat((sky + stars)[None], 3, axis=0)
        x = a_in * sg[:, None, None] + bk[:, None, None]
        nm = np.ones((S, S), np.float32)
        o = dn.apply_net(net, x, nm, bk, sg, DEV)
        out = (o.mean(0) - bk.mean()) / sg.mean()
        by = {}
        for h in heights:
            v = [(float(cv2.GaussianBlur(out, (0, 0), 1.0)[cy, cx]) / float(cv2.GaussianBlur(a_in[1], (0, 0), 1.0)[cy, cx]))
                 for cy, cx, hh in cents if hh == h]
            by[h] = round(float(np.mean(v)), 3)
        res[sigma] = by
        log(f"sigma {sigma}: peak kept {by}")
    with (HOME / "curve.jsonl").open("a") as fh:
        fh.write(json.dumps({"case": c["name"], "load": a.load, "tag": a.tag, "res": res}) + "\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("run", "curve", "table", "sheets"))
    ap.add_argument("--dest", default=str(HOME / "sheets"))
    ap.add_argument("--cfgs", default="pretrained,adapted")
    ap.add_argument("--load", default="", help="a network file for the curve command")
    ap.add_argument("--case", default="m51_rp")
    ap.add_argument("--cases", default="mono")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default="diag.jsonl")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--mod-a", default="", help="change half-stack A before adapting: blur=<sigma px>,gain=<factor>")
    ap.add_argument("--mod-b", default="", help="the same for half-stack B")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "table":
        return cmd_table(a)
    if a.cmd == "curve":
        return cmd_curve(a)
    if a.cmd == "sheets":
        return cmd_sheets(a)
    return cmd_run(a) if a.cmd == "run" else 2


if __name__ == "__main__":
    sys.exit(main())
