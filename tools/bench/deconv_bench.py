"""The bench of issue #92: the Deconvolution block on many targets, settings tuned on a few and judged on the others.

    python tools/bench/deconv_bench.py tune          the candidate settings on the tuning targets
    python tools/bench/deconv_bench.py judge NAME    one frozen setting on the judging targets, opened once

The targets are the half-stack pairs of the denoising dataset (colour and mono cameras, refractors, Newtonians, smart
telescopes), a central crop of each, in noise units. What is reported, against the stack: the share of stars with a dip
deeper than 5 noise units within 3 to 14 px, by brightness, after Denoise; star width; flux; injected filaments;
noise; the colour of the cores. Results in E:/float_train/bench/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astro_gpu.stages import deconv_lib as lib  # noqa: E402
from astro_gpu.stages import denoise as dn  # noqa: E402

PAIRS = Path("F:/astro_essais/denoise_dataset")
OUT = Path("E:/float_train/bench")
LEVELS = ("_half", "_quarter", "_eighth")
TUNING = ("bench_rosette", "bench_m8", "set_05_dslr_refractor_canon40da_m42", "set_31_smart_seestar_s30_m51")
CROP = 2048
DEV = "cuda"
#: the settings tried on the tuning targets
CANDIDATES = {
    "brut": dict(damping=0.0),
    "amorti3": dict(damping=3.0),
    "amorti5": dict(damping=5.0),
    "amorti3_arret": dict(damping=3.0, stop=0.003),
    "amorti3_global": dict(damping=3.0, global_dark=0.5),
    "amorti3_local": dict(damping=3.0, local_deringing=0.7),
    "amorti3_local_global": dict(damping=3.0, local_deringing=0.7, global_dark=0.5),
    "amorti3_masque": dict(damping=3.0, protect_low=3.0, protect_high=15.0),
    "complet": dict(damping=3.0, stop=0.003, local_deringing=0.7, global_dark=0.5, protect_low=3.0, protect_high=15.0),
    "local9": dict(damping=3.0, local_deringing=0.9),
    "local10": dict(damping=3.0, local_deringing=1.0),
    "local9_arret_masque": dict(damping=3.0, stop=0.003, local_deringing=0.9, protect_low=3.0, protect_high=15.0),
    "local10_arret_masque": dict(damping=3.0, stop=0.003, local_deringing=1.0, protect_low=3.0, protect_high=15.0),
    "fige": dict(),
    "complet_doux": dict(damping=5.0, stop=0.003, local_deringing=0.9, global_dark=0.8, protect_low=5.0, protect_high=25.0, iterations=20),
}
#: a dip counts when deeper than this many noise units of the stack (a half-stack's noise is 1 here)
DIP_NOISE, NOISE_UNIT = 5.0, 2 ** -0.5
YY, XX = np.mgrid[-16:17, -16:17]
RR = np.hypot(YY, XX)


def targets() -> list[str]:
    return [d.name for d in sorted(PAIRS.iterdir()) if (d / "pair.json").exists() and not d.name.endswith(LEVELS)]


def load(name: str):
    d = PAIRS / name
    a, b, nm = (np.load(d / f"{k}.npy", mmap_mode="r") for k in ("A", "B", "nm"))
    H, W = a.shape[1:]
    c = min(CROP, H, W)
    y, x = (H - c) // 2, (W - c) // 2
    g = lambda m: np.array(m[..., y:y + c, x:x + c], np.float32)
    meta = json.loads((d / "pair.json").read_text(encoding="utf-8"))
    return g(a), g(b), g(nm), np.array(meta["bk"], np.float32) / np.array(meta["sg"], np.float32)


def unsharp(x, amount, sigma=1.5):
    return x + amount * (x - cv2.GaussianBlur(x, (0, 0), sigma))


def bright_stars(g, valid, n=400):
    s = cv2.GaussianBlur(g, (0, 0), 1.0)
    ys, xs = np.nonzero((s == cv2.dilate(s, np.ones((9, 9)))) & valid & (s > np.median(s) + 8.0))
    m = 18
    k = (ys > m) & (ys < g.shape[0] - m) & (xs > m) & (xs < g.shape[1] - m)
    ys, xs = ys[k], xs[k]
    o = np.argsort(-s[ys, xs])[:n]
    return ys[o], xs[o]


def stamps(img, ys, xs):
    return np.stack([img[y - 16:y + 17, x - 16:x + 17] for y, x in zip(ys.tolist(), xs.tolist())])


def star_numbers(ref, img, ys, xs, sky_abs):
    """(dip share by brightness third, width ratio, flux median and scatter), green plane."""
    r, o = stamps(ref, ys, xs), stamps(img, ys, xs)
    ring = (RR > 11) & (RR < 16)
    bg_r = np.median(r[:, ring], axis=1)
    bg_o = np.median(o[:, ring], axis=1)
    blurred = np.stack([cv2.GaussianBlur(x, (0, 0), 1.0) for x in o])
    dips = blurred[:, (RR >= 3) & (RR <= 14)].min(1) - bg_r
    bad = dips < -DIP_NOISE * NOISE_UNIT                                   # (the stack's sky holds its pedestal: not a scale)
    thirds = np.array_split(np.arange(len(ys))[::-1], 3)                  # faint, middle, bright
    ap = RR < 10

    def width(s, bg):
        s = np.clip(s - bg[:, None, None], 0, None) * ap
        pk = s.max((1, 2), keepdims=True)
        s = np.where(s > 0.2 * pk, s, 0)
        t = s.sum((1, 2)).clip(1e-9)
        cy, cx = (s * YY).sum((1, 2)) / t, (s * XX).sum((1, 2)) / t
        v = (s * ((YY - cy[:, None, None]) ** 2 + (XX - cx[:, None, None]) ** 2)).sum((1, 2)) / t
        return 2.355 * np.sqrt(v / 2)

    f0, f1 = ((r - bg_r[:, None, None]) * ap).sum((1, 2)), ((o - bg_o[:, None, None]) * ap).sum((1, 2))
    good = f0 > 0
    return {"dip_%": [round(float(bad[t].mean()) * 100, 1) for t in thirds], "width": round(float(np.median(width(o, bg_o)[good] / np.maximum(width(r, bg_r)[good], 1e-3))), 3),
            "flux": round(float(np.median(f1[good] / f0[good])), 3), "flux_90": round(float(np.quantile(np.abs(f1[good] / f0[good] - 1), 0.9)), 3)}


def filaments(shape, sky, seed=2):
    rng = np.random.default_rng(seed)
    H, W = shape
    S = np.zeros((H, W), np.float32)
    out = []
    for _ in range(40):
        y, x = int(rng.integers(150, H - 150)), int(rng.integers(150, W - 150))
        a, amp = float(rng.uniform(0, np.pi)), float(rng.uniform(0.5, 3))
        yy, xx = np.mgrid[-120:121, -120:121]
        along, across = xx * np.cos(a) + yy * np.sin(a), -xx * np.sin(a) + yy * np.cos(a)
        t = (np.exp(-across ** 2 / (2 * 1.7 ** 2)) * np.exp(-np.maximum(np.abs(along) - 100, 0) ** 2 / 50.0)).astype(np.float32)
        if sky[y - 120:y + 121, x - 120:x + 121].mean() > 0.9:
            S[y - 120:y + 121, x - 120:x + 121] += amp * t
            out.append((y, x, t))
    return S, out


def evaluate(name: str, settings: dict, den, with_filaments: bool) -> dict:
    a, b, nm, sky_abs = load(name)
    full = 0.5 * (a + b)
    valid = nm > 0
    sat = np.zeros(valid.shape, bool)
    for c in range(3):
        sat |= full[c] > 0.7 * np.percentile(full[c][valid][::11], 99.999)
    st = lib.StackDeconvSettings(log=lambda *x, **k: None, **{k: v for k, v in settings.items()})
    t0 = time.time()
    of, oa, ob, info = lib.run(full, a, b, sat, valid, st, DEV)
    secs = time.time() - t0
    if not info.get("applied"):
        return {"name": name, "skipped": "too few stars"}
    bk, sg = np.zeros(3, np.float32), np.ones(3, np.float32)
    nm2 = (nm / np.sqrt(2)).astype(np.float32)
    denoised = dn.apply_net(den, of, nm2, bk, sg, torch.device(DEV))
    base_den = dn.apply_net(den, full, nm2, bk, sg, torch.device(DEV))
    ys, xs = bright_stars(full[1], valid)                                 # saturated stars included
    res = {"name": name, "s": round(secs, 1), "stars": len(ys)}
    res["sans_deconv"] = star_numbers(full[1], base_den[1], ys, xs, float(sky_abs[1]))
    res["avant_denoise"] = star_numbers(full[1], of[1], ys, xs, float(sky_abs[1]))
    res["apres_denoise"] = star_numbers(full[1], denoised[1], ys, xs, float(sky_abs[1]))
    n0 = 1.4826 * float(np.median(np.abs(0.5 * (a[1] - b[1])[valid])))
    n1 = 1.4826 * float(np.median(np.abs(0.5 * (oa[1] - ob[1])[valid])))
    res["bruit"] = round(n1 / max(n0, 1e-9), 3)
    core = stamps(of[0], ys[:100], xs[:100])[:, RR < 2].mean(1) / np.maximum(stamps(of[1], ys[:100], xs[:100])[:, RR < 2].mean(1), 1e-6)
    core0 = stamps(full[0], ys[:100], xs[:100])[:, RR < 2].mean(1) / np.maximum(stamps(full[1], ys[:100], xs[:100])[:, RR < 2].mean(1), 1e-6)
    res["couleur_coeurs"] = round(float(np.median(np.abs(core / np.maximum(core0, 1e-6) - 1))), 3)
    if with_filaments:
        sky = (np.abs(full[1] - cv2.GaussianBlur(full[1], (0, 0), 12)) < 3).astype(np.float32)
        S, fil = filaments(full.shape[1:], sky)
        i_f, *_ = lib.run(full + S[None], a + S[None], b + S[None], sat, valid, st, DEV)
        resp = (i_f - of)[1]
        noise = (oa - ob)[1] / 2
        rng = np.random.default_rng(7)
        snr = []
        for y, x, t in fil:
            vals = [float((noise[v - 120:v + 121, u - 120:u + 121] * t).sum()) for v, u in zip(rng.integers(130, full.shape[1] - 130, 30), rng.integers(130, full.shape[2] - 130, 30))]
            snr.append(float((resp[y - 120:y + 121, x - 120:x + 121] * t).sum()) / max(float(np.std(vals)), 1e-9))
        res["filaments"] = round(float(np.median(snr)), 2) if snr else None

        def matched(r, n):
            out = []
            for y, x, t in fil:
                vals = [float((n[v - 120:v + 121, u - 120:u + 121] * t).sum()) for v, u in zip(rng.integers(130, full.shape[1] - 130, 30), rng.integers(130, full.shape[2] - 130, 30))]
                out.append(float((r[y - 120:y + 121, x - 120:x + 121] * t).sum()) / max(float(np.std(vals)), 1e-9))
            return round(float(np.median(out)), 2) if out else None

        res["filaments_pile"] = matched(S, (a - b)[1] / 2)
        target = res["avant_denoise"]["width"]
        lo, hi = 0.0, 6.0                                                     # the unsharp mask of the same star width
        for _ in range(14):
            amount = (lo + hi) / 2
            w = star_numbers(full[1], unsharp(full[1], amount), ys, xs, float(sky_abs[1]))["width"]
            lo, hi = (amount, hi) if w > target else (lo, amount)
        res["masque_flou_montant"] = round(amount, 3)
        res["filaments_masque_flou"] = matched(unsharp(S, amount), unsharp((a - b)[1] / 2, amount))
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    den = dn.pretrained(DEV)
    den.eval()
    if sys.argv[1] == "tune":
        for label, st in CANDIDATES.items():
            if len(sys.argv) > 2 and label not in sys.argv[2:]:
                continue
            rows = []
            for name in TUNING:
                try:
                    rows.append(evaluate(name, st, den, False))
                except Exception as e:  # noqa: BLE001 - one target must not stop the bench
                    rows.append({"name": name, "error": repr(e)})
                print(label, json.dumps(rows[-1]), flush=True)
            (OUT / f"tune_{label}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    else:
        label = sys.argv[2]
        st = CANDIDATES[label]
        rows = []
        for name in targets():
            if name in TUNING:
                continue
            try:
                rows.append(evaluate(name, st, den, True))
            except Exception as e:  # noqa: BLE001
                rows.append({"name": name, "error": repr(e)})
            print(json.dumps(rows[-1]), flush=True)
        (OUT / f"judge_{label}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
