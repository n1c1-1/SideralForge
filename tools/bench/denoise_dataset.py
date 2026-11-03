"""The training set of the pretrained denoiser (docs/chantiers/debruitage.md): the colour sets of F:/dataset_astro
stacked by our own chain, their half-stack pairs exported, and the pretraining.

    python tools/bench/denoise_dataset.py stack  [--only 03,04,...]     stack every set (inventory -> stack)
    python tools/bench/denoise_dataset.py export                         the pairs (A, B, noise map) of every stack
    python tools/bench/denoise_dataset.py train  [--hours 4]             pretrain the U-Net on all the pairs
    python tools/bench/denoise_dataset.py night  [--hours 10]           all of it in a row
    python tools/bench/denoise_dataset.py report                         what the stacking met, per set (markdown)

``stack``: one project per set in F:/astro_essais/dataset_projets/<id> (its own engine state in _state), computed up
to the stack only; a set that fails is noted (stack.json) and the next one goes on (a registration that finds too
few stars is tried again with a lower detection threshold). Every set is stacked whole first, then from a half, a
quarter and an eighth of its lights (LEVELS): the same field at several signal-to-noise ratios, so that the network
learns the noise, not the fields. ``export``: for every stacked set,
for the user's projects (read only: their latest stack) and for the six bench cases (held out, for the validation),
drz_A, drz_B and the noise map, normalised as the denoiser normalises them (stages/denoise.norm_params), in float16,
in F:/astro_essais/denoise_dataset/<name>/. ``train``: the U-Net of stages/denoise.py, Noise2Noise on random tiles of
every training pair (the same input, loss and weights as the per-image training), validated on the bench pairs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

DATA = Path("F:/dataset_astro/data")
CATALOG = Path("F:/dataset_astro/catalog.csv")
PROJECTS = Path("F:/astro_essais/dataset_projets")
OUT = Path("F:/astro_essais/denoise_dataset")
USER_PROJECTS = Path("F:/astro_projets")
BENCH = Path("F:/astro_essais/bench")
BENCH_CASES = ("heart", "iris", "m51", "m8", "rosette", "seestar")
#: the colour sets our chain can stack (Bayer, not a fixed-foreground nightscape, a comet or a mosaic, 14 lights or
#: more); the monochrome sets and the X-Trans one are not (the chain refuses them)
#: the signal-to-noise levels: every set is stacked whole, then again from a half, a quarter and an eighth of its
#: lights (evenly spread over the night), each level only when it keeps MIN_LEVEL_LIGHTS lights or more (5 per
#: half-stack)
LEVELS = {"1/2": 2, "1/4": 4, "1/8": 8}
MIN_LEVEL_LIGHTS = 10
#: the star detection threshold of a second try when the registration finds too few stars (short or faint frames)
FAINT_SIGMA = 5.0
SETS = ("02", "03", "04", "05", "08", "14", "15", "16", "18", "22", "24", "25", "27", "29", "30", "31", "32", "33",
        "34", "36", "37", "38", "44", "47", "49", "52", "53", "55")


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def set_dirs(only: list[str] | None = None) -> list[Path]:
    """The set folders, smallest first (the training data come early)."""
    sizes = {r["id"]: float(r["size_gb"] or 0) for r in csv.DictReader(open(CATALOG, encoding="utf-8-sig"),
                                                                        delimiter=";")}
    dirs = [d for d in DATA.iterdir() if d.is_dir() and d.name[:2] in (only or SETS)]
    return sorted(dirs, key=lambda d: sizes.get(d.name, 1e9))


# ---------------------------------------------------------------------------------------------- stack
def stack(only: list[str] | None, device: str) -> int:
    os.environ["ASTRO_GPU_HOME"] = str(PROJECTS / "_state")
    import make_test_project as mtp

    from astro_gpu import api
    PROJECTS.mkdir(parents=True, exist_ok=True)
    report_f = PROJECTS / "stack.json"
    report = json.loads(report_f.read_text()) if report_f.exists() else {}
    def run(p, wf: int) -> tuple[bool, str]:
        cur = api.running(p)
        if cur is not None:                              # a run left by an earlier night: its end first
            log(f"{p.name}: a run is in progress, followed to its end")
            mtp.follow(api, p, cur)
            api.refresh(p)
        for attempt in (1, 2):
            plan = api.plan_to_block(p, wf, "stacking", device)
            if plan.jobs:
                log(f"{p.name}: {len(plan.jobs)} jobs: {', '.join(j.stage for j in plan.jobs)}")
                mtp.follow(api, p, api.start_run(p, plan, device))
                api.refresh(p)
            states = api.states(p)
            st, reg = states.get((wf, "stack")), states.get((wf, "register"))
            if st is not None and st.state == "done":
                return True, ""
            why = " ".join(x.reason for x in (reg, st) if x is not None and x.state == "failed")
            if attempt == 1 and ("too few stars" in why or "usable frames" in why):
                # short or faint frames: the detection threshold the registration's help advises for them
                log(f"{p.name}: {why}: again with the star detection threshold at {FAINT_SIGMA}")
                api.set_param(p, [wf], "register", "detect_sigma", FAINT_SIGMA)
                api.save(p)
                continue
            return False, why or (st.reason if st is not None else "no state")
        return False, "no state"

    def level_wf(p, frac: str, lights: int):
        name = f"Stack {frac}"
        w = next((x for x in p.workflows if x.name == name), None)
        if w is None:
            base = p.workflows[0]
            params = {k: dict(v) for k, v in base.params.items()}
            params["inventory"] = {**params.get("inventory", {}), "max_lights": lights // LEVELS[frac]}
            w = api.add_workflow(p, name, "Classic stack", params)
            api.save(p)
        return w

    for frac in (None, *LEVELS):
        for d in set_dirs(only):
            tag = d.name + (f" {frac}" if frac else "")
            if report.get(tag, {}).get("status") == "done":
                continue
            root = PROJECTS / d.name
            t0 = time.time()
            try:
                if frac is not None:
                    lights = int(report.get(d.name, {}).get("lights") or 0)
                    if report.get(d.name, {}).get("status") != "done" or lights // LEVELS[frac] < MIN_LEVEL_LIGHTS:
                        continue
                    p = api.open_project(root)
                    ok, why = run(p, level_wf(p, frac, lights).id)
                else:
                    if (root / "project.astroproj").exists():
                        p = api.open_project(root)
                    else:
                        p = api.create_project(root, d.name, d.name.split("_", 3)[-1])
                        scan = api.scan_folder(d / "lights")
                        log(f"{d.name}: {({k: len(v) for k, v in scan.frames.items()})}, "
                            f"{len(scan.errors)} unreadable")
                        api.add_frames(p, scan)
                        p.devices["main"] = device
                        api.add_workflow(p, "Stack", "Classic stack")
                        api.save(p)
                    ok, why = run(p, p.workflows[0].id)
                inv = api.stage_outputs(p, p.workflows[0].id, "inventory")
                report[tag] = {"status": "done" if ok else "failed", "minutes": round((time.time() - t0) / 60, 1),
                               "why": why, "lights": (inv.metrics.get("n_lights") if inv else None)}
            except Exception as e:  # noqa: BLE001 - one set failing does not stop the night
                report[tag] = {"status": "failed", "minutes": round((time.time() - t0) / 60, 1),
                               "why": f"{type(e).__name__}: {e}"[:300]}
            log(f"{tag}: {report[tag]}")
            report_f.write_text(json.dumps(report, indent=1))
    n = sum(r["status"] == "done" for r in report.values())
    log(f"stacked: {n} of {len(report)}")
    return 0


# ---------------------------------------------------------------------------------------------- report
#: the measures of the chain quoted per set: (stage, metric, column title)
REPORT_METRICS = [("register", "aligned", "aligned"), ("register", "rms_px", "align. RMS px"),
                  ("register", "fwhm_px", "FWHM px"), ("stack", "frames_used", "frames used"),
                  ("stack", "rejected_pct", "rejected %"), ("stack", "grid", "grid"),
                  ("calibrate", "bad_pixels_pct", "bad px %")]


def log_events(log_files: list[Path]) -> dict[str, list[str]]:
    """The warnings and errors of the night logs, per set (the lines between a set's first and last mention)."""
    out: dict[str, list[str]] = {}
    cur = None
    for lf in log_files:
        if not lf.exists():
            continue
        for line in lf.read_text(encoding="utf-8", errors="replace").splitlines():
            body = line[9:] if len(line) > 9 else ""
            head = body.split(":", 1)[0].strip()
            if head[:2].isdigit() and "_" in head:
                cur = head.split(" ")[0]
            elif cur and ("warning:" in body or "error:" in body):
                msg = body.split("]", 1)[-1].strip() if "]" in body else body.strip()
                stage = body.split("[", 1)[-1].split("]", 1)[0] if "[" in body else ""
                kind = "error" if "error:" in body else "warning"
                item = f"{kind} ({stage}): {msg}"
                if item not in out.setdefault(cur, []):
                    out[cur].append(item)
    return out


def report() -> int:
    os.environ["ASTRO_GPU_HOME"] = str(PROJECTS / "_state")
    from astro_gpu import api
    st = json.loads((PROJECTS / "stack.json").read_text()) if (PROJECTS / "stack.json").exists() else {}
    events = log_events([Path("F:/astro_essais/denoise_night_1.log"), Path("F:/astro_essais/denoise_night.log")])
    cat = {r["id"]: r for r in csv.DictReader(open(CATALOG, encoding="utf-8-sig"), delimiter=";")}
    lines = ["# L'empilement sur le jeu de données varié (rapport automatique)", "",
             f"Généré le {datetime.now():%d/%m/%Y à %H:%M} par `tools/bench/denoise_dataset.py report`, à partir de "
             "`stack.json`, des journaux de la nuit et des mesures de chaque projet "
             "(`F:\\astro_essais\\dataset_projets`).", "",
             "| Jeu | Appareil | Poses | Niveau | État | Minutes | " +
             " | ".join(t for _s, _m, t in REPORT_METRICS) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(REPORT_METRICS)]
    for tag, r in sorted(st.items()):
        sid = tag.split(" ")[0]
        level = tag[len(sid):].strip() or "toutes"
        vals = []
        try:
            p = api.open_project(PROJECTS / sid)
            w = next((x for x in p.workflows if x.name == ("Stack" if level == "toutes" else f"Stack {level}")), None)
            for stage, key, _t in REPORT_METRICS:
                o = api.stage_outputs(p, w.id, stage) if w is not None else None
                v = o.metrics.get(key) if o is not None else None
                vals.append("" if v is None else (", ".join(str(x) for x in v) if isinstance(v, list) else str(v)))
        except Exception:  # noqa: BLE001
            vals = [""] * len(REPORT_METRICS)
        c = cat.get(sid, {})
        state = "fait" if r.get("status") == "done" else f"**échec** : {r.get('why', '')}"
        lines.append(f"| {sid} | {c.get('camera', '')[:28]} | {r.get('lights') or c.get('n_lights', '')} | {level} | "
                     f"{state} | {r.get('minutes', '')} | " + " | ".join(vals) + " |")
    lines += ["", "## Avertissements et erreurs, par jeu", ""]
    for sid in sorted(events):
        lines.append(f"**{sid}**")
        lines += [f"- {e}" for e in events[sid]]
        lines.append("")
    out = ROOT / "docs" / "chantiers" / "empilement_jeu_varie.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"report: {out}")
    return 0


# ---------------------------------------------------------------------------------------------- export
def latest_stack(store: Path) -> Path | None:
    """The newest stack result of a project's store holding both half-stacks."""
    dirs = [d for d in (store / "stack").glob("*") if (d / "drz_A.npy").is_file() and (d / "drz_B.npy").is_file()]
    return max(dirs, key=lambda d: (d / "drz_A.npy").stat().st_mtime) if dirs else None


def export_one(name: str, stack_dir: Path, role: str) -> dict:
    from astro_gpu.stages.denoise import norm_params
    out = OUT / name
    if (out / "pair.json").exists():
        return json.loads((out / "pair.json").read_text())
    A = np.load(stack_dir / "drz_A.npy").astype(np.float32)
    B = np.load(stack_dir / "drz_B.npy").astype(np.float32)
    Wt = np.load(stack_dir / "drz_weight.npy").astype(np.float32)
    bk, sg, nm = norm_params(A, Wt)
    out.mkdir(parents=True, exist_ok=True)
    for k, a in (("A", A), ("B", B)):
        np.save(out / f"{k}.npy", ((a - bk[:, None, None]) / sg[:, None, None]).astype(np.float16))
    np.save(out / "nm.npy", nm.astype(np.float16))
    meta = {"name": name, "role": role, "source": str(stack_dir), "shape": list(A.shape), "bk": bk.tolist(),
            "sg": sg.tolist()}
    (out / "pair.json").write_text(json.dumps(meta, indent=1))
    return meta


def export() -> int:
    from astro_gpu import api
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = []
    for d in sorted(PROJECTS.glob("[0-9][0-9]_*")):
        try:
            p = api.open_project(d)
        except Exception:  # noqa: BLE001
            continue
        for w in p.workflows:
            o = api.stage_outputs(p, w.id, "stack")
            if o is not None and (Path(o.dir) / "drz_A.npy").is_file():
                level = {"Stack 1/2": "_half", "Stack 1/4": "_quarter", "Stack 1/8": "_eighth"}.get(w.name, "")
                pairs.append((f"set_{d.name}{level}", Path(o.dir), "train"))
    for d in sorted(USER_PROJECTS.iterdir()):
        s = latest_stack(d / "store") if (d / "store").is_dir() and not d.name.startswith("_") else None
        if s is not None:
            pairs.append((f"user_{d.name}", s, "train"))
    for c in BENCH_CASES:
        s = latest_stack(BENCH / c / "ours_express" / "store")
        if s is not None:
            pairs.append((f"bench_{c}", s, "validation"))
    for name, s, role in pairs:
        t = time.time()
        try:
            m = export_one(name, s, role)
            log(f"{name}: {m['shape']} ({role}) in {time.time() - t:.0f} s")
        except Exception as e:  # noqa: BLE001
            log(f"{name}: FAILED {type(e).__name__}: {e}")
    return 0


# ---------------------------------------------------------------------------------------------- train
def train(hours: float, device: str, patch: int, batch: int, lr: float, resume: bool) -> int:
    import torch
    import torch.nn.functional as F

    from astro_gpu.stages.denoise import UNet
    dev = torch.device(device)
    metas = [json.loads(f.read_text()) for f in sorted(OUT.glob("*/pair.json"))]
    tr = [m for m in metas if m["role"] == "train"]
    va = [m for m in metas if m["role"] == "validation"]
    log(f"training pairs: {len(tr)}, validation pairs: {len(va)}")

    def load(m):
        d = OUT / m["name"]
        return (np.load(d / "A.npy", mmap_mode="r"), np.load(d / "B.npy", mmap_mode="r"),
                np.load(d / "nm.npy", mmap_mode="r"))
    data = [load(m) for m in tr]
    vdata = [load(m) for m in va]
    rng = np.random.default_rng(0)

    def tiles(src, n, P, rng):
        """n random tiles (A or B as input, the other as target) from random pairs, inside the covered field."""
        xin, tgt, nmp = [], [], []
        while len(xin) < n:
            A, B, N = src[rng.integers(len(src))]
            H, W = A.shape[1:]
            if H <= P or W <= P:
                continue
            y, x = int(rng.integers(0, H - P)), int(rng.integers(0, W - P))
            c = float(N[y + P // 2, x + P // 2])
            if not 0 < c < 1.6:
                continue
            a, b, nn_ = (np.asarray(A[:, y:y + P, x:x + P], np.float32), np.asarray(B[:, y:y + P, x:x + P], np.float32),
                         np.asarray(N[y:y + P, x:x + P], np.float32)[None])
            if rng.random() < 0.5:
                a, b = b, a
            k = int(rng.integers(4))
            a, b, nn_ = (np.rot90(t, k, (1, 2)) for t in (a, b, nn_))
            if rng.random() < 0.5:
                a, b, nn_ = (t[:, :, ::-1] for t in (a, b, nn_))
            xin.append(np.ascontiguousarray(a))
            tgt.append(np.ascontiguousarray(b))
            nmp.append(np.ascontiguousarray(nn_))
        return (torch.from_numpy(np.stack(v)).to(dev) for v in (xin, tgt, nmp))

    def step_loss(net, xin, tgt, nmp):
        """The per-image training's loss (stages/denoise.train), on tiles already normalised (bk 0, sg 1)."""
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = net(torch.cat([torch.asinh(xin / 8.0), nmp], 1)).float()
        pred = xin + out
        with torch.no_grad():
            sm = F.avg_pool2d(xin, 5, 1, 2).clamp(min=0)
            wgt = 1.0 / ((nmp ** 2).clamp(min=0.25) + sm / 40.0)
        return (wgt * (pred - tgt) ** 2).mean()

    vrng = np.random.default_rng(1)
    vset = [tuple(tiles([v], 8, 256, vrng)) for v in vdata]           # fixed validation tiles, 8 per bench case
    ck = OUT / "pretrain.pt"
    hist_f = OUT / "pretrain_history.json"
    torch.manual_seed(0)
    net = UNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    start, hist = 0, []
    if resume and ck.exists():
        c = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(c["net"])
        opt.load_state_dict(c["opt"])
        start, hist = c["it"], json.loads(hist_f.read_text()) if hist_f.exists() else []
        log(f"resuming at step {start}")
    # the number of steps from the speed measured on the first 200 (cosine schedule to the end of the time given)
    t0 = time.time()
    it = start
    total = None
    run = None
    while True:
        if total is not None and it >= total:
            break
        if total is None and it == start + 200:
            per = (time.time() - t0) / 200
            total = start + int(hours * 3600 / per)
            log(f"{per * 1000:.0f} ms per step: {total - start} steps in {hours} h")
        if total is not None:
            frac = (it - start) / max(total - start, 1)
            for g in opt.param_groups:
                g["lr"] = lr * 0.5 * (1 + np.cos(np.pi * frac))
        net.train()
        xin, tgt, nmp = tiles(data, batch, patch, rng)
        loss = step_loss(net, xin, tgt, nmp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        it += 1
        if it % 2500 == 0:
            net.eval()
            with torch.no_grad():
                vl = [float(step_loss(net, *v)) for v in vset]
            hist.append({"step": it, "train": round(run, 4), "val": [round(v, 4) for v in vl],
                         "val_mean": round(float(np.mean(vl)), 4), "minutes": round((time.time() - t0) / 60, 1)})
            hist_f.write_text(json.dumps(hist, indent=1))
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "it": it}, ck)
            log(f"step {it}: train {run:.4f}, validation {np.mean(vl):.4f} ({', '.join(f'{v:.3f}' for v in vl)})")
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "it": it}, ck)
    torch.save({"net": net.state_dict()}, OUT / "pretrained_unet.pt")
    log(f"done: {it} steps in {(time.time() - t0) / 60:.0f} min")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("stack", "export", "train", "night", "report"))
    ap.add_argument("--only", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--patch", type=int, default=192)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    only = [x.strip() for x in a.only.split(",") if x.strip()] or None
    if a.cmd == "stack":
        return stack(only, a.device)
    if a.cmd == "export":
        return export()
    if a.cmd == "report":
        return report()
    if a.cmd == "night":
        stack(only, a.device)
        export()
        hours = a.hours
        log(f"training for {hours:.1f} h (a checkpoint and the validation every 2500 steps)")
        return train(hours, a.device, a.patch, a.batch, a.lr, a.resume)
    return train(a.hours, a.device, a.patch, a.batch, a.lr, a.resume)


if __name__ == "__main__":
    sys.exit(main())
