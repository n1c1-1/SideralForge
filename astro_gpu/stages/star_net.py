"""Learned star removal network: a U-Net trained on the project's own stack, which removes its stars.

The stage trains the network (library: star_net_lib.py) and writes the star layer it removes, in stack units, the far
halo of the stars excluded (``star_layer.npy``). The star removal stage ('starless') subtracts that layer from the two
half-stacks for its second, learned starless image when its own 'Learned star layer' setting is empty.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from ..core.params import ADVANCED, P, StageParams
from ..core.stage import Context, Stage
from . import star_net_lib as lib


class StarNetParams(StageParams):
    enabled: bool = P(True, label="Learn the star removal",
                      help="Train a star-removal network on this project's own stack and give its star layer to "
                           "the star removal stage (a second, learned starless image chosen at rendering). On by "
                           "default: both removals are computed at the start, to be compared as views. Off: the "
                           "stage does nothing.",
                      tip="About 35 minutes on one GPU (the training time below plus a few minutes of "
                          "preparation). Worth it with bright stars, spikes and halos.")
    minutes: float = P(32.0, label="Training time", unit="min", ge=1, le=240, step=1,
                       help="Wall-clock training budget of the network (the learning rate decays over it).",
                       tip="32 min gave the results measured on four projects; much less leaves faint halos.")
    device: str = P("auto", label="GPU", level=ADVANCED, choices=["auto", "main", "aux"],
                    help="Where to train: 'aux' = the second GPU of the project ('aux' in its devices), 'main' = "
                         "the main one, 'auto' = the second GPU when one is configured, else the main one.")
    batch: int = P(6, label="Batch size", level=ADVANCED, ge=1, le=32,
                   help="Training patches (384 px) per step; lower it if the GPU runs out of memory.")
    seed: int = P(0, label="Random seed", level=ADVANCED, ge=0,
                  help="Seed of the random patches and pasted stars (for reproducible runs).")
    glow: bool = P(True, label="Nebula peaked on its star", level=ADVANCED,
                   help="In training, half of the pasted bright stars sit on a smooth glow that the target keeps: "
                        "a nebula peaked on its star stays nebula.",
                   tip="On: the core of NGC 2023 is kept (off, the network took it for the light of its G 7.8 "
                       "star). It also keeps a round glow of about 40 px at the Iris's central star.")


# ------------------------------------------------------------------ the layer as seen by the star removal stage

_TAG = re.compile(r"^(.*\.npy) \(run [0-9a-fA-F]*\)$")


def stage_layer(project) -> str | None:
    """The star layer of this stage when it ran to the end and wrote one, as '<path> (run <id>)' (the run id makes a
    new layer a new value for the star removal stage), else None."""
    d = project.stage_dir("star_net")
    f = d / "star_layer.npy"
    try:
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if m.get("status") != "done" or not f.exists():
        return None
    return f"{f} (run {str(m.get('run_id', ''))[:8]})"


def layer_file(value: str) -> Path | None:
    """The file of a 'Learned star layer' value: a path, or this stage's '<path> (run <id>)'; '' or 'none' = no
    layer."""
    v = (value or "").strip()
    if not v or v.lower() in ("none", "off"):
        return None
    m = _TAG.match(v)
    return Path(m.group(1) if m else v)


def choose_device(ctx: Context, choice: str):
    """The training device: the project's second GPU ('aux' in its devices) for 'aux', and for 'auto' when one is
    configured; otherwise the main one."""
    aux = (ctx.project.config.devices or {}).get("aux", "")
    if choice in ("auto", "aux") and aux:
        if torch.cuda.is_available():
            try:
                d = torch.device(aux)
                if d.type != "cuda" or (d.index or 0) < torch.cuda.device_count():
                    return d
            except (RuntimeError, ValueError):
                pass
        ctx.log(f"the second GPU '{aux}' is not available: training on the main one", "warning")
    elif choice == "aux":
        ctx.log("no second GPU configured for this project ('aux' in its devices): training on the main one",
                "warning")
    return ctx.device


def _save_atomic(path: Path, arr) -> None:
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)


# ------------------------------------------------------------------------------------------------- the stage

class StarNet(Stage):
    name = "star_net"
    title = "Learned star removal network"
    description = ("Optional (off by default). Trains a neural network (U-Net) to remove the stars of this project, "
                   "on this project only: real stars cut from your stack are pasted at random places of the stack "
                   "(scaled, saturated like yours, with their photon noise) and the network learns to give back the "
                   "background, which the star-model removal provides away from the bright stars. Everything is "
                   "measured on your data: noise, saturation, how far the stars' light reaches (halo and spikes), "
                   "the colour of your brightest nebula, two hold-out regions that check the network. Its star layer "
                   "goes to the star removal stage, which makes a second, learned starless image from it.")
    deps = ("stars", "optics", "denoise", "color")
    Params = StarNetParams
    version = "3"          # 2: bright stamps = the star as the input shows it (light agreed by the interpolated and
                           # the model background, point-symmetric darkness), no more the input minus the starless
                           # image, whose holes the network learnt as star light; dark rings it had never seen;
                           # 3: stamps saturated as the rendering sees it (Alnitak, most of the Rosette's saturated
                           # stars passed for unsaturated); a glow centred on half of the pasted bright stars, in the
                           # input and the target alike (the network took the core of NGC 2023 for star light)

    def run(self, ctx: Context, p: StarNetParams) -> None:
        if not p.enabled:
            return
        dev = choose_device(ctx, p.device)
        ctx.log(f"learned star removal: training on {dev}, {p.minutes:.0f} min")
        ck = ctx.ckpt
        target, filled, dirs = self._target(ctx, ck)
        L = ctx.load("optics", "lat_full.npy", mmap=True)
        H, W = L.shape[1:]
        hf = ctx.project.stage_dir("stars") / "stars_model_halo.npy"
        halo = np.load(hf, mmap_mode="r") if hf.exists() else None
        if halo is not None and halo.shape != L.shape:
            halo = None
        ctx.progress(0.08, "the denoised stack, aligned in colour like the stack")
        from .optics import apply_lateral, lateral_from_json
        NL = apply_lateral(ctx.load("denoise", "n2n_full.npy").astype(np.float32),
                           lateral_from_json(ctx.input("optics", "lateral.json")))
        dm = float(ctx.metric_of("optics", "mag_offset", 0.0))
        fs = np.load(ctx.input("stars", "star_fluxes.npz"))
        x, y, g = fs["x"], fs["y"], fs["g"] - dm
        prep = self._prepare(ctx, ck, p, L, NL, halo, target, filled, dirs, x, y, g, H, W)

        ctx.progress(0.14, "training data")
        Lin = np.array(L, np.float32)                                   # input: the stack minus the far halo
        if halo is not None:
            Lin -= halo
        T = np.array(L, np.float32)                                     # target: starless + the input's own noise,
        T -= NL                                                         # none at the cores of the real stars
        T *= (1 - prep["cm"])[None]
        T += target
        del NL
        WM = prep["wm"] * (1 - prep["cm"])
        VAL = cv2.dilate(filled.astype(np.uint8), np.ones((7, 7), np.uint8)) == 0
        cys, cxs = lib.patch_centres(WM, x, y, g, H, W, lib.PATCH)
        m = prep["meta"]
        synth = lib.Synth(prep["lib"], m["sat99"], m["gain"], m["sg"], dev, glow=p.glow, neb_col=m["neb_col"])
        data = dict(L=Lin, T=T, S_den=target, WM=WM, VAL=VAL, cys=cys, cxs=cxs, bk=m["bk"], sg=m["sg"],
                    neb_col=m["neb_col"], items=lib.val_items(m["boxes"], synth))
        ctx.log(f"{len(cys)} training positions, {synth.n_bright} bright and {len(synth.faint)} faint star stamps")

        net, hist, steps = lib.train_net(data, synth, dev, p.minutes, batch=p.batch, patch=lib.PATCH, seed=p.seed,
                                         progress=lambda f, msg: ctx.progress(0.15 + 0.77 * f, msg), log=ctx.log,
                                         ckpt=ck / "train.pt")
        torch.save({"net": net.state_dict(), "bk": m["bk"], "sg": m["sg"], "widths": lib.WIDTHS,
                    "k_asinh": lib.K_ASINH, "hist": hist, "steps": steps}, ctx.out("star_net_model.pt"))
        del T, WM, VAL, data, synth

        ctx.progress(0.92, "removing the stars from the full frame")
        out = lib.apply_tiled(net, Lin, m["bk"], m["sg"], dev,
                              progress=lambda f, msg: ctx.progress(0.92 + 0.06 * f, msg))
        del net
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        layer = Lin - out
        ctx.save("star_layer.npy", layer.astype(np.float32))
        last = hist[-1] if hist else {}
        ctx.metric("training_steps", int(steps))
        ctx.metric("training_minutes", round(float(last.get("minutes", 0.0)), 1))
        ctx.metric("final_loss", round(float(last.get("loss") or 0.0), 4))
        if np.isfinite(last.get("val", np.nan)):
            ctx.metric("holdout_loss_with_stars", round(float(last["val"]), 4))
            ctx.metric("holdout_loss_without_stars", round(float(last["val0"]), 4))
        ctx.log(f"done: {steps} training steps; star layer written (the stack minus the far halo, minus the "
                f"network's starless image)")
        self._preview(ctx, L, out, layer, x, y, g, m)

    # ------------------------------------------------------------------------------------------------ steps
    @staticmethod
    def _target(ctx: Context, ck: Path):
        """The training target: the model-based starless image (the star removal's default settings), with its
        filled cores and the measured spike directions; kept in the checkpoint folder for a resume."""
        tf, mf, jf = ck / "target.npy", ck / "target_filled.npy", ck / "target.json"
        if tf.exists() and mf.exists() and jf.exists():
            ctx.log("training target taken back from the interrupted run")
            return np.load(tf), np.load(mf), json.loads(jf.read_text())["dirs"]
        ctx.log("training target: the star-model removal, with its default settings")
        from .starless import StarlessParams, model_starless
        r = model_starless(ctx, StarlessParams(), prog=lambda f, msg: ctx.progress(0.08 * f, f"training target: {msg}"))
        target, filled, dirs = r["sl_final"].astype(np.float32), r["mask"], [float(d) for d in r["dirs"]]
        del r
        _save_atomic(tf, target)
        _save_atomic(mf, filled)
        jf.write_text(json.dumps({"dirs": dirs}))
        return target, filled, dirs

    @staticmethod
    def _prepare(ctx, ck, p, L, NL, halo, target, filled, dirs, x, y, g, H, W) -> dict:
        """Measurements on the project's own data, kept in the checkpoint folder for a resume."""
        pf = ck / "prep.pt"
        if pf.exists():
            ctx.log("measurements taken back from the interrupted run")
            return torch.load(pf, map_location=ctx.device, weights_only=False)
        A, B = ctx.load("optics", "lat_A.npy", mmap=True), ctx.load("optics", "lat_B.npy", mmap=True)
        ctx.progress(0.09, "noise, saturation and photon noise")
        bk, sg, sat99 = lib.noise_stats(target, A, B, L)
        gain = lib.photon_gain(NL, target, A, B, x, y, g, H, W, log=ctx.log)
        NLh = NL if halo is None else NL - halo                       # the star light without the far halo

        ctx.progress(0.10, "how far the stars' light reaches")
        ext = lib.measure_extent(lib.Difference(NLh, target), sg, x, y, g, dirs, H, W)
        fit = lib.fit_extent(ext, g)
        R_all = lib.radius_of(g, fit)
        for i, v in ext.items():
            R_all[i] = float(np.clip(max(R_all[i], 1.15 * max(v)), 12, 600))
        law = {f"G{gg}": int(lib.radius_of(gg, fit)) for gg in (5, 7, 9, 11)}
        ctx.log(f"the stars' light (halo and spikes) reaches, measured on {len(ext)} bright stars: "
                + ", ".join(f"{k} {v} px" for k, v in law.items())
                + ("" if fit is not None else " (too few bright stars to measure: a fixed law)"))
        ctx.metric("star_extent_px", law)
        ctx.progress(0.11, "star stamps")
        stamps = lib.stamp_library(NLh, target, L, x, y, g, R_all.astype(int), sat99, H, W, seed=p.seed,
                                   fill=lib.fill_radii(ext, g, fit), filled=filled)
        del NLh
        ctx.log(f"{len(stamps['bright'])} bright star stamps (G < 11, "
                f"{sum(bool(np.asarray(b['sat']).any()) for b in stamps['bright'])} saturated) and "
                f"{len(stamps['faint'])} faint ones (G 11-15)")
        ctx.progress(0.12, "hold-out regions and nebula colour")
        wm, zone4 = lib.reliability(x, y, g, R_all, H, W)
        S4 = lib.down4(target)
        lum4 = ((S4 - bk[:, None, None]) / sg[:, None, None]).mean(0)
        boxes = lib.hold_out_boxes(lum4, zone4, H, W)
        if boxes:
            for (x0, y0, x1, y1) in boxes:
                wm[y0:y1, x0:x1] = 0
            ctx.log(f"hold-out regions (never training targets): structured {list(boxes[0])}, faint {list(boxes[1])}")
            ctx.metric("holdout_regions", [list(b) for b in boxes])
        neb_col, where = lib.nebula_colour(S4, lum4, bk, sg, x, y, g, R_all, filled)
        ctx.log(f"brightest extended nebula: {where}; its colour drives the brightness augmentation")
        ctx.metric("nebula_colour", [round(float(c), 3) for c in neb_col])
        cm = lib.star_cores(filled, x, y, g, R_all)
        prep = {"lib": stamps, "wm": wm, "cm": cm,
                "meta": dict(bk=bk, sg=sg, sat99=sat99, gain=gain, fit=fit, neb_col=neb_col,
                             boxes=[tuple(int(v) for v in b) for b in boxes] if boxes else None)}
        tmp = pf.with_suffix(".tmp")
        torch.save(prep, tmp)
        tmp.replace(pf)
        return prep

    @staticmethod
    def _preview(ctx: Context, L, out, layer, x, y, g, m) -> None:
        """Input | network's starless image | star layer, around the brightest star with room around it."""
        H, W = L.shape[1:]
        h = int(min(300, H // 2, W // 2))
        cand = [i for i in np.argsort(g) if h <= x[i] <= W - h and h <= y[i] <= H - h]
        cx, cy = (int(round(x[cand[0]])), int(round(y[cand[0]]))) if cand else (W // 2, H // 2)
        sl = (slice(None), slice(cy - h, cy + h), slice(cx - h, cx + h))
        bk, sg = np.asarray(m["bk"])[:, None, None], np.asarray(m["sg"])[:, None, None]

        def st(a, base=0.25, zero=False):
            v = np.arcsinh((np.asarray(a, np.float32) - (0 if zero else bk)) / sg) / np.arcsinh(3000.0)
            return np.clip(base + (1 - base) * v, 0, 1)
        sep = np.ones((3, 2 * h, 6), np.float32)
        ctx.preview("star_removal", np.concatenate([st(L[sl]), sep, st(out[sl]), sep, st(layer[sl], 0.5, True)], 2))
