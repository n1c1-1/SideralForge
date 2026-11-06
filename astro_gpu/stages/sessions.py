"""The sessions' join (issue #43): the sessions of a project, calibrated each with its own master frames, become one
set of calibrated frames, which registration and stacking read as they read a single session's.

The chain of a project whose sessions are calibrated separately holds, per session, an instance of the inventory and
of the calibration (``inventory@<id>``, ``calibrate@<id>``, core/workflow.py). This stage reads them all and writes
what the later stages read as "inventory" and "calibrate" (core/workflow.PROVIDES):

- frames.json: every session's frames, kind by kind, in the sessions' order;
- the metrics of both (the frame counts and the integration added up; the sensor of the first session, which every
  session must share; the time span of all);
- the calibrated cache: the sessions' calibrated frames one after the other in one file (the values on one scale:
  the smallest of the sessions'), with their frame statistics;
- master_dark.npy and master_flat.npy chosen so that the stack's saturation level, (white - dark) / flat x scale, is
  at every pixel the lowest of the sessions' (a pixel saturates in the joined set where it saturates in any session).

The sessions' own results are left as they are (Inspect shows each session's masters and calibrated frames).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..core.params import StageParams
from ..core.stage import Context, Stage, StageError
from .calibrate import SAT_MARK, VALID_MAX

KINDS = ("lights", "darks", "flats", "bias", "flat_darks")
#: the metrics of the inventory that every session must share
SAME = ("width", "height", "cfa")


class SessionsParams(StageParams):
    pass


def instances(ctx: Context, stage: str) -> list[str]:
    """The instances of ``stage`` in the chain of the job, in the sessions' order ("calibrate@1", ...)."""
    chain = getattr(ctx.project, "chain", None) or []
    return [x for x in chain if x.startswith(stage + "@")]


def _read(ctx: Context, name: str, file: str):
    return json.loads((ctx.project.stage_dir(name) / file).read_text(encoding="utf-8"))


def _metrics(ctx: Context, name: str) -> dict:
    return ctx.manifest(name).get("metrics") or {}


def joined_inventory(invs: list[dict], frames: list[dict]) -> tuple[dict, dict]:
    """(frames.json, metrics) of the joined sessions from each session's inventory metrics and frames.json."""
    out = {k: [f for fr in frames for f in fr.get(k) or []] for k in KINDS}
    first = invs[0]
    for k in SAME:
        values = {str(m.get(k)) for m in invs}
        if len(values) > 1:
            raise StageError(f"the sessions do not have the same sensor ({k}: {', '.join(sorted(values))}): sessions "
                             "taken with different cameras cannot be stacked together")
    m = dict(first)
    for k in ("n_lights", *(f"n_{x}" for x in KINDS[1:])):
        m[k] = int(sum(int(x.get(k) or 0) for x in invs))
    hours = [x.get("integration_h") for x in invs]
    m["integration_h"] = round(sum(h for h in hours if h), 2) if any(hours) else None
    texp = {x.get("exposure_s") for x in invs}
    m["exposure_s"] = texp.pop() if len(texp) == 1 else None
    whites = [float(x["white_level"]) for x in invs if x.get("white_level") is not None]
    if whites:
        m["white_level"] = min(whites)
    times = [t for x in invs for t in (x.get("first_time"), x.get("last_time")) if t]
    if times:
        m["first_time"], m["last_time"] = min(times), max(times)
    ra = [x["pointing_ra"] for x in invs if x.get("pointing_ra") is not None]
    dec = [x["pointing_dec"] for x in invs if x.get("pointing_dec") is not None]
    if ra and dec:
        m["pointing_ra"], m["pointing_dec"] = float(np.median(ra)), float(np.median(dec))
    return out, m


class Sessions(Stage):
    name = "sessions"
    title = "Sessions"
    description = ("Joins the sessions, each calibrated with its own master frames, into one set of calibrated "
                   "frames for the registration and the stack.")
    Params = SessionsParams
    version = "1"

    def run(self, ctx: Context, p: SessionsParams) -> None:
        inv_names, cal_names = instances(ctx, "inventory"), instances(ctx, "calibrate")
        if len(cal_names) < 2 or len(inv_names) != len(cal_names):
            raise StageError("the sessions' join needs the calibration of two sessions or more")
        invs = [_metrics(ctx, n) for n in inv_names]
        frames, m = joined_inventory(invs, [_read(ctx, n, "frames.json") for n in inv_names])
        caches = [_read(ctx, n, "cache.json") for n in cal_names]
        cals = [_metrics(ctx, n) for n in cal_names]
        H, W = int(caches[0]["height"]), int(caches[0]["width"])
        if any((int(c["height"]), int(c["width"])) != (H, W) for c in caches):
            raise StageError("the sessions' calibrated frames do not have the same size")
        scale = min(float(c["scale"]) for c in caches)
        white = min(float(c["white_level"]) for c in caches)
        n = sum(int(c["n"]) for c in caches)
        t0 = time.time()
        # the saturation level of the joined set: the lowest of the sessions' at each pixel, as dark and flat
        sat = None
        for name, c in zip(cal_names, caches):
            d = ctx.project.stage_dir(name)
            s = (float(c["white_level"]) - np.load(d / "master_dark.npy")) / np.load(d / "master_flat.npy") * float(c["scale"])
            sat = s if sat is None else np.minimum(sat, s)
        ctx.save("master_dark.npy", (white - sat / scale).astype(np.float32))
        ctx.save("master_flat.npy", np.ones((H, W), np.float32))
        bad = None
        for name in cal_names:
            f = ctx.project.stage_dir(name) / "badmask.npy"
            if f.is_file():
                b = np.load(f)
                bad = b if bad is None else bad | b
        if bad is not None:
            ctx.save("badmask.npy", bad)
        # the calibrated frames, one session after the other, on one scale
        cache = ctx.scratch / "lights.f16"
        out = np.memmap(cache, dtype=np.float16, mode="w+", shape=(n, H, W))
        stats = []
        at = 0
        for name, c in zip(cal_names, caches):
            k = int(c["n"])
            src = np.memmap(c["path"], dtype=np.float16, mode="r", shape=(k, H, W))
            f = scale / float(c["scale"])
            st = np.load(ctx.project.stage_dir(name) / "frame_stats.npy").astype(np.float32)
            for i in range(k):
                fr = src[i]
                if f != 1.0:
                    x = fr.astype(np.float32)
                    fr = np.where(x > VALID_MAX, SAT_MARK, x * f).astype(np.float16)
                out[at + i] = fr
                ctx.check_stop()
                ctx.progress((at + i + 1) / n, f"{at + i + 1}/{n} frames joined")
            stats.append(st * f)
            at += k
            del src
        out.flush()
        del out
        ctx.save("frame_stats.npy", np.concatenate(stats).astype(np.float32))
        ctx.save_json("frames.json", frames)
        ctx.save_json("cache.json", {"path": str(cache), "n": n, "height": H, "width": W, "scale": scale,
                                     "cfa": caches[0].get("cfa"), "sat_mark": SAT_MARK, "valid_max": VALID_MAX,
                                     "white_level": white, "dark_level": float(np.median([c["dark_level"] for c in caches])),
                                     "sessions": [{"name": nm, "n": int(c["n"])} for nm, c in zip(cal_names, caches)]})
        for k, v in m.items():
            ctx.metric(k, v)
        ctx.metric("sessions", len(cal_names))
        noise = [x["dark_frame_noise"] for x in cals if x.get("dark_frame_noise")]
        if noise:
            ctx.metric("dark_frame_noise", round(float(np.median(noise)), 2))
        ctx.metric("cache_scale", scale)
        ctx.metric("cache_gb", round(n * H * W * 2 / 1e9, 1))
        for k in ("bad_pixels_pct", "sky_noise_adu", "dark_level"):
            v = [x[k] for x in cals if x.get(k) is not None]
            if v:
                ctx.metric(k, round(float(np.median(v)), 2))
        ctx.log(f"{len(cal_names)} sessions joined: {n} calibrated frames in {time.time() - t0:.0f} s "
                f"(cache {n * H * W * 2 / 1e9:.1f} GB)")


def joined_sessions(cache_json: Path) -> list[dict]:
    """[{"name", "n"}] of the sessions in a joined cache (empty for one session's cache)."""
    try:
        return list(json.loads(Path(cache_json).read_text(encoding="utf-8")).get("sessions") or [])
    except (OSError, ValueError):
        return []
