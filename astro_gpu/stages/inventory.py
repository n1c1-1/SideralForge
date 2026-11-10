"""Inventory: list and check every input frame, read the metadata that drives the automatic settings."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..core import cfa as cfa_mod
from ..core.params import ADVANCED, P, StageParams
from ..core.project import frame_kind, list_frames
from ..core.stage import Context, Stage, StageError
from ..io.frames import fits_header, read_cfa, read_colour, read_info, useful_area


class InventoryParams(StageParams):
    max_lights: int = P(0, label="Frame limit", level=ADVANCED, ge=0,
                        help="Use only N light frames, evenly spaced over the whole session (so that drift, "
                             "transparency and sky changes stay representative). 0 uses all of them.",
                        tip="Useful for a quick test run of the whole pipeline on a subset; keep 0 for the real run.")


class Inventory(Stage):
    name = "inventory"
    title = "Inventory"
    description = ("Lists the light, dark, flat, bias and flat-dark frames, checks that they are consistent (same "
                   "sensor size, same colour pattern) and reads their metadata: exposure, ISO/gain, time, camera, "
                   "focal length, pointing. These measurements drive many automatic settings downstream.")
    Params = InventoryParams
    version = "1"

    def run(self, ctx: Context, p: InventoryParams) -> None:
        cfg = ctx.project.config.inputs
        kinds = ("lights", "darks", "flats", "bias", "flat_darks")
        for k in kinds:                        # a folder not there (its disk unplugged) is said, not "add frames" (7.9)
            gone = next((e for e in getattr(cfg, k) if not Path(e).exists()), None)
            if gone is not None:
                raise StageError(f"the folder of the {k.replace('_', ' ')} is not found: {Path(gone).as_posix()} "
                                 f"(is its disk plugged in?)")
        groups = {k: list_frames(getattr(cfg, k)) for k in kinds}
        if not groups["lights"]:
            raise StageError("no light frame found: add at least one folder of light frames")
        if p.max_lights and p.max_lights < len(groups["lights"]):
            keep = np.unique(np.linspace(0, len(groups["lights"]) - 1, p.max_lights).round().astype(int))
            groups["lights"] = [groups["lights"][i] for i in keep]
        out = {}
        total = sum(len(v) for v in groups.values())
        done = 0
        with ThreadPoolExecutor(8) as ex:
            for k, files in groups.items():
                infos = []
                if not files:
                    out[k] = infos
                    continue
                first = next((i for i in map(_readable, files) if i is not None), None)   # decoded once; the others
                bad = []                                                                  # only read their header
                for f, info in zip(files, ex.map(lambda f: _readable(f, like=first), files)):
                    if info is None:
                        bad.append(Path(f).name)
                    else:
                        infos.append(info.to_dict())
                    done += 1
                    ctx.progress(done / total, f"reading metadata {done}/{total}")
                # the night test pass: an empty or damaged file stopped the whole run without its name; it is left
                # out, by its name (all of them: what to do)
                what = k.replace("_", " ")
                if bad and not infos and k == "lights":
                    raise StageError(f"no light frame can be read (damaged or empty files?): {', '.join(bad)}; take "
                                     f"them out of the session")
                if bad:
                    ctx.log(f"{len(bad)} {what} cannot be read and are left out (damaged or empty files?): "
                            f"{', '.join(bad)}", "warning")
                out[k] = infos
        lights = out["lights"]
        sizes = Counter((f["width"], f["height"]) for f in lights)
        (w, h), _ = sizes.most_common(1)[0]
        cfas = Counter(f["cfa"] for f in lights)
        cfa = cfas.most_common(1)[0][0]
        if not cfa:                                   # no colour filter array: a monochrome camera (issue #43)
            cfa = cfa_mod.MONO
            for k in out:
                for f in out[k]:
                    f["cfa"] = f["cfa"] or cfa_mod.MONO
            lights = out["lights"]
            ctx.metric("mono", True)
        if not cfa_mod.is_supported(cfa):
            raise StageError(f"unsupported colour pattern {cfa!r}")
        same_kind(out, cfa, next((f for f in lights if f["cfa"] == cfa), lights[0]))
        if cfa_mod.is_colour(cfa):                    # said plainly: the frames are the camera's or a program's images
            ctx.metric("colour_frames", True)
            ctx.log("frames already in colour: no debayering, calibrated colour by colour, registered on their "
                    "luminance, stacked by the Classic method (no Bayer drizzle). They are not linear data when they "
                    "were stretched (every JPEG is): dark and flat correction are then approximate", "warning")
        for k in ("darks", "flats", "bias", "flat_darks"):
            bad = [f["path"] for f in out[k] if (f["width"], f["height"]) != (w, h)]
            if bad:
                raise StageError(f"{len(bad)} {k.replace('_', ' ')} have another size than the lights ({w}x{h}), e.g. "
                                 f"{Path(bad[0]).name}: take them out of the session, or use those of this "
                                 f"camera")
        odd = [f["path"] for f in lights if (f["width"], f["height"]) != (w, h) or f["cfa"] != cfa]
        if odd:
            ctx.log(f"{len(odd)} light frame(s) with a different size or pattern are ignored", "warning")
            out["lights"] = [f for f in lights if f["path"] not in set(odd)]
            lights = out["lights"]
        ctx.save_json("frames.json", out)

        exp = [f["exposure"] for f in lights if f["exposure"]]
        texp = float(np.median(exp)) if exp else None
        times = sorted(f["time"] for f in lights if f["time"])
        first = lights[0]
        ctx.metric("n_lights", len(lights))
        for k in ("darks", "flats", "bias", "flat_darks"):
            ctx.metric(f"n_{k}", len(out[k]))
        ctx.metric("width", w)
        ctx.metric("height", h)
        ctx.metric("cfa", cfa)
        ctx.metric("camera", first["camera"])
        ctx.metric("exposure_s", texp)
        ctx.metric("integration_h", round(texp * len(lights) / 3600, 2) if texp else None)
        ctx.metric("iso_gain", first["iso"])
        ctx.metric("white_level", _clip_level(ctx, lights, float(first["white"])))
        ctx.metric("black_level", first["black"])
        ctx.metric("focal_mm", first["focal_mm"])
        ctx.metric("pixel_um", first["pixel_um"])
        if first.get("pixel_scale"):
            ctx.metric("pixel_scale", first["pixel_scale"])
        ctx.metric("first_time", times[0] if times else None)
        ctx.metric("last_time", times[-1] if times else None)
        hints = [(f["ra"], f["dec"]) for f in lights if f.get("ra") is not None]
        if hints:
            ctx.metric("pointing_ra", float(np.median([a for a, _ in hints])))
            ctx.metric("pointing_dec", float(np.median([b for _, b in hints])))
        objs = Counter(f["object"] for f in lights if f.get("object"))
        if objs:
            ctx.metric("object", objs.most_common(1)[0][0])
        pattern = ("X-Trans" if cfa_mod.is_xtrans(cfa) else "monochrome" if cfa_mod.is_mono(cfa) else
                   "colour" if cfa_mod.is_colour(cfa) else cfa)
        ctx.log(f"{len(lights)} lights of {texp} s ({ctx.metrics['integration_h']} h), {w}x{h} {pattern}, "
                f"{len(out['darks'])} darks, {len(out['flats'])} flats, {len(out['bias'])} bias, "
                f"camera {first['camera'] or 'unknown'}")
        if frame_kind(first["path"]) == "fits":
            hd = fits_header(first["path"])
            if useful_area(hd) is not None:
                ctx.log(f"useful area from the frames' header: {w}x{h} of {hd['NAXIS1']}x{hd['NAXIS2']} pixels, the "
                        "overscan left out")
        # quick look at the middle frame (2x2 super-pixels, no calibration)
        read = read_colour if cfa_mod.is_colour(cfa) else read_cfa
        mid = read(lights[len(lights) // 2]["path"]) - first["black"]
        ctx.preview("middle_frame", _superpixel_rgb(mid, cfa))


def _readable(f, like=None):
    """A frame's metadata, None when the file cannot be read (empty, damaged, not what its name says)."""
    try:
        return read_info(f, like=like)
    except Exception:  # noqa: BLE001 - every reader's own error (LibRaw, FITS, TIFF, an empty file): the file is named
        return None


def same_kind(out: dict, cfa: str, light: dict) -> None:
    """Calibration frames that cannot calibrate these lights are refused, with the reason (issue #57): frames already
    in colour against a mosaic (one way or the other), or colour frames of another depth (8-bit darks under 16-bit
    lights): a dark is taken off and a flat divided by, value for value."""
    colour = cfa_mod.is_colour(cfa)
    for k in ("darks", "flats", "bias", "flat_darks"):
        word = k.replace("_", " ")
        for f in out[k]:
            if cfa_mod.is_colour(f["cfa"]) != colour:
                raise StageError(
                    f"the {word} ({Path(f['path']).name}) are "
                    f"{'already in colour' if not colour else 'a raw mosaic or monochrome'}, the lights "
                    f"{'already in colour' if colour else 'a raw mosaic or monochrome'}: they cannot calibrate them; "
                    f"take them out of the session, or use {word} of the same kind as the lights")
            if colour and float(f["white"]) != float(light["white"]):
                raise StageError(
                    f"the {word} ({Path(f['path']).name}) reach {float(f['white']):.0f} and the lights "
                    f"{float(light['white']):.0f} (another bit depth): they cannot calibrate them; take them out of "
                    f"the session, or use {word} of the same kind as the lights")


def _superpixel_rgb(cfa_img: np.ndarray, pattern: str) -> np.ndarray:
    from ..core.imgutil import asinh_preview
    return asinh_preview(np.moveaxis(cfa_mod.superpixel(cfa_img, pattern), -1, 0), k=30)


def _clip_level(ctx, lights, white: float) -> float:
    """The saturation level of FITS frames whose header does not give it (no DATAMAX: the type's maximum is
    assumed): a plateau of clipped pixels at the same highest value in a few frames is the real one (a 14-bit camera
    saved in 16-bit FITS saturates at 16383, not 65535)."""
    fits_l = [f for f in lights if frame_kind(f["path"]) == "fits"]
    if not fits_l:
        return white
    if "DATAMAX" in fits_header(fits_l[0]["path"]):
        return white
    tops = []
    for f in fits_l[:: max(1, len(fits_l) // 5)][:5]:
        d = read_cfa(f["path"])
        m = float(d.max())
        tops.append((m, int(np.count_nonzero(d == m))))
    plateau = [m for m, n in tops if n >= 20]
    if plateau and max(plateau) < white:
        level = max(plateau)
        ctx.log(f"saturation level {level:.0f} ADU found in the frames (a plateau of clipped pixels; the file type "
                f"allows {white:.0f})")
        return level
    return white
