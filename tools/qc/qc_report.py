"""Report cards and non-regression comparisons of v1 and v2 renders, written into the QC output folder only.

* ``bulletin``: the measurements the stages recorded (v1: ``work/*/manifest.json``; v2: the manifests of the done
  stages of the workflow's chain), the export's own checks (v1 settings file), and measurements taken on the render:
  sky flatness and colour cast, star size and elongation (the functions of tools/qc/bulletin.py, reused). The sheet
  that bulletin.py wrote into ``<project>/results/`` goes to ``<out>/bulletin/`` here.
* ``compare_renders`` / ``nonreg``: how far two renders of the same data differ (mean and 99.9th percentile of
  |after - before| in 8-bit levels, share of pixels over 2 levels, a map of the differences x20, where they differ
  most), after putting both in one frame with their own projects' files. The v1 nonreg.py did this on copies of the
  project; v2 stores are not rewritten, so two renders (a project before and after, or v1 and v2) are compared.

A module of tools/qc/qc2.py (``python tools/qc/qc2.py --help``): not run by itself.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import bulletin as old_bulletin
import cv2
import numpy as np
from qc_addr import QCError, Source, resolve
from qc_frame import View, load_render
from qc_out import BULLETIN, NONREG, slug, sub
from qc_sheets import changed_zones

QA_KEYS = ("pure_black_pct", "pure_white_pct", "burnt_highlights_worst_zone_pct", "dark_dips_at_stars",
           "dark_dips_blind", "warnings")
#: two renders "differ" for the procedure (blind sheets needed) above these (8-bit levels, share of pixels)
DIFFER_P999, DIFFER_SHARE = 1.0, 1e-4


# ---------------------------------------------------------------------------------------------- bulletin

def stage_metrics(src: Source) -> dict[str, Any]:
    """The recorded measurements of the stages behind a render, under bulletin.py's short names."""
    out: dict[str, Any] = {}
    status: dict[str, Any] = {}
    for stage, f in src.manifests.items():
        try:
            j = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        status[stage] = j.get("status")
        for (st, k), short in old_bulletin.KEYS.items():
            if st == stage and k in j.get("metrics", {}):
                out[short] = j["metrics"][k]
    out["stages_not_done"] = sorted({s for s, v in status.items() if v != "done"} | set(src.missing))
    return out


def card(src: Source, sheet_to: Path | None = None) -> dict[str, Any]:
    """The report card of one render. ``sheet_to``: where to write the sheet (whole image and 100 % crops)."""
    r = {"project": src.project_name, "render": src.name, "version": src.version, "address": src.address,
         **stage_metrics(src)}
    if src.settings:
        qa = src.settings.get("qa") or {}
        r.update({k: qa.get(k) for k in QA_KEYS})
    if src.image is None or not Path(src.image).is_file():
        r["render_image"] = "missing"
        return r
    rd = load_render(src)
    img = rd.img
    r.update(old_bulletin.sky_measures(img))
    m = old_bulletin.star_measures(img)
    # the star size in the render's pixels, and in the sensor's (the same for every render of the same stack)
    per_sensor = rd.geom.W * src.drizzle / (rd.geom.extent[1] - rd.geom.extent[0])
    if "star_fwhm_px" in m:
        m["star_fwhm_sensor_px"] = round(m["star_fwhm_px"] / per_sensor, 2)
    r.update(m)
    r["image_px_per_sensor_px"] = round(per_sensor, 3)
    if sheet_to is not None:
        old_bulletin.sheet(img, sheet_to, f"{src.project_name} - {src.name}")
        r["sheet"] = str(sheet_to)
    return r


def bulletin(out: str | Path, addresses: list[str], sheet: bool = False, sources: list[Source] | None = None,
             rules=None) -> dict[str, Any]:
    """Cards of several renders into ``<out>/bulletin/``: one JSON per render and ``bulletin.json`` with all."""
    d = sub(Path(out), BULLETIN)
    srcs = sources if sources is not None else [resolve(a, rules=rules) for a in addresses]
    allr: dict[str, Any] = {}
    for s in srcs:
        stem = f"{slug(s.project_name)}__{slug(s.name)}"
        r = card(s, d / f"{stem}_planche.jpg" if sheet else None)
        (d / f"{stem}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
        allr[stem] = r
    (d / "bulletin.json").write_text(json.dumps(allr, indent=1, ensure_ascii=False), encoding="utf-8")
    return allr


# ---------------------------------------------------------------------------------------------- non-regression

def difference(a: np.ndarray, b: np.ndarray, valid: np.ndarray, margin: int = 8) -> tuple[np.ndarray, dict[str, Any]]:
    """|b - a| in 8-bit levels (largest channel) over the pixels both renders cover (away from their edges), and the
    figures of nonreg.py."""
    ok = cv2.erode(valid.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)).astype(bool)
    d = np.abs(b - a).max(-1) * 255.0
    if not ok.any():
        raise QCError("the two renders have no common area")
    v = d[ok]
    m = {"mean_levels": round(float(v.mean()), 4), "p999_levels": round(float(np.percentile(v, 99.9)), 3),
         "max_levels": round(float(v.max()), 2), "share_over_2_levels": round(float((v > 2).mean()), 6),
         "common_share": round(float(ok.mean()), 4), "identical": bool(v.max() == 0)}
    m["differs"] = bool(m["p999_levels"] > DIFFER_P999 or m["share_over_2_levels"] > DIFFER_SHARE)
    return np.where(ok, d, 0.0), m


def compare_renders(a: Source, b: Source, out: str | Path, name: str, frame: str = "auto",
                    zones: int = 6) -> dict[str, Any]:
    """How far render ``b`` is from render ``a`` (each mapped with its own project's files), into ``<out>/nonreg/``."""
    ra, rb = load_render(a), load_render(b)
    view = View([ra, rb], frame, 0)
    (ia, va), (ib, vb) = view.remap(0), view.remap(1)
    d, m = difference(ia, ib, va & vb)
    nd = sub(Path(out), NONREG)
    w = 1600
    amp = np.clip(d * 20, 0, 255).astype(np.uint8)
    amp = cv2.resize(amp, (w, max(1, round(amp.shape[0] * w / amp.shape[1]))), interpolation=cv2.INTER_AREA)
    jpg = nd / f"{slug(name)}_diff.jpg"
    ok, buf = cv2.imencode(".jpg", amp, [cv2.IMWRITE_JPEG_QUALITY, 90])
    jpg.write_bytes(buf.tobytes())
    zs = changed_zones(ia, ib, va & vb, zones)
    m.update({"before": a.describe(), "after": b.describe(), "frame": view.summary(), "map": str(jpg),
              "zones": [f"{x:.3f}:{y:.3f}" for x, y, _ in zs], "zone_levels": [s for _, _, s in zs]})
    return m


def nonreg(out: str | Path, cases: dict[str, tuple[Source, Source]], frame: str = "auto",
           sheets: bool = False, seed: int | None = None) -> dict[str, Any]:
    """The comparisons of several data sets (the procedure's four: Heart, Iris, Rosette, Horsehead): per case the
    figures, and, for a case that differs and ``sheets``, the blind sheets at the zones that changed most. Writes
    ``<out>/nonreg/nonreg.json``."""
    report: dict[str, Any] = {}
    for name, (a, b) in cases.items():
        m = compare_renders(a, b, out, name, frame)
        if sheets and m["differs"]:
            from qc_sheets import blind_sheets
            m["sheets"] = blind_sheets(out, f"nr_{slug(name)}", [a, b], seed=seed, zones=m["zones"][:4], frame=frame,
                                       labels=["before", "after"], overwrite=True)
        report[name] = m
    nd = sub(Path(out), NONREG)
    (nd / "nonreg.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return report
