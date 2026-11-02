"""Stacking v2, the crossed bench (docs/spec/STACKING.md): the previous version, each change alone, all together, all
but one, against each other and against Siril, measured on the same sky.

    python tools/bench/crossed.py setup CASE LEVEL [--configs NAME ...]
    python tools/bench/crossed.py run CASE LEVEL [--device cuda:0] [--configs NAME ...] [--export NAME ...]
    python tools/bench/crossed.py measure CASE LEVEL [--configs NAME ...]
    python tools/bench/crossed.py sheet CASE LEVEL [--configs OLD V2]
    python tools/bench/crossed.py report [LEVEL ...]

One project per case and level, F:/astro_essais/bench/<case>/v2x_<level>/ (the same frozen frames as ours_<level> and
Siril), one workflow per configuration, named by it (OLD, S1 ... V2-S6): the store computes calibrate once and each
registration once per set of register switches. The configurations (CONFIGS, ``configs``):

- OLD: every switch at its default (the previous version);
- OLD-bis: OLD with the registration's star detection threshold at 7.9 sigma instead of 8: a few more faint stars in
  the registration, so a registration that differs by the small amount two passes differ by (the run-to-run noise;
  the number of workers is not part of the key, so it cannot give a second run);
- S1 ... S6, S4b, S7: one change alone (S1 = Moffat FWHM and the coverage rule together; S4b = the k x sigma
  selection without its transparency filter; S7 = the Classic method with its own rejection);
- V2: S1 to S6 together, Bayer drizzle; V2-auto: the same with the method the coverage rule advises (read from V2's
  stack once it is done: ``setup`` adds it then);
- V2-Sk: V2 with change k back to the previous version.

Every configuration but OLD skips the catalogue of the faint stars (catalogs ``g_faint`` 0: the stars and the colour
of the Classic stack workflow do not read it, and each new astrometric solution would otherwise fetch its sky tiles
again); the measures take the faint stars from OLD's catalogue for all images.

``measure`` writes <case>/crossed_<level>.json: every configuration and Siril on the common sky (one star list, one
target, boxes of the same angular size, as measure.py --all does for the variants), one image in memory at a time,
plus what the engine says of each stack (frames kept, rejection rate, scale, method, durations). ``report`` writes
docs/pilot/bench/crossed_<level>.md (one table per measure: configurations x cases, deltas against OLD and Siril)
and fills the blocks <!-- crossed:LEVEL --> ... <!-- /crossed --> of docs/pilot/BENCH.md. ``sheet`` writes
<case>/crossed_sheet_<level>.jpg (Siril, OLD, V2 side by side) from the exports (export/bench_<config>/).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as C  # noqa: E402

os.environ["USERPROFILE"] = str(C.HOME)          # before astro_gpu is imported: Path.home() -> the bench home
os.environ["ASTRO_GPU_HOME"] = str(C.AG_HOME)
os.environ["PYTHONUNBUFFERED"] = "1"
# the run's worker inherits this environment and prints its journal to a log file opened with the locale's encoding
# (cp1252 on Windows): a log line with a "σ" (the local normalisation's reason) would kill the stage
# (UnicodeEncodeError in core/journal.py, an engine bug reported to the orchestrator); UTF-8 avoids it
os.environ["PYTHONIOENCODING"] = "utf-8"
C.offline_env()
sys.path.insert(0, str(C.REPO))
C.utf8_console()

from astro_gpu import api  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

TEMPLATE = "Classic stack"
REGISTER_KEYS = ("fwhm_method", "channel_align", "detect_sigma")
CATALOG_KEYS = ("g_faint",)
#: the six changes of stacking v2, each with the switches it sets (the first value of each switch is the old one)
CHANGES: dict[str, dict[str, str]] = {
    "S1": {"fwhm_method": "moffat", "scale_rule": "coverage"},
    "S2": {"rejection": "hubble"},
    "S3": {"weighting": "noise"},
    "S4": {"selection": "sota"},
    "S5": {"normalisation": "local"},
    "S6": {"channel_align": "per_block"},
}
ALL_ON = {k: v for ch in CHANGES.values() for k, v in ch.items()}
#: since 2 November an empty configuration means the NEW defaults (Moffat star size, coverage rule, minimum-weight
#: selection): the previous version is now stored explicitly, so that it keeps the keys (stages/v2.KEY_BASELINE) of its
#: old results. The configurations of the first crossed pass (OLD, OLD-bis, S1 ... V2-Sk) are built on it; four of the
#: six changes (S2, S3, S5, S6) were removed from the engine on 2 November, their settings are read as unset.
LEGACY = {"fwhm_method": "legacy", "scale_rule": "legacy", "selection": "legacy"}
#: the bench of 2 November (docs/pilot/BENCH.md "Banc du 2 novembre"), with today's engine: our drizzle (today's
#: defaults), the same with PixInsight's rejection, our classic stack, the same without the white balance before the
#: debayering (exactly Siril's), and the previous version for reference
TODAY = ["PI-own", "PI-int", "CL", "CL-off", "LEG"]
TODAY_LABELS = {"PI-own": "notre drizzle", "PI-int": "notre drizzle avec le rejet de PixInsight",
                "CL": "notre empilement classique", "CL-off": "notre empilement classique sans équilibrage",
                "LEG": "l'ancienne version"}
#: the configurations that keep the catalogue of the faint stars (the star list of the measures comes from the first)
KEEP_FAINT = ("OLD", "LEG")
LABELS = {
    "OLD": "ancienne version (tout par défaut)",
    "OLD-bis": "ancienne version, recalage à peine différent (seuil 7,9 σ) : le bruit d'un passage à l'autre",
    "S1": "FWHM de Moffat + règle de couverture (échelle, goutte)",
    "S2": "rejet de Hubble",
    "S3": "pondération par le bruit, par couleur",
    "S4": "sélection k × σ (avec le filtre de transparence)",
    "S4b": "sélection k × σ sans le filtre de transparence",
    "S5": "normalisation locale, par couleur",
    "S6": "recalage des couleurs par blocs",
    "S7": "méthode Classique (dématriçage RCD, Lanczos-4, rejet WBPP)",
    "V2": "tout (S1 à S6), drizzle",
    "V2-auto": "tout, méthode conseillée par la règle de couverture",
    "V2-T": "tout, drizzle, sans le filtre de transparence",
    "V2-C": "tout, méthode Classique partout",
}
for _k in CHANGES:
    LABELS[f"V2-{_k}"] = f"tout sauf {_k}"
SIRIL = "Siril"


#: OLD-bis's only difference: the registration's star detection threshold (default 8 sigma)
REPLICATE_SIGMA = 7.9


def configs(rule_method: str | None = None) -> dict[str, dict]:
    """Every configuration: its switches (V2-auto only once the rule's advice is known)."""
    d: dict[str, dict] = {"OLD": dict(LEGACY), "OLD-bis": {**LEGACY, "detect_sigma": REPLICATE_SIGMA}}
    for k, ch in CHANGES.items():
        d[k] = {**LEGACY, **ch}
        if k == "S4":
            d["S4b"] = {**LEGACY, "selection": "sota", "selection_transparency": "off"}
    d["S7"] = {**LEGACY, "method": "classic"}
    d["V2"] = {**LEGACY, **ALL_ON}
    if rule_method:
        d["V2-auto"] = {**LEGACY, **ALL_ON, "method": rule_method}
    d["V2-T"] = {**LEGACY, **ALL_ON, "selection_transparency": "off"}
    d["V2-C"] = {**LEGACY, **ALL_ON, "method": "classic"}
    for k, ch in CHANGES.items():
        d[f"V2-{k}"] = {**LEGACY, **{a: b for a, b in ALL_ON.items() if a not in ch}}
    d["PI-own"] = {}
    d["PI-int"] = {"drizzle_rejection": "integration"}
    d["CL"] = {"method": "classic"}
    d["CL-off"] = {"method": "classic", "classic_cfa_balance": "off"}
    d["LEG"] = dict(LEGACY)
    return d


#: the configurations of a full Express pass (V2-T only on demand)
EXPRESS = ["OLD", "OLD-bis", "S1", "S2", "S3", "S4", "S4b", "S5", "S6", "S7", "V2", "V2-auto", "V2-C",
           "V2-S1", "V2-S2", "V2-S3", "V2-S4", "V2-S5", "V2-S6"]


def stage_params(cfg: dict, name: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for k, v in cfg.items():
        st = "register" if k in REGISTER_KEYS else "catalogs" if k in CATALOG_KEYS else "stack"
        out.setdefault(st, {})[k] = v
    if name not in KEEP_FAINT:
        out.setdefault("catalogs", {})["g_faint"] = 0.0
    return out


# ---------------------------------------------------------------------------------------------- the project

def root_of(case: str, level: str) -> Path:
    return C.BENCH / case / f"v2x_{level}"


def project(case: str, level: str):
    man = json.loads((C.BENCH / case / "manifest.json").read_text(encoding="utf-8"))
    root = root_of(case, level)
    if (root / "project.astroproj").exists():
        return api.open_project(root), man
    variant = C.norm_variant(case, None)
    lv = man["levels"][level]
    cal = C.calibration(man, variant)
    p = api.create_project(root, f"bench {case} {level} · stacking v2 crossed",
                           Target(name=man["target"], ra=man["ra"], dec=man["dec"], focal_mm=man["focal_mm"]))
    api.add_session(p, lights=[lv["folder"]], darks=[cal["darks"]["folder"]] if "darks" in cal else [],
                    flats=[cal["flats"]["folder"]] if "flats" in cal else [],
                    bias=[cal["bias"]["folder"]] if "bias" in cal else [], name=f"{case} {level}")
    p.description = (f"Stacking v2 crossed bench (tools/bench/crossed.py): {man['camera_label']}, {man['scope']}; "
                     f"{lv['count']} lights; one workflow per configuration")
    api.save(p)
    return p, man


def wf(p, name: str):
    return next((w for w in p.workflows if w.name == name), None)


def rule_method(p) -> str | None:
    """The method the coverage rule advises, from V2's stack (diagnostics.json coverage.rule), once it is done."""
    w = wf(p, "V2")
    if w is None:
        return None
    o = api.stage_outputs(p, w.id, "stack")
    if o is None:                                     # not stacked yet: the rule on V2's registration (stack.auto's)
        reg, cal = api.stage_outputs(p, w.id, "register"), api.stage_outputs(p, w.id, "calibrate")
        if reg is None or cal is None:
            return None
        from astro_gpu.stages.stack import drizzle_suggestion
        z = np.load(Path(reg.dir) / "reg.npz")
        info = json.loads((Path(cal.dir) / "cache.json").read_text(encoding="utf-8"))
        return drizzle_suggestion(z, info).method.value
    d = json.loads((Path(o.dir) / "diagnostics.json").read_text(encoding="utf-8"))
    r = ((d.get("coverage") or {}).get("rule") or {}).get("method") or {}
    return r.get("value")


def setup(case: str, level: str, names: list[str]) -> list[str]:
    """The workflows of these configurations, created or brought to exactly their switches. Returns the names set."""
    p, _ = project(case, level)
    cfgs = configs(rule_method(p))
    done = []
    for name in names:
        if name not in cfgs:
            print(f"{case} {level}: {name} not known yet (V2-auto needs V2's stack)")
            continue
        want = stage_params(cfgs[name], name)
        w = wf(p, name)
        if w is None:
            w = api.add_workflow(p, name, TEMPLATE)
        for st, kv in want.items():
            for k, v in kv.items():
                api.set_param(p, w.id, st, k, v)
        for st, kv in list(p.workflow(w.id).params.items()):
            for k in list(kv):
                if k not in want.get(st, {}):
                    api.set_param(p, w.id, st, k, None)
        done.append(name)
    api.save(p)
    print(f"{case} {level}: workflows {done}")
    return done


# ---------------------------------------------------------------------------------------------- running

def record(p, names: list[str]) -> dict:
    """Per configuration: each stage's key, duration and metrics; the stack's scale, method, frames, rejection."""
    out = {}
    for name in names:
        w = wf(p, name)
        if w is None:
            continue
        e: dict = {"stages": {}}
        for s in api.chain(p, w.id):
            o = api.stage_outputs(p, w.id, s)
            if o is None:
                continue
            e["stages"][s] = {"key": o.key, "duration": o.duration, "dir": o.dir,
                              "metrics": o.metrics if s in ("register", "stack", "astrometry") else None}
        out[name] = e
    return out


def prefill(case: str, level: str) -> dict:
    """The Gaia cones of every workflow whose astrometry is done and catalogs not, cut from the cones the bench cache
    already holds for this field (prefetch.py --pool's method, for every workflow of the project at once): the
    catalogs stage then reads them instead of the network (a new solution asks for new cells; m8's sky tiles were
    never cached, and the Gaia services did not answer on 2026-11-01)."""
    import prefetch as PF
    from astropy.io import fits
    from astropy.wcs import WCS

    from astro_gpu.stages.catalogs import CatalogsParams, _cells
    p, _ = project(case, level)
    cache = C.AG_HOME / "cache"
    fields = []
    for w in p.workflows:
        ast, stk = api.stage_outputs(p, w.id, "astrometry"), api.stage_outputs(p, w.id, "stack")
        if ast is None or stk is None or api.stage_outputs(p, w.id, "catalogs") is not None:
            continue
        wcs = WCS(fits.getheader(Path(ast.dir) / "wcs.fits"))
        meta = json.loads((Path(stk.dir) / "drz_meta.json").read_text(encoding="utf-8"))
        Wd, Hd = meta["width"], meta["height"]
        pr = CatalogsParams(**{k: v for k, v in (w.params.get("catalogs") or {}).items()})
        centres, radius = _cells(wcs, Wd, Hd, float(ast.metrics["scale_arcsec"]), pr.cell_deg)
        fields.append({"wf": w.name, "centres": centres, "radius": radius, "p": pr, "wcs": wcs, "W": Wd, "H": Hd})
    report: dict = {"workflows": [f["wf"] for f in fields]}
    if not fields:
        return report
    f0 = fields[0]
    centre = f0["wcs"].pixel_to_world(f0["W"] / 2, f0["H"] / 2)
    reach = 0.5 * np.hypot(f0["W"], f0["H"]) * float(np.abs(np.linalg.det(f0["wcs"].pixel_scale_matrix)) ** 0.5) + 1.5
    pr = f0["p"]
    for label, cols, gf in (("deep", PF.DEEP_COLS, f"<{pr.g_deep}"),
                            ("faint", PF.FAINT_COLS, f"{pr.g_deep}..{CatalogsParams().g_faint}")):
        need = [f for f in fields if label == "deep" or f["p"].g_faint > f["p"].g_deep]
        if not need:
            continue
        parts, cones = [], []
        for fz in (cache / "gaia").glob("*.npz"):
            try:
                z = dict(np.load(fz))
            except (OSError, ValueError):
                continue
            if set(z) != set(cols) or not len(z["Gmag"]):
                continue
            if label == "faint" and float(np.nanmin(z["Gmag"])) < pr.g_deep - 0.05:
                continue
            c = PF._cone_of(z["RA_ICRS"], z["DE_ICRS"])
            if PF._sep(c[0], c[1], centre.ra.deg, centre.dec.deg) > reach + c[2]:
                continue
            parts.append(z)
            cones.append(c)
        if not parts:
            report[label] = "no cached cone near the field"
            continue
        pl = {k: np.concatenate([q[k] for q in parts]) for k in cols}
        key = np.round(pl["RA_ICRS"] * 3.6e5).astype(np.int64) * 10 ** 9 + np.round(pl["DE_ICRS"] * 3.6e5).astype(
            np.int64)
        _, idx = np.unique(key, return_index=True)
        pl = {k: v[idx] for k, v in pl.items()}
        cc = np.array([[c[0], c[1]] for c in cones])
        rr = np.array([c[2] for c in cones])
        made = kept = out = forced = 0
        for fd in need:
            for c in fd["centres"]:
                f = cache / "gaia" / f"{PF.gaia_key(cols, gf, c, fd['radius'])}.npz"
                if f.exists():
                    kept += 1
                    continue
                if not PF.covered(c, fd["radius"], cc, rr, fd):
                    if label != "faint":
                        out += 1
                        continue
                    # the faint list (OLD's, the measures' star list for every image) cut even where the cached
                    # cones do not cover the whole cell: a few faint stars missing at the edge, the same for all
                    # images, rather than a stage waiting on Gaia services that do not answer
                    forced += 1
                sel = PF._sep(pl["RA_ICRS"], pl["DE_ICRS"], c[0], c[1]) <= fd["radius"]
                np.savez(f, **{k: v[sel] for k, v in pl.items()})
                made += 1
        report[label] = {"cached_cones_used": len(parts), "pool_stars": int(len(idx)), "made": made,
                         "already": kept, "left_to_network": out, "cut_partly_covered": forced}
    print(json.dumps({"case": case, "level": level, "prefill": report}), flush=True)
    return report


def run(case: str, level: str, device: str, names: list[str], export: list[str],
        stop_before: str | None = None) -> int:
    from astro_gpu.cli2 import follow
    p, _ = project(case, level)

    def log(msg: str) -> None:
        print(f"{datetime.now():%H:%M:%S} {case} {level} {msg}", flush=True)
    ids = [wf(p, n).id for n in names if wf(p, n) is not None]
    t0 = time.time()
    info = api.running(p)
    status = "up to date"
    if info is None:
        plan = api.plan_workflows(p, ids, device)
        if stop_before:                  # only the stages before it (the catalogue pre-fill goes in between)
            ch = api.chain(p, ids[0])
            cut = ch.index(stop_before)
            plan.jobs = [j for j in plan.jobs if ch.index(j.stage) < cut]
            plan.targets = [t for t in plan.targets if t[1] in ch[:cut]]
        if plan:
            info = api.start_run(p, plan, device)
            log(f"run {info.run_id}: {len(plan)} job(s) on {device}, pid {info.pid}")
    else:
        log(f"attached to the live run {info.run_id}")
    if info is not None:
        status = follow(p, info.offset, info.run_id, info.pid, poll=3.0, echo=lambda s: log(s.strip()))
    api.refresh(p)
    rec_f = root_of(case, level) / "crossed_run.json"
    hist = json.loads(rec_f.read_text(encoding="utf-8")) if rec_f.exists() else {"runs": []}
    r = {"when": datetime.now().isoformat(timespec="seconds"), "device": device, "status": status,
         "configs": names, "wall_s": round(time.time() - t0, 1)}
    for name in export:
        w = wf(p, name)
        if w is None or api.stage_outputs(p, w.id, "render_simple") is None:
            continue
        dst = root_of(case, level) / "export" / f"bench_{name}"
        if dst.is_dir() and any(dst.glob("*.tif")):
            continue
        te = time.time()
        files = api.export_workflow(p, w.id, None, None, {"name": f"bench_{name}", "formats": ["tiff16"]},
                                    device="cpu")
        log(f"exported {name}: {[Path(f).name for f in files]} in {time.time() - te:.0f} s")
    hist["runs"].append(r)
    hist["configs"] = record(p, [w.name for w in p.workflows])
    rec_f.write_text(json.dumps(hist, indent=1, default=str), encoding="utf-8")
    log(f"{status} in {(time.time() - t0) / 60:.1f} min")
    return 0 if status in ("done", "up to date") else 1


# ---------------------------------------------------------------------------------------------- measuring

def _siril_rejection(case: str, level: str) -> float | None:
    """Siril's rejection rate of the lights' stack: the mean of the last three channels' low + high (pre.log)."""
    f = C.siril_dir(case, level, C.norm_variant(case, None)) / "pre.log"
    if not f.exists():
        return None
    m = re.findall(r"Pixel rejection in channel #(\d): ([\d.]+)% - ([\d.]+)%", f.read_text(encoding="utf-8",
                                                                                      errors="replace"))
    if len(m) < 3:
        return None
    return float(np.mean([float(a) + float(b) for _, a, b in m[-3:]]))


def engine_info(p, name: str) -> dict:
    """What the engine says of a configuration's stack: frames kept, rejection, scale, method, durations."""
    w = wf(p, name)
    out: dict = {}
    cal = api.stage_outputs(p, w.id, "calibrate")
    reg = api.stage_outputs(p, w.id, "register")
    stk = api.stage_outputs(p, w.id, "stack")
    if cal is not None:
        out["calibrate_s"] = cal.duration
    if reg is not None:
        out["register_s"] = reg.duration
        out["register_key"] = reg.key
        out["fwhm_moffat_px"] = (reg.metrics or {}).get("fwhm_moffat_px")
        out["fwhm_px"] = (reg.metrics or {}).get("fwhm_px")
    if stk is None:
        return out
    m = stk.metrics or {}
    d = Path(stk.dir)
    meta = json.loads((d / "drz_meta.json").read_text(encoding="utf-8"))
    diag = json.loads((d / "diagnostics.json").read_text(encoding="utf-8")) if (d / "diagnostics.json").exists() else {}
    out.update({"stack_s": stk.duration, "stack_key": stk.key, "frames_used": m.get("frames_used"),
                "rejected_pct": m.get("rejected_pct"), "untested_pct": m.get("untested_pct"),
                "coverage_variation_pct": m.get("coverage_variation_pct"),
                "scale": meta.get("scale"), "pixfrac": meta.get("pixfrac"),
                "method": (meta.get("switches") or {}).get("method", "drizzle")})
    if out["frames_used"] is None:
        out["frames_used"] = meta.get("n") or len(diag.get("frames") or [])
    rj = diag.get("rejection") or {}
    if rj.get("classic"):
        out["classic_rejection"] = rj.get("classic")
    cov = diag.get("coverage") or {}
    if cov.get("rule"):
        out["rule"] = {k: (v.get("value") if isinstance(v, dict) else v) for k, v in cov["rule"].items()}
    sel = diag.get("selection") or {}
    if sel:
        out["selection_kept"] = sel.get("kept")
        counts: dict[str, int] = {}
        for rows in (sel.get("excluded") or {}).values():
            for code in {r.get("code") for r in rows}:
                counts[code] = counts.get(code, 0) + 1
        out["excluded_codes"] = counts
    out["chain_s"] = sum((api.stage_outputs(p, w.id, s).duration or 0)
                         for s in api.chain(p, w.id) if api.stage_outputs(p, w.id, s) is not None)
    return out


def measure_crossed(case: str, level: str, names: list[str], extra_siril: dict[str, str] | None = None) -> dict:
    """``extra_siril``: more Siril stacks to measure on the same sky, {label: its folder's level name} (Siril's
    drizzle: {"Siril drizzle": "express_drizzle"})."""
    import measure as M
    extra = dict(extra_siril or {})
    t0 = time.time()
    sv = C.norm_variant(case, None)
    p, _ = project(case, level)
    dirs: dict[str, dict[str, Path]] = {}
    for n in names:
        w = wf(p, n)
        if w is None:
            continue
        d = {}
        for s in api.chain(p, w.id):
            o = api.stage_outputs(p, w.id, s)
            if o is not None:
                d[s] = Path(o.dir)
        if "render_simple" in d:
            dirs[n] = d
    same: dict[str, str] = {}
    seen: dict[str, str] = {}
    for n, d in dirs.items():                         # a configuration whose stack is another's: measured once
        k = f"{d['stack']}|{d['render_simple']}"
        if k in seen:
            same[n] = seen[k]
        else:
            seen[k] = n
    ours = [n for n in dirs if n not in same]
    first = next((n for n in KEEP_FAINT if n in ours), ours[0])     # the star list: the faint catalogue's

    class Geo(M.Img):
        def __init__(self, name, wcs, cover, H, W):        # noqa: D401 - the geometry of an image, no pixels
            self.name, self.wcs, self.cover, self.H, self.W = name, wcs, cover, H, W
            self.scale = float(np.abs(np.linalg.det(wcs.pixel_scale_matrix)) ** 0.5 * 3600)
            self.pixarea = self.scale ** 2
            self.dx = self.dy = 0.0

    ref = M.load_siril(case, level, "stack_linear", sv)
    geos: dict[str, Geo] = {}
    for n, folder in extra.items():
        im = M.load_siril(case, folder, "stack_linear", sv)
        geos[n] = Geo(n, im.wcs, im.cover, im.H, im.W)
        del im
    for n in ours:
        im = M.load_ours(dirs[n], "stack")
        geos[n] = Geo(n, im.wcs, im.cover, im.H, im.W)
        del im
    common_ref = ref.cover.copy()
    for g in geos.values():
        common_ref &= M.common_cover(ref, g)
    refc = Geo("ref", ref.wcs, common_ref, ref.H, ref.W)
    for g in geos.values():
        g.cover = g.cover & M.common_cover(g, refc)
    ref.cover = common_ref
    print(f"{case} {level}: {len(ours)} stacks + Siril, common sky ({time.time() - t0:.0f} s)", flush=True)
    st = M.gaia(dirs[first], geos[first])
    allg = {SIRIL: ref, **geos}
    inside = []
    for g in allg.values():
        x, y = g.pix(st["ra"], st["dec"])
        inside.append(g.inside(x, y, 40))
    common = np.logical_and.reduce(inside)
    bright = common & (st["g"] < 14) & (st["g"] > 9)
    iso_shape = M.isolated(st, common & (st["g"] < 16.5) & (st["g"] > 8), 40 / 3600, 3.0)

    def load(n: str, which: str = "stack"):
        if n in extra:
            im = M.load_siril(case, extra[n], "stack_linear" if which == "stack" else "color_linear", sv)
        else:
            im = M.load_ours(dirs[n], which)
        if which == "stack":
            im.cover = geos[n].cover
            im.dx, im.dy = geos[n].dx, geos[n].dy
        return im
    shapes = {}
    M.refine_shift(ref, st, bright)
    shapes[SIRIL] = M.star_shapes(ref, st, iso_shape, ref)
    for n in list(extra) + ours:
        im = load(n)
        M.refine_shift(im, st, bright)
        geos[n].dx, geos[n].dy = im.dx, im.dy
        shapes[n] = M.star_shapes(im, st, iso_shape, ref)
        del im
    print(f"stars fitted ({time.time() - t0:.0f} s)", flush=True)
    fwhm_as = max(s.get("fwhm_arcsec", 5) for s in shapes.values())
    gmin = max(s.get("max_g_unsaturated_bright", 10) for s in shapes.values())
    iso_f = M.isolated(st, common & (st["g"] > gmin), 3 * fwhm_as / 3600, 1.5)
    iso_z = M.isolated(st, common & (st["g"] > gmin + 0.3) & (st["g"] < gmin + 3.5), 30 / 3600, 2.5)
    iso_c = M.isolated(st, common & np.isfinite(st["bp"]) & (st["g"] > gmin + 0.3) & (st["g"] < gmin + 5),
                       30 / 3600, 2.5)
    dr = M.drift_direction(dirs[first])
    bs = 4
    maps: dict = {}
    ref_mask = M.star_mask(ref, st, fwhm_as)
    M.normalise_channels(ref, st, iso_z, shapes[SIRIL].get("fwhm_px", 3.0))
    ref_ph = M.photometry(ref, st, iso_f, shapes[SIRIL].get("fwhm_px", 3.0), ref_mask, gmin)
    out: dict = {}
    snrs, fsel = {}, {}
    for n in [SIRIL] + list(extra) + ours:
        fw = shapes[n].get("fwhm_px", 3.0)
        if n == SIRIL:
            im, mask, ph = ref, ref_mask, ref_ph
        else:
            im = load(n)
            mask = M.star_mask(im, st, fwhm_as)
            M.normalise_channels(im, st, iso_z, fw)
            ph = M.photometry(im, st, iso_f, fw, mask, gmin)
        e: dict = {"stars": shapes[n]}
        snr = np.full(len(st["g"]), np.nan)
        snr[ph["fsel"]] = ph["snr"]
        snrs[n], fsel[n] = snr, ph["fsel"]
        bex = bs * ref.scale / im.scale
        b = max(2, int(round(bex)))
        bn = M.box_noise(im, ph["R"], mask, b, bex)
        zp = ph["zp"]
        bn["sb_mag_arcsec2"] = float(zp - 2.5 * np.log10(bn["sigma_box"] / im.pixarea))
        bn["sb_large_mag_arcsec2"] = float(zp - 2.5 * np.log10(
            M.box_noise(im, ph["R"], mask, max(2, int(round(4 * bex))), 4 * bex)["sigma_box"] / im.pixarea))
        sky = float(np.median(ph["B"][im.cover & ~mask]))
        bn["sky_sb"] = float(zp - 2.5 * np.log10(sky / im.pixarea)) if sky > 0 else float("nan")
        bn.update({k: ph[k] for k in ("zp", "zp_stars", "r_det_px", "depth5_mag", "aperture_fraction", "growth",
                                  "depth5_uncorrected_mag")})
        e["noise"] = bn
        maps[n] = M.box_map(im, mask, b, bex, zp)
        ang = M.angle_in(im, geos[first], dr["v"])
        e["streaks"] = M.streaks(im, ph["R"], mask, ph["B"], ang, zp)
        e["coloured_spots"] = M.coloured_spots(im, mask, fw)
        rs = M.residuals(im, mask, fw)
        e["residuals"] = {"trails": len(rs["trails"]), "area_deg2": rs["area_deg2"],
                          "trail_length_arcmin": float(sum(x["length_arcmin"] for x in rs["trails"])),
                          "hot_spots_per_deg2": len(rs["spots"]["hot"]) / max(rs["area_deg2"], 1e-6),
                          "cold_spots_per_deg2": len(rs["spots"]["cold"]) / max(rs["area_deg2"], 1e-6)}
        im2 = M.load_siril(case, level, "color_linear", sv) if n == SIRIL else load(n, "lin")
        im2.dx, im2.dy, im2.cover = im.dx, im.dy, im.cover & im2.cover
        e["colours"] = M.colours(im2, st, iso_c, fw)
        M.normalise_channels(im2, st, iso_z, fw)
        ph2 = M.photometry(im2, st, iso_f, fw, mask, gmin)
        e["flatness"] = M.flatness(im2, mask, ph2["zp"], bn["sky_sb"])
        del im2, ph2
        e["engine"] = (engine_info(p, n) if n not in extra and n != SIRIL
                       else {"rejected_pct": _siril_rejection(case, extra.get(n, level))})
        out[n] = e
        print(f"{n}: depth {bn['depth5_mag']:.2f} noise {bn['sb_mag_arcsec2']:.2f}/{bn['sb_large_mag_arcsec2']:.2f} "
              f"fwhm {shapes[n].get('fwhm_centre_arcsec')} "
              f"spots {e['coloured_spots']['all_per_deg2']:.0f} "
              f"B-R {(e['colours'].get('b_r') or {}).get('scatter_bp_rp')} ({time.time() - t0:.0f} s)", flush=True)
        if n != SIRIL:
            del im, mask, ph
    # the target's SNR: regions chosen on an image independent of the stacks, the same for all (measure.target_measures)
    tm, tinfo = M.target_measures(case, level, sv, maps, SIRIL, st, bright, fwhm_as, bs, ref,
                                  dump=C.BENCH / case / f"target_maps_{level}.npz")
    for n in out:
        out[n]["target"] = M.target_entry(tm[n], tm[SIRIL].get("snr"), tinfo)
    print("target SNR:", {n: out[n]["target"]["snr"] for n in out}, tinfo.get("source"), flush=True)
    common_f = np.logical_and.reduce([fsel[n] for n in out])
    for n in out:
        s = snrs[n][common_f]
        out[n]["faint_stars"] = {"found": int((s >= 5).sum()), "of": int(len(s)),
                                 "g50": M._g50(s >= 5, st["g"][common_f])}
    return {"case": case, "level": level, "images": [SIRIL] + list(extra) + ours, "same_as": same, "first": first,
            "box_arcsec": bs * ref.scale, "gaia_stars": int(len(st["g"])), "drift": {k: (v.tolist() if isinstance(
                v, np.ndarray) else v) for k, v in dr.items()}, "by_image": out, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------------------------- the sheet

def sheet(case: str, level: str, names: list[str]) -> Path:
    """Siril and these configurations side by side, on Siril's framing: the finals, the raw linear stacks shown by
    their own noise, 100 % crops of the target and a corner (finals), and the same corner of the linear stacks."""
    import sheet as SH
    from astropy.io import fits
    from astropy.wcs import WCS
    from PIL import Image, ImageDraw, ImageFont
    from scipy import ndimage as ndi
    man = json.loads((C.BENCH / case / "manifest.json").read_text(encoding="utf-8"))
    sv = C.norm_variant(case, None)
    p, _ = project(case, level)
    s_img, s_wcs, s_H, s_flip = SH._siril_any(case, level, sv)
    SHh, SWw = s_img.shape[:2]
    s_scale = float(np.abs(np.linalg.det(s_wcs.pixel_scale_matrix)) ** 0.5 * 3600)
    finals, lins, labels = [SH.Src(s_img, SH._wcs_map(s_wcs, s_H if s_flip else None))], [], [SIRIL]
    with fits.open(C.siril_dir(case, level, sv) / "stack_linear.fit", memmap=False) as h:
        d = np.asarray(h[0].data, np.float32)
        d = d if d.ndim == 3 else np.repeat(d[None], 3, 0)
        lins.append((d.mean(0), WCS(h[0].header, naxis=2), d))
    o_scale = s_scale
    for n in names:
        w = wf(p, n)
        rs = Path(api.stage_outputs(p, w.id, "render_simple").dir)
        ast = Path(api.stage_outputs(p, w.id, "astrometry").dir)
        stk = Path(api.stage_outputs(p, w.id, "stack").dir)
        x0, x1, y0, y1 = json.loads((rs / "prep_simple.json").read_text(encoding="utf-8"))["crop"]
        img = SH._tif(sorted((root_of(case, level) / "export" / f"bench_{n}").glob("*.tif"))[-1])
        wcs = WCS(fits.getheader(ast / "wcs.fits"), naxis=2)
        finals.append(SH.Src(img, SH._wcs_map(wcs, None, (x0, y0))))
        o_scale = float(np.abs(np.linalg.det(wcs.pixel_scale_matrix)) ** 0.5 * 3600)
        rgb = np.load(stk / "drz_full.npy", mmap_mode="r")
        lins.append((np.asarray(rgb, np.float32).mean(0), wcs, rgb))
        labels.append(n)
    CW, gap, top, lab = 560, 10, 52, 26
    ov_scale = CW / SWw
    ovh = int(SHh * ov_scale)

    def grid(u0, v0, w_, h_, scale):
        GU, GV = np.meshgrid(np.arange(w_, dtype=float), np.arange(h_, dtype=float))
        fx, fy = u0 + GU / scale, v0 + GV / scale
        fits_y = (s_H - 1 - fy) if s_flip else fy
        ra, dec = s_wcs.all_pix2world(fx.ravel(), fits_y.ravel(), 0)
        return ra, dec, GU.shape
    with fits.open(C.siril_dir(case, level, sv) / "color_linear.fit", memmap=False) as h:
        cl = np.asarray(h[0].data, np.float32)
    cl = cl.mean(0) if cl.ndim == 3 else cl
    sm = ndi.median_filter(SH._bin(cl, 16), 7)
    m = np.zeros_like(sm, bool)
    m[int(0.15 * sm.shape[0]):int(0.85 * sm.shape[0]), int(0.15 * sm.shape[1]):int(0.85 * sm.shape[1])] = True
    ty, tx = np.unravel_index(np.argmax(np.where(m, sm, -np.inf)), sm.shape)
    tx, ty = (tx + 0.5) * 16, (ty + 0.5) * 16
    ty = (s_H - 1 - ty) if s_flip else ty
    del cl, sm
    cs = s_scale / o_scale
    half = CW / cs / 2
    rows = []
    ra, dec, shp = grid(0, 0, CW, ovh, ov_scale)
    rows.append(("Images finales (rendus par défaut)", [np.clip(f.sample(ra, dec, shp, 1), 0, 1) for f in finals]))
    ov_lin = []
    for (L, lw, _rgb) in lins:
        lsc = float(np.abs(np.linalg.det(lw.pixel_scale_matrix)) ** 0.5 * 3600)
        f = max(1, int(round(SWw / CW * s_scale / lsc)))
        ov_lin.append(SH.Src(SH._zstretch(SH._bin(L, f), 32), SH._wcs_map(lw, None, (0, 0), f)).sample(ra, dec, shp, 1))
    rows.append(("Piles linéaires, fond étiré par son propre bruit (−2,5σ noir, +5σ blanc)",
                 [np.repeat(o, 3, -1) for o in ov_lin]))
    for title, cx, cy in (("cible : la région la plus brillante", tx, ty),
                          ("coin haut gauche", 0.15 * SWw, 0.15 * SHh)):
        ra, dec, shp = grid(cx - half, cy - half, CW, CW, cs)
        rows.append((f"100 % : {title} (images finales)", [np.clip(f.sample(ra, dec, shp, 1), 0, 1) for f in finals]))
    cx, cy = 0.3 * SWw, 0.3 * SHh
    ra, dec, shp = grid(cx - half, cy - half, CW, CW, cs)
    lin_c = []
    for (L, lw, rgb) in lins:                     # each colour stretched by its own noise: one-colour dots show
        x, y = SH._wcs_map(lw)(ra, dec)
        x0, x1 = int(max(0, np.nanmin(x) - 40)), int(min(L.shape[1], np.nanmax(x) + 40))
        y0, y1 = int(max(0, np.nanmin(y) - 40)), int(min(L.shape[0], np.nanmax(y) + 40))
        sub = np.concatenate([SH._zstretch(np.asarray(rgb[c, y0:y1, x0:x1], np.float32), 16) for c in range(3)], -1)
        lin_c.append(SH.Src(sub, SH._wcs_map(lw, None, (x0, y0))).sample(ra, dec, shp, 1))
    rows.append(("100 % : un coin de ciel, pile linéaire, chaque couleur étirée par son bruit (points d'une couleur, "
                 "grain)", lin_c))
    n_c = len(finals)
    Wt = n_c * CW + (n_c + 1) * gap
    Ht = top + sum(lab + r[1][0].shape[0] + gap for r in rows) + 30
    out_img = Image.new("RGB", (Wt, Ht), (24, 24, 24))
    dr = ImageDraw.Draw(out_img)
    f1, f2 = ImageFont.truetype(SH.FONT, 22), ImageFont.truetype(SH.FONT, 16)
    lv = man["levels"][level]
    dr.text((gap, 12), f"{case} · {level} · {lv['count']} poses · {man['camera_label']} · même zone du ciel",
            fill=(235, 235, 235), font=f1)
    y = top
    for title, ims in rows:
        dr.text((gap, y + 3), title, fill=(220, 220, 220), font=f2)
        y += lab
        for i, a in enumerate(ims):
            x = gap + i * (CW + gap)
            out_img.paste(Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)), (x, y))
            if title.startswith("Images finales"):
                who = "Siril 1.4" if labels[i] == SIRIL else f"astro_gpu · {labels[i]}"
                dr.text((x + 6, y + 6), who, fill=(255, 220, 120) if i == 0 else (120, 220, 255), font=f1)
                if labels[i] != SIRIL:
                    dr.text((x + 6, y + 32), LABELS.get(labels[i], ""), fill=(230, 230, 230), font=f2)
        y += ims[0].shape[0] + gap
    out = C.BENCH / case / f"crossed_sheet_{level}.jpg"
    out_img.save(out, quality=88)
    print(f"sheet {out}")
    return out


# ---------------------------------------------------------------------------------------------- the report

def _fr(v: float | None, fmt: str, sign: bool = False) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    s = (("{:+" + fmt[2:]) if sign else fmt).format(v)
    return s.replace("-", "−").replace(".", ",")


def _get(e: dict, path: str):
    cur = e
    for k in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


#: the measures of the report: key, label, path in a measure, format, kind of delta ("mag": difference, "rel": %,
#: "abs": difference), higher is better (None: no better or worse), tolerance (the run-to-run noise), in the
#: criterion
MEASURES = [
    ("depth", "Profondeur : étoile détectable à 5σ (magnitude G ; plus haut = plus profond)", "noise.depth5_mag",
     "{:.2f}", "mag", True, 0.04, True),
    ("fine", "Bruit du fond, petite échelle (mag/arcsec² ; plus haut = moins de bruit)", "noise.sb_mag_arcsec2",
     "{:.2f}", "mag", True, 0.04, True),
    ("large", "Bruit du fond, échelle 4 fois plus grande (mag/arcsec²)", "noise.sb_large_mag_arcsec2", "{:.2f}",
     "mag", True, 0.04, True),
    ("snr", "Signal/bruit sur la cible", "target.snr", "{:.1f}", "rel", True, 0.03, True),
    ("faint", "Étoiles faibles trouvées à 5σ (les mêmes étoiles Gaia pour toutes)", "faint_stars.found", "{:.0f}",
     "rel", True, 0.03, True),
    ("fwhm", "Finesse des étoiles : FWHM au centre (″ ; plus bas = plus fines)", "stars.fwhm_centre_arcsec", "{:.2f}",
     "rel", False, 0.03, True),
    ("round", "Rondeur des étoiles au centre (1 = rondes)", "stars.roundness_centre", "{:.2f}", "abs", True, 0.02,
     True),
    ("dots", "Points d'une seule couleur (pixels chauds restants, par degré²)", "coloured_spots.all_per_deg2",
     "{:.0f}", "dots", False, 0.15, True),
    ("reject", "Taux de rejet (% des échantillons ; repère de Siril : 0,1 à 0,5 %)", "engine.rejected_pct", "{:.3f}",
     "abs", None, 0.0, False),
    ("br", "Couleurs des étoiles : erreur B−R face à Gaia (BP−RP ; plus bas = plus fidèle)",
     "colours.b_r.scatter_bp_rp", "{:.3f}", "abs", False, 0.005, True),
    ("gr", "Couleurs des étoiles : erreur V−R face à Gaia (BP−RP)", "colours.g_r.scatter_bp_rp", "{:.3f}", "abs", False,
     0.005, True),
    ("flat", "Planéité du fond après dégradé (mag/arcsec² ; plus haut = plus plat)", "flatness.spread_sb", "{:.2f}",
     "mag", True, 0.10, True),
    ("streak", "Traînées le long de la dérive (« walking noise » ; 1 = aucune)", "streaks.ratio_along", "{:.2f}", "abs",
     False, 0.03, False),
    ("frames", "Poses gardées", "engine.frames_used", "{:.0f}", "abs", None, 0.0, False),
    ("time", "Durée de l'empilement seul (min ; machine partagée)", "engine.stack_s", "{:.1f}", "abs", None, 0.0,
     False),
    ("time_pre", "Durée calibration + alignement + empilement (min ; Siril : conversion comprise, 8 cœurs)",
     "engine.pre_s", "{:.1f}", "abs", None, 0.0, False),
]


def load_level(level: str) -> dict[str, dict]:
    out = {}
    for case in C.CASES:
        f = C.BENCH / case / f"crossed_{level}.json"
        if f.exists():
            out[case] = json.loads(f.read_text(encoding="utf-8"))
    return out


def value(m: dict, name: str, path: str, case: str = "", level: str = ""):
    """A configuration's value (one measured once for another: that one's), Siril's frames from its log."""
    n = m["same_as"].get(name, name)
    e = m["by_image"].get(n)
    if e is None:
        return None
    if path == "engine.frames_used" and n == SIRIL:
        import report as R
        return R.siril_frames(case, level, "")          # "" = the case's main variant's folder
    if path == "engine.stack_s":
        if n == SIRIL:
            return None
        v = _get(e, path)
        return v / 60 if v else None
    if path == "engine.pre_s":                          # calibration + registration + stacking
        if n == SIRIL:
            sj = C.siril_dir(case, level, "") / "siril.json"
            if sj.exists():
                return (json.loads(sj.read_text(encoding="utf-8")).get("pre_s") or 0) / 60
            return None
        eg = e.get("engine") or {}
        parts = [eg.get(k) for k in ("calibrate_s", "register_s", "stack_s")]
        return sum(parts) / 60 if all(x is not None for x in parts) else None
    return _get(e, path)


def delta(v, ref, kind: str):
    if v is None or ref is None or not (math.isfinite(v) and math.isfinite(ref)):
        return None
    if kind == "rel":
        return 100 * (v / ref - 1) if ref else None
    return v - ref


#: the run-to-run noise measured by OLD-bis against OLD (the largest gap over the cases), per measure: the criterion's
#: tolerance is the larger of it and the nominal one (MEASURES); set by ``report`` (``measured_noise``)
_NOISE: dict[str, float] = {}


def measured_noise(level: str, data: dict[str, dict]) -> dict[str, float]:
    out = {}
    for key, _l, path, _f, kind, higher, _t, _c in MEASURES:
        if higher is None:
            continue
        gaps = []
        for c, m in data.items():
            v, o = value(m, "OLD-bis", path, c, level), value(m, "OLD", path, c, level)
            if v is None or o is None or not (math.isfinite(v) and math.isfinite(o)):
                continue
            gaps.append(abs(v / o - 1) if kind == "rel" and o else abs(v - o))
        if gaps:
            out[key] = float(max(gaps))
    return out


def tol_of(key: str, tol: float) -> float:
    return max(tol, _NOISE.get(key, 0.0))


def judge(v, ref, kind: str, higher, tol: float, key: str = "") -> int:
    """+1 better, -1 worse, 0 equal (within the tolerance, at least the measured run-to-run noise of the measure)."""
    if higher is None or v is None or ref is None or not (math.isfinite(v) and math.isfinite(ref)):
        return 0
    if kind == "dots":
        if abs(v - ref) <= max(3.0, _NOISE.get(key, 0.0)) or abs(v - ref) <= tol * max(abs(ref), 1e-9):
            return 0
        return 1 if (v > ref) == higher else -1
    d = (v - ref) / abs(ref) if kind == "rel" and ref else v - ref
    if abs(d) <= tol_of(key, tol) + 1e-9:
        return 0
    return 1 if (d > 0) == higher else -1


def dfmt(d, kind: str, fmt: str) -> str:
    if d is None:
        return "–"
    if kind == "rel":
        return _fr(d, "{:.0f}", True) + " %"
    if kind == "dots":
        return _fr(d, "{:.0f}", True)
    return _fr(d, fmt, True)


def table(level: str, data: dict[str, dict], key: str, names: list[str]) -> str:
    mk = next(x for x in MEASURES if x[0] == key)
    _, label, path, fmt, kind, higher, tol, _ = mk
    cases = list(data)
    head = f"**{label}**\n\n"
    head += "| | " + " | ".join(cases) + " |\n|---|" + "---|" * len(cases) + "\n"
    rows = []
    for n in [SIRIL] + names:
        cells = []
        for c in cases:
            m = data[c]
            v = value(m, n, path, c, level)
            if n == SIRIL:
                cells.append(_fr(v, fmt))
                continue
            o = value(m, "OLD", path, c, level)
            s = value(m, SIRIL, path, c, level)
            if v is None:
                cells.append("–")
                continue
            if n == "OLD":
                cells.append(f"{_fr(v, fmt)} (– / {dfmt(delta(v, s, kind), kind, fmt)})")
                continue
            jo = judge(v, o, kind, higher, tol, key)
            mark = " ▲" if jo > 0 else " ▼" if jo < 0 else ""
            cells.append(f"{_fr(v, fmt)} ({dfmt(delta(v, o, kind), kind, fmt)} / "
                         f"{dfmt(delta(v, s, kind), kind, fmt)}){mark}")
        lab = "Siril" if n == SIRIL else f"{n}"
        rows.append(f"| {lab} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


#: cells left out of the criterion: M8's one-colour dots (its field is full of red nebulosity and faint stars: the
#: count says nothing there, BENCH.md)
NOT_IN_CRITERION = {("dots", "m8")}


def verdicts(level: str, data: dict[str, dict], names: list[str]) -> dict[str, dict]:
    """Per configuration: the cells worse than OLD beyond the run-to-run noise (the criterion), better than OLD, and
    better / worse than Siril."""
    out = {}
    for n in names:
        if n == "OLD":
            continue
        worse, better, s_better, s_worse = [], [], [], []
        for key, _label, path, _fmt, kind, higher, tol, crit in MEASURES:
            if not crit:
                continue
            for c, m in data.items():
                if (key, c) in NOT_IN_CRITERION:
                    continue
                v = value(m, n, path, c, level)
                o = value(m, "OLD", path, c, level)
                s = value(m, SIRIL, path, c, level)
                if v is None:
                    continue
                jo, js = judge(v, o, kind, higher, tol, key), judge(v, s, kind, higher, tol, key)
                tag = f"{key}:{c}"
                (worse if jo < 0 else better if jo > 0 else []).append(tag)
                (s_better if js > 0 else s_worse if js < 0 else []).append(tag)
        out[n] = {"worse_than_old": worse, "better_than_old": better, "better_than_siril": s_better,
                  "worse_than_siril": s_worse}
    return out


#: the measures of the summaries
SUMMARY_KEYS = ["depth", "fine", "large", "snr", "faint", "fwhm", "round", "dots", "br", "gr", "flat"]
SHORT = {"depth": "Profondeur", "fine": "Bruit fin", "large": "Bruit large", "snr": "S/B cible",
         "faint": "Étoiles faibles", "fwhm": "FWHM", "round": "Rondeur", "dots": "Points 1 couleur",
         "br": "Couleur B−R", "gr": "Couleur V−R", "flat": "Planéité", "streak": "Traînées (dérive)", "frames": "Poses",
         "reject": "Rejet %", "time": "Empilement (min)"}


def summary(level: str, data: dict[str, dict], pairs: list[tuple[str, str, str]]) -> str:
    """One row per (label, config, against): over the cases, the median of the delta config - against and its range,
    and how many cases are better (▲) or worse (▼) beyond the run-to-run noise."""
    head = "| | " + " | ".join(SHORT[k] for k in SUMMARY_KEYS) + " |\n|---|" + "---|" * len(SUMMARY_KEYS) + "\n"
    rows = []
    for label, n, against in pairs:
        cells = []
        for key in SUMMARY_KEYS:
            _, _lab, path, fmt, kind, higher, tol, _ = next(x for x in MEASURES if x[0] == key)
            ds, up, down = [], 0, 0
            for c, m in data.items():
                if (key, c) in NOT_IN_CRITERION:
                    continue
                v, o = value(m, n, path, c, level), value(m, against, path, c, level)
                d = delta(v, o, kind)
                if d is None:
                    continue
                ds.append(d)
                j = judge(v, o, kind, higher, tol, key)
                up, down = up + (j > 0), down + (j < 0)
            if not ds:
                cells.append("–")
                continue
            f = "{:.0f}" if kind in ("rel", "dots") else fmt
            unit = " %" if kind == "rel" else ""
            marks = ("" if not up else f" {up}▲") + ("" if not down else f" {down}▼")
            cells.append(f"{_fr(float(np.median(ds)), f, True)}{unit} [{_fr(min(ds), f, True)} ; "
                         f"{_fr(max(ds), f, True)}]{marks}")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def case_table(level: str, data: dict[str, dict], n: str) -> str:
    """One configuration per case: its value, its gap to OLD and to Siril, for the main measures."""
    cases = list(data)
    head = "| | " + " | ".join(cases) + " |\n|---|" + "---|" * len(cases) + "\n"
    rows = []
    for key in SUMMARY_KEYS + ["streak", "frames", "reject", "time"]:
        _, _lab, path, fmt, kind, higher, tol, _ = next(x for x in MEASURES if x[0] == key)
        cells = []
        for c in cases:
            m = data[c]
            v, o, s = value(m, n, path, c, level), value(m, "OLD", path, c, level), value(m, SIRIL, path, c, level)
            if v is None:
                cells.append("–")
                continue
            jo, js = judge(v, o, kind, higher, tol, key), judge(v, s, kind, higher, tol, key)
            mo = " ▲" if jo > 0 else " ▼" if jo < 0 else ""
            ms = " ▲" if js > 0 else " ▼" if js < 0 else ""
            cells.append(f"{_fr(v, fmt)} · OLD {dfmt(delta(v, o, kind), kind, fmt)}{mo} · Siril "
                         f"{dfmt(delta(v, s, kind), kind, fmt)}{ms}")
        rows.append(f"| {SHORT[key]} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def durations(level: str, data: dict[str, dict], names: list[str]) -> str:
    """Minutes per case: Siril's preparation (conversion, calibration, registration, stacking, 8 cores) and, per
    configuration, our calibration + registration + stacking (the stacking alone in brackets)."""
    head = "| Cas | Siril (préparation) | " + " | ".join(names) + " |\n|---|---|" + "---|" * len(names) + "\n"
    rows = []
    for c, m in data.items():
        cells = [_fr(value(m, SIRIL, "engine.pre_s", c, level), "{:.1f}")]
        for n in names:
            pre, stk = value(m, n, "engine.pre_s", c, level), value(m, n, "engine.stack_s", c, level)
            cells.append("–" if pre is None else f"{_fr(pre, '{:.1f}')} ({_fr(stk, '{:.1f}')})")
        rows.append(f"| {c} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def report(levels: list[str]) -> int:
    docs = C.REPO / "docs" / "pilot" / "bench"
    docs.mkdir(parents=True, exist_ok=True)
    md = C.REPO / "docs" / "pilot" / "BENCH.md"
    text = md.read_text(encoding="utf-8")
    _NOISE.clear()
    _NOISE.update(measured_noise("express", load_level("express")))    # Base has no OLD-bis: Express's noise
    print("run-to-run noise (OLD-bis against OLD, largest gap):", json.dumps({k: round(v, 4) for k, v in
                                                                            _NOISE.items()}))
    for level in levels:
        data = load_level(level)
        if not data:
            print(f"{level}: no crossed measures")
            continue
        names = []
        for m in data.values():
            for n in m["images"] + list(m["same_as"]):
                if n != SIRIL and n not in names:
                    names.append(n)
        order = [n for n in list(configs("classic")) if n in names] + [n for n in names if n not in configs("classic")]
        parts = [table(level, data, k, order) for k, *_ in MEASURES]
        v = verdicts(level, data, order)
        body = "\n".join(parts)
        (docs / f"crossed_{level}.md").write_text(body, encoding="utf-8")
        alone = [(k, k, "OLD") for k in order if re.fullmatch(r"S\d+b?|OLD-bis|V2|V2-auto|V2-C|V2-T", k)]
        within = [(f"{k} (V2 − V2-{k})", "V2", f"V2-{k}") for k in CHANGES if f"V2-{k}" in order]
        extra = {"alone": summary(level, data, alone), "within": summary(level, data, within),
                 "durations": durations(level, data, [n for n in ("OLD", "S1", "V2", "V2-C", "S7") if n in order])}
        for n in order:
            extra[f"case:{n}"] = case_table(level, data, n)
        (docs / f"crossed_{level}_summaries.json").write_text(json.dumps(extra, indent=1, ensure_ascii=False),
                                                               encoding="utf-8")
        for k, t in extra.items():
            text = re.sub(rf"<!-- crossed-{re.escape(k)}:{level} -->.*?<!-- /crossed -->",
                          lambda _m, k=k, t=t: f"<!-- crossed-{k}:{level} -->\n{t}<!-- /crossed -->", text,
                          flags=re.S)
        (docs / f"crossed_{level}_verdicts.json").write_text(json.dumps(v, indent=1), encoding="utf-8")
        for n, x in v.items():
            print(f"{level} {n}: worse than OLD {len(x['worse_than_old'])} {x['worse_than_old']}; better than OLD "
                  f"{len(x['better_than_old'])}; better / worse than Siril {len(x['better_than_siril'])} / "
                  f"{len(x['worse_than_siril'])}")
        text = re.sub(rf"<!-- crossed:{level} -->.*?<!-- /crossed -->",
                      lambda _m: f"<!-- crossed:{level} -->\n{body}<!-- /crossed -->", text, flags=re.S)
        for case in data:
            sh = C.BENCH / case / f"crossed_sheet_{level}.jpg"
            if sh.exists():
                shutil.copy2(sh, docs / f"crossed_{case}_{level}.jpg")
    md.write_text(text, encoding="utf-8")
    print(f"written {docs} and the crossed blocks of {md}")
    return 0


# ---------------------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "run", "phased", "prefill", "measure", "sheet", "report", "rule"])
    ap.add_argument("--stop-before", help="run only the stages before this one (e.g. catalogs)")
    ap.add_argument("case", nargs="?")
    ap.add_argument("level", nargs="?", default="express")
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", help="measure: the name of the json (default crossed_<level>)")
    ap.add_argument("--siril-drizzle", action="store_true", help="measure: Siril's drizzle stack too")
    ap.add_argument("--export", nargs="*", default=["OLD", "V2", "V2-auto"])
    a = ap.parse_args()
    if a.cmd == "report":
        return report([a.case] if a.case else ["express", "base"])
    names = a.configs or EXPRESS
    if a.cmd == "setup":
        setup(a.case, a.level, names)
        return 0
    if a.cmd == "rule":
        p, _ = project(a.case, a.level)
        print(a.case, a.level, rule_method(p))
        return 0
    if a.cmd == "run":
        return run(a.case, a.level, a.device, names, a.export, a.stop_before)
    if a.cmd == "prefill":
        prefill(a.case, a.level)
        return 0
    if a.cmd == "phased":             # to the astrometry, the cones cut from the cache, then the rest
        r1 = run(a.case, a.level, a.device, names, [], "catalogs")
        prefill(a.case, a.level)
        r2 = run(a.case, a.level, a.device, names, a.export)
        return r1 or r2
    if a.cmd == "measure":
        r = measure_crossed(a.case, a.level, names,
                            {"Siril drizzle": f"{a.level}_drizzle"} if a.siril_drizzle else None)
        out = C.BENCH / a.case / f"{a.out or 'crossed_' + a.level}.json"
        out.write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")
        print(f"written {out} ({r['seconds']:.0f} s)")
        return 0
    if a.cmd == "sheet":
        sheet(a.case, a.level, a.configs or ["OLD", "V2"])
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
