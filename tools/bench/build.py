"""Build the quality bench: choose and copy the frames of each case, write its manifest, prepare the bench homes.

    python tools/bench/build.py [CASE ...]          (every case by default; bench.json keeps the others' lines)

For each case (tools/bench/cases.py):
- the lights of each level are chosen evenly over the session (sorted by file name: 30 for Express, 100 for Base, all
  of them when there are fewer) and copied to <case>/data/lights_<level>/;
- every calibration frame of the source folders (darks, flats, bias) is copied to <case>/data/<kind>/; a variant's
  own frames (rosette: the darks of B and C) to <case>/data/<kind>_<variant>/, listed under "variants";
- <case>/manifest.json lists the files with their sizes, the camera read from the headers, and the calibration the
  case expects.
Both programs then read exactly these copies. A file already copied with the same size is not copied again.

The bench homes:
- _home/.astro_gpu (our engine's ASTRO_GPU_HOME): the user's camera profiles copied, plus a read-only copy of the
  catalogue items of the user's cache that do not depend on the field (XP spectra by Gaia id, the blind-solve quads,
  Hipparcos cones). The Gaia cones are fetched by the first run of each case and level (their cache key is the exact
  cell centre, which moves with the stack's astrometric solution).
- _siril/config.ini: a copy of the user's Siril settings, for siril-cli -i (Siril never writes the user's own).
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as C  # noqa: E402

C.engine_home(profile=False)                           # before astro_gpu is imported (nothing is written by it here)
sys.path.insert(0, str(C.REPO))
C.utf8_console()


def frames(folder: str, ext: tuple[str, ...]) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in ext)


def pick(files: list[Path], n: int) -> list[Path]:
    """n files evenly spread over the list (all of them when there are fewer)."""
    if n >= len(files):
        return list(files)
    return [files[round(i * (len(files) - 1) / (n - 1))] for i in range(n)]


def copy(files: list[Path], dst: Path) -> list[dict]:
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for f in files:
        t = dst / f.name
        if not t.exists() or t.stat().st_size != f.stat().st_size:
            shutil.copy2(f, t)
        out.append({"name": f.name, "size": f.stat().st_size, "source": f.as_posix()})
    # anything else in the folder would be read by Siril's convert: refuse it
    extra = sorted(set(p.name for p in dst.iterdir()) - {f.name for f in files})
    if extra:
        raise SystemExit(f"{dst} holds files that are not in the frozen list: {extra[:5]}")
    return out


def info(path: Path) -> dict:
    from astro_gpu.io.frames import read_info
    i = read_info(path)
    return {"camera": i.camera, "width": i.width, "height": i.height, "cfa": i.cfa, "black": i.black,
            "white": i.white, "exposure_s": i.exposure, "iso_or_gain": i.iso, "time": i.time,
            "focal_mm_header": i.focal_mm, "pixel_um_header": i.pixel_um, "temperature": i.temperature}


def expected_calibration(c: dict, black: float) -> str:
    have = [k for k in C.KINDS if c.get(k)]
    if not have:
        return "none: lights only (debayer, register, stack)"
    parts = []
    if "bias" in have:
        parts.append("master bias (offset) subtracted from flats and lights")
    else:
        parts.append(f"no bias frames: Siril uses a synthetic offset at the raw black level ({black:g} ADU)")
    if "darks" in have:
        parts.append("master dark subtracted from the lights (with its bias), hot pixels from the dark")
    if "flats" in have:
        parts.append("master flat (bias-subtracted) divides the lights")
    return "; ".join(parts)


def build_case(name: str) -> dict:
    c = C.CASES[name]
    root = C.BENCH / name
    lights = frames(c["lights"], c["ext"])
    man: dict = {"case": name, "target": c["target"], "ra": c["ra"], "dec": c["dec"], "centre_from": c["centre_from"],
                 "focal_mm": c["focal"], "pixel_um": c["pixel"], "camera_label": c["camera"], "scope": c["scope"],
                 "note": c.get("note", ""), "source_lights": c["lights"], "source_lights_count": len(lights),
                 "levels": {}, "calibration": {}, "built": datetime.now().isoformat(timespec="seconds")}
    chosen_prev: list[str] | None = None
    prev_level = None
    for level, n in C.LEVELS.items():
        sel = pick(lights, n)
        names = [p.name for p in sel]
        entry: dict = {"count": len(sel), "folder": C.lights_dir(name, level).as_posix()}
        if chosen_prev is not None and names == chosen_prev:
            entry["same_as"] = prev_level                    # m8: 15 lights, both levels are the same
            entry["folder"] = C.lights_dir(name, prev_level).as_posix()
        else:
            entry["files"] = copy(sel, C.lights_dir(name, level))
            (root / f"lights_{level}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        man["levels"][level] = entry
        chosen_prev, prev_level = names, level
        print(f"{name}: {level} {len(sel)} lights of {len(lights)}" + (f" (= {entry['same_as']})"
                                                                         if "same_as" in entry else ""), flush=True)
    for kind in C.KINDS:
        if not c.get(kind):
            continue
        fl = frames(c[kind], c["ext"])
        man["calibration"][kind] = {"count": len(fl), "folder": C.kind_dir(name, kind).as_posix(),
                                    "files": copy(fl, C.kind_dir(name, kind))}
        (root / f"{kind}.txt").write_text("\n".join(p.name for p in fl) + "\n", encoding="utf-8")
        print(f"{name}: {len(fl)} {kind}", flush=True)
    first = C.lights_dir(name, "express") / man["levels"]["express"]["files"][0]["name"]
    man["camera"] = info(first)
    man["calibration_info"] = {k: info(C.kind_dir(name, k) / v["files"][0]["name"])
                               for k, v in man["calibration"].items()}
    man["expected_calibration"] = expected_calibration(c, man["camera"]["black"])
    # variants: the same lights, each with its own calibration frames (data/<kind>_<variant>)
    for vname, v in (c.get("variants") or {}).items():
        ve: dict = {"label": v["label"], "short": v.get("short", ""), "levels": list(C.variant_levels(name, vname)),
                    "siril": C.variant_has_siril(name, vname), "calibration": {}, "calibration_info": {}}
        for kind in C.KINDS:
            if not v.get(kind):
                continue
            fl = frames(v[kind], c["ext"])
            dst = C.data_dir(name) / f"{kind}_{vname}"
            ve["calibration"][kind] = {"count": len(fl), "folder": dst.as_posix(), "files": copy(fl, dst)}
            ve["calibration_info"][kind] = info(dst / fl[0].name)
            (root / f"{kind}_{vname}.txt").write_text("\n".join(p.name for p in fl) + "\n", encoding="utf-8")
            print(f"{name} {vname}: {len(fl)} {kind}", flush=True)
        ve["expected_calibration"] = expected_calibration({**c, **v}, man["camera"]["black"])
        man.setdefault("variants", {})[vname] = ve
    if c.get("main"):
        man["main_variant"] = c["main"]            # its projects are the case's main ones: ours_<level>, siril_<level>
    man["bytes"] = sum(f["size"] for lv in man["levels"].values() for f in lv.get("files", [])) + \
        sum(f["size"] for k in man["calibration"].values() for f in k["files"]) + \
        sum(f["size"] for v in (man.get("variants") or {}).values() for k in v["calibration"].values()
            for f in k["files"])
    (root / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


def homes() -> None:
    """The bench home of our runs and the Siril settings copy (the user's are only read)."""
    prof = C.AG_HOME / "camera_profiles"
    prof.mkdir(parents=True, exist_ok=True)
    for f in sorted((C.USER_AG / "camera_profiles").iterdir()):
        if f.is_file() and not (prof / f.name).exists():
            shutil.copy2(f, prof / f.name)
    cache = C.AG_HOME / "cache"
    src = C.USER_AG / "cache"
    for item in ("quads_hip_k8.npz", "quads_tyc_k7.npz", "hip_quads_v9.5.npz"):
        if (src / item).is_file() and not (cache / item).exists():
            cache.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / item, cache / item)
    for sub in ("xp", "hipparcos"):
        if (src / sub).is_dir():
            d = cache / sub
            d.mkdir(parents=True, exist_ok=True)
            have = {p.name for p in d.iterdir()}
            for f in (src / sub).iterdir():
                if f.name not in have:
                    shutil.copy2(f, d / f.name)
    print(f"bench home {C.AG_HOME}: profiles {sorted(p.name for p in prof.iterdir())}; cache "
          f"xp {len(list((cache / 'xp').iterdir()))} spectra", flush=True)
    C.SIRIL_CFG.parent.mkdir(parents=True, exist_ok=True)
    if not C.SIRIL_CFG.exists() and C.SIRIL_CFG_USER.is_file():
        shutil.copy2(C.SIRIL_CFG_USER, C.SIRIL_CFG)
        print(f"Siril settings copied to {C.SIRIL_CFG}")


def main() -> int:
    names = sys.argv[1:] or list(C.CASES)
    homes()
    bj = C.BENCH / "bench.json"
    old = json.loads(bj.read_text(encoding="utf-8")) if bj.exists() else {}
    summary = dict(old.get("cases") or {})           # building one case keeps the others' lines
    for n in names:
        m = build_case(n)
        summary[n] = {lv: e["count"] for lv, e in m["levels"].items()}
        summary[n].update({k: v["count"] for k, v in m["calibration"].items()})
        for vname, v in (m.get("variants") or {}).items():
            summary[n].setdefault("variants", {})[vname] = {
                "short": v["short"], "levels": v["levels"], "siril": v["siril"],
                **{k: e["count"] for k, e in v["calibration"].items()}}
        summary[n]["GB"] = round(m["bytes"] / 1e9, 2)
        print(f"{n}: {summary[n]}; calibration: {m['expected_calibration']}", flush=True)
    bj.write_text(json.dumps({"built": old.get("built") or datetime.now().isoformat(timespec="seconds"),
                              "updated": datetime.now().isoformat(timespec="seconds"),
                              "cases": {k: summary[k] for k in C.CASES if k in summary}}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
