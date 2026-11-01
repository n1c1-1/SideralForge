"""The Siril reference of the bench: Siril 1.4 (siril-cli) on the frozen frames of a case and level.

    python tools/bench/siril/run_siril.py CASE LEVEL [--variant V] [--post-only] [--threads 8]

Writes F:/astro_essais/bench/<case>/siril_<level>/ (siril_<level>_<V> for a variant of a case that has some):
    pre.ssf, post.ssf         the two scripts (pre-processing up to the stack; post: astrometry, background, colour,
                              stretch), pre.log, post.log
    stack_linear.fit          the stack (32-bit, linear, plate-solved: it carries Siril's WCS)
    bkg_linear.fit            after the background extraction (polynomial, degree 2)
    color_linear.fit          after the colour calibration (SPCC or PCC): the linear end of the pipeline
    final.tif, final.png      linked autostretch: the final (TIFF 16 bits, PNG)
    siril.json                durations, colour method, the commands, whether every step succeeded
The recipe in plain words is in docs/pilot/BENCH.md (§ La recette Siril). Siril reads a copy of the user's settings
(-i, _siril/config.ini), never the user's own file.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cases as C  # noqa: E402

C.utf8_console()

#: SPCC where Siril's database has the sensor; PCC otherwise (why: in BENCH.md)
COLOUR = {
    "heart": ("pcc", "Sony A7S (astro-modified): no curve of this sensor in Siril's SPCC database, and the "
                     "modification changes the red response anyway"),
    "iris": ("pcc", "Sony A7S (astro-modified): no curve of this sensor in Siril's SPCC database, and the "
                    "modification changes the red response anyway"),
    "m51": ("pcc", "Canon EOS 6D: not in Siril's SPCC database (only older Canons are)"),
    "m8": ("spcc", "Sony IMX571 (the ASI2600MC's sensor) is in the database; filter assumed UV/IR block (the "
                   "camera has a plain AR window, broadband imaging adds an IR-cut)"),
    "seestar": ("spcc", "ZWO Seestar S50 is in the database; the frames say FILTER=IRCUT: UV/IR block"),
    "rosette": ("pcc", "Sony A7S (astro-modified): no curve of this sensor in Siril's SPCC database, and the "
                       "modification changes the red response anyway"),
}
#: flats without darks (rosette A): Siril's cosmetic correction of the lights themselves, as a Siril user without darks
#: does it (Image processing > Cosmetic correction on the sequence, CFA): cold 3 sigma, hot 3 sigma, before the
#: calibration (on the raw mosaic, so that a hot pixel is not spread by the demosaicing)
COSME = (3, 3)
SPCC_ARGS = {
    "m8": '"-oscsensor=Sony IMX571" "-oscfilter=UV/IR Block"',
    "seestar": '"-oscsensor=ZWO Seestar S50" "-oscfilter=UV/IR Block"',
}


def q(p: Path | str) -> str:
    return str(p).replace("\\", "/")


def pre_script(case: str, level: str, man: dict, threads: int, variant: str = "") -> str:
    lv = man["levels"][level]
    lights = Path(lv["folder"])
    cal = C.calibration(man, variant)
    wk = C.siril_dir(case, level, variant).name            # the work folder, next to data/ (../../<wk> from a kind)
    L = ["requires 1.3.4", f"setcpu {threads}", "setmem 0.4"]
    seestar = not cal                                     # no calibration frames: the Seestar script
    if "bias" in cal:
        L += [f"cd {q(cal['bias']['folder'])}", f"convert bias -out=../../{wk}/process",
              f"cd ../../{wk}/process", "stack bias rej w 3 3 -nonorm -out=../masters/bias_stacked",
              "cd .."]
        bias = "-bias=../masters/bias_stacked"
    else:
        bias = f'-bias="={man["camera"]["black"]:g}"'      # synthetic offset at the raw black level
    if "flats" in cal:
        L += [f"cd {q(cal['flats']['folder'])}", f"convert flat -out=../../{wk}/process",
              f"cd ../../{wk}/process", f"calibrate flat {bias}",
              "stack pp_flat rej w 3 3 -norm=mul -out=../masters/pp_flat_stacked", "cd .."]
    if "darks" in cal:
        L += [f"cd {q(cal['darks']['folder'])}", f"convert dark -out=../../{wk}/process",
              f"cd ../../{wk}/process", "stack dark rej w 3 3 -nonorm -out=../masters/dark_stacked",
              "cd .."]
    L += [f"cd {q(lights)}", f"convert light -out=../../{wk}/process", f"cd ../../{wk}/process"]
    if seestar:
        L += ["calibrate light -debayer", "register pp_light -2pass", "seqapplyreg pp_light -filter-round=2.5k"]
        seq = "r_pp_light"
    else:
        opts, src = [], "light"
        if "darks" in cal:
            opts += ["-dark=../masters/dark_stacked", "-cc=dark"]
        else:
            # no darks: the offset is subtracted from the lights too, and the hot pixels are found in the lights
            L += [f"seqfind_cosme_cfa light {COSME[0]} {COSME[1]}"]
            src = "cc_light"
            opts += [bias]
        if "flats" in cal:
            opts += ["-flat=../masters/pp_flat_stacked"]
        L += [f"calibrate {src} {' '.join(opts)} -cfa -equalize_cfa -debayer", f"register pp_{src}"]
        seq = f"r_pp_{src}"
    L += [f"stack {seq} rej w 3 3 -norm=addscale -output_norm -rgb_equal -32b -out=../result", "cd .."]
    return "\n".join(L) + "\n"


def post_script(case: str, man: dict, threads: int) -> str:
    how = COLOUR[case][0]
    # the catalogues: Siril's own local extracts of Gaia DR3 (install_catalogues.py); VizieR, which Siril queries
    # online, did not answer on 2026-11-01
    colour = f"spcc -catalog=localgaia {SPCC_ARGS[case]}" if how == "spcc" else "pcc -catalog=localgaia"
    return "\n".join([
        "requires 1.3.4", f"setcpu {threads}", "setmem 0.4",
        "load result",
        "mirrorx -bottomup",
        f"platesolve {man['ra']},{man['dec']} -focal={man['focal_mm']:g} -pixelsize={man['pixel_um']:g} "
        "-catalog=localgaia",
        "save stack_linear",
        "subsky 2 -samples=20 -tolerance=1.0",
        "save bkg_linear",
        colour,
        "save color_linear",
        "autostretch -linked",
        "savetif final",
        "savepng final",
    ]) + "\n"


def run_script(work: Path, name: str, text: str) -> tuple[float, int, str]:
    (work / f"{name}.ssf").write_text(text, encoding="utf-8")
    log = work / f"{name}.log"
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as lf:
        rc = subprocess.call([str(C.SIRIL), "-d", str(work), "-i", str(C.SIRIL_CFG), "-s", str(work / f"{name}.ssf")],
                             cwd=work, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    dt = time.time() - t0
    txt = log.read_text(encoding="utf-8", errors="replace")
    ok = "Script execution finished successfully" in txt
    return dt, (0 if ok else (rc or 1)), txt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=list(C.CASES))
    ap.add_argument("level", choices=list(C.LEVELS))
    ap.add_argument("--post-only", action="store_true", help="re-run the post-processing on the existing stack")
    ap.add_argument("--pre-only", action="store_true", help="stop after the stack")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--variant", help="the variant of a case that has some (rosette: A, B)")
    a = ap.parse_args()
    variant = C.norm_variant(a.case, a.variant)
    if variant and (not C.variant_has_siril(a.case, variant) or a.level not in C.variant_levels(a.case, variant)):
        print(f"{a.case} {variant} {a.level}: no Siril run for this variant")
        return 1
    man = json.loads((C.BENCH / a.case / "manifest.json").read_text(encoding="utf-8"))
    if man["levels"][a.level].get("same_as"):
        print(f"{a.case} {a.level}: same lights as {man['levels'][a.level]['same_as']}: nothing to run")
        return 0
    work = C.siril_dir(a.case, a.level, variant)
    (work / "process").mkdir(parents=True, exist_ok=True)
    (work / "masters").mkdir(parents=True, exist_ok=True)
    rec_f = work / "siril.json"
    rec = json.loads(rec_f.read_text(encoding="utf-8")) if rec_f.exists() else {}
    rec.update({"case": a.case, "level": a.level, "siril": "1.4.0-beta3", "threads": a.threads,
                "colour": COLOUR[a.case][0], "colour_why": COLOUR[a.case][1]})
    if variant:
        rec.update({"variant": variant, "variant_label": man["variants"][variant]["label"]})
        if "darks" not in C.calibration(man, variant):
            rec["cosmetic"] = f"seqfind_cosme_cfa on the raw lights, cold {COSME[0]} sigma, hot {COSME[1]} sigma"
        else:
            rec["cosmetic"] = "-cc=dark (hot pixels from the master dark, 3 sigma); no dark optimisation (-opt)"
    if not a.post_only:
        dt, rc, txt = run_script(work, "pre", pre_script(a.case, a.level, man, a.threads, variant))
        rec["pre_s"], rec["pre_ok"] = round(dt, 1), rc == 0
        rec["pre_notes"] = [ln.strip() for ln in txt.splitlines()
                            if re.search(r"(?i)excluded|rejected|stacked|error|failed|registered|corrected|"
                                         r"deviant", ln)][-30:]
        rec["when_pre"] = datetime.now().isoformat(timespec="seconds")
        rec_f.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"{a.case} {a.level}: pre-processing {'ok' if rc == 0 else 'FAILED'} in {dt / 60:.1f} min", flush=True)
        if rc:
            return 1
        if a.pre_only:
            return 0
    dt, rc, txt = run_script(work, "post", post_script(a.case, man, a.threads))
    rec["post_s"], rec["post_ok"] = round(dt, 1), rc == 0
    rec["when_post"] = datetime.now().isoformat(timespec="seconds")
    # what Siril said about the frames it kept and the colour calibration
    rec["log_notes"] = [ln.strip() for ln in txt.splitlines()
                        if re.search(r"(?i)images? (were|was) (excluded|stacked)|pixel scale|catalog|"
                                     r"white balance|factors|error|failed|solved", ln)][:40]
    rec_f.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"{a.case} {a.level}: post-processing {'ok' if rc == 0 else 'FAILED'} in {dt / 60:.1f} min", flush=True)
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
