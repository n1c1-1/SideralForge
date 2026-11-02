"""Siril 1.4's own Bayer drizzle on the frozen frames of a case: the classic recipe (run_siril.py), changed only where
the drizzle requires it, as Siril's documentation and its script OSC_Preprocessing_BayerDrizzle.ssf say.

    python tools/bench/siril/run_siril_drizzle.py CASE LEVEL [--post-only] [--pre-only] [--threads 8]

Writes F:/astro_essais/bench/<case>/siril_<level>_drizzle/ (same files as siril_<level>/, same post-processing).
What changes against the classic recipe:
- calibrate: no -debayer (the frames stay CFA; -cfa -equalize_cfa kept);
- register: -drizzle -scale=1.0 -pixfrac=1.0 -kernel=square -flat=<master flat> (the official script's line; the doc:
  "it is recommended to apply CFA drizzle with scale = 1.0 and pixfrac = 1.0"). The stack line is unchanged;
- Seestar (no calibration frames): `calibrate light` without -debayer, `register pp_light -2pass`, then
  `seqapplyreg pp_light -drizzle -scale=1.0 -pixfrac=1.0 -kernel=square -filter-round=2.5k` (same filter as classic).
Nothing under siril_<level>/ is touched.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as C  # noqa: E402
import run_siril as R  # noqa: E402

DRZ = "-drizzle -scale=1.0 -pixfrac=1.0 -kernel=square"
_siril_dir = C.siril_dir
_classic_pre = R.pre_script


def drizzle_dir(case: str, level: str, variant: str = "") -> Path:
    return _siril_dir(case, level, variant).with_name(_siril_dir(case, level, variant).name + "_drizzle")


def pre_script(case: str, level: str, man: dict, threads: int, variant: str = "") -> str:
    text = _classic_pre(case, level, man, threads, variant)         # work folder = the drizzle one (C.siril_dir patched)
    if "-debayer" not in text:
        raise SystemExit("unexpected classic script")
    text = text.replace(" -cfa -equalize_cfa -debayer", " -cfa -equalize_cfa")
    text = text.replace("calibrate light -debayer", "calibrate light")
    flat = "-flat=../masters/pp_flat_stacked" if "masters/pp_flat_stacked" in text else ""
    text = re.sub(r"^register (pp_\S+)$", lambda m: f"register {m.group(1)} {DRZ} {flat}".rstrip(), text, flags=re.M)
    text = text.replace("seqapplyreg pp_light -filter-round=2.5k", f"seqapplyreg pp_light {DRZ} -filter-round=2.5k")
    return text


if __name__ == "__main__":
    C.siril_dir = drizzle_dir
    R.pre_script = pre_script
    rc = R.main()
    case = sys.argv[1]
    level = sys.argv[2]
    f = drizzle_dir(case, level, C.norm_variant(case, None)) / "siril.json"
    if f.exists():
        rec = json.loads(f.read_text(encoding="utf-8"))
        rec["drizzle"] = "Bayer drizzle, scale 1.0, pixfrac 1.0, square kernel (Siril doc / OSC_Preprocessing_BayerDrizzle)"
        f.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    sys.exit(rc)
