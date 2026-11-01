"""Siril's own offline catalogues for the bench, in F:/astro_essais/bench/_siril/catalogues (not the user's profile).

    python tools/bench/siril/install_catalogues.py

What Siril's Catalog_Installer script does, without its window:
- the astrometric extract of Gaia DR3 (Zenodo 14692304, 1.1 GB compressed): plate solving and PCC offline;
- the chunks of the spectrophotometric extract (Zenodo 14738271, HEALPix level 1) that cover the five fields: SPCC
  offline.
Each archive is checked against its sha256 and decompressed. The bench's copy of the Siril settings (_siril/config.ini)
then points core.catalogue_gaia_astro and core.catalogue_gaia_photo at them. Why: the online catalogues Siril uses
(VizieR) did not answer on 2026-11-01; Siril itself recommends these local extracts.
"""
from __future__ import annotations

import bz2
import hashlib
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cases as C  # noqa: E402

DIR = C.BENCH / "_siril" / "catalogues"
ASTRO = ("14692304", "siril_cat_healpix8_astro.dat.bz2")
SPCC_RECORD = "14738271"
SPCC_DIR = DIR / "siril_cat1_healpix8_xpsamp"


def chunks() -> list[int]:
    import astropy.units as u
    from astropy_healpix import HEALPix
    h = HEALPix(nside=2, order="nested", frame="icrs")
    out: set[int] = set()
    for c in C.CASES.values():
        out |= set(h.cone_search_lonlat(c["ra"] * u.deg, c["dec"] * u.deg, 5 * u.deg).tolist())
    return sorted(out)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fetch(record: str, name: str, dst: Path) -> Path:
    """Download (resuming), check the sha256, decompress; returns the .dat file."""
    dst.mkdir(parents=True, exist_ok=True)
    dat = dst / name[:-4]
    if dat.exists() and dat.stat().st_size > 0:
        return dat
    url = f"https://zenodo.org/records/{record}/files/{name}"
    arc = dst / name
    want = subprocess.run(["curl", "-s", "-L", "-m", "60", url + ".sha256sum"], capture_output=True,
                          text=True).stdout.split()[0]
    if not (arc.exists() and sha(arc) == want):
        subprocess.run(["curl", "-s", "-L", "-C", "-", "-o", str(arc), url], check=False)
        if sha(arc) != want:
            raise SystemExit(f"checksum mismatch for {arc}")
    tmp = dat.with_suffix(".part")
    with bz2.open(arc, "rb") as fi, open(tmp, "wb") as fo:
        for b in iter(lambda: fi.read(1 << 22), b""):
            fo.write(b)
    tmp.replace(dat)
    arc.unlink()
    print(f"installed {dat} ({dat.stat().st_size / 1e9:.2f} GB)", flush=True)
    return dat


def set_config(key: str, value: Path) -> None:
    txt = C.SIRIL_CFG.read_text(encoding="utf-8")
    v = str(value).replace("/", "\\").replace("\\", "\\\\")
    txt, n = re.subn(rf"(?m)^{key}=.*$", lambda m: f"{key}={v}", txt)
    if not n:
        raise SystemExit(f"{key} not found in {C.SIRIL_CFG}")
    C.SIRIL_CFG.write_text(txt, encoding="utf-8")


def main() -> int:
    astro = fetch(ASTRO[0], ASTRO[1], DIR)
    px = chunks()
    print(f"SPCC chunks for the five fields: {px}", flush=True)
    for p in px:
        fetch(SPCC_RECORD, f"siril_cat1_healpix8_xpsamp_{p}.dat.bz2", SPCC_DIR)
    set_config("catalogue_gaia_astro", astro)
    set_config("catalogue_gaia_photo", SPCC_DIR)
    print(f"{C.SIRIL_CFG}: catalogue_gaia_astro and catalogue_gaia_photo set", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
