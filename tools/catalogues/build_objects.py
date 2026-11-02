"""Make the bundled catalogues of named deep-sky objects again (astro_gpu/resources/catalogues/).

    python tools/catalogues/build_objects.py [--from FOLDER] [--openngc TAG] [--vizier HOST]

Each source is downloaded from its original publisher (or read from FOLDER, where earlier downloads were kept under
their original names) and written gzip-compressed, without any change to its content, so that the reader of the
package (astro_gpu/core/skyobjects.py) parses the original columns:

- steinicke/ngcic.csv.gz: W. Steinicke's Revised NGC/IC, http://www.klima-luft.de/steinicke/ngcic/ngcic_e.htm
  ("Revised NGC/IC Data", a zipped xls: rev2000/NI<year>.zip). The sheet is written as CSV (comma, UTF-8), its 38
  columns as they are (numbers written back as the sheet has them: 1.0 -> 1). Needs ``xlrd`` (pip install xlrd; a
  tool dependency only, the package never reads xls). Plain http: the site does not answer https.
- openngc/NGC.csv.gz, openngc/addendum.csv.gz, openngc/LICENSE.txt: OpenNGC (github.com/mattiaverga/OpenNGC) at a
  release tag (database_files/NGC.csv, addendum.csv; LICENSES/CC-BY-SA-4.0.txt).
- sharpless/vii20.tsv.gz: VizieR VII/20 (Sharpless 1959), every column plus _RAJ2000 / _DEJ2000 in degrees.
- lynds/vii7a.tsv.gz: VizieR VII/7A (Lynds 1962), table ldn, every column plus _RAJ2000 / _DEJ2000 in degrees.

The VizieR host: the first of HOSTS that answers (Strasbourg, Harvard, Cape Town, Tokyo), or --vizier. The README.md of
each folder (origin, version, licence, citation) is written by hand: update its version / date lines when a source
changes. gzip files are written with mtime 0, so that the same content gives the same bytes.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "astro_gpu" / "resources" / "catalogues"
STEINICKE_PAGE = "http://www.klima-luft.de/steinicke/ngcic/ngcic_e.htm"
STEINICKE_ZIP = "http://www.klima-luft.de/steinicke/ngcic/rev2000/NI2026.zip"
OPENNGC_TAG = "v20260501"
OPENNGC_RAW = "https://raw.githubusercontent.com/mattiaverga/OpenNGC/{tag}/{path}"
HOSTS = ("vizier.cds.unistra.fr", "vizier.cfa.harvard.edu", "vizier.idia.ac.za", "vizier.nao.ac.jp")
VIZIER = {"sharpless/vii20.tsv.gz": "VII/20/catalog", "lynds/vii7a.tsv.gz": "VII/7A/ldn"}
VIZIER_QUERY = ("https://{host}/viz-bin/asu-tsv?-source={table}&-out.all&-out.add=_RAJ2000,_DEJ2000&-oc.form=d"
                "&-out.max=unlimited")


def fetch(url: str, timeout: float = 120.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "astro_gpu catalogue builder"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def write_gz(rel: str, data: bytes) -> None:
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with gzip.GzipFile(filename=p.name[:-3], mode="wb", fileobj=buf, mtime=0, compresslevel=9) as g:
        g.write(data)
    p.write_bytes(buf.getvalue())
    print(f"{rel}: {len(data) / 1e3:.0f} kB -> {p.stat().st_size / 1e3:.0f} kB")


def steinicke_csv(xls: bytes) -> bytes:
    import xlrd
    sheet = xlrd.open_workbook(file_contents=xls).sheet_by_index(0)
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    for r in range(sheet.nrows):
        row = []
        for v in sheet.row_values(r):
            if isinstance(v, float):
                v = int(v) if v.is_integer() else repr(v)
            row.append(str(v).strip())
        w.writerow(row)
    return out.getvalue().encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", type=Path, help="a folder of earlier downloads (NI2026.zip or .xls, "
                    "NGC.csv, addendum.csv, CC-BY-SA-4.0.txt, vii20.tsv, vii7a.tsv)")
    ap.add_argument("--openngc", default=OPENNGC_TAG, help="OpenNGC release tag")
    ap.add_argument("--vizier", help="the VizieR host")
    a = ap.parse_args()

    def get(local: str, url: str) -> bytes:
        if a.src is not None and (a.src / local).is_file():
            return (a.src / local).read_bytes()
        print(f"downloading {url}")
        return fetch(url)

    # Steinicke
    if a.src is not None and (a.src / "NI2026.xls").is_file():
        xls = (a.src / "NI2026.xls").read_bytes()
    else:
        z = zipfile.ZipFile(io.BytesIO(get("NI2026.zip", STEINICKE_ZIP)))
        xls = z.read(next(n for n in z.namelist() if n.lower().endswith(".xls")))
    write_gz("steinicke/ngcic.csv.gz", steinicke_csv(xls))
    # OpenNGC
    for local, path in (("NGC.csv", "database_files/NGC.csv"), ("addendum.csv", "database_files/addendum.csv")):
        data = get(local, OPENNGC_RAW.format(tag=a.openngc, path=path))
        if not data.startswith(b"Name;Type;RA;Dec"):
            sys.exit(f"{path} is not OpenNGC's table")
        write_gz(f"openngc/{local}.gz", data)
    lic = get("CC-BY-SA-4.0.txt", OPENNGC_RAW.format(tag=a.openngc, path="LICENSES/CC-BY-SA-4.0.txt"))
    (OUT / "openngc" / "LICENSE.txt").write_bytes(lic)
    # VizieR
    for rel, table in VIZIER.items():
        local = Path(rel).name[:-3]
        if a.src is not None and (a.src / local).is_file():
            write_gz(rel, (a.src / local).read_bytes())
            continue
        for host in ([a.vizier] if a.vizier else HOSTS):
            try:
                data = fetch(VIZIER_QUERY.format(host=host, table=table), timeout=60)
            except OSError as e:
                print(f"{host}: {e}")
                continue
            if b"_RAJ2000" in data and data.count(b"\n") > 300:
                print(f"{table} from {host}")
                write_gz(rel, data)
                break
        else:
            sys.exit(f"no VizieR host gave {table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
