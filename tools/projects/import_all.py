"""P8 step 2b: the import of the 9 projects, with tools/import_v1.py (the committed import tool).

    python tools/projects/import_all.py batch [PROJECT ...]     writes _p8/batch.json (old, new, path map)
    python tools/projects/import_all.py check REPORT.json        the table, checked against P6's dry run

Then, with ASTRO_GPU_HOME=F:/astro_projets/_home:

    python tools/import_v1.py --batch F:/astro_projets/_p8/batch.json --dry-run \
        --out F:/astro_projets/_p8/dryrun.json --proof F:/astro_projets/_p8/proof_dryrun
    python tools/import_v1.py --batch F:/astro_projets/_p8/batch.json \
        --out F:/astro_projets/_p8/import.json --proof F:/astro_projets/_p8/proof_import

The path maps point the dead or moved inputs at the copies of _raw/ (copy_raws.py); Heart and NGC 7822 read their
raws in place. The import runs under its write guard; ``--proof`` lists the old project's folders (project,
scratch, inputs, reference, star layers) and the new raw places before and after.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p8  # noqa: E402


def batch(names: list[str]) -> Path:
    projs = [p8.BY_NAME[n] for n in names] if names else p8.PROJECTS
    jobs = [{"old": p.old, "new": p.new.as_posix(), "map": p.path_map()} for p in projs]
    out = p8.P8 / ("batch.json" if not names else f"batch_{'_'.join(names)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    print(out)
    return out


def _name_of(report: dict) -> str:
    for p in p8.PROJECTS:
        if Path(p.old).resolve() == Path(report.get("old_root", "")).resolve():
            return p.name
    return "?"


def check(report_file: str) -> int:
    reports = json.loads(Path(report_file).read_text(encoding="utf-8"))
    status = 0
    print(f"{'project':12s} {'adopted':34s} {'first recomputed':44s} {'copy':>7s}  proof   P6")
    for r in reports:
        name = _name_of(r)
        if "error" in r:
            print(f"{name:12s} FAILED {r['error']}")
            status = 1
            continue
        wid = next(iter(r["workflows"]))
        st = [s for s in r["stages"] if str(s["workflow"]) == wid]
        ad = [s["stage"] for s in st if s["adopted"]]
        rec = [(s["stage"], s["reason"]) for s in st if not s["adopted"]]
        first = f"{rec[0][0]} ({rec[0][1]})" if rec else "-"
        proof = r.get("proof") or {}
        ok_proof = "ok" if proof and not proof.get("changes") else ("-" if not proof else "CHANGED")
        same = ad == p8.EXPECTED_ADOPTED.get(name)
        span = f"{ad[0]} .. {ad[-1]} ({len(ad)})" if ad else "-"
        print(f"{name:12s} {span:34s} {first[:44]:44s} {r['copy_bytes'] / 1e9:5.1f} GB  {ok_proof:7s} "
              f"{'same' if same else 'DIFFERENT'}")
        for n in r.get("notes") or []:
            print(f"{'':12s} note: {n}")
        for k, v in (r.get("layers") or {}).items():
            print(f"{'':12s} layer {Path(k).name}: stack {v}")
        if not same or ok_proof == "CHANGED":
            status = 2
    return status


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "batch":
        batch(sys.argv[2:])
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "check":
        return check(sys.argv[2])
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
