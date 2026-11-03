"""Import v1 projects into v2 projects with their still valid results, or report what an import would adopt.

    python tools/import_v1.py OLD NEW [--map OLD=NEW[|NEW2]]... [--dry-run] [--out REPORT.json] [--proof DIR]
    python tools/import_v1.py --batch PROJECTS.json [--dry-run] [--out REPORT.json] [--proof DIR]

PROJECTS.json: [{"old": "<v1 folder>", "new": "<v2 folder>", "map": {"<old input folder or file>": "<new place>" or
["<place>", ...]}}, ...]. ``--map`` and ``map`` move the inputs whose old place is gone (DESIGN §1.6,
core/import_v1.map_path).

Nothing is written into the old projects: the import runs under a write guard (core/import_v1.WriteGuard). With
``--dry-run`` nothing is written at all. ``--proof DIR`` lists every file of the old projects' folders (project,
scratch, inputs, reference, star layers) with its size and date before and after, writes both lists into DIR and
reports any difference. The stage durations of the old manifests (estimate.learn_from_v1) are printed (a timing history can be
seeded with them); this tool writes no history.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from astro_gpu.core import import_v1 as iv  # noqa: E402


def _maps(items: list[str]) -> dict[str, str | list[str]]:
    out: dict[str, str | list[str]] = {}
    for it in items:
        old, sep, new = it.partition("=")
        if not sep or not old or not new:
            raise SystemExit(f"--map expects OLD=NEW (or OLD=NEW1|NEW2): {it!r}")
        news = new.split("|")
        out[old] = news if len(news) > 1 else news[0]
    return out


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def _summary(r: iv.ImportReport) -> str:
    wid = next(iter(r.workflows), None)
    ad, rec = r.adopted(wid), r.recomputed(wid)
    first = f"{rec[0][0]} ({rec[0][1]})" if rec else "-"
    return (f"{r.name:32s} adopted {len(ad):2d}: {(ad[0] + ' .. ' + ad[-1]) if ad else '-':24s} "
            f"recomputed {len(rec):2d} from {first}; {_gb(r.copy_bytes)} to copy")


def _timings(r: iv.ImportReport) -> str:
    rows = sorted(r.timings, key=lambda t: t[0])
    return ", ".join(f"{s} {sec:.0f} s/{u:g}" for s, sec, u, _ in rows) or "-"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old", nargs="?")
    ap.add_argument("new", nargs="?")
    ap.add_argument("--map", action="append", default=[], help="OLD=NEW or OLD=NEW1|NEW2")
    ap.add_argument("--batch", help="a JSON list of {old, new, map}")
    ap.add_argument("--dry-run", action="store_true", help="report only: write nothing")
    ap.add_argument("--out", help="write the reports (JSON) here")
    ap.add_argument("--proof", help="folder for the before/after file lists of the old projects")
    a = ap.parse_args()
    if a.batch:
        jobs = json.loads(Path(a.batch).read_text(encoding="utf-8"))
    elif a.old and a.new:
        jobs = [{"old": a.old, "new": a.new, "map": _maps(a.map)}]
    else:
        ap.error("give OLD NEW, or --batch")
    reports, status = [], 0
    for n, job in enumerate(jobs, 1):
        old = job["old"]
        print(f"\n=== {old}", flush=True)
        t0 = time.time()
        moved = [v for vs in (job.get("map") or {}).values() for v in ([vs] if isinstance(vs, str) else vs)]
        watched = list(dict.fromkeys(iv.folders_of(old) + moved))       # the old project and the raws it reads
        before = iv.snapshot(watched) if a.proof else None
        try:
            r = iv.import_v1(old, job["new"], job.get("map") or None, dry_run=a.dry_run,
                             progress=lambda m: print(f"  {m}", flush=True))
        except Exception as e:  # noqa: BLE001 - report it, go on with the next project
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            reports.append({"old_root": old, "error": f"{type(e).__name__}: {e}"})
            status = max(status, 1)
            r = None
        if r is not None:
            print(r.text())
            print(f"  stage durations in the old manifests (stage seconds/units): {_timings(r)}")
            print(f"  ({time.time() - t0:.0f} s)", flush=True)
            reports.append(r.to_dict())
        if a.proof:
            after = iv.snapshot(watched)
            changes = iv.snapshot_changes(before, after)
            proof = Path(a.proof)
            proof.mkdir(parents=True, exist_ok=True)
            name = f"{n}_{Path(old).parent.name}_{Path(old).name}"
            for tag, snap in (("before", before), ("after", after)):
                (proof / f"{name}_{tag}.json").write_text(json.dumps(snap, indent=0), encoding="utf-8")
            files = sum(1 for v in after.values() if not v[0])
            size = sum(v[1] for v in after.values())
            print(f"  proof: {files} files, {_gb(size)} in {len(watched)} watched places: "
                  + ("unchanged" if not changes else f"{len(changes)} CHANGES, first {changes[:3]}"), flush=True)
            reports[-1]["proof"] = {"files": files, "bytes": size, "changes": changes}
            if changes:
                status = 2
    print("\n=== summary" + (" (dry run)" if a.dry_run else ""))
    for d in reports:
        if "error" in d:
            print(f"{d['old_root']}: FAILED {d['error']}")
        else:
            r = iv.ImportReport(d["old_root"], d["new_root"], d["name"], d["dry_run"],
                                {int(k): v for k, v in d["workflows"].items()},
                                [iv.StageReport(**s) for s in d["stages"]], d["notes"], d["layers"], d["copy_bytes"],
                                d["free_bytes"], d["timings"], d["written"])
            print(_summary(r))
    if a.out:
        Path(a.out).write_text(json.dumps(reports, indent=1, default=str), encoding="utf-8")
    return status


if __name__ == "__main__":
    sys.exit(main())
