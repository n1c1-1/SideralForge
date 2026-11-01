"""P8 step 2a: copy the raw frames that are not on F: to F:/astro_projets/_raw/<project>/<kind>/.

    python tools/projects/copy_raws.py [--dry-run] [PROJECT ...]

- only frame files (core.project.list_frames: the camera JPEGs beside the ARW stay behind), names and dates kept
  (shutil.copy2), so the inventory keys and the import's EXIF check hold;
- the 28.12.24 darks: only the 100 files the projects list, not the 2467 of the folder (review point 12);
- Flaming Star: its two G: folders merged back into one, as its project.json had them (same names, same order);
- a file already copied (same size and date) is skipped, so the script resumes;
- the sources are read under the import's write guard, and listed (size and date of every file) before and after:
  _p8/proof_copy/<project>_{before,after}.json; the summary goes to _p8/raw_copies.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p8  # noqa: E402  (first: it takes this folder, which holds queue.py, off sys.path)

p8.use_home()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from datetime import datetime  # noqa: E402

from astro_gpu.core import import_v1 as iv  # noqa: E402
from astro_gpu.core.project import list_frames  # noqa: E402


def _norm(p: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def planned(proj: p8.Proj) -> list[tuple[p8.Copy, list[Path]]]:
    """(copy, its source files in order) of a project."""
    d = json.loads((Path(proj.old) / "project.json").read_text(encoding="utf-8"))
    out = []
    for c in proj.copies:
        if c.listed:
            base = _norm(c.old)
            names = [Path(e).name for k in (d.get("inputs") or {}).values() for e in (k or [])
                     if _norm(e).startswith(base + os.sep)]
            files = [Path(c.sources[0]) / n for n in names]
            missing = [f for f in files if not f.is_file()]
            if missing:
                raise FileNotFoundError(f"{proj.name}: {len(missing)} listed {c.kind} missing, first {missing[0]}")
        else:
            files = list_frames(c.sources)
        names = [f.name.lower() for f in files]
        if len(set(names)) != len(names):
            raise ValueError(f"{proj.name} {c.kind}: two source files share a name")
        out.append((c, files))
    return out


def _same(src: Path, dst: Path) -> bool:
    try:
        a, b = src.stat(), dst.stat()
    except OSError:
        return False
    if a.st_size != b.st_size:
        return False
    if a.st_mtime_ns // 10**9 == b.st_mtime_ns // 10**9:
        return True
    # exFAT keeps dates on a 2-second grid: a staged input's date is rounded up (p8.staged_inputs)
    return dst.name.lower() == src.name.lower() and dst.parent.name == "inputs" and \
        0 <= b.st_mtime_ns - a.st_mtime_ns < 2_000_000_000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projects", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    projs = [p8.BY_NAME[n] for n in a.projects] if a.projects else [p for p in p8.PROJECTS if p.copies]
    proof = p8.P8 / "proof_copy"
    tasks: dict[str, list[tuple[Path, Path]]] = defaultdict(list)     # source drive -> [(src, dst)]
    summary: dict[str, dict] = {}
    watched: dict[str, list[str]] = {}
    total = 0
    for proj in projs:
        summary[proj.name] = {}
        for c, files in planned(proj):
            dst_dir = proj.raw_dir(c.kind)
            assert p8.under_ws(dst_dir), dst_dir
            size = sum(f.stat().st_size for f in files)
            total += size
            summary[proj.name][c.kind] = {"files": len(files), "bytes": size, "sources": c.sources,
                                          "listed": c.listed, "dest": dst_dir.as_posix(), "old": c.old}
            for f in files:
                tasks[Path(f).drive.upper()].append((f, dst_dir / f.name))
        staged = p8.staged_inputs(proj)
        if staged:
            size = sum(src.stat().st_size for _, src, _ in staged)
            total += size
            summary[proj.name]["inputs"] = {"files": len(staged), "bytes": size, "dest": staged[0][2].parent.as_posix(),
                                            "staged": {old: [str(src), dst.as_posix()] for old, src, dst in staged}}
            for _, src, dst in staged:
                assert p8.under_ws(dst), dst
                tasks[src.drive.upper()].append((src, dst))
        watched[proj.name] = p8.protected(proj)
        print(f"{proj.name}: " + ", ".join(f"{k} {v['files']} ({v['bytes'] / 1e9:.1f} GB)"
                                         for k, v in summary[proj.name].items()), flush=True)
    free = iv.disk_free(p8.WS)
    print(f"to copy: {total / 1e9:.1f} GB; free on F: {free / 1e9:.0f} GB", flush=True)
    if total * 1.1 > free:
        print("not enough free space", file=sys.stderr)
        return 1
    if a.dry_run:
        return 0
    proof.mkdir(parents=True, exist_ok=True)
    before = {n: iv.snapshot(w) for n, w in watched.items()}
    for n, s in before.items():
        (proof / f"{n}_before.json").write_text(json.dumps(s, indent=0), encoding="utf-8")
    all_protected = [f for w in watched.values() for f in w]
    t0 = time.time()
    done = defaultdict(int)

    def run(drive: str) -> int:
        n = 0
        for src, dst in tasks[drive]:
            if _same(src, dst):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            n += 1
            done[drive] += 1
            if done[drive] % 200 == 0:
                print(f"  {drive} {done[drive]}/{len(tasks[drive])} copied ({time.time() - t0:.0f} s)", flush=True)
        return n

    with iv.WriteGuard(all_protected):
        with ThreadPoolExecutor(len(tasks) or 1) as ex:
            copied = dict(zip(tasks, ex.map(run, list(tasks))))
    print(f"copied {sum(copied.values())} files in {time.time() - t0:.0f} s ({copied})", flush=True)
    status = 0
    for proj in projs:
        after = iv.snapshot(watched[proj.name])
        (proof / f"{proj.name}_after.json").write_text(json.dumps(after, indent=0), encoding="utf-8")
        changes = iv.snapshot_changes(before[proj.name], after)
        summary[proj.name]["_proof"] = {"places": watched[proj.name], "files": len(after), "changes": changes}
        bad = []
        for c, files in planned(proj):
            got = list_frames([str(proj.raw_dir(c.kind))])
            if [g.name for g in got] != [f.name for f in files] or not all(_same(f, g) for f, g in zip(files, got)):
                bad.append(c.kind)
        for _, src, dst in p8.staged_inputs(proj):
            if not _same(src, dst) or dst.stat().st_mtime_ns % 2_000_000_000:
                bad.append(f"inputs/{dst.name}")
        summary[proj.name]["_check"] = "ok" if not bad else f"MISMATCH in {bad}"
        print(f"{proj.name}: copies {summary[proj.name]['_check']}; sources "
              + ("unchanged" if not changes else f"{len(changes)} CHANGES {changes[:3]}"), flush=True)
        if bad or changes:
            status = 2
    out = p8.P8 / "raw_copies.json"
    old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    old.update(summary)
    old["_written"] = datetime.now().isoformat(timespec="seconds")
    out.write_text(json.dumps(old, indent=1), encoding="utf-8")
    return status


if __name__ == "__main__":
    sys.exit(main())
