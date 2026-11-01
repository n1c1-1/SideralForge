"""Quality control of v1 and v2 renders: pairing, blind sheets, report cards, non-regression. The v2 port of
judge.py, blind.py, bulletin.py and nonreg.py (P17).

    python tools/qc/qc2.py pairs OLD [OLD ...] [--new NEW] [--out DIR]
    python tools/qc/qc2.py sheets --out DIR --short NAME [--seed N] [--zones x:y ...] CAND CAND ...
    python tools/qc/qc2.py pair-sheets --out DIR --old OLD --new NEW [--new-image NAME=PATH ...] [--candidates DIR]
    python tools/qc/qc2.py bulletin --out DIR [--sheet] CAND ...
    python tools/qc/qc2.py nonreg --out DIR [--sheets] --case NAME BEFORE AFTER ...
    python tools/qc/qc2.py render --out DIR CAND [--base-from OLD_CAND] [--device cuda:1]

CAND is ``PROJECT::WORKFLOW`` (v2: the workflow's name, or #id) or ``PROJECT::DELIVERABLE`` (v1: results/<name>/), with
``=IMAGE`` to name the file to judge (a v2 workflow's exported image, e.g. in F:/astro_projets/_qc/<name>/candidates/;
by default export/<workflow>/<workflow>_16bit.tif; a v1 deliverable's is results/<name>/<name>.png). ``--render DEVICE``
renders a v2 workflow that has no image, into DIR/renders/.

Everything is written into DIR (F:/astro_projets/_qc/ for P14): sheets/ for the judges, _keys/ for the key (never given
to them), bulletin/, nonreg/, pairs/, renders/. DIR inside a project, or in the folder of an image read, is refused,
and no write under a project is possible during the run (the write guard of core/import_v1).

Each render is mapped with the files of its own project (qc_frame.py), and an old image is paired with the v2 workflow
of the same preparation (qc_pairs.py), not with "Full" by default.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from qc_addr import QCError, Source, resolve, resolve_v1, resolve_v2  # noqa: E402
from qc_out import PAIRS, OutputRefused, protect, slug, sub  # noqa: E402
from qc_pairs import Pairing, old_images, pair, table, within_project  # noqa: E402

IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg")


def _inputs(sources: list[Source]) -> list[Path]:
    return [p for s in sources for p in s.read_only()]


def ensure_images(sources: list[Source], out: Path, device: str | None, bases: list[Any] | None = None,
                  names: list[str] | None = None) -> None:
    """A v2 workflow with no image is rendered into the output folder when ``device`` is given, else refused.
    ``names``: the file name of each render (by default project__workflow: two images of one workflow need two)."""
    from qc_render import render_v2
    svc = None
    for i, s in enumerate(sources):
        if s.image is not None and Path(s.image).is_file():
            continue
        if s.version != 2 or device is None:
            raise QCError(f"{s.address}: no image to judge (give ADDRESS=IMAGE" +
                          (", or --render DEVICE" if s.version == 2 else "") + ")")
        if svc is None:
            from astro_gpu.render.layers import LayerService
            svc = LayerService(device)
        render_v2(s, out, bases[i] if bases else None, device=device, service=svc, name=names[i] if names else None)


# ---------------------------------------------------------------------------------------------- pairs

def pairs(old_root: str | Path, new_root: str | Path | None = None, out: str | Path | None = None) -> dict[str, Any]:
    """The pairing table of a v1 project's images with the v2 workflows (or the workflow each should have)."""
    from astro_gpu.core import project2 as p2
    new = p2.load(new_root) if new_root else None
    prs = [pair(o, new) for o in old_images(old_root)]
    rows = [p.row() for p in prs]
    res: dict[str, Any] = {"old": str(old_root), "new": str(new_root) if new_root else None, "rows": rows,
                           "text": table(rows) if rows else "(no rendered image in this project)"}
    if new is not None:
        res["within"] = [{"classic": c.name, "full": f.name} for c, f in within_project(new)]
    if out:
        with protect(out, [old_root, new_root]) as o:
            f = sub(o, PAIRS) / f"{slug(Path(old_root).name)}_pairs.json"
            f.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
            res["file"] = str(f)
    return res


def pairs_all(old_roots: list[str | Path], out: str | Path | None = None) -> dict[str, Any]:
    """The pairing table of several v1 projects (the expected v2 workflow of each old image, from its own settings
    file). With ``out``: ``pairs/pairing_table.json`` and ``pairing_table.txt`` there. Reads only."""
    rows: list[dict[str, Any]] = []
    empty: list[str] = []
    for r in old_roots:
        got = [pair(o).row() for o in old_images(r)]
        rows += got
        if not got:
            empty.append(str(r))
    res: dict[str, Any] = {"rows": rows, "no_image": empty, "text": table(rows) if rows else "(no rendered image)"}
    if out:
        with protect(out, list(old_roots)) as o:
            d = sub(o, PAIRS)
            (d / "pairing_table.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
            (d / "pairing_table.txt").write_text(res["text"] + "\n", encoding="utf-8")
            res["file"] = str(d / "pairing_table.json")
    return res


# ---------------------------------------------------------------------------------------------- sheets

def run_sheets(out: str | Path, short: str, addresses: list[str], seed: int | None = None, frame: str = "auto",
               zones: list[str] | None = None, device: str | None = None, overwrite: bool = False,
               labels: list[str] | None = None, rules=None) -> dict[str, Any]:
    from qc_sheets import blind_sheets, check_short
    check_short(short)
    srcs = [resolve(a, rules=rules) for a in addresses]
    with protect(out, _inputs(srcs)) as o:
        ensure_images(srcs, o, device)
        return blind_sheets(o, short, srcs, seed=seed, frame=frame, zones=zones, labels=labels or addresses,
                            overwrite=overwrite)


def neutral_short(project: str, n: int) -> str:
    """The name of the sheet set of the n-th old image of a project: the project's name (when it tells nothing about
    which render is which) and the image's number. The image's own name ("natural_v2", "old") stays out of the sheets."""
    from qc_sheets import check_short
    base = slug(project)
    try:
        check_short(base)
    except QCError:
        base = "set"
    return f"{base}_{n:02d}"


def find_new_image(workflow_name: str, deliverable: str, explicit: dict[str, Path],
                   candidates: Path | None) -> Path | None:
    """The v2 render to put against an old deliverable: --new-image DELIVERABLE=PATH, else
    ``<candidates>/<slug(workflow)>__<deliverable>.(tif|png|jpg)``."""
    if deliverable in explicit:
        return explicit[deliverable]
    if candidates is not None:
        for ext in IMAGE_SUFFIXES:
            f = Path(candidates) / f"{slug(workflow_name)}__{deliverable}{ext}"
            if f.is_file():
                return f
    return None


def run_pair_sheets(out: str | Path, old_root: str | Path, new_root: str | Path, new_images: dict[str, Path] | None = None,
                    candidates: str | Path | None = None, device: str | None = None, seed: int | None = None,
                    frame: str = "auto", only: list[str] | None = None, overwrite: bool = False,
                    zones: list[str] | None = None, rules=None) -> dict[str, Any]:
    """Blind sheets of each old image against the v2 workflow of the same preparation. One sheet set per image; the key
    of each says which letter is the old image and which the workflow."""
    from qc_sheets import blind_sheets

    from astro_gpu.core import project2 as p2
    new = p2.load(new_root)
    todo: list[tuple[Pairing, Source, Source, str]] = []
    report: dict[str, Any] = {}
    for n, o in enumerate(old_images(old_root), 1):
        if only and o.name not in only:
            continue
        pr = pair(o, new)
        if pr.status != "paired":
            report[o.name] = {"status": pr.status, "reason": pr.reason}
            continue
        img = find_new_image(pr.workflow.name, o.name, {k: Path(v) for k, v in (new_images or {}).items()},
                             Path(candidates) if candidates else None)
        try:
            old_src = resolve_v1(Path(old_root), o.name)
            new_src = resolve_v2(Path(new_root), f"#{pr.workflow.id}", img, rules)
        except QCError as e:
            report[o.name] = {"status": "not comparable", "reason": str(e)}
            continue
        todo.append((pr, old_src, new_src, neutral_short(o.project, n)))
    inputs = [p for _, a, b, _ in todo for p in (*a.read_only(), *b.read_only())] or [old_root, new_root]
    with protect(out, [*inputs, old_root, new_root]) as od:
        for pr, old_src, new_src, short in todo:
            if new_src.image is None or not Path(new_src.image).is_file():
                if device is None:
                    report[pr.old.name] = {"status": "no image", "reason": (
                        f"no render of '{pr.workflow.name}' for {pr.old.name}: give --new-image {pr.old.name}=PATH, "
                        f"put {slug(pr.workflow.name)}__{pr.old.name}.tif in --candidates, or use --render DEVICE")}
                    continue
                # like for like: the new workflow is rendered with the settings the old image was made with
                ensure_images([new_src], od, device, [old_src.settings.get("settings") if old_src.settings else None],
                              [f"{slug(new_src.project_name)}__{slug(new_src.name)}__{slug(pr.old.name)}"])
            res = blind_sheets(od, short, [old_src, new_src], seed=seed, frame=frame, zones=zones,
                               labels=[f"old {pr.old.name} ({pr.old.stage})", f"new {pr.workflow.name}"],
                               extra={"pairing": pr.row()}, overwrite=overwrite)
            res["pairing"] = pr.row()           # which image the neutral name stands for: here, not in the sheets
            report[pr.old.name] = res
        f = sub(od, PAIRS) / f"{slug(Path(old_root).name)}_pair_sheets.json"
        f.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------------------------- cards and nonreg

def run_bulletin(out: str | Path, addresses: list[str], sheet: bool = False, device: str | None = None,
                 rules=None) -> dict[str, Any]:
    from qc_report import bulletin
    srcs = [resolve(a, rules=rules) for a in addresses]
    with protect(out, _inputs(srcs)) as o:
        ensure_images(srcs, o, device)
        return bulletin(o, addresses, sheet, srcs)


def run_nonreg(out: str | Path, cases: dict[str, tuple[str, str]], frame: str = "auto", sheets: bool = False,
               device: str | None = None, seed: int | None = None, rules=None) -> dict[str, Any]:
    from qc_report import nonreg
    srcs = {n: (resolve(a, rules=rules), resolve(b, rules=rules)) for n, (a, b) in cases.items()}
    with protect(out, _inputs([s for ab in srcs.values() for s in ab])) as o:
        ensure_images([s for ab in srcs.values() for s in ab], o, device)
        return nonreg(o, srcs, frame, sheets, seed)


def run_render(out: str | Path, address: str, base_from: str | None = None, device: str = "cpu",
               rules=None) -> str:
    from qc_render import render_v2
    s = resolve(address, rules=rules)
    base = None
    inputs = list(s.read_only())
    if base_from:
        old = resolve(base_from, rules=rules)
        base = (old.settings or {}).get("settings")
        inputs += old.read_only()
    with protect(out, inputs) as o:
        return str(render_v2(s, o, base, device=device))


# ---------------------------------------------------------------------------------------------- command line

def _cases(items: list[list[str]]) -> dict[str, tuple[str, str]]:
    out = {}
    for name, a, b in items:
        out[name] = (a, b)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, help_: str, out: bool = True) -> argparse.ArgumentParser:
        p = sp.add_parser(name, help=help_)
        p.add_argument("--out", required=out, help="the output folder (never inside a project)")
        return p

    p = add("pairs", "which v2 workflow each old image is compared with", out=False)
    p.add_argument("old", nargs="+", help="v1 project folders (several: the table of all, without --new)")
    p.add_argument("--new", help="the v2 project to pair with (one old project only)")
    p = add("sheets", "blind sheets of any renders")
    p.add_argument("--short", required=True, help="neutral name of the sheet set (not old/new/full/...)")
    p.add_argument("--seed", type=int)
    p.add_argument("--frame", choices=["auto", "reference", "sky"], default="auto")
    p.add_argument("--zones", nargs="*", default=[], help="extra zones, x:y fractions of the image")
    p.add_argument("--render", metavar="DEVICE", help="render the v2 workflows that have no image")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("cands", nargs="+")
    p = add("pair-sheets", "blind sheets of each old image against its paired v2 workflow")
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--new-image", action="append", default=[], metavar="DELIVERABLE=PATH")
    p.add_argument("--candidates", help="a folder of <workflow>__<deliverable>.tif exports")
    p.add_argument("--render", metavar="DEVICE")
    p.add_argument("--only", nargs="*")
    p.add_argument("--seed", type=int)
    p.add_argument("--frame", choices=["auto", "reference", "sky"], default="auto")
    p.add_argument("--zones", nargs="*", default=[])
    p.add_argument("--overwrite", action="store_true")
    p = add("bulletin", "report cards")
    p.add_argument("--sheet", action="store_true", help="a sheet per render, written into OUT/bulletin/")
    p.add_argument("--render", metavar="DEVICE")
    p.add_argument("cands", nargs="+")
    p = add("nonreg", "how far two renders of the same data differ")
    p.add_argument("--case", nargs=3, action="append", required=True, metavar=("NAME", "BEFORE", "AFTER"))
    p.add_argument("--sheets", action="store_true", help="blind sheets for the cases that differ")
    p.add_argument("--frame", choices=["auto", "reference", "sky"], default="auto")
    p.add_argument("--render", metavar="DEVICE")
    p.add_argument("--seed", type=int)
    p = add("render", "render a v2 workflow into OUT/renders/")
    p.add_argument("cand")
    p.add_argument("--base-from", help="an old deliverable whose saved render settings are used")
    p.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "pairs":
            if len(a.old) > 1 and a.new:
                raise QCError("--new pairs one old project with one v2 project: give a single old folder")
            r = pairs(a.old[0], a.new, a.out) if len(a.old) == 1 else pairs_all(a.old, a.out)
            print(r["text"])
            for w in r.get("within", []):
                print(f"also: {w['classic']}  against  {w['full']}")
        elif a.cmd == "sheets":
            r = run_sheets(a.out, a.short, a.cands, a.seed, a.frame, a.zones, a.render, a.overwrite)
            print(json.dumps({k: r[k] for k in ("short", "letters", "files", "frame")}, indent=1))
        elif a.cmd == "pair-sheets":
            ni = {}
            for it in a.new_image:
                k, _, v = it.partition("=")
                ni[k] = Path(v)
            r = run_pair_sheets(a.out, a.old, a.new, ni, a.candidates, a.render, a.seed, a.frame, a.only, a.overwrite,
                                a.zones)
            for k, v in r.items():
                print(k, v.get("status") or v.get("letters"), v.get("reason", ""))
        elif a.cmd == "bulletin":
            r = run_bulletin(a.out, a.cands, a.sheet, a.render)
            print(json.dumps(r, indent=1, ensure_ascii=False))
        elif a.cmd == "nonreg":
            r = run_nonreg(a.out, _cases(a.case), a.frame, a.sheets, a.render, a.seed)
            print(json.dumps({k: {x: v[x] for x in ("mean_levels", "p999_levels", "max_levels", "share_over_2_levels",
                                                    "differs", "zones")} for k, v in r.items()}, indent=1))
        else:
            print(run_render(a.out, a.cand, a.base_from, a.device))
    except (QCError, OutputRefused) as e:
        print(f"qc2: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
