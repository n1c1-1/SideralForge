"""Which v2 workflow an old image is compared with: the one that has the same preparation.

An image rendered from ``render_prep_learned`` (the preparation that removes the stars with a learned layer) must be
compared with a workflow that ends in ``render_prep_learned`` ("Full + neural stars", or "Full + earlier neural layer"
when the old project used a layer trained before), not with "Full" (model removal): the judges would see two changes
at once. Heart's, Horsehead's, Flaming Star's and Iris's images, and Rosette's faithful one, were rendered that way;
Rosette's artistic image was rendered from ``render_prep`` (see each ``results/*/*_settings.json``, key
``preparation``), so it goes with "Full".

The rule reads the old image's own settings file, never the current state of the old project: a project can hold
images of both preparations. A settings file that predates the key is taken as ``render_prep`` (the learned
preparation came later) and the table says so.

Nothing here writes. The old project is read through ``project.json`` and ``results/*/*_settings.json`` only.

A module of tools/qc/qc2.py (``python tools/qc/qc2.py --help``): not run by itself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qc_addr import OUTPUT_STAGES, QCError, kind_of, v1_preparation

from astro_gpu.core import project2 as p2
from astro_gpu.core import workflow as wfl

NAME_LEARNED, NAME_OWN, NAME_MODEL = "Full + earlier neural layer", "Full + neural stars", "Full"


@dataclass
class OldImage:
    """An image of a v1 project and what it was made from."""

    root: Path
    project: str
    name: str                     # the deliverable: results/<name>/
    stage: str                    # the preparation it was rendered from
    inferred: bool                # the settings file does not say: taken as render_prep
    date: str
    variant: str
    learned_layer: bool           # the project used an earlier learned star layer (starless.learned_stars)
    gradient: str                 # halpha, or polynomial (surveys.survey 'none')
    settings: Path

    @property
    def address(self) -> str:
        return f"{self.root.as_posix()}::{self.name}"


@dataclass
class Pairing:
    old: OldImage
    expected: str                                   # the v2 workflow name the import / P8 give to this preparation
    stage: str
    workflow: p2.Workflow | None = None
    alternatives: list[p2.Workflow] = field(default_factory=list)
    status: str = "expected"                        # expected (no v2 project given), paired, ambiguous, missing
    reason: str = ""

    def row(self) -> dict[str, Any]:
        return {"project": self.old.project, "image": self.old.name, "date": self.old.date,
                "preparation": self.old.stage + (" (inferred)" if self.old.inferred else ""),
                "learned_layer": self.old.learned_layer, "v2_workflow": self.workflow.name if self.workflow else None,
                "v2_expected": self.expected, "status": self.status, "reason": self.reason,
                "alternatives": [w.name for w in self.alternatives]}


def old_images(root: str | Path) -> list[OldImage]:
    """The rendered deliverables of a v1 project (a results/<name>/<name>_settings.json with its render settings),
    oldest first."""
    root = Path(root)
    if kind_of(root) != 1:
        raise QCError(f"{root} is not a v1 project")
    cfg = json.loads((root / p2.V1_FILE).read_text(encoding="utf-8"))
    main = cfg.get("params") or {}
    out = []
    for f in sorted((root / "results").glob("*/*_settings.json")) if (root / "results").is_dir() else []:
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(s.get("settings"), dict) or f.stem != f"{f.parent.name}_settings":
            continue
        stage, inferred = v1_preparation(s)
        variant = s.get("variant") or "main"
        params = main
        if variant != "main" and variant in (cfg.get("variants") or {}):
            params = p2._merge_variant(main, (cfg["variants"][variant] or {}).get("params") or {})
        out.append(OldImage(
            root=root, project=cfg.get("name") or root.name, name=f.parent.name, stage=stage, inferred=inferred,
            date=str(s.get("date") or ""), variant=variant, learned_layer=wfl.has_learned_layer(params),
            gradient="polynomial" if (params.get("surveys") or {}).get("survey") == "none" else "halpha", settings=f))
    return sorted(out, key=lambda o: (o.date, o.name))


def output_stage(options: dict[str, Any], end: str | None = None) -> str:
    """The output stage a workflow with these options ends in."""
    o = wfl.check_options(options)
    if end:
        return "render_simple"
    return {"psf_neural": "render_prep_learned", "psf": "render_prep"}.get(o["stars"], "render_simple")


def expected_name(old: OldImage) -> str:
    if old.stage == "render_prep_learned":
        return NAME_LEARNED if old.learned_layer else NAME_OWN
    return NAME_MODEL if old.stage == "render_prep" else "Classic stack"


def pair(old: OldImage, new: p2.Project2 | None = None) -> Pairing:
    """The workflow of ``new`` an old image is compared with. Without ``new``, the name it should have."""
    exp = expected_name(old)
    pr = Pairing(old, exp, old.stage)
    if old.stage not in OUTPUT_STAGES or old.stage == "render_simple":
        pr.status, pr.reason = "missing", f"{old.stage} is not a preparation of the old engine"
        return pr
    how = ("settings.preparation" if not old.inferred else "no preparation recorded: taken as render_prep, older than "
           "the learned preparation")
    pr.reason = f"{old.stage} ({how})"
    if new is None:
        pr.reason += f" -> a workflow ending in {old.stage}: '{exp}'"
        return pr
    cands = [w for w in new.workflows if w.end is None and output_stage(w.options) == old.stage]
    if not cands:
        pr.status = "missing"
        pr.reason += f"; {new.name} has no workflow ending in {old.stage} (add '{exp}' and run it)"
        return pr

    def rank(w: p2.Workflow) -> tuple[int, int, int, int]:
        return (int(w.options["gradient"] == old.gradient), int(wfl.has_learned_layer(w.params) == old.learned_layer),
                int(w.name == exp), int(w.variant_of is None))

    best = max(rank(w) for w in cands)
    top = [w for w in cands if rank(w) == best]
    pr.workflow = top[0]
    pr.alternatives = [w for w in cands if w is not top[0]]
    if len(top) > 1:
        pr.status = "ambiguous"
        pr.reason += f"; {len(top)} workflows fit equally ({', '.join(repr(w.name) for w in top)}): name one"
        return pr
    pr.status = "paired"
    notes = []
    if rank(top[0])[0] == 0:
        notes.append(f"its gradient is {top[0].options['gradient']}, the old image's is {old.gradient}")
    if rank(top[0])[1] == 0:
        notes.append("the earlier neural layer of the old project is not in this workflow" if old.learned_layer else
                     "this workflow uses an earlier neural layer, the old image trained its own")
    pr.reason += f" -> '{top[0].name}'" + (f" (but {'; '.join(notes)})" if notes else "")
    return pr


def pair_project(old_root: str | Path, new_root: str | Path | p2.Project2 | None = None) -> list[Pairing]:
    """Every old image of a v1 project with its v2 workflow (or the workflow it should have, without ``new_root``)."""
    new = new_root if isinstance(new_root, p2.Project2) else p2.load(new_root) if new_root else None
    return [pair(o, new) for o in old_images(old_root)]


def within_project(new: p2.Project2) -> list[tuple[p2.Workflow, p2.Workflow]]:
    """Pairs of workflows of one v2 project for the Classic-against-Full comparison: each Classic workflow (no Stars)
    with the plain Full one (else the first workflow with Stars)."""
    full = [w for w in new.workflows if w.end is None and w.options["stars"]]
    if not full:
        return []
    ref = next((w for w in full if w.name == NAME_MODEL), full[0])
    return [(w, ref) for w in new.workflows if w.end is None and not w.options["stars"]]


def table(rows: list[dict[str, Any]]) -> str:
    """The pairing table as text (the first lines of a report)."""
    head = ("project", "old image", "rendered from", "v2 workflow", "status")
    body = [(r["project"], r["image"], r["preparation"], r["v2_workflow"] or f"({r['v2_expected']})", r["status"])
            for r in rows]
    w = [max(len(str(x)) for x in col) for col in zip(head, *body)]
    line = lambda c: "  ".join(str(x).ljust(n) for x, n in zip(c, w)).rstrip()      # noqa: E731
    return "\n".join([line(head), line(["-" * n for n in w]), *map(line, body)])
