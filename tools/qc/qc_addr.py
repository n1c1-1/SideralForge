"""Addresses of renders, v1 and v2: ``PROJECT::WORKFLOW`` (v2) and ``PROJECT::DELIVERABLE`` (v1), and what each one
brings to a comparison: its image, its own geometry, its own alignment and astrometry, its own star list.

    F:/astro_projets/Horsehead::Full + earlier neural layer         a v2 workflow (name, or #id)
    F:/astro_projets/Horsehead::Full=F:/astro_projets/_qc/H/full_faithful.tif    ... with the image to judge
    E:/astro_gpu/projects/Horsehead::faithful                       a v1 deliverable (results/faithful/)

Every file a render is mapped with comes from the project that made it, never from the project it is compared with
(the framing of two stacks of the same field moves by a few pixels). The project is read, never written.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qc_out import slug

from astro_gpu.core import project2 as p2
from astro_gpu.core.store import KeyRules, Keys, Store

SEP = "::"
OUTPUT_STAGES = ("render_prep", "render_prep_learned", "render_simple")
IMAGE_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


class QCError(ValueError):
    """An address that cannot be resolved, or a render that cannot be compared (the message says why)."""


# ---------------------------------------------------------------------------------------------- addresses

@dataclass(frozen=True)
class Address:
    project: Path
    name: str
    image: Path | None = None

    def __str__(self) -> str:
        return f"{self.project.as_posix()}{SEP}{self.name}" + (f"={self.image.as_posix()}" if self.image else "")


def parse_address(text: str) -> Address:
    """``PROJECT::NAME`` or ``PROJECT::NAME=IMAGE``. A Windows drive letter is no problem: the separator is two colons."""
    if SEP not in text:
        raise QCError(f"an address is PROJECT{SEP}WORKFLOW (or PROJECT{SEP}DELIVERABLE for a v1 project): {text!r}")
    project, _, rest = text.partition(SEP)
    name, eq, image = rest.partition("=")
    if not project.strip() or not name.strip():
        raise QCError(f"an address needs a project and a name: {text!r}")
    return Address(Path(project.strip()), name.strip(), Path(image.strip()) if eq and image.strip() else None)


def kind_of(root: str | Path) -> int:
    """2 for a v2 project folder (project.astroproj), 1 for a v1 one (project.json), else QCError."""
    r = Path(root)
    if (r / p2.FILE).is_file():
        return 2
    if (r / p2.V1_FILE).is_file():
        return 1
    raise QCError(f"not a project folder (no {p2.FILE}, no {p2.V1_FILE}): {r}")


# ---------------------------------------------------------------------------------------------- sources

@dataclass
class Source:
    """One render, with the files of the project that made it."""

    address: str
    version: int                      # 1 or 2
    root: Path
    project_name: str
    name: str                         # v1: the deliverable; v2: the workflow
    stage: str                        # the output stage: render_prep, render_prep_learned, render_simple
    out_dir: Path                     # its folder (prep.json / prep_simple.json)
    prep: dict[str, Any]              # crop (x0, x1, y0, y1), drizzle_scale, width, height
    image: Path | None = None
    align: Path | None = None         # the project's own align_ref.json (reference alignment), when it has one
    wcs: Path | None = None           # its own astrometric solution
    stars: Path | None = None         # its own star_fluxes.npz (x, y, g in its grid pixels)
    reference: Path | None = None     # the reference image the alignment was made with
    rot90: int = 0
    flip: bool = False
    workflow: p2.Workflow | None = None
    settings: dict[str, Any] | None = None          # v1: the deliverable's *_settings.json
    manifests: dict[str, Path] = field(default_factory=dict)    # done stages of its chain
    missing: list[str] = field(default_factory=list)            # stages of its chain that are not done
    notes: list[str] = field(default_factory=list)

    @property
    def crop(self) -> tuple[int, int, int, int]:
        x0, x1, y0, y1 = (int(v) for v in self.prep["crop"])
        return x0, x1, y0, y1

    @property
    def drizzle(self) -> float:
        return float(self.prep.get("drizzle_scale") or 1.0)

    def read_only(self) -> list[Path]:
        """What this render reads: its project, its image, its reference image."""
        return [p for p in (self.root, self.image, self.reference) if p]

    def describe(self) -> dict[str, Any]:
        """The identity of the render, for the key file (never for the sheets)."""
        return {"address": self.address, "version": self.version, "project": self.project_name, "name": self.name,
                "stage": self.stage, "image": self.image.as_posix() if self.image else None,
                "workflow": ({"id": self.workflow.id, "name": self.workflow.name} if self.workflow else None),
                "out_dir": self.out_dir.as_posix(), "crop": list(self.crop), "drizzle_scale": self.drizzle,
                "align": self.align.as_posix() if self.align else None,
                "wcs": self.wcs.as_posix() if self.wcs else None, "notes": list(self.notes)}


def _json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise QCError(f"cannot read {path}: {e}") from None


def _aligned(path: Path) -> tuple[Path | None, Path | None]:
    """(align_ref.json, the reference image) when the alignment holds a solution, else (None, None)."""
    if not path.is_file():
        return None, None
    d = _json(path)
    if not d.get("reference") or "cx" not in d:
        return None, None
    ref = Path(d["reference"])
    return path, (ref if ref.is_file() else None)


def find_image(folder: Path, stem: str) -> Path | None:
    """The first image ``<folder>/<stem>`` in the order of preference: sensor-scale PNG, JPEG, 16-bit TIFF."""
    for suffix in (".png", ".jpg", "_16bit.tif", "_full_resolution_16bit.tif", ".tif"):
        f = folder / f"{stem}{suffix}"
        if f.is_file():
            return f
    return None


# ---------------------------------------------------------------------------------------------- v1

def v1_settings(root: Path, name: str) -> tuple[Path, dict[str, Any]]:
    f = root / "results" / name / f"{name}_settings.json"
    if not f.is_file():
        have = sorted(p.parent.name for p in (root / "results").glob("*/*_settings.json")) if (root / "results").is_dir() else []
        raise QCError(f"{root}: no deliverable {name!r} (results/{name}/{name}_settings.json); it has {have}")
    return f, _json(f)


def v1_preparation(settings: dict[str, Any]) -> tuple[str, bool]:
    """(the preparation a v1 deliverable was rendered from, whether that is inferred). The settings file records it
    since the views were added; older ones predate the learned preparation, so they come from render_prep."""
    prep = settings.get("preparation")
    if prep in OUTPUT_STAGES:
        return prep, False
    return "render_prep", True


def resolve_v1(root: Path, name: str, image: Path | None = None) -> Source:
    cfg = _json(root / p2.V1_FILE)
    sf, settings = v1_settings(root, name)
    stage, inferred = v1_preparation(settings)
    variant = settings.get("variant") or "main"

    def folder(s: str) -> Path:
        if variant != "main" and variant in (cfg.get("variants") or {}):
            d = root / "variants" / cfg["variants"][variant]["slug"] / s
            if d.is_dir():
                return d
        return root / "work" / s

    out_dir = folder(stage)
    pf = out_dir / "prep.json"
    if not pf.is_file():
        raise QCError(f"{root}: the {stage} result this deliverable came from is gone ({pf}): its geometry is unknown")
    notes = [f"preparation not recorded in the settings file: taken as {stage}"] if inferred else []
    align, ref = _aligned(folder("reference") / "align_ref.json")
    wcs = folder("astrometry") / "wcs.fits"
    stars = folder("stars") / "star_fluxes.npz"
    img = image or find_image(sf.parent, name)
    if img is None:
        raise QCError(f"{root}: no image for the deliverable {name!r} in {sf.parent}")
    orient = settings.get("orientation") or {}
    done = {m.parent.name: m for m in sorted((root / "work").glob("*/manifest.json"))}
    return Source(address=f"{root.as_posix()}{SEP}{name}", version=1, root=root, project_name=cfg.get("name") or root.name,
                  name=name, stage=stage, out_dir=out_dir, prep=_json(pf), image=img, align=align,
                  wcs=wcs if wcs.is_file() else None, stars=stars if stars.is_file() else None, reference=ref,
                  rot90=int(orient.get("rot90", 0)), flip=bool(orient.get("flip", False)), settings=settings,
                  manifests=done, notes=notes)


# ---------------------------------------------------------------------------------------------- v2

def workflow_of(p: p2.Project2, name: str) -> p2.Workflow:
    """A workflow by name (exact, then ignoring case) or by ``#id`` / id."""
    m = re.fullmatch(r"#?(\d+)", name.strip())
    if m and any(w.id == int(m.group(1)) for w in p.workflows) and not any(w.name == name for w in p.workflows):
        return p.workflow(int(m.group(1)))
    for same in (lambda w: w.name == name, lambda w: w.name.casefold() == name.casefold()):
        hit = [w for w in p.workflows if same(w)]
        if len(hit) == 1:
            return hit[0]
        if len(hit) > 1:
            raise QCError(f"{p.name}: several workflows are called {name!r}: use #{' or #'.join(str(w.id) for w in hit)}")
    raise QCError(f"{p.name}: no workflow {name!r}; it has {[f'#{w.id} {w.name}' for w in p.workflows]}")


def orientation_of(image: Path | None) -> tuple[int, bool]:
    """(rot90, flip) an exported image was turned by: from its ``<name>_history.json`` (export_rgb), else none."""
    if image is None:
        return 0, False
    for f in (image.with_name(re.sub(r"(_16bit|_32bit_float)$", "", image.stem) + "_history.json"),
              image.with_suffix(".json")):
        if f.is_file():
            h = _json(f)
            o = h.get("options") or h.get("export") or h
            if "rot90" in o or "flip" in o:
                return int(o.get("rot90", 0)), bool(o.get("flip", False))
    return 0, False


def resolve_v2(root: Path, name: str, image: Path | None = None, rules: KeyRules | None = None,
               rot90: int | None = None, flip: bool | None = None) -> Source:
    p = p2.load(root)
    w = workflow_of(p, name)
    keys = Keys(p, rules)
    store = Store(root)
    chain = keys.chain(w)
    outs = [s for s in chain if keys.rules.chains.block_of(s) == "output"]
    if not outs:
        raise QCError(f"{p.name}: workflow {w.name!r} has no output stage")
    stage = outs[-1]
    done: dict[str, Path] = {}
    missing: list[str] = []
    dirs: dict[str, Path] = {}
    for s in chain:
        k = keys.key(w, s)
        if store.done(s, k):
            dirs[s] = store.dir(s, k)
            done[s] = dirs[s] / "manifest.json"
        else:
            missing.append(s)
    if stage not in dirs:
        raise QCError(f"{p.name}: the {stage} result of workflow {w.name!r} is not computed (not done: "
                      f"{', '.join(missing)})")
    out_dir = dirs[stage]
    pf = out_dir / ("prep_simple.json" if stage == "render_simple" else "prep.json")
    if not pf.is_file():
        raise QCError(f"{p.name}: {pf} is missing")
    align, ref = (None, None)
    if "reference" in dirs:
        align, ref = _aligned(dirs["reference"] / "align_ref.json")
    wcs = dirs["astrometry"] / "wcs.fits" if "astrometry" in dirs else None
    stars = dirs["stars"] / "star_fluxes.npz" if "stars" in dirs else None
    img = image
    if img is None:
        d = root / "export" / slug(w.name)
        img = find_image(d, slug(w.name))
    notes = []
    if img is None:
        notes.append("no image yet: give one (ADDRESS=IMAGE) or let the tool render it")
    o_rot, o_flip = orientation_of(img)
    return Source(address=f"{root.as_posix()}{SEP}{w.name}", version=2, root=root, project_name=p.name, name=w.name,
                  stage=stage, out_dir=out_dir, prep=_json(pf), image=img, align=align,
                  wcs=wcs if wcs and wcs.is_file() else None, stars=stars if stars and stars.is_file() else None,
                  reference=ref, rot90=o_rot if rot90 is None else rot90, flip=o_flip if flip is None else flip,
                  workflow=w, manifests=done, missing=missing, notes=notes)


def resolve(address: str | Address, image: str | Path | None = None, rules: KeyRules | None = None,
            rot90: int | None = None, flip: bool | None = None) -> Source:
    """The render an address names. ``image`` overrides the one in the address."""
    a = parse_address(address) if isinstance(address, str) else address
    img = Path(image) if image else a.image
    if kind_of(a.project) == 2:
        return resolve_v2(a.project, a.name, img, rules, rot90, flip)
    s = resolve_v1(a.project, a.name, img)
    if rot90 is not None:
        s.rot90 = rot90
    if flip is not None:
        s.flip = flip
    return s


def sources_of(addresses: list[str], rules: KeyRules | None = None) -> list[Source]:
    return [resolve(a, rules=rules) for a in addresses]
