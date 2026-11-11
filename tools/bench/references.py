"""Give each bench project Siril's results for the same frames as comparison images ("Siril (reference)").

    python tools/bench/references.py [CASE ...]          (every case by default)

For every <case>/ours_<level>* project of the bench (F:/astro_essais/bench), the files Siril wrote for that level in
<case>/siril_<level>/ are listed in the project's definition (``post["references"]``, read by
``client/model.ProjectModel.references``): every viewer of the client then offers them in its Compare with list, the
block's own first. Only the list is written; the images stay where Siril put them. A file Siril did not write is left
out. Running it again replaces the list.

What Siril's post-processing script saves, in its order (siril_<level>/post.ssf): the stack (stack_linear.fit, the
camera's colours), the background extracted (bkg_linear.fit, still the camera's colours), the colour calibrated
(color_linear.fit, PCC), the stretched picture (final.tif / final.png).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as C  # noqa: E402

C.engine_home(False, os.environ.get("ASTRO_GPU_HOME"))      # before astro_gpu is imported: nothing of the user's
sys.path.insert(0, str(C.REPO))

#: Siril's file, what it is (English, the client translates it), how to show it, the blocks it compares with first.
#: stack and gradient removed are in the camera's colours (linear), colour-calibrated is linear, final is the picture
FILES = [
    ("stack_linear.fit", "Siril (reference) · stack", "camera", ["empilement", "astrometrie"]),
    ("bkg_linear.fit", "Siril (reference) · gradient removed", "camera", ["fond"]),
    ("color_linear.fit", "Siril (reference) · colour-calibrated", "linear", ["couleur", "fond"]),
    ("final.tif", "Siril (reference) · final image", "picture", ["post"]),
]


def level_of(project: Path) -> str | None:
    """ours_base, ours_base_gui -> base; ours_express -> express."""
    parts = project.name.split("_")
    return parts[1] if len(parts) >= 2 and parts[0] == "ours" else None


def references(case_dir: Path, level: str) -> list[dict]:
    siril = case_dir / f"siril_{level}"
    out = []
    for name, label, kind, blocks in FILES:
        f = siril / name
        if not f.is_file() and name == "final.tif" and (siril / "final.png").is_file():
            f = siril / "final.png"
        if not f.is_file():
            continue
        ref = {"name": label, "path": f.as_posix(), "kind": kind, "blocks": blocks}
        if kind == "picture":
            ref["white"] = 65535.0 if _sixteen_bits(f) else 255.0
        out.append(ref)
    return out


def _sixteen_bits(f: Path) -> bool:
    if f.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        with tifffile.TiffFile(f) as t:
            return t.pages[0].dtype.itemsize == 2
    from PIL import Image
    with Image.open(f) as im:
        return im.mode.startswith("I;16") or im.mode == "I"


def main(names: list[str]) -> int:
    from astro_gpu.core import project2
    root = Path(C.BENCH) if hasattr(C, "BENCH") else Path("F:/astro_essais/bench")
    cases = names or sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    for case in cases:
        cd = root / case
        for proj in sorted(cd.glob("ours_*")):
            level = level_of(proj)
            if level is None or not (proj / "project.astroproj").is_file():
                continue
            refs = references(cd, level)
            p = project2.load(proj)
            post = dict(p.post or {})
            post["references"] = refs
            p.post = post
            project2.save(p)
            print(f"{case}/{proj.name}: {len(refs)} Siril images" + (" (none: Siril has no result at this level)"
                                                                       if not refs else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
