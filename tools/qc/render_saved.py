"""Re-render a project's deliverables with the exact settings saved next to them.

    python tools/qc/render_saved.py PROJECT NAME [NAME ...] [--suffix S]

Reads results/<NAME>/<NAME>_settings.json and exports again (into results/<NAME>, or results/<NAME>_<S> with
--suffix, so the delivered files are kept). Used after a change of the rendering engine or of the layers.
"""
import json
import sys
from pathlib import Path

from astro_gpu.core.project import Project
from astro_gpu.render.engine import StarRemovalEngines
from astro_gpu.render.export import ExportSettings, export
from astro_gpu.render.settings import RenderSettings


def main():
    args = sys.argv[1:]
    suffix = ""
    if "--suffix" in args:
        i = args.index("--suffix")
        suffix = args[i + 1]
        del args[i:i + 2]
    p = Project(args[0])
    col = json.loads((p.stage_dir("color") / "manifest.json").read_text())["params"]
    eng = StarRemovalEngines(p.stage_dir("render_prep"), device="cuda:0", calibrated_white=col.get("white_temp", 5778.0))
    for name in args[1:]:
        saved = json.loads((Path(args[0]) / "results" / name / f"{name}_settings.json").read_text())
        e = dict(saved["export"], name=f"{name}_{suffix}" if suffix else name)
        meta = export(p, eng, RenderSettings(**saved["settings"]), ExportSettings(**e))
        print(e["name"], json.dumps(meta.get("qa", {})))


if __name__ == "__main__":
    main()
