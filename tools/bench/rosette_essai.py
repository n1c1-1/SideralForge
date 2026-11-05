"""One trial of the Rosette verdict: a Post composition (layers and filters, as the GUI makes them) exported with the
GUI's own export code, compared to render2, kept in its own folder, and saved as the project's Post composition so
that the GUI opens it as it was.

    python tools/bench/rosette_essai.py ESSAI.json

ESSAI.json: {"name": "01_...", "note": "what is tried and why", "layers": [...]} with layers top first, each one of
    {"workflow": "Full", "part": null | "starless" | "stars" | "combined", "sliders": {...}, "blend": "Screen",
     "mode": "Normal", "opacity": 100, "name": "..."}
    {"filter": "local_contrast", "params": {...}, "on": "previous" | "all", "mode": "Normal", "opacity": 100,
     "mask": {"parts": [{"src": "halpha"}], "feather": 0}}
A filter "on": "previous" applies to the pixel layer just below it in the list (its clipping target), "all" to
everything below. Writes into F:/astro_essais/rosette_verdict/essais/<name>/: the export (TIFF 16 bits, JPEG), the
composition (composition.json), the comparison (board.jpg, measures.json).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ours  # noqa: E402,F401  (the bench environment, before the engine)
import rosette_verdict as RV  # noqa: E402

from astro_gpu import api  # noqa: E402
from astro_gpu.client import postengine as pe  # noqa: E402
from astro_gpu.client.services import render as rsv  # noqa: E402
from astro_gpu.render.layers import LayerService  # noqa: E402

ESSAIS = Path("F:/astro_essais/rosette_verdict/essais")


def build(p, spec: list[dict]) -> list[dict]:
    """The composition's layers (postengine's dicts), top first."""
    wf = {w.name: w.id for w in p.workflows}
    layers: list[dict] = []
    pending: list[dict] = []                         # filters waiting for the pixel layer below them
    for d in spec:
        if "workflow" in d:
            lay = pe.new_workflow_layer(wf[d["workflow"]], d.get("name") or d["workflow"], True, d.get("part"))
            lay["sliders"] = dict(d.get("sliders") or {})
            for k in ("mode", "opacity", "blend", "base"):
                if k in d:
                    lay[k] = d[k]
            for f in pending:
                f["target"] = lay["id"]
            pending = []
            layers.append(lay)
        else:
            f = pe.new_filter_layer(d["filter"], None)
            f["params"].update(d.get("params") or {})
            for k in ("mode", "opacity", "name"):
                if k in d:
                    f[k] = d[k]
            if d.get("mask"):                        # {"parts": [{"src": ..., settings}], "feather", "invert", ...}
                m = pe.new_mask()
                m.update({k: v for k, v in d["mask"].items() if k != "parts"})
                m["parts"] = [dict(pe.new_part(q["src"]), **q) for q in d["mask"]["parts"]]
                f["mask"] = m
            if d.get("on", "previous") == "previous":
                pending.append(f)
            layers.append(f)
    return pe.normalize(layers)


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = ESSAIS / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    p = api.open_project(RV.ROOT)
    layers = build(p, spec["layers"])
    dev = rsv.choose_device("auto")
    src = rsv.LayerSource(rsv.api_resolver(p), LayerService(dev), rsv.bases_of(p))
    r = rsv.Renderer(src, device=dev)
    t0 = time.time()
    files = rsv.run_export(r, rsv.ExportJob(layers=layers, folder=str(out), name=spec["name"],
                                            format=["tiff16", "jpeg"], wcs=False, project_name=p.name),
                           lambda f, m: None)
    print(f"exported in {time.time() - t0:.0f} s: {[Path(f).name for f in files]}", flush=True)
    (out / "composition.json").write_text(json.dumps({**spec, "layers_saved": pe.save_layers(layers)}, indent=1,
                                                     ensure_ascii=False), encoding="utf-8")
    shutil.copy(sys.argv[1], out / "essai.json")
    p.post = {**(p.post or {}), "layers": pe.save_layers(layers)}       # the GUI opens the last trial as it was
    api.save(p)
    tif = next(f for f in map(Path, files) if f.suffix.lower() in (".tif", ".tiff"))
    subprocess.run([sys.executable, str(Path(__file__).with_name("rosette_compare.py")), str(tif), str(out),
                    "--name", spec["name"][:2] and "essai " + spec["name"][:2]], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
