"""Check a code change on a copy of a project: the deliverables rendered before and after the out-of-date stages are
recomputed, and how much they differ.

    python tools/qc/nonreg.py SOURCE_PROJECT COPY_DIR [--name faithful --name artistic] [--set stage.key=value ...]

1. Copies the project (project.json, work/, reference/ and the deliverables' settings) into COPY_DIR, unless COPY_DIR
   already holds a copy. The copy's scratch folder is inside COPY_DIR and the copy is never added to the interface's
   list of projects: nothing is written to the source project.
2. Renders each deliverable with its saved settings (results/<name>/<name>_settings.json), the current code and the
   copied layers -> results/<name>_before.
3. Applies the --set parameters to the copy, then runs every stage that is out of date (new stage versions, changed
   parameters).
4. Renders them again -> results/<name>_after, and writes COPY_DIR/nonreg.json: per deliverable the mean and the
   99.9th percentile of |after - before| in 8-bit levels, the share of pixels changed by more than 2 levels, and a
   map of the changes (results/<name>_diff.jpg, 20x amplified); plus the stages recomputed and their metrics.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def copy_project(src: Path, dst: Path, names: list[str]) -> None:
    if (dst / "project.json").exists():
        print(f"copy already there: {dst}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("work", "reference"):
        if (src / sub).exists():
            # robocopy: multi-threaded, returns < 8 on success
            r = subprocess.run(["robocopy", str(src / sub), str(dst / sub), "/E", "/MT:8", "/R:1", "/W:1",
                                "/NFL", "/NDL", "/NJH", "/NJS", "/NP"], capture_output=True, text=True)
            if r.returncode >= 8:
                raise RuntimeError(f"copy of {src / sub} failed: {r.stdout[-2000:]}")
    for n in names:
        f = src / "results" / n / f"{n}_settings.json"
        (dst / "results" / n).mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst / "results" / n / f.name)
    cfg = json.loads((src / "project.json").read_text(encoding="utf-8"))
    cfg["scratch_dir"] = str(dst / "scratch")
    cfg["devices"] = {"main": "cuda:0"}
    cfg["description"] = f"Test copy of {src} (non-regression), {time.strftime('%Y-%m-%d %H:%M')}."
    (dst / "project.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"copied {src} -> {dst}")


def render(proj, name: str, suffix: str) -> Path:
    from astro_gpu.render.engine import StarRemovalEngines
    from astro_gpu.render.export import ExportSettings, export
    from astro_gpu.render.settings import RenderSettings
    col = json.loads((proj.stage_dir("color") / "manifest.json").read_text())["params"]
    eng = StarRemovalEngines(proj.stage_dir("render_prep"), device="cuda:0",
                             calibrated_white=col.get("white_temp", 5778.0))
    saved = json.loads((proj.results / name / f"{name}_settings.json").read_text())
    e = dict(saved["export"], name=f"{name}_{suffix}", sensor_png=True, jpeg=True, full_tiff=True)
    meta = export(proj, eng, RenderSettings(**saved["settings"]), ExportSettings(**e))
    del eng
    import torch
    torch.cuda.empty_cache()
    print(f"rendered {name}_{suffix}: {json.dumps(meta.get('qa', {}))[:300]}")
    return proj.results / f"{name}_{suffix}" / f"{name}_{suffix}_full_resolution_16bit.tif"


def compare(a: Path, b: Path, out_jpg: Path) -> dict:
    import cv2
    import tifffile
    A = tifffile.imread(a).astype(np.float32) / 257.0
    B = tifffile.imread(b).astype(np.float32) / 257.0
    if A.shape != B.shape:
        return {"error": f"sizes differ {A.shape} vs {B.shape}"}
    d = np.abs(B - A).max(-1)
    m = np.clip(d * 20, 0, 255).astype(np.uint8)
    w = 1600
    m = cv2.resize(m, (w, round(m.shape[0] * w / m.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_jpg), m, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"mean_levels": round(float(d.mean()), 4), "p999_levels": round(float(np.percentile(d, 99.9)), 3),
            "max_levels": round(float(d.max()), 2), "share_over_2_levels": round(float((d > 2).mean()), 6)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source")
    ap.add_argument("copy")
    ap.add_argument("--name", action="append", help="deliverables to render (default: faithful and artistic)")
    ap.add_argument("--set", action="append", default=[], help="stage.key=value applied to the copy before the run")
    ap.add_argument("--skip-before", action="store_true", help="the 'before' renders exist already")
    a = ap.parse_args()
    names = a.name or ["faithful", "artistic"]
    src, dst = Path(a.source), Path(a.copy)

    import astro_gpu.core.project as project_mod
    project_mod.REGISTRY = dst / "_registry_unused.json"     # a test copy never appears in the interface's list
    from astro_gpu.core.project import Project
    from astro_gpu.core.runner import Pipeline
    from astro_gpu.stages import all_stages

    copy_project(src, dst, names)
    proj = Project(dst)
    report = {"source": str(src), "copy": str(dst), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not a.skip_before:
        for n in names:
            render(proj, n, "before")
    for kv in a.set:
        k, v = kv.split("=", 1)
        st, key = k.split(".", 1)
        try:
            v = json.loads(v)
        except ValueError:
            pass
        proj.config.params.setdefault(st, {})[key] = v
        report.setdefault("params_set", {})[k] = v
    if a.set:
        proj.save()
        proj = Project(dst)
    pipe = Pipeline(proj, all_stages())
    todo = pipe.plan()
    print("stages to run:", todo)
    t0 = time.time()
    status = pipe.run()
    report["run_status"] = status
    report["run_minutes"] = round((time.time() - t0) / 60, 1)
    report["stages"] = {}
    for st in todo:
        m = json.loads((proj.stage_dir(st) / "manifest.json").read_text(encoding="utf-8"))
        report["stages"][st] = {"version": m.get("version"), "minutes": round((m.get("duration") or 0) / 60, 1),
                                "metrics": m.get("metrics", {})}
    if status != "done":
        print("run did not finish:", status)
    else:
        proj = Project(dst)
        report["renders"] = {}
        for n in names:
            after = render(proj, n, "after")
            before = proj.results / f"{n}_before" / f"{n}_before_full_resolution_16bit.tif"
            report["renders"][n] = compare(before, after, proj.results / f"{n}_diff.jpg")
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (dst / "nonreg.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report.get("renders", {}), indent=1))


if __name__ == "__main__":
    sys.exit(main())
