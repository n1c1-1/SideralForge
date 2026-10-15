#!/usr/bin/env bash
# Run E recipe on several projects, one after another, on the second GPU only.
#   bash experiments/own_starnet/run_projects.sh <project folder> [<project folder> ...]
# Per project (outputs in D:/astro_gpu_diag/own_starnet/<folder name>, the project folder is only read):
#   prep --minus-halo (skipped when it is already there), train runE (32 min), infer, eval, comparison sheets.
# Progress: <out>/driver.log for each project, D:/astro_gpu_diag/own_starnet/projects_driver.log overall.
PY=C:/Users/Nicolas/sandbox/astro_GPU/.venv/Scripts/python.exe
HERE="$(cd "$(dirname "$0")" && pwd)"
S="$HERE/own_starnet.py"
ALL=D:/astro_gpu_diag/own_starnet/projects_driver.log
export CUDA_VISIBLE_DEVICES=1
for P in "$@"; do
  name=$(basename "$P")
  out=D:/astro_gpu_diag/own_starnet/$name
  mkdir -p "$out"
  lg=$out/driver.log
  echo "$(date +%T) $name: start" | tee -a "$ALL" >> "$lg"
  step() {  # step <label> <command...>
    local label=$1; shift
    echo "$(date +%T) $name: $label" | tee -a "$ALL" >> "$lg"
    if ! "$@" >> "$lg" 2>&1; then
      echo "$(date +%T) $name: $label FAILED" | tee -a "$ALL" >> "$lg"
      return 1
    fi
  }
  if [ ! -f "$out/prep/meta.json" ] || [ ! -f "$out/prep/library_nohalo.pt" ]; then
    step prep "$PY" "$S" prep --minus-halo --project "$P" || continue
  fi
  step train "$PY" "$S" train --run runE --minus-halo --noisy-target --bright-aug --core-mask --minutes 32 \
    --batch 6 --project "$P" || continue
  step infer "$PY" "$S" infer --run runE --project "$P" || continue
  step eval "$PY" "$S" eval --run runE --project "$P" || continue
  step sheets "$PY" "$HERE/compare_runs.py" runE --project "$P"
  step overview "$PY" "$HERE/prep_overview.py" --project "$P"
  echo "$(date +%T) $name: done" | tee -a "$ALL" >> "$lg"
done
echo "$(date +%T) ALL DONE" >> "$ALL"
