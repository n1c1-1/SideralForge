#!/usr/bin/env bash
# Run F: the run E recipe with the bright stamps built against an interpolated background (own_starnet.py stamps),
# for the projects given as "<name>" (inputs: D:/astro_gpu_diag/own_starnet/<name>/snapshot, read-only copies of the
# project files; outputs: D:/astro_gpu_diag/own_starnet/<name>/runF). Second GPU only.
#   bash experiments/own_starnet/run_F.sh FlamingStar Horsehead
PY=C:/Users/Nicolas/sandbox/astro_GPU/.venv/Scripts/python.exe
HERE="$(cd "$(dirname "$0")" && pwd)"
S="$HERE/own_starnet.py"
ALL=D:/astro_gpu_diag/own_starnet/runF_driver.log
export CUDA_VISIBLE_DEVICES=1
for name in "$@"; do
  out=D:/astro_gpu_diag/own_starnet/$name
  P=$out/snapshot
  lg=$out/runF_driver.log
  step() {
    local label=$1; shift
    echo "$(date +%T) $name: $label" | tee -a "$ALL" >> "$lg"
    if ! "$@" >> "$lg" 2>&1; then
      echo "$(date +%T) $name: $label FAILED" | tee -a "$ALL" >> "$lg"
      return 1
    fi
  }
  [ -f "$out/prep/library_nohalo_membrane.pt" ] || step stamps "$PY" "$S" stamps --project "$P" --out "$out" || continue
  step train "$PY" "$S" train --run runF --minus-halo --noisy-target --bright-aug --core-mask --stamps membrane \
    --minutes 32 --batch 6 --project "$P" --out "$out" || continue
  step infer "$PY" "$S" infer --run runF --project "$P" --out "$out" || continue
  step eval "$PY" "$S" eval --run runF --inject-stamps "" --project "$P" --out "$out" || continue
  step sheets "$PY" "$HERE/compare_runs.py" runE,runF --project "$P" --out "$out"
  echo "$(date +%T) $name: done" | tee -a "$ALL" >> "$lg"
done
echo "$(date +%T) ALL DONE" >> "$ALL"
