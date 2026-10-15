#!/usr/bin/env bash
# Run H: the network removes only the compact light of the stars. Input: the stack minus the corrected far halo
# (experiments/halo2/halo_lab.py, outputs in D:/astro_gpu_diag/halo2/<name>); bright stamps split into compact
# light (removed) and diffuse light (pasted into the input and the target alike: own_starnet.py stamps
# --stamp-variant compact); otherwise the run E recipe. Outputs: D:/astro_gpu_diag/own_starnet/<name>/H (snapshot
# with hard links, prep, runH). Second GPU only.
#   bash experiments/own_starnet/run_H.sh Horsehead Rosette FlamingStar Iris
PY=C:/Users/Nicolas/sandbox/astro_GPU/.venv/Scripts/python.exe
HERE="$(cd "$(dirname "$0")" && pwd)"
S="$HERE/own_starnet.py"
ALL=D:/astro_gpu_diag/own_starnet/runH_driver.log
export CUDA_VISIBLE_DEVICES=1
for name in "$@"; do
  out=D:/astro_gpu_diag/own_starnet/$name/H
  P=$out/snap
  mkdir -p "$out"
  lg=$out/runH_driver.log
  step() {
    local label=$1; shift
    echo "$(date +%T) $name: $label" | tee -a "$ALL" >> "$lg"
    if ! "$@" >> "$lg" 2>&1; then
      echo "$(date +%T) $name: $label FAILED" | tee -a "$ALL" >> "$lg"
      return 1
    fi
  }
  [ -f "$P/work/starless/sl_final.npy" ] || step snapshot "$PY" "$HERE/make_snap_halo.py" D:/astro_gpu_diag/halo2/$name "$P" || continue
  [ -f "$out/prep/library_nohalo.pt" ] || step prep "$PY" "$S" prep --minus-halo --project "$P" --out "$out" || continue
  early=D:/astro_gpu_diag/own_starnet/$name/prep
  [ "$name" = Iris ] && early=D:/astro_gpu_diag/own_starnet/prep          # the first experiment's project
  if [ ! -f "$out/prep/same_holdouts.done" ]; then                        # measured on the earlier runs' regions
    step holdouts "$PY" "$HERE/same_holdouts.py" "$early" "$out/prep" && touch "$out/prep/same_holdouts.done" || continue
  fi
  [ -f "$out/prep/library_nohalo_compact.pt" ] || step stamps "$PY" "$S" stamps --stamp-variant compact --project "$P" --out "$out" || continue
  [ -f "$out/runH/model.pt" ] || step train "$PY" "$S" train --run runH --minus-halo --noisy-target --bright-aug --core-mask \
    --stamps compact --minutes 32 --batch 6 --project "$P" --out "$out" || continue
  [ -f "$out/runH/star_layer_net.npy" ] || step infer "$PY" "$S" infer --run runH --project "$P" --out "$out" || continue
  step eval "$PY" "$S" eval --run runH --project "$P" --out "$out"
  step lost "$PY" "$HERE/lost_light.py" --snap "$P" --nl "$out/prep/n2n_lat.npy" \
    --layers runH=$out/runH/star_layer_net.npy --out "$out/runH/lost_light.json"
  echo "$(date +%T) $name: done" | tee -a "$ALL" >> "$lg"
done
echo "$(date +%T) ALL DONE" >> "$ALL"
