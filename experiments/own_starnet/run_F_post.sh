#!/usr/bin/env bash
# After run F: the Flaming Star inference / evaluation from its last checkpoint (its training was cut by a GPU fault
# at 28.7 of 32 min), evaluations of runE and runF with the new stamps in the known-truth test (suffix _minj), faint
# stars inside the glare, tables, region sheets. Second GPU only.
PY=C:/Users/Nicolas/sandbox/astro_GPU/.venv/Scripts/python.exe
HERE="$(cd "$(dirname "$0")" && pwd)"
S="$HERE/own_starnet.py"
D=D:/astro_gpu_diag/own_starnet
export CUDA_VISIBLE_DEVICES=1
set -x
P=$D/FlamingStar/snapshot; O=$D/FlamingStar
[ -f $O/runF/starless_net.npy ] || "$PY" "$S" infer --run runF --project $P --out $O
[ -f $O/runF/metrics.json ] || "$PY" "$S" eval --run runF --inject-stamps "" --project $P --out $O
for name in FlamingStar Horsehead; do
  P=$D/$name/snapshot; O=$D/$name
  for r in runE runF; do
    "$PY" "$S" eval --run $r --inject-stamps membrane --eval-suffix _minj --project $P --out $O
  done
  "$PY" "$S" table --run runE,runF --project $P --out $O > /dev/null
  "$PY" "$S" table --run runE,runF --eval-suffix _minj --project $P --out $O > /dev/null
done
"$PY" "$HERE/faint_in_glare.py" --project $D/Horsehead/snapshot --out $D/Horsehead --runs runE,runF \
  --stars 2569,1135,Alnitak --stars 4825,583,sigmaOri --stars 1050,2303,G5.7
"$PY" "$HERE/faint_in_glare.py" --project $D/FlamingStar/snapshot --out $D/FlamingStar --runs runE,runF \
  --stars 5763,3787,G4.1 --stars 4598,3794,G5.3 --stars 3786,1719,G5.9
"$PY" "$HERE/compare_runs.py" runE,runF --project $D/FlamingStar/snapshot --out $D/FlamingStar
"$PY" "$HERE/compare_runs.py" runE,runF --project $D/FlamingStar/snapshot --out $D/FlamingStar \
  --at 5763,3787,300,"G4.1" --at 4598,3794,300,"G5.3" --at 3786,1719,250,"G5.9 (AE Aur) in IC 405" \
  --at 3000,1300,250,"red filaments north-west" --at 3400,1450,250,"filaments between the two stars"
"$PY" "$HERE/compare_runs.py" runE,runF --project $D/Horsehead/snapshot --out $D/Horsehead \
  --at 2569,1135,350,"Alnitak" --at 2400,1500,300,"Flame nebula side of Alnitak" --at 4825,583,300,"sigma Ori" \
  --at 3140,2060,250,"NGC 2023" --at 3800,1880,250,"Horsehead and the IC 434 edge" \
  --at 1920,1920,200,"structured hold-out (bright nebula)"
echo POST DONE
