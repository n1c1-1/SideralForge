#!/bin/bash
# after run H of a project: the light lost at rendering (base layer and run H), comparison sheet, tables
# post_H.sh <project> <base name> <base layer> <base run dir with metrics.json> <spots>
P=$1; BN=$2; BL=$3; BR=$4; SPOTS=$5
HERE=/c/Users/Nicolas/sandbox/astro_GPU_halo2/experiments/own_starnet
PY=C:/Users/Nicolas/sandbox/astro_GPU/.venv/Scripts/python.exe
O=D:/astro_gpu_diag/own_starnet/$P/H
S=D:/astro_gpu_diag/halo2/$P
export CUDA_VISIBLE_DEVICES=1
$PY $HERE/lost_light.py --snap $S/snapshot --meta $O/prep/meta.json --layers $BN=$BL,runH=$O/runH/star_layer_net.npy --out $O/lost_${BN}_runH.json
$PY $HERE/compare_GH.py --snap $S/snapshot --base $BN=$BL,$S/halo_old.npy --new runH=$O/runH/star_layer_net.npy,$S/halo_new.npy --meta $O/prep/meta.json --out $O --spots "$SPOTS" --gmax ${GMAX:-6.0}
$PY $HERE/table_GH.py --runs $BN=$BR,runH=$O/runH --lost $BN=$O/lost_${BN}_runH.json,runH=$O/lost_${BN}_runH.json > $O/tables_${BN}_runH.md
echo "post done $P"
