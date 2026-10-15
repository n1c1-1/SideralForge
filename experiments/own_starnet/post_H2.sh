#!/bin/bash
# post_H2.sh <project> <base name> <base run dir> <spots>
P=$1; BN=$2; BR=$3; SPOTS=$4
HERE=/c/Users/Nicolas/sandbox/astro_GPU_halo2/experiments/own_starnet
PY=C:/Users/Nicolas/sandbox/astro_GPU/.venv/Scripts/python.exe
O=D:/astro_gpu_diag/own_starnet/$P/H
S=D:/astro_gpu_diag/halo2/$P
BL=$BR/star_layer_net.npy
LAY="$BN=$BL,runH2=$O/runH2/star_layer_net.npy"
[ -f $O/runH/star_layer_net.npy ] && LAY="$BN=$BL,runH=$O/runH/star_layer_net.npy,runH2=$O/runH2/star_layer_net.npy"
$PY $HERE/lost_light.py --snap $S/snapshot --meta $O/prep/meta.json --layers $LAY --out $O/lost_${BN}_runH2.json
$PY $HERE/compare_GH.py --snap $S/snapshot --base $BN=$BL,$S/halo_old.npy --new runH2=$O/runH2/star_layer_net.npy,$S/halo_new.npy --meta $O/prep/meta.json --out $O --spots "$SPOTS" --gmax ${GMAX:-6.0}
RUNS="$BN=$BR,runH2=$O/runH2"; LST="$BN=$O/lost_${BN}_runH2.json,runH2=$O/lost_${BN}_runH2.json"
[ -f $O/runH/metrics.json ] && RUNS="$BN=$BR,runH=$O/runH,runH2=$O/runH2" && LST="$BN=$O/lost_${BN}_runH2.json,runH=$O/lost_${BN}_runH2.json,runH2=$O/lost_${BN}_runH2.json"
$PY $HERE/table_GH.py --runs $RUNS --lost $LST --stars "3898,406;2968,1731;3133,2055" > $O/tables_${BN}_runH2.md
RC="$BN=$BL,$S/halo_old.npy;runH2=$O/runH2/star_layer_net.npy,$S/halo_new.npy"
[ -f $O/runH/star_layer_net.npy ] && RC="$BN=$BL,$S/halo_old.npy;runH=$O/runH/star_layer_net.npy,$S/halo_new.npy;runH2=$O/runH2/star_layer_net.npy,$S/halo_new.npy"
$PY C:/Users/Nicolas/sandbox/astro_GPU_halo2/experiments/own_starnet/ring_glows.py $P "$RC" > $O/rings_check_runH2.txt
cat $O/rings_check_runH2.txt
echo "post H2 done $P"
