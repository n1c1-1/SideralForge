#!/bin/sh
# block_actions: run scenarios one after the other, each in its own process with a hard timeout.
# usage: sh explore_blocks_batch.sh <lang> <scenario>...
cd /c/Users/Nicolas/sandbox/astro || exit 1
OUT=F:/astro_essais/A_REGARDER/exploration/block_actions
lang=$1
shift
for sc in "$@"; do
  PYTHONIOENCODING=utf-8 timeout 2500 C:/Users/Nicolas/sandbox/ComfyUI/.venv/Scripts/python.exe \
    tools/gui_explore/explore_blocks_run.py "$sc" "$lang" 2400 > "$OUT/log_${sc}_${lang}.txt" 2>&1
  echo "$sc exit $?" >> "$OUT/batch_status.txt"
done
