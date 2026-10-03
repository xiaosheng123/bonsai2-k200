#!/bin/bash
# run_golden.sh <binary> <outlog> —— 离线黄金测试 ("你好"), 包在 safe_run 里
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
cd /home/caden/orn_engine || exit 1
BIN=${1:-/home/caden/orn_engine/orn3new}
OUT=${2:-/tmp/golden_new.log}
if [ -n "${K200_CH:-}" ]; then export K200_CH; fi
if [ -n "${K200_CH_DOWN:-}" ]; then export K200_CH_DOWN; fi
SAFE_TEE_STDOUT=1 bash safe_run.sh -n orn_golden -t 1800 -- "$BIN" \
  --model /home/caden/orn/Ornith-1.5-9B-Q8_0.gguf --gen "你好" --n 24 > "$OUT" 2>&1
echo "RC=$?" >> "$OUT"
