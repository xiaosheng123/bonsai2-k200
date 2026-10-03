#!/bin/bash
# runprobe.sh <probe args...> — 单个探针, 走 safe_run 安全网
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
cd /home/caden/orn_engine
NM="p_$1"
bash safe_run.sh -n "$NM" -t 120 -- ./probe "$@"
echo "[runprobe] rc=$? arg=$*"
