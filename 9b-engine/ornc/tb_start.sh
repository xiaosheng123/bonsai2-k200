#!/bin/bash
# tb_start.sh — 启动 8090 服务, 引擎用 orn3.batch; 额外 env 由调用方 export
cd /home/caden/ornc || exit 1
export K200_ENGINE=${K200_ENGINE:-/home/caden/orn_engine/orn3.batch}
export K200_MODEL_PATH=/home/caden/orn/Ornith-1.5-9B-Q8_0.gguf
export K200_PORT=8090
export K200_NGEN=64
export K200_MODEL=ornith-1.5-9b-k200
export K200_SAFEMAXSEC=28800
export K200_RAWTOK=1
echo "[tb_start] ENGINE=$K200_ENGINE BMIN=${K200_BMIN:-def} BMAX=${K200_BMAX:-def} PROF=${K200_PROF:-0} LDUMP=${K200_LDUMP:-none}"
setsid nohup python3 serve2.py >> serve2.log 2>&1 < /dev/null &
disown 2>/dev/null || true
echo "[tb_start] launched pid=$!"
exit 0
