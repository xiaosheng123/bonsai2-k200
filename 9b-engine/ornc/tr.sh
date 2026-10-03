#!/bin/bash
# tr.sh —— 中间张量 trace: 同一个 token 在逐 token 路径 vs 批量 T=2 下的 layer0 输出 / 末层 x
set -u
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
cd /home/caden/orn_engine || exit 1
K200_TRACE=1 ./orn3.batch --n 1 --gen "你好"
echo "[tr] rc=$?"
