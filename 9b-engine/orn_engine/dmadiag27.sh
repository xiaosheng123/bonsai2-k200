#!/bin/bash
# dmadiag27.sh — 第27轮: DMA 通路诊断 (厂商工具 test_dma) + 等待后再试一次引擎 load
#   目的: 把"H2D 到底坏在哪一层"钉死成证据; 只做诊断, 不改任何配置, 不动驱动模块。
set -u
L=/home/caden/ornc/dmadiag27.log
exec >> "$L" 2>&1
echo "=================================================================="
echo "=== dmadiag27 start $(date '+%F %T') ==="
echo "--- test_dma 用法 (裸跑) ---"
timeout 60 /usr/local/xpu-4.33.0/tools/test_dma 2>&1 | head -12
echo "--- test_dma 0 (safe_run 内) ---"
cd /home/caden/orn_engine
timeout 120 bash safe_run.sh -n p27dma -t 60 -- /usr/local/xpu-4.33.0/tools/test_dma 0 2>&1 | tail -15
echo "--- 等 60s ---"; sleep 60
echo "--- 再试一次: 老生产件 orn3 离线 load (safe_run -t 45) ---"
timeout 180 bash safe_run.sh -n p27retry -t 45 -- /bin/sh -c 'exec env LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib VIS_CH=96 VIS_CHA=0 VIS_FOLD=1 ./orn3 --n 8 --model /home/caden/orn/Ornith-1.5-9B-Q8_0.gguf < /dev/null' 2>&1 | tail -8
echo "  异常累计 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log 2>/dev/null)"
echo "  reset_count = $(cat /proc/xpu/dev0/reset_count 2>/dev/null) / $(cat /proc/xpu/dev1/reset_count 2>/dev/null)"
echo "  state = $(cat /proc/xpu/dev0/state 2>/dev/null) / $(cat /proc/xpu/dev1/state 2>/dev/null)"
echo "--- dmesg 尾 6 ---"; dmesg | tail -6
echo "=== dmadiag27 done $(date '+%F %T') ==="
