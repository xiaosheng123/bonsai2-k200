#!/bin/bash
# =============================================================================
# build_atn_probe.sh — 编译 ATN_ON_CARD 阶段①探针 (纯主机编译, 【不产生设备码】=> 不碰卡)
#   产物: /home/caden/orn_engine/atn_probe
#   ★ 只用 nice -n 19 + -O1: 编译期间生产服务可能正在跑, 不许抢 CPU (参考 bench-vs-service-conflict)
# =============================================================================
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
echo "== compile (nice -19, -O1) =="
nice -n 19 g++ -std=c++11 -O1 -fopenmp -I$X/include -I$RT/include -I. -c atn_probe.cpp -o atn_probe.o 2> atn_probe_cpp.log || { cat atn_probe_cpp.log; exit 1; }
echo "== link =="
nice -n 19 g++ atn_probe.o -o atn_probe -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> atn_probe_link.log || { cat atn_probe_link.log; exit 1; }
ls -la atn_probe
md5sum atn_probe
echo BUILD_ATN_PROBE_OK
