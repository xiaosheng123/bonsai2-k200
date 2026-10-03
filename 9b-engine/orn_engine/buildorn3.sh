#!/bin/bash
# buildorn3.sh — 编译 orn3 (本轮: GEMV 换成官方 gemm_int8 ⇒ 需要 xtdk 的 api 头/库)
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
echo "== host compile orn3.cpp =="
g++ -std=c++11 -O2 -march=native -I$X/include -I$RT/include -I. -c orn3.cpp -o orn3.o 2> orn3_cpp.log || { tail -40 orn3_cpp.log; exit 1; }
echo "== link (复用 kq8 设备内核 + 官方 libxpuapi) =="
g++ orn3.o kq8.proxy.o kq8_host.o -o orn3 -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> orn3_link.log || { tail -30 orn3_link.log; exit 1; }
ls -la orn3
echo BUILD_OK
