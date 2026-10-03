#!/bin/bash
# 构建 mb3 (K200 CL1 微基准)
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine/mb
echo "--- device compile"
$X/bin/clang -I$RT/include -I. -std=c++11 -O2 -fno-builtin -o mb3.sec mb3.xpu --xpu-device-only -c > mb3_dev.log 2>&1 || { tail -40 mb3_dev.log; exit 1; }
echo "--- elfconv"
$X/bin/xpu-elfconv mb3.sec mb3.proxy.o $X/bin/clang > mb3_elf.log 2>&1 || { tail -20 mb3_elf.log; exit 1; }
echo "--- host stub"
$X/bin/clang -I$RT/include -I. -std=c++11 -o mb3_host.o mb3.xpu --xpu-host-only -fPIC -c > mb3_host.log 2>&1 || { tail -40 mb3_host.log; exit 1; }
echo "--- driver"
g++ -std=c++11 -O2 -march=native -I$RT/include -I. -c mb3.cpp -o mb3.o > mb3_cpp.log 2>&1 || { tail -40 mb3_cpp.log; exit 1; }
echo "--- link"
g++ mb3.o mb3.proxy.o mb3_host.o -o mb3 -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp > mb3_link.log 2>&1 || { tail -30 mb3_link.log; exit 1; }
ls -la mb3
echo BUILD_OK
