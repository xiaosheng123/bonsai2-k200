#!/bin/bash
# 构建 mbench
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine/mb
$X/bin/clang -I$RT/include -I. -std=c++11 -O2 -fno-builtin -o mbench.sec mbench.xpu --xpu-device-only -c > mb_dev.log 2>&1 || { tail -40 mb_dev.log; exit 1; }
$X/bin/xpu-elfconv mbench.sec mbench.proxy.o $X/bin/clang > mb_elf.log 2>&1 || { tail -20 mb_elf.log; exit 1; }
$X/bin/clang -I$RT/include -I. -std=c++11 -o mbench_host.o mbench.xpu --xpu-host-only -fPIC -c > mb_host.log 2>&1 || { tail -40 mb_host.log; exit 1; }
g++ -std=c++11 -O2 -march=native -I$RT/include -I. -c mbench.cpp -o mb.o > mb_cpp.log 2>&1 || { tail -40 mb_cpp.log; exit 1; }
g++ mb.o mbench.proxy.o mbench_host.o -o mbench -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp > mb_link.log 2>&1 || { tail -30 mb_link.log; exit 1; }
ls -la mbench
