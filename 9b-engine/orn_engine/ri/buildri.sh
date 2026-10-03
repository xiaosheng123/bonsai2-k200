#!/bin/bash
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine/ri
echo "== device compile rdinfo =="
$X/bin/clang -I$RT/include -I. -std=c++11 -O2 -fno-builtin -o rdinfo.sec rdinfo.xpu --xpu-device-only -c > dev.log 2>&1 || { tail -30 dev.log; exit 1; }
$X/bin/xpu-elfconv rdinfo.sec rdinfo.proxy.o $X/bin/clang > elf.log 2>&1 || { tail -20 elf.log; exit 1; }
$X/bin/clang -I$RT/include -I. -std=c++11 -o rdinfo_host.o rdinfo.xpu --xpu-host-only -fPIC -c > host.log 2>&1 || { tail -30 host.log; exit 1; }
echo "== host =="
g++ -std=c++11 -O2 -march=native -I$RT/include -I. -c ri.cpp -o ri.o > cpp.log 2>&1 || { tail -40 cpp.log; exit 1; }
g++ ri.o rdinfo.proxy.o rdinfo_host.o -o ri -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp > link.log 2>&1 || { tail -30 link.log; exit 1; }
ls -la ri
echo BUILD_OK
