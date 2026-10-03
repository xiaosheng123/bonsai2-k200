#!/bin/bash
# 构建 orn 引擎: kq8.xpu (设备内核) + orn.cpp (主机)
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine

echo "== device compile kq8 =="
$X/bin/clang -I$RT/include -I. -std=c++11 -O2 -fno-builtin -o kq8.sec kq8.xpu --xpu-device-only -c > dev.log 2>&1 || { tail -30 dev.log; exit 1; }
tail -3 dev.log || true
$X/bin/xpu-elfconv kq8.sec kq8.proxy.o $X/bin/clang > elf.log 2>&1 || { tail -20 elf.log; exit 1; }
echo "== host stub =="
$X/bin/clang -I$RT/include -I. -std=c++11 -o kq8_host.o kq8.xpu --xpu-host-only -fPIC -c > host.log 2>&1 || { tail -30 host.log; exit 1; }
echo "== compile orn.cpp =="
g++ -std=c++11 -O2 -march=native -I$RT/include -I. -c orn.cpp -o orn.o > cpp.log 2>&1 || { tail -40 cpp.log; exit 1; }
echo "== link =="
g++ orn.o kq8.proxy.o kq8_host.o -o orn -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp > link.log 2>&1 || { tail -30 link.log; exit 1; }
ls -la orn
