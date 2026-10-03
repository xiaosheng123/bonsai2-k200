#!/bin/bash
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
BT=${1:-256}
cd /home/caden/orn_engine
VISOBJ=vis23.o
VF=$(nm -S --defined-only $VISOBJ | awk '/ T _Z11vis_forward/ {print $2; exit}')
if [ "$VF" != "00000000000052f4" ]; then echo "FATAL: $VISOBJ vis_forward=$VF (期望 52f4)"; exit 3; fi
echo "== 编译 orn3.cb33.cpp (-DBATCH_MAXT=$BT) =="
nice -n 19 g++ -std=c++11 -O2 -march=native -fopenmp -DBATCH_MAXT=$BT -I$X/include -I$RT/include -I. -c orn3.cb33.cpp -o orn3cb33.o 2> orn3cb33_cpp.log \
  || { echo "!! 编译失败"; tail -60 orn3cb33_cpp.log; exit 1; }
echo "== 链接 -> orn3.b$BT =="
nice -n 19 g++ orn3cb33.o vis23.o kq8.proxy.o kq8_host.o -o orn3.b$BT -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> orn3cb33_link.log \
  || { echo "!! 链接失败"; tail -40 orn3cb33_link.log; exit 1; }
ls -la orn3.b$BT; md5sum orn3.b$BT
echo BUILD33_OK
