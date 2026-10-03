#!/bin/bash
# build22.sh — 编译第 21 轮 v2 (accdst bugfix + mode4/5)
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
g++ -std=c++11 -O3 -march=native -fopenmp -I$X/include -I$RT/include -I. -x c++ -c vis.cpp.cards24 -o vis24.o 2> vis24_cpp.log || { tail -60 vis24_cpp.log; exit 1; }
g++ vis24.o vistestc21.o -o vistest.cards.new24 -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> vis24_link.log || { tail -30 vis24_link.log; exit 1; }
g++ orn3.o vis24.o kq8.proxy.o kq8_host.o -o orn3.new24 -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> orn3_link22.log || { tail -30 orn3_link22.log; echo "orn3 环节失败(不影响 vistest)"; }
ls -la vistest.cards.new24 orn3.new24 2>/dev/null
md5sum vistest.cards.new24
echo BUILD24_OK
