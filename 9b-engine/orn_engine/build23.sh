#!/bin/bash
# build23.sh — 编译第 21 轮 v3 (准备上生产: (B) 默认开启 + useCA 门控)
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
g++ -std=c++11 -O3 -march=native -fopenmp -I$X/include -I$RT/include -I. -x c++ -c vis.cpp.cards23 -o vis23.o 2> vis23_cpp.log || { tail -60 vis23_cpp.log; exit 1; }
g++ vis23.o vistestc21.o -o vistest.cards.new23 -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> vis23_link.log || { tail -30 vis23_link.log; exit 1; }
g++ orn3.o vis23.o kq8.proxy.o kq8_host.o -o orn3.new23 -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> orn3_link23.log || { tail -30 orn3_link23.log; exit 1; }
ls -la vistest.cards.new23 orn3.new23; md5sum vistest.cards.new23 orn3.new23
echo BUILD23_OK
