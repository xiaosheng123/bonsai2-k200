#!/bin/bash
# build26.sh — 编译第 23 轮工作副本 vis.cpp.cards26 (抽干点变稀 + H2D 目的地乒乓)
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
g++ -std=c++11 -O3 -march=native -fopenmp -I$X/include -I$RT/include -I. -x c++ -c vis.cpp.cards26 -o vis26.o 2> vis26_cpp.log || { tail -60 vis26_cpp.log; exit 1; }
g++ vis26.o vistestc21.o -o vistest.cards.new26 -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> vis26_link.log || { tail -30 vis26_link.log; exit 1; }
g++ orn3.o vis26.o kq8.proxy.o kq8_host.o -o orn3.new26 -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> orn3_link26.log || { tail -30 orn3_link26.log; exit 1; }
ls -la vistest.cards.new26 orn3.new26
md5sum vis.cpp.cards26 vistest.cards.new26 orn3.new26
echo BUILD26_OK
