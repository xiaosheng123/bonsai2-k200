#!/bin/bash
# build25.sh — 第 22 轮: 编译 vis.cpp.cards25 -> vistest.cards.new25 + orn3.new25
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
g++ -std=c++11 -O3 -march=native -fopenmp -I$X/include -I$RT/include -I. -x c++ -c vis.cpp.cards25 -o vis25.o 2> vis25_cpp.log || { tail -60 vis25_cpp.log; exit 1; }
g++ vis25.o vistestc21.o -o vistest.cards.new25 -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> vis25_link.log || { tail -30 vis25_link.log; exit 1; }
g++ orn3.o vis25.o kq8.proxy.o kq8_host.o -o orn3.new25 -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> orn3_link25.log || { tail -30 orn3_link25.log; echo "orn3 环节失败(不影响 vistest)"; }
ls -la vistest.cards.new25 orn3.new25 2>/dev/null
md5sum vistest.cards.new25 orn3.new25
echo BUILD25_OK
