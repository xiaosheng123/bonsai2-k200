#!/bin/bash
# buildvis.sh — 编译 vistest (视觉塔单机验证程序)
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
echo "== compile =="
g++ -std=c++11 -O3 -march=native -fopenmp -I$X/include -I$RT/include -I. -c vis.cpp -o vis.o 2> vis_cpp.log || { cat vis_cpp.log; exit 1; }
g++ -std=c++11 -O3 -march=native -fopenmp -I$X/include -I$RT/include -I. -c vistest.cpp -o vistest.o 2> vistest_cpp.log || { cat vistest_cpp.log; exit 1; }
echo "== link =="
g++ vis.o vistest.o -o vistest -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> vis_link.log || { cat vis_link.log; exit 1; }
ls -la vistest
echo BUILD_OK
