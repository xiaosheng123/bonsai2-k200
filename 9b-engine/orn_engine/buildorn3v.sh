#!/bin/bash
# buildorn3v.sh — 编译带视觉塔的 orn3 (文本路径源码零改动, 仅增量)
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
echo "== compile orn3.cpp =="
g++ -std=c++11 -O2 -march=native -fopenmp -I$X/include -I$RT/include -I. -c orn3.cpp -o orn3.o 2> orn3_cpp.log || { tail -40 orn3_cpp.log; exit 1; }
echo "== compile vis.cpp =="
g++ -std=c++11 -O3 -march=native -fopenmp -I$X/include -I$RT/include -I. -c vis.cpp -o vis.o 2> vis_cpp.log || { tail -40 vis_cpp.log; exit 1; }
echo "== link =="
g++ orn3.o vis.o kq8.proxy.o kq8_host.o -o orn3 -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> orn3_link.log || { tail -30 orn3_link.log; exit 1; }
ls -la orn3
echo BUILD_OK
