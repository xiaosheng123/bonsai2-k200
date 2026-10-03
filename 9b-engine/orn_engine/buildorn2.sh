#!/bin/bash
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
echo "== host compile orn2.cpp =="
g++ -std=c++11 -O2 -march=native -I$RT/include -I. -c orn2.cpp -o orn2.o 2> orn2_cpp.log || { tail -40 orn2_cpp.log; exit 1; }
echo "== link (复用 kq8 设备内核) =="
g++ orn2.o kq8.proxy.o kq8_host.o -o orn2 -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> orn2_link.log || { tail -30 orn2_link.log; exit 1; }
ls -la orn2
echo BUILD_OK
