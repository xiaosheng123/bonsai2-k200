#!/bin/bash
# buildorn3b.sh — 编译 orn3.batch.cpp (prefill 批量化版) -> orn3.batch ; 不改动 orn3/or3.cpp
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
OPT="${OPT:--O2}"
cd /home/caden/orn_engine
echo "== host compile orn3.batch.cpp  OPT=$OPT =="
g++ -std=c++11 $OPT -march=native -fopenmp -I$X/include -I$RT/include -I. -c orn3.batch.cpp -o orn3b.o 2> orn3b_cpp.log || { tail -60 orn3b_cpp.log; exit 1; }
echo "== link =="
g++ orn3b.o kq8.proxy.o kq8_host.o -o orn3.batch -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> orn3b_link.log || { tail -30 orn3b_link.log; exit 1; }
ls -la orn3.batch
echo BUILD_OK
