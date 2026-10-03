#!/bin/bash
# buildprobe.sh — 编译 probe (新算子极小尺寸单发验证)
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
echo "== compile =="
g++ -std=c++11 -O3 -march=native -fopenmp -I$X/include -I$RT/include -I. -c probe.cpp -o probe.o 2> probe_cpp.log || { cat probe_cpp.log; exit 1; }
echo "== link =="
g++ probe.o -o probe -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> probe_link.log || { cat probe_link.log; exit 1; }
ls -la probe
echo BUILD_OK
