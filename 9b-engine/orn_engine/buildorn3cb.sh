#!/bin/bash
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
echo "== compile orn3.cb.cpp =="
g++ -std=c++11 -O2 -march=native -ffp-contract=off -fopenmp -I$X/include -I$RT/include -I. -c orn3.cb.cpp -o orn3cb.o 2> orn3cb_cpp.log || { tail -60 orn3cb_cpp.log; exit 1; }
echo "== link =="
g++ delta_net_merged.host.o delta_net_merged.proxy.o delta_net_merged_multi.host.o delta_net_merged_multi.proxy.o orn3cb.o vis.o kq8.proxy.o kq8_host.o delta_net.host.o delta_net.proxy.o -o orn3.cb.new -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> orn3cb_link.log || { tail -30 orn3cb_link.log; exit 1; }
md5sum orn3.cb.new
ls -la orn3.cb.new
echo BUILD_OK