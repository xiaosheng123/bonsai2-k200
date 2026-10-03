#!/bin/bash
# build_fresh.sh
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
C=$X/bin/clang
echo "== device compile =="
$C -I$X/include -I$RT/include -std=c++11 -c --xpu-device-only fresh.xpu -o fresh.sec 2>&1 | head
echo "== elfconv =="
cd $X/bin
PATH=$X/bin:$PATH $X/bin/xpu-elfconv /home/caden/orn_engine/fresh.sec /home/caden/orn_engine/fresh.proxy.o clang 2>&1 | head
cd /home/caden/orn_engine
echo "== host compile =="
$C --xpu-host-only -I$X/include -I$RT/include -std=c++11 -c fresh.xpu -o fresh.host.o 2>&1 | head
echo "== link =="
g++ -std=c++11 -O2 -I$X/include -I$RT/include -I. fresh.cpp fresh.host.o fresh.proxy.o -o fresh -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm 2>&1 | head
echo "== run =="
LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib ./fresh 2>&1 | grep -v WARN
echo DONE