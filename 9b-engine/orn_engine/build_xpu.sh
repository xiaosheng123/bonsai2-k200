#!/bin/bash
# build_xpu.sh NAME — builds NAME.xpu + NAME.cpp -> NAME binary
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
N=$1
cd /home/caden/orn_engine
C=$X/bin/clang
echo "== device compile =="
$C -I$X/include -I$RT/include -std=c++11 -c --xpu-device-only $N.xpu -o $N.sec 2>&1 | head
echo "== elfconv =="
cd $X/bin
PATH=$X/bin:$PATH $X/bin/xpu-elfconv /home/caden/orn_engine/$N.sec /home/caden/orn_engine/$N.proxy.o clang 2>&1 | head
cd /home/caden/orn_engine
echo "== host compile =="
$C --xpu-host-only -I$X/include -I$RT/include -std=c++11 -c $N.xpu -o $N.host.o 2>&1 | head
echo "== link =="
g++ -std=c++11 -O2 -I$X/include -I$RT/include -I. $N.cpp $N.host.o $N.proxy.o -o $N -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm 2>&1 | head
echo "== run =="
LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib ./$N 2>&1 | grep -viE "warn|loaded"
echo DONE