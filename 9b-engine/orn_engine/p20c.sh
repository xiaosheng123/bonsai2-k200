#!/bin/bash
# p20c.sh — 编译 vis.cpp.cards -> vistest.cards + orn3.new20 (不碰线上 orn3)
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
echo '== compile vis.cpp.cards =='
g++ -std=c++11 -O3 -march=native -fopenmp -I$X/include -I$RT/include -I. -x c++ -c vis.cpp.cards -o vis20.o 2> vis20_cpp.log || { tail -60 vis20_cpp.log; exit 1; }
echo '== compile vistest.cpp =='
g++ -std=c++11 -O3 -march=native -fopenmp -I$X/include -I$RT/include -I. -c vistest.cpp -o vistestc20.o 2> vistestc20_cpp.log || { tail -40 vistestc20_cpp.log; exit 1; }
echo '== link vistest.cards =='
g++ vis20.o vistestc20.o -o vistest.cards -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> viscards20_link.log || { tail -30 viscards20_link.log; exit 1; }
echo '== link orn3.new20 =='
[ -f orn3.o ] || g++ -std=c++11 -O2 -march=native -fopenmp -I$X/include -I$RT/include -I. -c orn3.cpp -o orn3.o 2> orn3_cpp.log
g++ orn3.o vis20.o kq8.proxy.o kq8_host.o -o orn3.new20 -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> orn3_link20.log || { tail -30 orn3_link20.log; exit 1; }
ls -la vistest.cards orn3.new20
md5sum vistest.cards orn3.new20
echo BUILD_OK
