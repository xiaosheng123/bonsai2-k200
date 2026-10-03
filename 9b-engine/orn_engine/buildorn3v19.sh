#!/bin/bash
# buildorn3v19.sh — 用 vis.cpp.cards 预编译带视觉塔的 orn3 (输出 orn3.new19, 不碰线上 orn3)
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
echo "== compile vis.cpp.cards =="
g++ -std=c++11 -O3 -march=native -fopenmp -I$X/include -I$RT/include -I. -x c++ -c vis.cpp.cards -o vis19.o 2> vis19_cpp.log || { tail -40 vis19_cpp.log; exit 1; }
echo "== check orn3.o =="
[ -f orn3.o ] || g++ -std=c++11 -O2 -march=native -fopenmp -I$X/include -I$RT/include -I. -c orn3.cpp -o orn3.o 2> orn3_cpp.log || { tail -40 orn3_cpp.log; exit 1; }
echo "== link =="
g++ orn3.o vis19.o kq8.proxy.o kq8_host.o -o orn3.new19 -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> orn3_link19.log || { tail -30 orn3_link19.log; exit 1; }
ls -la orn3.new19
md5sum orn3.new19
echo BUILD_OK
