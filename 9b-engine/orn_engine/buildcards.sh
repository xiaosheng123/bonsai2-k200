#!/bin/bash
# buildcards.sh — 编译 vis.cpp.cards (不碰线上 vis.cpp / orn3)
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
g++ -std=c++11 -O3 -march=native -fopenmp -I$X/include -I$RT/include -I. -x c++ -c vis.cpp.cards -o viscards.o 2> viscards_cpp.log || { cat viscards_cpp.log; exit 1; }
g++ -std=c++11 -O3 -march=native -fopenmp -I$X/include -I$RT/include -I. -c vistest.cpp -o vistestc.o 2> vistestc_cpp.log || { cat vistestc_cpp.log; exit 1; }
g++ viscards.o vistestc.o -o vistest.cards -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> viscards_link.log || { cat viscards_link.log; exit 1; }
ls -la vistest.cards
echo BUILD_OK
