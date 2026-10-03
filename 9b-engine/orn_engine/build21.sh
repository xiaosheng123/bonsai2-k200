#!/bin/bash
# build21.sh — 编译第 21 轮工作副本 (不碰线上 vis.cpp / orn3 / vistest.cards)
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
echo "== compile vis.cpp.cards21 =="
g++ -std=c++11 -O3 -march=native -fopenmp -I$X/include -I$RT/include -I. -x c++ -c vis.cpp.cards21 -o vis21.o 2> vis21_cpp.log || { tail -60 vis21_cpp.log; exit 1; }
echo "== compile vistest.cpp =="
g++ -std=c++11 -O3 -march=native -fopenmp -I$X/include -I$RT/include -I. -c vistest.cpp -o vistestc21.o 2> vistestc21_cpp.log || { tail -40 vistestc21_cpp.log; exit 1; }
echo "== link vistest.cards.new21 =="
g++ vis21.o vistestc21.o -o vistest.cards.new21 -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> vis21_link.log || { tail -30 vis21_link.log; exit 1; }
echo "== compile probe.cpp21 (必须 -x c++, 否则 .cpp21 后缀会被 g++ 当成目标文件而静默跳过) =="
g++ -std=c++11 -O3 -march=native -fopenmp -I$X/include -I$RT/include -I. -x c++ -c probe.cpp21 -o probe21.o 2> probe21_cpp.log || { tail -60 probe21_cpp.log; exit 1; }
echo "== link probe21 =="
g++ probe21.o -o probe21 -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> probe21_link.log || { tail -30 probe21_link.log; exit 1; }
echo "== link orn3.new21 (预备; 不安装) =="
[ -f orn3.o ] || g++ -std=c++11 -O2 -march=native -fopenmp -I$X/include -I$RT/include -I. -c orn3.cpp -o orn3.o 2> orn3_cpp.log
g++ orn3.o vis21.o kq8.proxy.o kq8_host.o -o orn3.new21 -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> orn3_link21.log || { tail -30 orn3_link21.log; echo "orn3 环节失败(不影响 vistest/probe)"; }
ls -la vistest.cards.new21 probe21 orn3.new21 2>/dev/null
md5sum vistest.cards.new21 probe21 2>/dev/null
echo BUILD21_OK
