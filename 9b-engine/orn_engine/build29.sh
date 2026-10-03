#!/bin/bash
# build29.sh — 第24轮: 主机侧编译 MAXT=2048 + rope 查表 + PPROF 插桩版
#   只产出 orn3.m29 (新的工作二进制), 绝不碰线上 orn3 / orn3.o。
#   ★ 必须复用生产同款编译参数: g++ -std=c++11 -O2 -march=native -fopenmp
#   ★ 链接必须复用生产同款目标文件 vis24.o (orn3.new24 == 线上 orn3, md5 24a92510…)
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
echo "== 预检: 生产件 md5 =="
md5sum orn3 orn3.o vis24.o kq8.proxy.o kq8_host.o
echo "== 编译 orn3_m.cpp (与生产同参) =="
g++ -std=c++11 -O2 -march=native -fopenmp -I$X/include -I$RT/include -I. -c orn3_m.cpp -o orn3_m.o 2> orn3_m_cpp.log || { tail -60 orn3_m_cpp.log; exit 1; }
echo "== 链接 =="
g++ orn3_m.o vis24.o kq8.proxy.o kq8_host.o -o orn3.m29 -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> orn3_m_link.log || { tail -40 orn3_m_link.log; exit 1; }
ls -la orn3.m29
md5sum orn3.m29
echo BUILD29_OK
