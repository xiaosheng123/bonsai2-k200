#!/bin/bash
# buildorn4.sh <RB> — 构建带"多行批量搬运"内核的引擎 orn4_<RB>
#   RB=1 时等价于老 V2 路径 (kq8r 内核仍编进去, 但不会被调用), 用于先验证符号/链接管线
set -e
RB="${1:-1}"
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine

echo "== device compile kq8r (RB=$RB) =="
$X/bin/clang -I$RT/include -I. -DRB=$RB -std=c++11 -O2 -fno-builtin -o kq8r_$RB.sec kq8r.xpu --xpu-device-only -c > k4_dev_$RB.log 2>&1 || { tail -30 k4_dev_$RB.log; exit 1; }
$X/bin/xpu-elfconv kq8r_$RB.sec kq8r_$RB.proxy.o $X/bin/clang > k4_elf_$RB.log 2>&1 || { tail -20 k4_elf_$RB.log; exit 1; }
$X/bin/clang -I$RT/include -I. -DRB=$RB -std=c++11 -o kq8r_${RB}_host.o kq8r.xpu --xpu-host-only -fPIC -c > k4_host_$RB.log 2>&1 || { tail -30 k4_host_$RB.log; exit 1; }
echo "== host compile orn4.cpp (G_RB_MAX=$RB) =="
g++ -std=c++11 -O2 -march=native -I$RT/include -I. -DG_RB_MAX=$RB -c orn4.cpp -o orn4_$RB.o > o4_cpp_$RB.log 2>&1 || { tail -40 o4_cpp_$RB.log; exit 1; }
echo "== link =="
g++ orn4_$RB.o kq8r_$RB.proxy.o kq8r_${RB}_host.o kq8.proxy.o kq8_host.o -o orn4_$RB \
    -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp > o4_link_$RB.log 2>&1 || { tail -30 o4_link_$RB.log; exit 1; }
ls -la orn4_$RB
echo BUILD_OK
