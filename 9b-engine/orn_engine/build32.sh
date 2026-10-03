#!/bin/bash
# build32.sh — 第32轮编译: orn3.cb32.cpp -> orn3.new32 (链接生产同款 vis23.o, 硬校验符号大小)
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
echo "== 视觉目标文件校验 (只允许 vis23.o, 防上一轮的退化事故) =="
for f in vis.o vis23.o vis26.o; do
  [ -f "$f" ] || continue
  printf "%-10s md5=%s vis_forward=%s\n" "$f" "$(md5sum $f | cut -c1-8)" \
    "$(nm -S --defined-only $f 2>/dev/null | awk '/ T _Z11vis_forward/ {print $2; exit}')"
done
VISOBJ=vis23.o
VF=$(nm -S --defined-only $VISOBJ | awk '/ T _Z11vis_forward/ {print $2; exit}')
if [ "$VF" != "00000000000052f4" ]; then echo "FATAL: $VISOBJ 不是生产同款视觉塔 (vis_forward=$VF, 期望 52f4)"; exit 3; fi
echo "== 编译 orn3.cb32.cpp (-O2 -march=native -fopenmp, 与生产同参) =="
nice -n 19 g++ -std=c++11 -O2 -march=native -fopenmp -I$X/include -I$RT/include -I. -c orn3.cb32.cpp -o orn3cb32.o 2> orn3cb32_cpp.log || { echo "!! 编译失败"; tail -60 orn3cb32_cpp.log; exit 1; }
echo "== 链接 (vis23.o) =="
nice -n 19 g++ orn3cb32.o vis23.o kq8.proxy.o kq8_host.o -o orn3.new32 -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> orn3cb32_link.log || { echo "!! 链接失败"; tail -40 orn3cb32_link.log; exit 1; }
echo "== 结果 =="
ls -la orn3.new32
md5sum orn3.new32
nm -S --defined-only orn3.new32 | awk '/ T _Z11vis_forward/ {print "orn3.new32 vis_forward = " $2}'
echo BUILD32_OK
