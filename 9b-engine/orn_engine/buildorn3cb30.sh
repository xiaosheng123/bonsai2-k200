#!/bin/bash
# buildorn3cb30.sh — 第30轮: 编译 orn3.cb30.cpp 并【链接正确的视觉目标文件 vis23.o】
#   ★ 教训: 上一轮 buildorn3cb.sh 链接的是过期的 `vis.o`(Sep22 05:03, vis_forward=0x3488),
#     而不是生产同款 `vis23.o`(vis_forward=0x52f4) ⇒ 视觉塔整体退化(表格读数错位/形状位置错)。
#     这里硬校验 vis_forward 的符号大小, 不符就直接失败。
set -e
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
echo "== 视觉目标文件校验 =="
for f in vis.o vis23.o vis26.o; do
  [ -f "$f" ] || continue
  printf "%-10s md5=%s vis_forward=%s\n" "$f" "$(md5sum $f | cut -c1-8)" \
    "$(nm -S --defined-only $f 2>/dev/null | awk '/ T _Z11vis_forward/ {print $2; exit}')"
done
VISOBJ=vis23.o
VF=$(nm -S --defined-only $VISOBJ | awk '/ T _Z11vis_forward/ {print $2; exit}')
if [ "$VF" != "00000000000052f4" ]; then echo "FATAL: $VISOBJ 不是生产同款视觉塔 (vis_forward=$VF, 期望 52f4)"; exit 3; fi
echo "== 编译 orn3.cb30.cpp (-O2, 与生产同参) =="
nice -n 19 g++ -std=c++11 -O2 -march=native -fopenmp -I$X/include -I$RT/include -I. -c orn3.cb30.cpp -o orn3cb30.o 2> orn3cb30_cpp.log || { tail -60 orn3cb30_cpp.log; exit 1; }
echo "== 链接 (vis23.o) =="
nice -n 19 g++ orn3cb30.o vis23.o kq8.proxy.o kq8_host.o -o orn3.exp -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm -fopenmp 2> orn3cb30_link.log || { tail -40 orn3cb30_link.log; exit 1; }
echo "== 结果 =="
ls -la orn3.exp
md5sum orn3.exp
nm -S --defined-only orn3.exp | awk '/ T _Z11vis_forward/ {print "orn3.exp vis_forward = " $2}'
echo BUILD30_OK
