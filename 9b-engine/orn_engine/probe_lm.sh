#!/bin/bash
# 定位: 模块加载失败是不是因为 __local__ 太大 / 多 kernel 同文件
set -u
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
X=/home/caden/xtdk/xtdk-x86_64; RT=/usr/local/xpu-4.33.0
cd /home/caden/orn_engine
CC="$X/bin/clang -I$RT/include -I. -std=c++11 -O2 -fno-builtin"

build_run () {   # $1=名字  $2=设备源码
  local n=$1 src=$2
  $CC -o $n.sec $src.xpu --xpu-device-only -c > /dev/null 2>&1 || { echo "  [$n] device compile FAIL"; return; }
  $X/bin/xpu-elfconv $n.sec $n.proxy.o $X/bin/clang > /dev/null 2>&1 || { echo "  [$n] elfconv FAIL"; return; }
  $CC -o $n.host.o $src.xpu --xpu-host-only -fPIC -c > /dev/null 2>&1 || { echo "  [$n] host compile FAIL"; return; }
  g++ -std=c++11 -O2 -march=native -I$RT/include probe_lm.cpp $n.proxy.o $n.host.o -o $n -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm >/dev/null 2>&1 || { echo "  [$n] link FAIL"; return; }
  /usr/local/xpu-4.33.0/tools/soft_reset 1 >/dev/null 2>&1; sleep 1
  echo "  [$n] -> $(./$n 2>&1 | grep -av XPURT | tr '\n' ' ')"
}

build_run m_none  k_none   # 只有 map
build_run m_rd16  k_rd16   # map + 16KB local 读
build_run m_lm4   k_lm4
build_run m_lm8   k_lm8
build_run m_lm16  k_lm16
build_run m_lm32  k_lm32
echo DONE
