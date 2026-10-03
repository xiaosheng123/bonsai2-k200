#!/bin/bash
# go23.sh — 单实例后台启动 stage27
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
n=$(pgrep -fc 'stage2[0-9].sh')
if [ "$n" -gt 0 ]; then echo "ABORT: 已有 stage 实例 $n 个在跑"; exit 1; fi
: > /home/caden/orn_engine/stage27.out
setsid nohup bash /home/caden/orn_engine/stage27.sh </dev/null >>/home/caden/orn_engine/stage27.out 2>&1 &
echo "launched pid=$!"
