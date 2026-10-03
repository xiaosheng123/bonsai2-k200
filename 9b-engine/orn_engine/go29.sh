#!/bin/bash
# go29.sh — 单实例后台启动 stage29 (卡窗口作业跑过 ssh 会话寿命也不会被掐断)
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
n=$(pgrep -fc 'stage29.sh')
if [ "${n:-0}" -gt 0 ]; then echo "ABORT: 已有 stage29 实例 $n 个在跑"; exit 1; fi
: > /home/caden/orn_engine/stage29.out
setsid nohup bash /home/caden/orn_engine/stage29.sh </dev/null >>/home/caden/orn_engine/stage29.out 2>&1 &
echo "launched pid=$!"
sleep 5
tail -5 /home/caden/orn_engine/stage29.out
