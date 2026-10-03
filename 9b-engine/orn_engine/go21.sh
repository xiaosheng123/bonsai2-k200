#!/bin/bash
# go21.sh — 后台启动 stage21.sh (彻底脱离 ssh 会话: setsid + 全部 fd 重定向)
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
: > /home/caden/orn_engine/stage21.out
setsid nohup bash /home/caden/orn_engine/stage21.sh </dev/null >>/home/caden/orn_engine/stage21.out 2>&1 &
echo "launched pid=$!"
