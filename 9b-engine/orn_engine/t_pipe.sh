#!/bin/bash
# t_pipe.sh — 诊断: orn 引擎在管道 stdin 下是否保持存活并正确服务一个请求
# 用法: t_pipe.sh <模式>  模式: hold=只等不喂; feed=喂一个请求; feed2=喂两个请求
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
B64="PHxpbV9zdGFydHw+dXNlcgoxKzHnrYnkuo7lh6A8fGltX2VuZHw+Cjx8aW1fc3RhcnR8PmFzc2lzdGFudAo8dGhpbms+Cgo8L3RoaW5rPgoK"
MODE="${1:-hold}"
case "$MODE" in
  hold)
    { sleep 30; } | /home/caden/orn_engine/orn --n 4 --model /home/caden/orn/Ornith-1.5-9B-Q8_0.gguf
    echo "ENGINE_RC=$?"
    ;;
  feed)
    { sleep 1; printf '@4@b64:%s\n' "$B64"; sleep 240; } | /home/caden/orn_engine/orn --n 4 --model /home/caden/orn/Ornith-1.5-9B-Q8_0.gguf
    echo "ENGINE_RC=$?"
    ;;
  feed2)
    { sleep 1; printf '@4@b64:%s\n' "$B64"; sleep 200; printf '@8@b64:%s\n' "$B64"; sleep 300; } | /home/caden/orn_engine/orn --n 8 --model /home/caden/orn/Ornith-1.5-9B-Q8_0.gguf
    echo "ENGINE_RC=$?"
    ;;
esac
