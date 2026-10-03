#!/bin/bash
# 恢复旧 MiniCPM5-2B 服务 (8091/8092 引擎 + 8090 网关) —— 不依赖 sudo
set -u
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
cd /home/caden/k200llm

echo "== stop everything =="
pkill -9 -f 'server\.py' ; pkill -9 -f 'lb\.py' ; pkill -9 -f 'mini\.th' ; pkill -9 -f '/orn ' ; sleep 3
ps -ef | grep -aE 'server\.py|lb\.py|mini\.th' | grep -v grep | head -3

echo "== soft_reset 两芯 (无需 sudo) =="
/usr/local/xpu-4.33.0/tools/soft_reset 0 ; echo "  reset0 rc=$?"
/usr/local/xpu-4.33.0/tools/soft_reset 1 ; echo "  reset1 rc=$?"
sleep 3

waitup () {
  for i in $(seq 1 $2); do
    if curl -s -m 2 "http://127.0.0.1:$1/health" >/dev/null 2>&1; then echo "  port $1 ready (${i}s)"; return 0; fi
    sleep 1
  done
  echo "  port $1 NOT ready after ${2}s"; return 1
}

echo "== start 8091 (chip0) =="
XPU_VISIBLE_DEVICES=0 K200_PORT=8091 K200_ENGINE=/home/caden/k200llm/mini.th nohup python3 server.py > y0.log 2>&1 &
waitup 8091 240
sleep 2
echo "== start 8092 (chip1) =="
XPU_VISIBLE_DEVICES=1 K200_PORT=8092 K200_ENGINE=/home/caden/k200llm/mini.th nohup python3 server.py > y1.log 2>&1 &
waitup 8092 240
sleep 1
echo "== start 8090 lb =="
nohup python3 lb.py > lb.log 2>&1 &
waitup 8090 30
echo "== health =="
curl -s -m 5 http://127.0.0.1:8090/v1/models ; echo
