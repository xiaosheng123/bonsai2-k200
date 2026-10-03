#!/bin/bash
# svc_guard.sh —— 8090 服务健康守卫（夜间无人值守保险）
# 只有"8090 不健康 + 卡上没有任何作业在跑"时才自动 start.sh；卡上有活就不碰
LOG=/home/caden/ornc/svc_guard.log
H=$(curl -s --max-time 6 http://127.0.0.1:8090/health 2>/dev/null)
case "$H" in *'"status": "ok"'*|*'"status":"ok"'*) exit 0;; esac
BUSY=0
for d in /proc/[0-9]*/fd/*; do
  t=$(readlink "$d" 2>/dev/null)
  case "$t" in /dev/xpu*) BUSY=1; break;; esac
done
pgrep -f "safe_run.sh -n orn_serve" >/dev/null 2>&1 && BUSY=1
pgrep -f "orn_engine/orn3" >/dev/null 2>&1 && BUSY=1
if [ "$BUSY" = 1 ]; then
  echo "$(date '+%F %T') 8090 不健康, 但卡上有作业在跑 => 不重启(防抢卡)" >> "$LOG"; exit 0
fi
echo "$(date '+%F %T') 8090 不健康且卡空闲 => 自动 start.sh" >> "$LOG"
bash /home/caden/ornc/start.sh >> "$LOG" 2>&1
echo "$(date '+%F %T') 重启后 health: $(curl -s --max-time 6 http://127.0.0.1:8090/health | head -c 120)" >> "$LOG"
