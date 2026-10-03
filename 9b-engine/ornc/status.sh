#!/bin/bash
# status.sh — 8090 服务 + 两芯 + 卡异常计数 一览 (只读, 不碰卡)
set -u
echo "=== status.sh $(date '+%F %T') ==="
echo "--- 8090 /health ---"
timeout 6 curl -s http://127.0.0.1:8090/health || echo "  (无响应)"
echo
echo "--- 进程 ---"
pgrep -a -f 'serve[0-9]*\.py' || echo "  server: 无"
pgrep -a -f 'safe_run\.sh -n orn' || echo "  safe_run: 无"
pgrep -a -f 'orn_engine/orn[0-9]*' || echo "  engine: 无"
echo "--- 持卡者 (按 /proc/<pid>/fd 实测) ---"
n=0
for p in /proc/[0-9]*; do
  pid=${p#/proc/}
  [ "$pid" = "$$" ] && continue
  [ "$pid" = "$PPID" ] && continue
  if ls -l "$p/fd" 2>/dev/null | grep -q '/dev/xpu'; then
    echo "  pid=$pid $(cat "$p/cmdline" 2>/dev/null | tr -d '\0' | head -c 90)"; n=$((n+1))
  fi
done
echo "  持卡进程数=$n"
echo "--- 两芯 state ---"
printf '  dev0=%s dev1=%s\n' "$(cat /proc/xpu/dev0/state 2>/dev/null)" "$(cat /proc/xpu/dev1/state 2>/dev/null)"
echo "--- 卡异常累计 ---"
echo "  Exception in kernel execution = $(grep -ac 'Exception in kernel execution' /var/log/kern.log 2>/dev/null)"
echo "  distinct exception token     = $(grep -a 'exception token=' /var/log/kern.log 2>/dev/null | sed -E 's/.*exception token=([0-9]+).*/\1/' | sort -nu | wc -l)"
echo "--- watchdog ---"
pgrep -a -f watchdog.py || echo "  watchdog: 无"
