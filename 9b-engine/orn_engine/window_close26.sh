#!/bin/bash
# window_close26.sh — 收尾: 原样恢复 crontab + 拉回 8090 服务 + 报卡状态
set -u
cd /home/caden/ornc
echo "=== window_close26 $(date '+%F %T') ==="
if [ -f /home/caden/ornc/crontab.bak.agent26 ]; then
  crontab /home/caden/ornc/crontab.bak.agent26 && echo "crontab 已原样恢复"
fi
echo "--- crontab ---"; crontab -l; echo "--- 行数 ---"; crontab -l | wc -l
for d in /proc/[0-9]*/fd/*; do t=$(readlink "$d" 2>/dev/null); case "$t" in /dev/xpu*) echo "  持卡: $d -> $t";; esac; done
bash start.sh
for i in $(seq 1 40); do
  H=$(curl -s --max-time 5 http://127.0.0.1:8090/health 2>/dev/null)
  case "$H" in *'ready": true'*|*'ready":true'*) echo "READY: $H"; break;; esac
  sleep 5
done
echo "--- 最终 health ---"; curl -s --max-time 6 http://127.0.0.1:8090/health; echo
echo "异常累计 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log 2>/dev/null)"
echo "reset_count dev0/1 = $(cat /proc/xpu/dev0/reset_count 2>/dev/null) / $(cat /proc/xpu/dev1/reset_count 2>/dev/null)"
echo "state dev0/1 = $(cat /proc/xpu/dev0/state 2>/dev/null) / $(cat /proc/xpu/dev1/state 2>/dev/null)"
echo "=== window_close26 done $(date '+%F %T') ==="
