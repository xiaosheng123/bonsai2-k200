#!/bin/bash
# det_restart2.sh — 第26轮: 正确的诊断重启 (复用 start.sh, 额外 export K200_PROF=1 → 透传给引擎)
set -u
cd /home/caden/ornc
TS(){ date '+%F %T'; }
echo "=== det_restart2 START $(TS) ==="
crontab -l > /home/caden/ornc/crontab.bak.agent26d 2>/dev/null
crontab -l 2>/dev/null | sed 's|^\(\*/5 \* \* \* \* /bin/bash /home/caden/ornc/svc_guard.sh\)|#AGENT26D# \1|' | crontab -
echo "--- T_STOP $(TS) ---"
bash stop.sh
sleep 3
n=0; for d in /proc/[0-9]*/fd/*; do t=$(readlink "$d" 2>/dev/null); case "$t" in /dev/xpu*) echo "  持卡: $d -> $t"; n=$((n+1));; esac; done
echo "  停后持卡 fd 数 = $n"
echo "--- T_START $(TS) ---"
export K200_PROF=1
bash start.sh
echo "--- T_READY $(TS) ---"
ps aux | grep -a "orn3 --n" | grep -v grep | head -2
if [ -f /home/caden/ornc/crontab.bak.agent26d ]; then crontab /home/caden/ornc/crontab.bak.agent26d; echo "crontab 恢复行数 = $(crontab -l | wc -l)"; fi
curl -s -m 6 http://127.0.0.1:8090/health; echo
echo "异常累计 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log 2>/dev/null)"
echo "reset_count = $(cat /proc/xpu/dev0/reset_count 2>/dev/null) / $(cat /proc/xpu/dev1/reset_count 2>/dev/null)"
echo "=== det_restart2 END $(TS) ==="
