#!/bin/bash
# det_restart.sh — 第26轮确定性排查: 诊断重启窗口 (停 → 起(带 K200_PROF=1) → 健康 → 恢复 crontab)
#   单一窗口, 目标停机 <=90s。失败则退回 start.sh 常规启动, 绝不留停机。
set -u
cd /home/caden/ornc
TS(){ date '+%F %T'; }
echo "=== det_restart START $(TS) ==="
crontab -l > /home/caden/ornc/crontab.bak.agent26d 2>/dev/null
crontab -l 2>/dev/null | sed 's|^\(\*/5 \* \* \* \* /bin/bash /home/caden/ornc/svc_guard.sh\)|#AGENT26D# \1|' | crontab -
echo "--- crontab(窗口内) ---"; crontab -l
echo "--- T_STOP $(TS) ---"
bash stop.sh
sleep 3
n=0; for d in /proc/[0-9]*/fd/*; do t=$(readlink "$d" 2>/dev/null); case "$t" in /dev/xpu*) echo "  持卡: $d -> $t"; n=$((n+1));; esac; done
echo "  停后持卡 fd 数 = $n"
echo "--- T_START $(TS) ---"
setsid nohup env K200_PROF=1 python3 /home/caden/ornc/serve2.py >> /home/caden/ornc/serve2.log 2>&1 < /dev/null &
RDY=0
for i in $(seq 1 60); do
  sleep 2
  H=$(curl -s --max-time 5 http://127.0.0.1:8090/health 2>/dev/null)
  case "$H" in *'ready": true'*) echo "[det_restart] READY after $((i*2))s --- T_READY $(TS)"; RDY=1; break;; esac
done
if [ "$RDY" != 1 ]; then echo "[det_restart] 未就绪 => 退回 start.sh"; bash start.sh; fi
if [ -f /home/caden/ornc/crontab.bak.agent26d ]; then crontab /home/caden/ornc/crontab.bak.agent26d && echo "crontab 已原样恢复"; fi
echo "--- crontab 行数 = $(crontab -l | wc -l) ---"
echo "--- T_HEALTH $(TS) ---"
curl -s --max-time 6 http://127.0.0.1:8090/health; echo
echo "异常累计 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log 2>/dev/null)"
echo "reset_count = $(cat /proc/xpu/dev0/reset_count 2>/dev/null) / $(cat /proc/xpu/dev1/reset_count 2>/dev/null)"
echo "state = $(cat /proc/xpu/dev0/state 2>/dev/null) / $(cat /proc/xpu/dev1/state 2>/dev/null)"
echo "=== det_restart END $(TS) ==="
