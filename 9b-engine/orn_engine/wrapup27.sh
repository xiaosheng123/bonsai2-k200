#!/bin/bash
# wrapup27.sh — 第27轮收尾: 恢复 crontab + 干净重启服务 (老生产件) + 形成最终台账
#   注意: 卡 H2D DMA 已坏 (厂商 test_dma: HOST_TO_DEVICE fail errno=809), 引擎起不来;
#         本脚本只把系统恢复到"生产件 + crontab 三行 + 服务进程干净"的可自愈状态。
set -u
L=/home/caden/ornc/wrapup27.log
exec >> "$L" 2>&1
echo "=================================================================="
echo "=== wrapup27 start $(date '+%F %T') ==="
cd /home/caden/ornc
echo "--- 恢复 crontab (svc_guard 取消注掉) ---"
if [ -f /home/caden/ornc/crontab.bak.agent27 ]; then
  crontab /home/caden/ornc/crontab.bak.agent27 && echo "  已恢复"; else echo "  备份不在, 手工改回"; fi
echo "  crontab 行数 = $(crontab -l | wc -l)"; crontab -l
echo "--- 生产件 md5 (应为 6785f070… / 247fe23e…) ---"
md5sum /home/caden/orn_engine/orn3 /home/caden/ornc/serve2.py
echo "--- T_STOP $(date '+%F %T') ---"
bash stop.sh
sleep 3
echo "--- 持卡残留 ---"; n=0; for d in /proc/[0-9]*/fd/*; do t=$(readlink "$d" 2>/dev/null); case "$t" in /dev/xpu*) echo "  $d -> $t"; n=$((n+1));; esac; done; echo "  持卡 fd=$n"
echo "--- T_START $(date '+%F %T') ---"
bash start.sh
echo "--- 结果 $(date '+%F %T') ---"
curl -s -m 8 http://127.0.0.1:8090/health; echo
echo "  引擎进程: $(pgrep -af 'orn_engine/orn3 --n' | head -1)"
echo "  异常累计 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log 2>/dev/null)"
echo "  reset_count = $(cat /proc/xpu/dev0/reset_count 2>/dev/null) / $(cat /proc/xpu/dev1/reset_count 2>/dev/null)"
echo "  state = $(cat /proc/xpu/dev0/state 2>/dev/null) / $(cat /proc/xpu/dev1/state 2>/dev/null)"
echo "=== wrapup27 done $(date '+%F %T') ==="
