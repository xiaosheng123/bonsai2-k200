#!/bin/bash
# det_install26.sh — 第26轮上线窗口: 装修复件 orn3.det26 -> orn3, 重启, 健康检查, 失败即回滚
#   单一窗口, 目标停机 <=120s; 失败自动回 24a92510 并重启, 绝不留停机/坏件。
set -u
cd /home/caden/ornc
TS(){ date '+%F %T'; }
echo "=== det_install26 START $(TS) ==="
echo "--- 上线前 md5 ---"; md5sum /home/caden/orn_engine/orn3 /home/caden/ornc/serve2.py
cp -a /home/caden/orn_engine/orn3 /home/caden/orn_engine/orn3.pre26.bak
echo "--- 备份 orn3.pre26.bak ---"; md5sum /home/caden/orn_engine/orn3.pre26.bak
crontab -l > /home/caden/ornc/crontab.bak.agent26i 2>/dev/null
crontab -l 2>/dev/null | sed 's|^\(\*/5 \* \* \* \* /bin/bash /home/caden/ornc/svc_guard.sh\)|#AGENT26I# \1|' | crontab -
echo "--- T_STOP $(TS) ---"
bash stop.sh
sleep 3
n=0; for d in /proc/[0-9]*/fd/*; do t=$(readlink "$d" 2>/dev/null); case "$t" in /dev/xpu*) echo "  持卡: $d -> $t"; n=$((n+1));; esac; done
echo "  停后持卡 fd 数 = $n"
if [ "$n" != "0" ]; then
  echo "!! 有残留持卡进程 => 放弃上线, 直接拉回服务"
  crontab /home/caden/ornc/crontab.bak.agent26i
  bash start.sh
  exit 2
fi
cp -f /tmp/orn3_det /home/caden/orn_engine/orn3 && chmod +x /home/caden/orn_engine/orn3
echo "--- 上线后 md5 ---"; md5sum /home/caden/orn_engine/orn3
echo "--- T_START $(TS) ---"
bash start.sh
echo "--- T_READY $(TS) ---"
H=$(curl -s -m 6 http://127.0.0.1:8090/health)
echo "health: $H"
case "$H" in
  *'"status": "ok"'*) ;;
  *) echo "!! 服务未就绪 => 回滚 24a92510 并重启"
     cp -f /home/caden/orn_engine/orn3.pre26.bak /home/caden/orn_engine/orn3
     bash stop.sh; sleep 3; bash start.sh ;;
esac
if [ -f /home/caden/ornc/crontab.bak.agent26i ]; then crontab /home/caden/ornc/crontab.bak.agent26i; echo "crontab 恢复 $(crontab -l|wc -l) 行"; fi
echo "--- 最终 ---"; md5sum /home/caden/orn_engine/orn3; curl -s -m 6 http://127.0.0.1:8090/health; echo
echo "异常累计 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log 2>/dev/null)"
echo "reset_count = $(cat /proc/xpu/dev0/reset_count 2>/dev/null) / $(cat /proc/xpu/dev1/reset_count 2>/dev/null)"
echo "=== det_install26 END $(TS) ==="
