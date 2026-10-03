#!/bin/bash
# =============================================================================
# det_install27.sh — 第27轮上线窗口: LCP 前缀复用件 orn3.new27 -> orn3 (+ 网关 serve2.py)
#   单一窗口: 停服务 -> 换件 -> 拉服务 -> 等 ready -> 报 md5/卡台账; 未就绪自动回滚。
#   目标停机 <=120s (实测 stop->ready 见 T_STOP/T_READY)。
# =============================================================================
set -u
cd /home/caden/ornc
TS(){ date '+%F %T'; }
T0=$(date +%s)
echo "=== det_install27 START $(TS) ==="
echo "--- 上线前 md5 ---"; md5sum /home/caden/orn_engine/orn3 /home/caden/ornc/serve2.py
[ -f /home/caden/orn_engine/orn3.pre27.bak ] || cp -a /home/caden/orn_engine/orn3 /home/caden/orn_engine/orn3.pre27.bak
[ -f /home/caden/ornc/serve2.pre27.bak ] || cp -a /home/caden/ornc/serve2.py /home/caden/ornc/serve2.pre27.bak
echo "--- 备份 ---"; md5sum /home/caden/orn_engine/orn3.pre27.bak /home/caden/ornc/serve2.pre27.bak
crontab -l > /home/caden/ornc/crontab.bak.agent27 2>/dev/null
crontab -l 2>/dev/null | sed 's|^\(\*/5 \* \* \* \* /bin/bash /home/caden/ornc/svc_guard.sh\)|#AGENT27# \1|' | crontab -
echo "--- crontab 已临时注掉 svc_guard ---"; crontab -l | head -5
echo "--- T_STOP $(TS) ---"
bash stop.sh
sleep 3
n=0; for d in /proc/[0-9]*/fd/*; do t=$(readlink "$d" 2>/dev/null); case "$t" in /dev/xpu*) echo "  持卡: $d -> $t"; n=$((n+1));; esac; done
echo "  停后持卡 fd 数 = $n"
if [ "$n" != "0" ]; then
  echo "!! 有残留持卡进程 => 放弃上线, 直接拉回服务"
  crontab /home/caden/ornc/crontab.bak.agent27; bash start.sh; exit 2
fi
cp -f /home/caden/orn_engine/orn3.new27 /home/caden/orn_engine/orn3 && chmod +x /home/caden/orn_engine/orn3
cp -f /tmp/serve2.py /home/caden/ornc/serve2.py
echo "--- 上线后 md5 ---"; md5sum /home/caden/orn_engine/orn3 /home/caden/ornc/serve2.py
echo "--- T_START $(TS) ---"
bash start.sh
echo "--- T_READY $(TS) ---"
T1=$(date +%s); echo "  停机时长(stop->ready) = $((T1-T0)) s"
H=$(curl -s -m 8 http://127.0.0.1:8090/health)
echo "health: $H"
case "$H" in
  *'"status": "ok'*) echo "  ready ✓";;
  *) echo "!! 服务未就绪 => 回滚到 orn3.pre27.bak + serve2.pre27.bak 并重启"
     cp -f /home/caden/orn_engine/orn3.pre27.bak /home/caden/orn_engine/orn3
     cp -f /home/caden/ornc/serve2.pre27.bak /home/caden/ornc/serve2.py
     bash stop.sh; sleep 3; bash start.sh; T1=$(date +%s); echo "  回滚后停机时长 = $((T1-T0)) s";;
esac
if [ -f /home/caden/ornc/crontab.bak.agent27 ]; then crontab /home/caden/ornc/crontab.bak.agent27; echo "crontab 恢复行数 = $(crontab -l | wc -l)"; fi
echo "--- 最终 ---"; md5sum /home/caden/orn_engine/orn3 /home/caden/ornc/serve2.py; curl -s -m 8 http://127.0.0.1:8090/health; echo
echo "异常累计 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log 2>/dev/null)"
echo "reset_count = $(cat /proc/xpu/dev0/reset_count 2>/dev/null) / $(cat /proc/xpu/dev1/reset_count 2>/dev/null)"
echo "state = $(cat /proc/xpu/dev0/state 2>/dev/null) / $(cat /proc/xpu/dev1/state 2>/dev/null)"
echo "=== det_install27 END $(TS) (总耗时 $(( $(date +%s) - T0 ))s) ==="
