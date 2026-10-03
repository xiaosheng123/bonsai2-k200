#!/bin/bash
# =============================================================================
# install32.sh [冷预填充分上限]  — 第32轮上线窗口: 停服务 -> 换件 -> 拉服务 -> 等 ready
#   目标: 单次停机 <=120s; 未就绪自动回滚旧件; 停机开始/结束逐笔记账。
#   用法: K200_MAX_COLD_PREFILL_TOK=3000 bash install32.sh 3000
# =============================================================================
set -u
GUARD="${1:-3000}"
cd /home/caden/ornc
TS(){ date '+%F %T'; }
T0=$(date +%s)
echo "=== install32 START $(TS)  (K200_MAX_COLD_PREFILL_TOK=$GUARD) ==="
echo "--- 上线前 md5 ---"; md5sum /home/caden/orn_engine/orn3 /home/caden/ornc/serve2.py 2>/dev/null
[ -f /home/caden/orn_engine/orn3.pre32.bak ] || cp -a /home/caden/orn_engine/orn3 /home/caden/orn_engine/orn3.pre32.bak
[ -f /home/caden/ornc/serve2.pre32.bak ]     || cp -a /home/caden/ornc/serve2.py /home/caden/ornc/serve2.pre32.bak
echo "--- 备份 ---"; md5sum /home/caden/orn_engine/orn3.pre32.bak /home/caden/ornc/serve2.pre32.bak
crontab -l > /home/caden/ornc/crontab.bak.agent32 2>/dev/null
crontab -l 2>/dev/null | sed 's|^\(.*svc_guard.sh.*\)$|#AGENT32# \1|; s|^\(@reboot.*\)$|#AGENT32# \1|' | crontab -
echo "--- crontab 已临时注掉 svc_guard/@reboot (收尾必须用 restore_crontab32.sh 恢复) ---"
crontab -l | grep -c AGENT32
echo "--- T_STOP $(TS) ---"
bash stop.sh
sleep 2
n=0; for d in /proc/[0-9]*/fd/*; do t=$(readlink "$d" 2>/dev/null); case "$t" in /dev/xpu*) echo "  持卡: $d -> $t"; n=$((n+1));; esac; done
echo "  停后持卡 fd 数 = $n"
if [ "$n" != "0" ]; then
  echo "!! 有残留持卡进程 => 放弃上线, 直接拉回旧服务"
  bash start.sh; exit 2
fi
cp -f /home/caden/orn_engine/orn3.new32 /home/caden/orn_engine/orn3 && chmod +x /home/caden/orn_engine/orn3
cp -f /home/caden/ornc/serve2.new32.py /home/caden/ornc/serve2.py
echo "--- 上线后 md5 ---"; md5sum /home/caden/orn_engine/orn3 /home/caden/ornc/serve2.py
echo "--- T_START $(TS) ---"
K200_MAX_COLD_PREFILL_TOK=$GUARD bash start.sh
echo "--- T_READY $(TS) ---"
T1=$(date +%s); echo "  停机时长(stop->ready) = $((T1-T0)) s"
H=$(curl -s -m 8 http://127.0.0.1:8090/health)
echo "health: $H"
case "$H" in
  *'"status": "ok'*) echo "  ready ✓";;
  *) echo "!! 服务未就绪 => 回滚 orn3.pre32.bak + serve2.pre32.bak 并重启"
     cp -f /home/caden/orn_engine/orn3.pre32.bak /home/caden/orn_engine/orn3
     cp -f /home/caden/ornc/serve2.pre32.bak /home/caden/ornc/serve2.py
     bash stop.sh; sleep 2; bash start.sh; T1=$(date +%s); echo "  回滚后停机时长 = $((T1-T0)) s";;
esac
echo "--- 引擎新功能自检 (日志行) ---"
grep -a "冷预填充上限" /home/caden/ornc/serve2.log | tail -1
grep -a "前缀 KV 复用" /home/caden/ornc/serve2.log | tail -1
grep -a "多槽位(第29轮)" /home/caden/ornc/serve2.log | tail -1
echo "--- 最终 ---"; md5sum /home/caden/orn_engine/orn3 /home/caden/ornc/serve2.py; curl -s -m 8 http://127.0.0.1:8090/health; echo
echo "异常累计 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log 2>/dev/null)"
echo "reset_count = $(cat /proc/xpu/dev0/reset_count 2>/dev/null) / $(cat /proc/xpu/dev1/reset_count 2>/dev/null)"
echo "state = $(cat /proc/xpu/dev0/state 2>/dev/null) / $(cat /proc/xpu/dev1/state 2>/dev/null)"
echo "=== install32 END $(TS) (总耗时 $(( $(date +%s) - T0 ))s) ==="
