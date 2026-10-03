#!/bin/bash
# cb_vaseline29.sh — 单窗口: 换回第27轮生产件(LCP 引擎+老网关) 跑【表格图】取基线, 然后立刻装回第29轮新件
#   目的: 判定"表格读数退化"是不是第29轮引入的 (第27轮那批件从未跑过 det_vis)
set -u
ORN=/home/caden/orn_engine
ORN_DEV=/home/caden/ornc
ACC=/home/caden/ornc/accept.post29
LOG="$ACC/vaseline.log"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
T0=$(date +%s)
say "=== cb_vaseline29 开始 ==="
bash "$ORN_DEV/stop.sh" >/dev/null 2>&1 || true
sleep 1
cp -f "$ORN/orn3.pre29.bak" "$ORN/orn3"
cp -f "$ORN_DEV/serve2.py.pre29.bak" "$ORN_DEV/serve2.py"
say "已换回第27轮件: engine=$(md5sum $ORN/orn3|awk '{print $1}') gateway=$(md5sum $ORN_DEV/serve2.py|awk '{print $1}')"
bash "$ORN_DEV/start.sh" 2>&1 | tail -2 | tee -a "$LOG"
for i in $(seq 1 45); do sleep 2; timeout 5 curl -s http://127.0.0.1:8090/health 2>/dev/null | grep -q '"ready": true' && break; done
say "第27轮件 ready (${i}0s); 跑表格图"
python3 /home/caden/sdnn/cb_vistab.py 8090 400 2>&1 | tee -a "$LOG"
T1=$(date +%s)
say "第27轮件窗口耗时 $((T1-T0))s (停机+起+测)"
say "现在装回第29轮新件"
bash "$ORN_DEV/cb_install29.sh" 2>&1 | tail -4 | tee -a "$LOG"
say "=== 完成; 总 $(( $(date +%s) - T0 ))s ==="
