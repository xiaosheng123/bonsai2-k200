#!/bin/bash
# finish21.sh — 上线: ①卡上自检(修好的 selftest) ②安装 orn3(v24) ③拉回服务并等 ready
set -u
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
LOG=/home/caden/orn_engine/finish21.log
: > "$LOG"
say() { echo "$*" | tee -a "$LOG"; }
say "=== finish21 开始 $(date '+%F %T') | 异常起 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log) | reset $(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) ==="
bash window_open.sh "finish21-install" >>"$LOG" 2>&1
say "--- [1] 卡上自检 cardAtest (selftest eps 修复后) ---"
bash safe_run.sh -n p_self2 -t 150 -- env VIS_CH=96 VIS_CHA=0 ./vistest.cards.new24 cardAtest >>"$LOG" 2>&1
RC=$?
say "[finish21] cardAtest safe_run rc=$RC (3=卡异常 4=目标非0退出)"
if [ "$RC" -eq 3 ]; then say "!! 卡异常 -> 停手 !!"; bash window_close.sh >>"$LOG" 2>&1; echo ABORT; exit 3; fi
say "--- [2] 安装 orn3.new24 -> orn3 (旧件已备份 orn3.pre21.bak) ---"
cp -a orn3 orn3.pre21b.bak 2>/dev/null
cp -a orn3.new24 orn3
md5sum orn3 orn3.new24 orn3.pre21.bak
say "--- [3] 收尾 (恢复 crontab + 拉服务) ---"
bash window_close.sh >>"$LOG" 2>&1
say "=== finish21 结束 $(date '+%F %T') | 异常 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log) | reset $(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) ==="
echo "=== finish21 done ==="
