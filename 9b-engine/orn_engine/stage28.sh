#!/bin/bash
# stage28.sh — 第23轮卡窗口 (C): 耗时复测 (每档 2~3 次) 以判定"抽稀的收益是否在噪声内"
set -u
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
LOG=/home/caden/orn_engine/stage28.log
: > "$LOG"
MM=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
V=./vistest.cards.new26
say() { echo "$*" | tee -a "$LOG"; }
exc() { grep -ac 'Exception in kernel execution' /var/log/kern.log; }
run() { local nm="$1"; local tt="$2"; shift 2
  echo "########## $nm $(date '+%T') ##########" | tee -a "$LOG"
  bash safe_run.sh -n "$nm" -t "$tt" -- "$@" >>"$LOG" 2>&1
  local rc=$?; echo "[stage28] $nm rc=$rc" | tee -a "$LOG"
  if [ "$rc" -eq 3 ]; then echo "CARD_EXCEPTION" >>"$LOG"; return 3; fi; return 0; }
abort() { say "!!! 卡异常 -> 停手收尾 !!!"; bash window_close26.sh >>"$LOG" 2>&1; echo ABORT; exit 3; }
C="env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=1 VIS_ONLY=27 VIS_PROF=1 OMP_NUM_THREADS=4"
say "=== stage28 开始 $(date '+%F %T') | 异常起 = $(exc) | reset $(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) ==="
bash window_open26.sh "stage28-timing" >>"$LOG" 2>&1
run p_self 150 env VIS_CH=96 VIS_CHA=0 $V cardAtest || true; grep -q CARD_EXCEPTION "$LOG" && abort
run w1r2 300 $C VIS_HEAPD2H=4 VIS_HEAPW=1 $V run "$MM" /tmp/inp768.f32 768 768 || true; grep -q CARD_EXCEPTION "$LOG" && abort
run w1r3 300 $C VIS_HEAPD2H=4 VIS_HEAPW=1 $V run "$MM" /tmp/inp768.f32 768 768 || true; grep -q CARD_EXCEPTION "$LOG" && abort
run w4r2 300 $C VIS_HEAPD2H=4 VIS_HEAPW=4 $V run "$MM" /tmp/inp768.f32 768 768 || true; grep -q CARD_EXCEPTION "$LOG" && abort
run m1r2 300 $C VIS_HEAPD2H=1 $V run "$MM" /tmp/inp768.f32 768 768 || true; grep -q CARD_EXCEPTION "$LOG" && abort
say ""; say "=========== 耗时 (自报 / wall) ==========="
grep -a -E "^########## |^\[vis\] 完成: |^\[safe_run\] 目标结束 rc=" "$LOG" | tee -a "$LOG"
say "=== stage28 结束 $(date '+%F %T') | 异常 = $(exc) | reset $(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) ==="
bash window_close26.sh >>"$LOG" 2>&1
echo "=== stage28 done ==="
