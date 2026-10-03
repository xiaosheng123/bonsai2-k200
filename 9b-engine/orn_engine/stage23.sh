#!/bin/bash
# stage23.sh — 上线前最后一道卡验证: 768² (堆叠默认开) 与 512² (必须走主机路径, 位级不变)
set -u
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
LOG=/home/caden/orn_engine/stage23.log
: > "$LOG"
MM=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
D=/tmp/vs23
rm -rf $D; mkdir -p $D/n768 $D/n512
V=./vistest.cards.new23
say() { echo "$*" | tee -a "$LOG"; }
run() { local nm="$1"; local tt="$2"; shift 2
  echo "########## $nm $(date '+%T') ##########" | tee -a "$LOG"
  bash safe_run.sh -n "$nm" -t "$tt" -- "$@" >>"$LOG" 2>&1
  local rc=$?; echo "[stage23] $nm safe_run rc=$rc" | tee -a "$LOG"
  [ "$rc" -eq 3 ] && { echo "CARD_EXCEPTION" >>"$LOG"; return 3; }; return 0; }
abort() { say "!!! 卡异常 -> 停手收尾 !!!"; bash window_close.sh >>"$LOG" 2>&1; echo ABORT; exit 3; }

say "=== stage23 开始 $(date '+%F %T') | 异常起 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log) | reset $(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) ==="
bash window_open.sh "stage23-prodcheck" >>"$LOG" 2>&1
say "--- [0] 卡上自检 cardAtest (含新的分母夹逼) ---"
run p_self 150 env VIS_CH=96 VIS_CHA=0 $V cardAtest || true; grep -q CARD_EXCEPTION "$LOG" && abort
say "--- [1] 768²: 默认开关 (VIS_HEAP=1 VIS_HEAPD2H=4) ---"
run p768 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 $D/n768 || true; grep -q CARD_EXCEPTION "$LOG" && abort
say "--- [2] 512²: 默认开关 (useCA=0 ⇒ 必须自动退回主机路径) ---"
run p512 600 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp512.f32 512 512 $D/n512 || true; grep -q CARD_EXCEPTION "$LOG" && abort
say ""
say "=== 对拍: 768² vs 第19章生产落盘 (期望 13/13 字节全同) ==="
python3 cmp20.py $D/n768 /tmp/vsw/n7c/new "768 默认(HEAP mode4) vs ch19" 2>&1 | tee -a "$LOG"
say "=== 对拍: 512² vs 第19章生产落盘 (期望 13/13 字节全同) ==="
python3 cmp20.py $D/n512 /tmp/vsw/n1c/new "512 默认 vs ch19" 2>&1 | tee -a "$LOG"
say ""
grep -a -E '完成: |PROF:|PROF2:' "$LOG" | tail -8 | tee -a "$LOG"
say "=== stage23 结束 $(date '+%F %T') | 异常 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log) | reset $(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) ==="
bash window_close.sh >>"$LOG" 2>&1
echo "=== stage23 done ==="
