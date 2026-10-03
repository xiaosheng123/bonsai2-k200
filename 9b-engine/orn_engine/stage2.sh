#!/bin/bash
# stage2.sh — 卡上折回 (vis.cpp.cards) 离线对拍: selftest → 512² → 768², 全程 safe_run
set -u
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
LOG=/home/caden/orn_engine/stage2.log
: > "$LOG"
MM=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
mkdir -p /tmp/vsw/cards

bash window_open.sh "stage2-cards" >>"$LOG" 2>&1

run() {  # run <name> <timeout> <env...> -- cmd
  local nm="$1"; local tt="$2"; shift 2
  echo "########## $nm $(date '+%T') ##########" | tee -a "$LOG"
  bash safe_run.sh -n "$nm" -t "$tt" -- "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "[stage2] $nm safe_run rc=$rc" | tee -a "$LOG"
  if [ "$rc" -eq 3 ]; then echo "CARD_EXCEPTION" >>"$LOG"; return 3; fi
  if [ "$rc" -eq 2 ]; then echo "PRECHECK_REFUSE" >>"$LOG"; return 2; fi
  return 0
}

echo "=== 0) 极小尺寸 selftest ===" >>"$LOG"
run cards_st 300 ./vistest.cards gemmtest || true
grep -q CARD_EXCEPTION "$LOG" && { bash window_close.sh >>"$LOG" 2>&1; echo "ABORT"; exit 3; }

echo "=== 1) 512² ===" >>"$LOG"
run cards512 900 env VIS_CH=96 VIS_CHA=0 VIS_PROF=1 OMP_NUM_THREADS=4 \
    ./vistest.cards run "$MM" /tmp/inp512.f32 512 512 /tmp/vsw/cards/o512 || true
grep -q CARD_EXCEPTION "$LOG" && { bash window_close.sh >>"$LOG" 2>&1; echo "ABORT"; exit 3; }

echo "=== 2) 768² ===" >>"$LOG"
run cards768 1200 env VIS_CH=96 VIS_CHA=0 VIS_PROF=1 OMP_NUM_THREADS=4 \
    ./vistest.cards run "$MM" /tmp/inp768.f32 768 768 /tmp/vsw/cards/o768 || true

echo "=== 3) 对拍 ===" >>"$LOG"
python3 /home/caden/orn_engine/vscmp2.py /tmp/vsw/cards 512 96_0 >>"$LOG" 2>&1
python3 /home/caden/orn_engine/vscmp2.py /tmp/vsw/cards 768 96_0 >>"$LOG" 2>&1

bash window_close.sh >>"$LOG" 2>&1
echo "=== stage2 done $(date '+%F %T') ===" | tee -a "$LOG"
