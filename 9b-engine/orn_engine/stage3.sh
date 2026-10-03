#!/bin/bash
# stage3.sh — 定位卡上折回的多块 bug + 拿 relrms 对拍 (全程 safe_run)
set -u
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
LOG=/home/caden/orn_engine/stage3.log
: > "$LOG"
MM=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
rm -rf /tmp/vsw/c1 /tmp/vsw/c7; mkdir -p /tmp/vsw/c1/o /tmp/vsw/c7/o

bash window_open.sh "stage3-cards" >>"$LOG" 2>&1

run() {
  local nm="$1"; local tt="$2"; shift 2
  echo "########## $nm $(date '+%T') ##########" | tee -a "$LOG"
  bash safe_run.sh -n "$nm" -t "$tt" -- "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "[stage3] $nm safe_run rc=$rc" | tee -a "$LOG"
  if [ "$rc" -eq 3 ]; then echo "CARD_EXCEPTION" >>"$LOG"; return 3; fi
  if [ "$rc" -eq 2 ]; then echo "PRECHECK_REFUSE" >>"$LOG"; return 2; fi
  return 0
}

# 1) 新算子极小尺寸: reduce(dim0) / mul_2d 原地
run pr_red0  120 ./probe red0   || true
run pr_red0b 120 ./probe red0b  || true
run pr_inpl  120 ./probe inpl   || true
grep -q CARD_EXCEPTION "$LOG" && { bash window_close.sh >>"$LOG" 2>&1; echo ABORT; exit 3; }

# 2) 自检: nc=1 vs nc>1 (定位多块 bug)
run st_nc1 300 env VIS_CH=768 VIS_CHA=768 ./vistest.cards gemmtest || true
run st_nc2 300 env VIS_CH=384 VIS_CHA=36  ./vistest.cards gemmtest || true
run st_nc8 300 env VIS_CH=96  VIS_CHA=24  ./vistest.cards gemmtest || true
grep -q CARD_EXCEPTION "$LOG" && { bash window_close.sh >>"$LOG" 2>&1; echo ABORT; exit 3; }

# 3) 全量对拍
run cards512 900 env VIS_CH=96 VIS_CHA=0 VIS_PROF=1 OMP_NUM_THREADS=4 \
    ./vistest.cards run "$MM" /tmp/inp512.f32 512 512 /tmp/vsw/c1/o || true
grep -q CARD_EXCEPTION "$LOG" && { bash window_close.sh >>"$LOG" 2>&1; echo ABORT; exit 3; }
echo "=== cmp512 ===" >>"$LOG"
python3 /home/caden/orn_engine/vscmp2.py /tmp/vsw/c1 512 96_0 >>"$LOG" 2>&1

run cards768 1200 env VIS_CH=96 VIS_CHA=0 VIS_PROF=1 OMP_NUM_THREADS=4 \
    ./vistest.cards run "$MM" /tmp/inp768.f32 768 768 /tmp/vsw/c7/o || true
echo "=== cmp768 ===" >>"$LOG"
python3 /home/caden/orn_engine/vscmp2.py /tmp/vsw/c7 768 96_0 >>"$LOG" 2>&1

bash window_close.sh >>"$LOG" 2>&1
echo "=== stage3 done $(date '+%F %T') ===" | tee -a "$LOG"
