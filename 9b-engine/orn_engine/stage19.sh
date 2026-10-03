#!/bin/bash
# stage19.sh — 第 19 轮卡窗口: 新算子探针 -> cardA 自检 -> 512²/768² A/B 对拍 (全程 safe_run)
set -u
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
LOG=/home/caden/orn_engine/stage19.log
: > "$LOG"
MM=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
rm -rf /tmp/vsw/n1 /tmp/vsw/n7; mkdir -p /tmp/vsw/n1/o /tmp/vsw/n1/old /tmp/vsw/n7/o /tmp/vsw/n7/old

bash window_open.sh "stage19-cardA" >>"$LOG" 2>&1

run() {
  local nm="$1"; local tt="$2"; shift 2
  echo "########## $nm $(date '+%T') ##########" | tee -a "$LOG"
  bash safe_run.sh -n "$nm" -t "$tt" -- "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "[stage19] $nm safe_run rc=$rc" | tee -a "$LOG"
  if [ "$rc" -eq 3 ]; then echo "CARD_EXCEPTION" >>"$LOG"; return 3; fi
  if [ "$rc" -eq 2 ]; then echo "PRECHECK_REFUSE" >>"$LOG"; return 2; fi
  return 0
}
abort() { bash window_close.sh >>"$LOG" 2>&1; echo ABORT; exit 3; }

# 1) 唯一的新算子: elementwise_div_2d 极小尺寸单发
run p_div2d   90 ./probe div2d  || true
run p_div2dM  90 ./probe div2dM || true
grep -q CARD_EXCEPTION "$LOG" && abort

# 2) 老算子回归 (reduce/transpose/mul2d)
run p_red     90 ./probe reduce || true
run p_trans   90 ./probe trans  || true
grep -q CARD_EXCEPTION "$LOG" && abort

# 3) 新路径自检: 卡上归一 vs 老路径 (极小尺寸, 同一份 A/B)
run caself   150 ./vistest.cards cardAtest || true
grep -q CARD_EXCEPTION "$LOG" && abort

# 4) 老自检回归 (分块 gemm)
run st_nc1   200 env VIS_CH=768 VIS_CHA=768 ./vistest.cards gemmtest || true
run st_nc8   200 env VIS_CH=96  VIS_CHA=24  ./vistest.cards gemmtest || true
grep -q CARD_EXCEPTION "$LOG" && abort

# 5) 512² A/B (新 vs 老路径, 同一份输入)
run n512card 600 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_PROF=1 OMP_NUM_THREADS=4 \
    ./vistest.cards run "$MM" /tmp/inp512.f32 512 512 /tmp/vsw/n1/o || true
grep -q CARD_EXCEPTION "$LOG" && abort
run n512old  600 env VIS_CARDA=0 VIS_CH=96 VIS_CHA=0 VIS_PROF=1 OMP_NUM_THREADS=4 \
    ./vistest.cards run "$MM" /tmp/inp512.f32 512 512 /tmp/vsw/n1/old || true
grep -q CARD_EXCEPTION "$LOG" && abort
echo "=== cmp512 ===" >>"$LOG"
python3 /home/caden/orn_engine/vscmp2.py /tmp/vsw/n1 512 o >>"$LOG" 2>&1

# 6) 768² A/B
run n768card 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_PROF=1 OMP_NUM_THREADS=4 \
    ./vistest.cards run "$MM" /tmp/inp768.f32 768 768 /tmp/vsw/n7/o || true
grep -q CARD_EXCEPTION "$LOG" && abort
run n768old  900 env VIS_CARDA=0 VIS_CH=96 VIS_CHA=0 VIS_PROF=1 OMP_NUM_THREADS=4 \
    ./vistest.cards run "$MM" /tmp/inp768.f32 768 768 /tmp/vsw/n7/old || true
echo "=== cmp768 ===" >>"$LOG"
python3 /home/caden/orn_engine/vscmp2.py /tmp/vsw/n7 768 o >>"$LOG" 2>&1

bash window_close.sh >>"$LOG" 2>&1
echo "=== stage19 done $(date '+%F %T') ===" | tee -a "$LOG"
