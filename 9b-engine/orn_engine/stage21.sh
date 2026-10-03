#!/bin/bash
# stage21.sh — 第 21 轮卡窗口: ①分母夹逼探针 ②VIS_HEAP 三档 D2H + 哨兵诊断 ③FFN 收益测量(小层数)
#   全部占卡作业走 safe_run.sh; 出现任何卡异常立即 abort (停止本窗口 + 收尾拉回服务)
set -u
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
LOG=/home/caden/orn_engine/stage21.log
: > "$LOG"
MM=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
D=/tmp/vs21
rm -rf $D; mkdir -p $D/n7/{a,b,c,d} $D/only
V=./vistest.cards.new21
say() { echo "$*" | tee -a "$LOG"; }
run() {
  local nm="$1"; local tt="$2"; shift 2
  echo "########## $nm $(date '+%T') ##########" | tee -a "$LOG"
  bash safe_run.sh -n "$nm" -t "$tt" -- "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "[stage21] $nm safe_run rc=$rc" | tee -a "$LOG"
  if [ "$rc" -eq 3 ]; then echo "CARD_EXCEPTION" >>"$LOG"; return 3; fi
  if [ "$rc" -eq 2 ]; then echo "PRECHECK_REFUSE" >>"$LOG"; return 2; fi
  if [ "$rc" -eq 1 ]; then echo "TARGET_FAIL" >>"$LOG"; return 1; fi
  return 0
}
abort() { say "!!! 卡异常 -> 立即停手, 收尾 !!!"; bash window_close.sh >>"$LOG" 2>&1; echo ABORT; exit 3; }

say "=== stage21 开始 $(date '+%F %T') | 异常起 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log) | reset $(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) ==="
bash window_open.sh "stage21-dumps" >>"$LOG" 2>&1

say "--- [1] ★①分母夹逼探针 (含负值/全零块/整块全0/正常对照) ---"
run p_divclampS 90 ./probe21 divclampS  || true; grep -q CARD_EXCEPTION "$LOG" && abort
run p_divclamp  90 ./probe21 divclamp   || true; grep -q CARD_EXCEPTION "$LOG" && abort
run p_divclampZ 90 ./probe21 divclampZ  || true; grep -q CARD_EXCEPTION "$LOG" && abort
run p_divclampN 90 ./probe21 divclampN  || true; grep -q CARD_EXCEPTION "$LOG" && abort
say "--- [1b] 我新用到的算子回归 (位置新: 夹逼在 Mc reduce 之前) ---"
run p_maxabs    90 ./probe21 maxabs     || true; grep -q CARD_EXCEPTION "$LOG" && abort
run p_maxabsM   90 ./probe21 maxabsM    || true; grep -q CARD_EXCEPTION "$LOG" && abort
run p_cardself 150 env VIS_CH=96 VIS_CHA=0 $V cardAtest || true; grep -q CARD_EXCEPTION "$LOG" && abort
say "--- 探针结果 ---"
grep -a '^\[probe\]\|^\[vis\] ★ cardA' "$LOG" | tail -30 | tee -a "$LOG"

say "--- [2] 768² a: 线上档 (VIS_HEAP=0) —— 窗口内基线 ---"
run n7a 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=0 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 $D/n7/a || true; grep -q CARD_EXCEPTION "$LOG" && abort
say "--- [3] 768² b: HEAP=1 + HEAPD2H=1 (★候选修复: 每芯先同步再大 D2H) ---"
run n7b 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=1 VIS_HEAPD2H=1 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 $D/n7/b || true; grep -q CARD_EXCEPTION "$LOG" && abort
say "--- [4] 768² c: HEAP=1 + HEAPD2H=2 (逐头小块 D2H, 对照) ---"
run n7c 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=1 VIS_HEAPD2H=2 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 $D/n7/c || true; grep -q CARD_EXCEPTION "$LOG" && abort
say "--- [5] 768² d: HEAP=1 + HEAPD2H=1 + HEAPDBG=1 (哨兵 + 原始堆叠缓冲落盘) ---"
run n7d 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=1 VIS_HEAPD2H=1 VIS_HEAPDBG=1 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 $D/n7/d || true; grep -q CARD_EXCEPTION "$LOG" && abort

say "--- [6] ★③FFN 收益测量: 只跑前 3 层 (避开上一轮崩卡的第 21 层), 只看层耗时 ---"
run only0 300 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=0 VIS_ONLY=3 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 $D/only/o0 || true; grep -q CARD_EXCEPTION "$LOG" && abort
run only2 300 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=2 VIS_HEAP=0 VIS_PROF=1 VIS_ONLY=3 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 $D/only/o2 || true; grep -q CARD_EXCEPTION "$LOG" && abort
run only1 300 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=1 VIS_HEAP=0 VIS_PROF=1 VIS_ONLY=3 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 $D/only/o1 || true; grep -q CARD_EXCEPTION "$LOG" && abort

say ""
say "=========== 对拍 A: 768² HEAP=0 (本轮基线) vs 第19章生产落盘 ==========="
python3 /home/caden/orn_engine/cmp20.py $D/n7/a /tmp/vsw/n7c/new "n7a(HEAP=0) vs ch19" 2>&1 | tee -a "$LOG"
say "=========== 对拍 B: 768² HEAP=1/D2H=1 vs 基线 n7a ==========="
python3 /home/caden/orn_engine/cmp20.py $D/n7/b $D/n7/a "HEAP1/D2H1 vs n7a" 2>&1 | tee -a "$LOG"
say "=========== 对拍 C: 768² HEAP=1/D2H=2 (逐头) vs 基线 n7a ==========="
python3 /home/caden/orn_engine/cmp20.py $D/n7/c $D/n7/a "HEAP1/D2H2 vs n7a" 2>&1 | tee -a "$LOG"
say "=========== 对拍 D: 768² HEAPDBG=1 vs 基线 n7a ==========="
python3 /home/caden/orn_engine/cmp20.py $D/n7/d $D/n7/a "HEAPDBG vs n7a" 2>&1 | tee -a "$LOG"
say "=========== 对拍 E: 各档逐 head/chip 归因 ==========="
for t in b c d; do
  say "--- n7$t ---"
  python3 /home/caden/orn_engine/an20.py $D/n7/$t/l0_ao.f32 $D/n7/a/l0_ao.f32 2>&1 | tail -22 | tee -a "$LOG"
done
say ""
say "=========== 耗时 ==========="
grep -a -E '完成: |PROF:|PROF2:|层  0|层  1|层  2|只跑|哨兵' "$LOG" | tail -60 | tee -a "$LOG"
say ""
say "=== stage21 结束 $(date '+%F %T') | 异常 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log) | reset $(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) ==="
bash window_close.sh >>"$LOG" 2>&1
echo "=== stage21 done ==="
