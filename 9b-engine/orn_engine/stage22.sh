#!/bin/bash
# stage22.sh — 第 21 轮第二窗口: ①HEAP=0 基线修复验证 ②(B) 修复候选 mode4/5 ③FFN 收益 (VIS_ONLY=22, 含上一轮崩卡的第21层)
set -u
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
LOG=/home/caden/orn_engine/stage22.log
: > "$LOG"
MM=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
D=/tmp/vs22
rm -rf $D; mkdir -p $D/n7/{a,b,e,f}
V=./vistest.cards.new22
say() { echo "$*" | tee -a "$LOG"; }
run() {
  local nm="$1"; local tt="$2"; shift 2
  echo "########## $nm $(date '+%T') ##########" | tee -a "$LOG"
  bash safe_run.sh -n "$nm" -t "$tt" -- "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "[stage22] $nm safe_run rc=$rc" | tee -a "$LOG"
  if [ "$rc" -eq 3 ]; then echo "CARD_EXCEPTION" >>"$LOG"; return 3; fi
  return 0
}
abort() { say "!!! 卡异常 -> 立即停手, 收尾 !!!"; bash window_close.sh >>"$LOG" 2>&1; echo ABORT; exit 3; }

say "=== stage22 开始 $(date '+%F %T') | 异常起 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log) | reset $(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) ==="
bash window_open.sh "stage22-heapfix" >>"$LOG" 2>&1

say "--- [1] 768² a: HEAP=0 基线 (accdst bugfix 后应能跑通) ---"
run n7a 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=0 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 $D/n7/a || true; grep -q CARD_EXCEPTION "$LOG" && abort
say "--- [2] 768² b: HEAP=1 HEAPD2H=1 (复测, 看错误头集合是否随跑次变化) ---"
run n7b 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=1 VIS_HEAPD2H=1 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 $D/n7/b || true; grep -q CARD_EXCEPTION "$LOG" && abort
say "--- [3] 768² e: HEAP=1 HEAPD2H=4 (★候选1: 每头结束两芯 device 同步) ---"
run n7e 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=1 VIS_HEAPD2H=4 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 $D/n7/e || true; grep -q CARD_EXCEPTION "$LOG" && abort
say "--- [4] 768² f: HEAP=1 HEAPD2H=5 (★候选2: 每头同步 + 立即取回该头) ---"
run n7f 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=1 VIS_HEAPD2H=5 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 $D/n7/f || true; grep -q CARD_EXCEPTION "$LOG" && abort

say "--- [5] ③FFN 收益: VIS_ONLY=22 (含第20轮崩卡的第21层) CARDFFN=0 基线 ---"
run f0 600 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=0 VIS_ONLY=22 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 "" || true; grep -q CARD_EXCEPTION "$LOG" && abort
say "--- [6] ③FFN 收益: VIS_ONLY=22 CARDFFN=2 (主机 bias/gelu + 卡上归一/gemm; 位级最保守的搬卡档) ---"
run f2 600 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=2 VIS_HEAP=0 VIS_ONLY=22 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 "" || true; grep -q CARD_EXCEPTION "$LOG" && abort
say "--- [7] ③FFN 收益 + ★①验死: VIS_ONLY=22 CARDFFN=1 (全卡 bias/gelu; 第20轮在此档第21层 FP_DIV0 崩卡) ---"
run f1 600 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=1 VIS_HEAP=0 VIS_ONLY=22 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 "" || true; grep -q CARD_EXCEPTION "$LOG" && abort

say ""
say "=========== 对拍 A: HEAP=0 (本窗口基线, accdst 修复后) vs 第19章生产落盘 ==========="
python3 /home/caden/orn_engine/cmp20.py $D/n7/a /tmp/vsw/n7c/new "n7a(HEAP=0,accdst fix) vs ch19" 2>&1 | tee -a "$LOG"
for t in b e f; do
  say "=========== 对拍 $t: HEAP=1 档 $t vs 本窗口 HEAP=0 基线 ==========="
  python3 /home/caden/orn_engine/cmp20.py $D/n7/$t $D/n7/a "heap-$t vs n7a" 2>&1 | tee -a "$LOG"
  say "--- 逐 head/chip 归因 ($t) ---"
  python3 /home/caden/orn_engine/an20.py $D/n7/$t/l0_ao.f32 $D/n7/a/l0_ao.f32 2>&1 | tail -20 | tee -a "$LOG"
done
say ""
say "=========== 耗时 ==========="
grep -a -E '完成: |PROF:|PROF2:|层 +(0|1|20|21|22) |只跑' "$LOG" | tail -50 | tee -a "$LOG"
say ""
say "=== stage22 结束 $(date '+%F %T') | 异常 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log) | reset $(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) ==="
bash window_close.sh >>"$LOG" 2>&1
echo "=== stage22 done ==="
