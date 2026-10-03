#!/bin/bash
# stage27.sh — 第23轮卡窗口 (B): ① 512² 生产/新件 13/13 复核  ② 768 每2头抽干复跑(证伪)
set -u
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
LOG=/home/caden/orn_engine/stage27.log
: > "$LOG"
MM=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
D=/tmp/vs27
rm -rf $D; mkdir -p $D/n512/prod $D/n512/new26 $D/n768/w2b
P=./vistest.cards.new24
N=./vistest.cards.new26
say() { echo "$*" | tee -a "$LOG"; }
exc() { grep -ac 'Exception in kernel execution' /var/log/kern.log; }
run() {
  local nm="$1"; local tt="$2"; shift 2
  echo "########## $nm $(date '+%T') ##########" | tee -a "$LOG"
  bash safe_run.sh -n "$nm" -t "$tt" -- "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "[stage27] $nm safe_run rc=$rc" | tee -a "$LOG"
  if [ "$rc" -eq 3 ]; then echo "CARD_EXCEPTION" >>"$LOG"; return 3; fi
  return 0
}
abort() { say "!!! 卡异常 -> 立即停手, 收尾拉服务 !!!"; bash window_close26.sh >>"$LOG" 2>&1; echo ABORT; exit 3; }
C="env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=1 VIS_PROF=1 OMP_NUM_THREADS=4"
say "=== stage27 开始 $(date '+%F %T') | 异常起 = $(exc) | reset $(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) ==="
bash window_open26.sh "stage27-512chk" >>"$LOG" 2>&1
run p_self 150 env VIS_CH=96 VIS_CHA=0 $N cardAtest || true; grep -q CARD_EXCEPTION "$LOG" && abort

say "--- [1] 512² 生产档 (线上同一份源码 cards24 编的 vistest.cards.new24, mode4) 落盘 ---"
run n512prod 300 $C VIS_HEAPD2H=4 $P run "$MM" /tmp/inp512.f32 512 512 $D/n512/prod || true; grep -q CARD_EXCEPTION "$LOG" && abort

say "--- [2] 512² 新件 (cards26, mode4 W=2) 落盘 ---"
run n512new 300 $C VIS_HEAPD2H=4 VIS_HEAPW=2 $N run "$MM" /tmp/inp512.f32 512 512 $D/n512/new26 || true; grep -q CARD_EXCEPTION "$LOG" && abort

say "--- [3] 768² mode4 W=2 复跑 (证伪: 是否稳定不复现) ---"
run w2b 300 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=1 VIS_ONLY=27 VIS_PROF=1 OMP_NUM_THREADS=4 VIS_HEAPD2H=4 VIS_HEAPW=2 $N run "$MM" /tmp/inp768.f32 768 768 $D/n768/w2b || true; grep -q CARD_EXCEPTION "$LOG" && abort

say ""
say "=========== 对拍 A: 512² 生产档 vs 第21章生产落盘 /tmp/vs23/n512 ==========="
python3 cmp20.py $D/n512/prod /tmp/vs23/n512 "512 PROD(cards24) vs ch21PROD" 2>&1 | tee -a "$LOG"
say "=========== 对拍 B: 512² 新件(cards26, W=2) vs 第21章生产落盘 ==========="
python3 cmp20.py $D/n512/new26 /tmp/vs23/n512 "512 NEW(cards26 W=2) vs ch21PROD" 2>&1 | tee -a "$LOG"
say "=========== 对拍 C: 768² W=2 复跑 vs 生产落盘 ==========="
python3 cmp20.py $D/n768/w2b /tmp/vs23/n768 "768 w2b vs PROD" 2>&1 | tee -a "$LOG"
say "=========== 对拍 D: 768² W=2 复跑 vs W=2 第一次 ==========="
python3 cmp20.py $D/n768/w2b /tmp/vs26/n768/b4w2 "768 w2b vs w2(first)" 2>&1 | tee -a "$LOG"
say ""
say "=========== 耗时 ==========="
grep -a -E "^##########|^\[vis\] 完成: |^\[safe_run\] 目标结束 rc=" "$LOG" | tee -a "$LOG"
say "=== stage27 结束 $(date '+%F %T') | 异常 = $(exc) | reset $(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) ==="
bash window_close26.sh >>"$LOG" 2>&1
echo "=== stage27 done ==="
