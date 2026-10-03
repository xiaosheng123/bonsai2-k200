#!/bin/bash
# stage26.sh — 第23轮卡窗口 (A): 各同步点方案 768² 落盘对拍 + 耗时 (先测, 不装)
set -u
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
LOG=/home/caden/orn_engine/stage26.log
: > "$LOG"
MM=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
D=/tmp/vs26
rm -rf $D; mkdir -p $D/n768/b4w1 $D/n768/b4w2 $D/n768/b4w4 $D/n768/b1 $D/n768/pp $D/n768/ppr
V=./vistest.cards.new26
say() { echo "$*" | tee -a "$LOG"; }
exc() { grep -ac 'Exception in kernel execution' /var/log/kern.log; }
run() {
  local nm="$1"; local tt="$2"; shift 2
  echo "########## $nm $(date '+%T') ##########" | tee -a "$LOG"
  bash safe_run.sh -n "$nm" -t "$tt" -- "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "[stage26] $nm safe_run rc=$rc" | tee -a "$LOG"
  if [ "$rc" -eq 3 ]; then echo "CARD_EXCEPTION" >>"$LOG"; return 3; fi
  return 0
}
abort() { say "!!! 卡异常 -> 立即停手, 不装, 收尾拉服务 !!!"; bash window_close26.sh >>"$LOG" 2>&1; echo ABORT; exit 3; }
C="env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=1 VIS_ONLY=27 VIS_PROF=1 OMP_NUM_THREADS=4"

say "=== stage26 开始 $(date '+%F %T') | 异常起 = $(exc) | reset $(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) ==="
bash window_open26.sh "stage26-sync" >>"$LOG" 2>&1

say "--- [0] 卡上自检 cardAtest ---"
run p_self 150 env VIS_CH=96 VIS_CHA=0 $V cardAtest || true; grep -q CARD_EXCEPTION "$LOG" && abort

say "--- [1] 768 mode4 VIS_HEAPW=1 (第21轮上线行为; 本条顺带自校新基线) ---"
run b4w1 300 $C VIS_HEAPD2H=4 VIS_HEAPW=1 $V run "$MM" /tmp/inp768.f32 768 768 $D/n768/b4w1 || true; grep -q CARD_EXCEPTION "$LOG" && abort

say "--- [2] 768 mode4 VIS_HEAPW=2 (每2头抽干) ---"
run b4w2 300 $C VIS_HEAPD2H=4 VIS_HEAPW=2 $V run "$MM" /tmp/inp768.f32 768 768 $D/n768/b4w2 || true; grep -q CARD_EXCEPTION "$LOG" && abort

say "--- [3] 768 mode4 VIS_HEAPW=4 (每4头抽干) ---"
run b4w4 300 $C VIS_HEAPD2H=4 VIS_HEAPW=4 $V run "$MM" /tmp/inp768.f32 768 768 $D/n768/b4w4 || true; grep -q CARD_EXCEPTION "$LOG" && abort

say "--- [4] 768 mode1 (零抽干, 已知数值错档; 耗时下界/负对照) ---"
run b1 300 $C VIS_HEAPD2H=1 $V run "$MM" /tmp/inp768.f32 768 768 $D/n768/b1 || true; grep -q CARD_EXCEPTION "$LOG" && abort

say "--- [5] 768 mode8 = 零抽干 + H2D 目的地乒乓 (VIS_PP=1) ---"
run pp 300 $C VIS_HEAPD2H=8 VIS_PP=1 $V run "$MM" /tmp/inp768.f32 768 768 $D/n768/pp || true; grep -q CARD_EXCEPTION "$LOG" && abort

say "--- [6] 768 mode8 复跑一次 (确定性检查) ---"
run ppr 300 $C VIS_HEAPD2H=8 VIS_PP=1 $V run "$MM" /tmp/inp768.f32 768 768 $D/n768/ppr || true; grep -q CARD_EXCEPTION "$LOG" && abort

say ""
for t in b4w1 b4w2 b4w4 b1 pp; do
  say "=========== 对拍: 768 $t vs 第21章生产落盘 /tmp/vs23/n768 ==========="
  python3 cmp20.py $D/n768/$t /tmp/vs23/n768 "768 $t vs PROD" 2>&1 | tee -a "$LOG"
done
say "=========== 对拍: pp vs ppr (两次跑) ==========="
python3 cmp20.py $D/n768/ppr $D/n768/pp "pprun1 vs pprun2" 2>&1 | tee -a "$LOG"
say "=========== 对拍: pp vs b4w2 / pp vs b4w4 ==========="
python3 cmp20.py $D/n768/pp $D/n768/b4w2 "pp vs b4w2" 2>&1 | tee -a "$LOG"
python3 cmp20.py $D/n768/pp $D/n768/b4w4 "pp vs b4w4" 2>&1 | tee -a "$LOG"

say ""
say "=========== 耗时 (vistest 自报 / safe_run wall) ==========="
grep -a -E "^\[vis\] 完成: |^\[safe_run\] 目标结束 rc=|^##########" "$LOG" | tee -a "$LOG"
say "=========== PROF 口径 ==========="
grep -a -E "PROF: " "$LOG" | tee -a "$LOG"
say "=========== 开关生效确认 ==========="
grep -a -E "第 23 轮乒乓缓冲|VIS_HEAPD2H=|抽干步长" "$LOG" | tee -a "$LOG"
say ""
say "=== stage26 结束 $(date '+%F %T') | 异常 = $(exc) | reset $(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) ==="
bash window_close26.sh >>"$LOG" 2>&1
echo "=== stage26 done ==="
