#!/bin/bash
# stage25.sh — 第22轮卡窗口: FFN 全卡路径 (VIS_CARDFFN=1) 全 27 层单发 + 三档对拍
#   流程: 卡上自检 -> VIS_ONLY=27 (CARDFFN=1, 含曾崩的第21层) -> VIS_ONLY=27 (CARDFFN=0 基线)
#         -> 768² ref/card 落盘 -> 512² ref/card 落盘 -> 对拍 -> 安装 orn3.new25 -> 收尾拉服务
set -u
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
LOG=/home/caden/orn_engine/stage25.log
: > "$LOG"
MM=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
D=/tmp/vs25
rm -rf $D; mkdir -p $D/n768/ref $D/n768/card $D/n512/ref $D/n512/card
V=./vistest.cards.new25
say() { echo "$*" | tee -a "$LOG"; }
run() {
  local nm="$1"; local tt="$2"; shift 2
  echo "########## $nm $(date '+%T') ##########" | tee -a "$LOG"
  bash safe_run.sh -n "$nm" -t "$tt" -- "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "[stage25] $nm safe_run rc=$rc" | tee -a "$LOG"
  if [ "$rc" -eq 3 ]; then echo "CARD_EXCEPTION" >>"$LOG"; return 3; fi
  return 0
}
abort() { say "!!! 卡异常 -> 立即停手, 不安装, 收尾拉回旧服务 !!!"; bash window_close25.sh >>"$LOG" 2>&1; echo ABORT; exit 3; }
exc() { grep -ac 'Exception in kernel execution' /var/log/kern.log; }

say "=== stage25 开始 $(date '+%F %T') | 异常起 = $(exc) | reset $(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) ==="
bash window_open25.sh "stage25-cardffn" >>"$LOG" 2>&1

say "--- [1] 卡上自检 cardAtest (含分母夹逼 + eps 源缓冲修复) ---"
run p_self 150 env VIS_CH=96 VIS_CHA=0 $V cardAtest || true; grep -q CARD_EXCEPTION "$LOG" && abort

say "--- [2] ★ VIS_ONLY=27 (全 27 层, 含第 21 层) CARDFFN=1 ---"
run only27c 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=1 VIS_HEAP=1 VIS_HEAPD2H=4 VIS_ONLY=27 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 "" || true; grep -q CARD_EXCEPTION "$LOG" && abort

say "--- [3] VIS_ONLY=27 CARDFFN=0 (基线, 逐层耗时对照) ---"
run only27b 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=1 VIS_HEAPD2H=4 VIS_ONLY=27 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 "" || true; grep -q CARD_EXCEPTION "$LOG" && abort

say "--- [4] 768² 参考档 CARDFFN=0 (生产行为基线, 落盘) ---"
run n768ref 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=1 VIS_HEAPD2H=4 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 $D/n768/ref || true; grep -q CARD_EXCEPTION "$LOG" && abort

say "--- [5] 768² FFN 全卡档 CARDFFN=1 (落盘) ---"
run n768card 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=1 VIS_HEAP=1 VIS_HEAPD2H=4 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 $D/n768/card || true; grep -q CARD_EXCEPTION "$LOG" && abort

say "--- [6] 512² 参考档 CARDFFN=0 (落盘) ---"
run n512ref 600 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=1 VIS_HEAPD2H=4 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp512.f32 512 512 $D/n512/ref || true; grep -q CARD_EXCEPTION "$LOG" && abort

say "--- [7] 512² FFN 全卡档 CARDFFN=1 (落盘) ---"
run n512card 600 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=1 VIS_HEAP=1 VIS_HEAPD2H=4 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp512.f32 512 512 $D/n512/card || true; grep -q CARD_EXCEPTION "$LOG" && abort

say ""
say "=========== 对拍 1: 768² 参考档(CARDFFN=0) vs 第21轮生产落盘 /tmp/vs23/n768 (期望 13/13 字节全同) ==========="
python3 cmp20.py $D/n768/ref /tmp/vs23/n768 "768 ref(CARDFFN=0) vs ch21生产落盘" 2>&1 | tee -a "$LOG"
say "=========== 对拍 2: 768² 全卡档(CARDFFN=1) vs 768² 参考档 ==========="
python3 cmp20.py $D/n768/card $D/n768/ref "768 card(CARDFFN=1) vs ref" 2>&1 | tee -a "$LOG"
say "=========== 对拍 3: 512² 参考档(CARDFFN=0) vs 第21轮生产落盘 /tmp/vs23/n512 ==========="
python3 cmp20.py $D/n512/ref /tmp/vs23/n512 "512 ref(CARDFFN=0) vs ch21生产落盘" 2>&1 | tee -a "$LOG"
say "=========== 对拍 4: 512² 全卡档(CARDFFN=1) vs 512² 参考档 ==========="
python3 cmp20.py $D/n512/card $D/n512/ref "512 card(CARDFFN=1) vs ref" 2>&1 | tee -a "$LOG"

say ""
say "=========== 逐层耗时 (VIS_ONLY=27) ==========="
grep -a -E '^\[vis\]   层 (0|1|20|21|22|25|26) |完成: |^\[vis\] ★ PROF' "$LOG" | tee -a "$LOG"

say ""
say "--- [8] 安装 orn3.new25 -> orn3 (旧件 orn3.pre25.bak 已在; 再存 orn3.pre25b.bak) ---"
cp -a orn3 orn3.pre25b.bak 2>/dev/null
cp -a orn3.new25 orn3
md5sum orn3 orn3.new25 orn3.pre25.bak | tee -a "$LOG"

say ""
say "=== stage25 结束 $(date '+%F %T') | 异常 = $(exc) | reset $(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) ==="
bash window_close25.sh >>"$LOG" 2>&1
echo "=== stage25 done ==="
