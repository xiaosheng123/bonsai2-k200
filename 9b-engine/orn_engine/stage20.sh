#!/bin/bash
# stage20.sh — 第 20 轮卡窗口: 新算子探针 + 768²/512² 四档对拍 (全部走 safe_run 安全网)
set -u
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
LOG=/home/caden/orn_engine/stage20.log
: > "$LOG"
MM=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
D=/tmp/vs20
rm -rf $D; mkdir -p $D/n7/a $D/n7/b $D/n7/c $D/n7/d $D/n1/a $D/n1/c
say() { echo "$*" | tee -a "$LOG"; }
run() {
  local nm="$1"; local tt="$2"; shift 2
  echo "########## $nm $(date '+%T') ##########" | tee -a "$LOG"
  bash safe_run.sh -n "$nm" -t "$tt" -- "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "[stage20] $nm safe_run rc=$rc" | tee -a "$LOG"
  if [ "$rc" -eq 3 ]; then echo "CARD_EXCEPTION" >>"$LOG"; return 3; fi
  if [ "$rc" -eq 2 ]; then echo "PRECHECK_REFUSE" >>"$LOG"; return 2; fi
  if [ "$rc" -eq 1 ]; then echo "TARGET_FAIL" >>"$LOG"; return 1; fi
  return 0
}
abort() { bash window_close.sh >>"$LOG" 2>&1; echo ABORT; exit 3; }

say "=== stage20 开始 $(date '+%F %T') | 异常起 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log) | reset $(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) ==="
bash window_open.sh "stage20-dumps" >>"$LOG" 2>&1

say "--- [1] 第 20 轮新算子/新用法 极小尺寸单发 ---"
run p_redmin3 90  ./probe redmin3  || true; grep -q CARD_EXCEPTION "$LOG" && abort
run p_addrow  90  ./probe addrow   || true; grep -q CARD_EXCEPTION "$LOG" && abort
run p_addrowM 90  ./probe addrowM  || true; grep -q CARD_EXCEPTION "$LOG" && abort
run p_maxabs  90  ./probe maxabs   || true; grep -q CARD_EXCEPTION "$LOG" && abort
run p_maxabsM 90  ./probe maxabsM  || true; grep -q CARD_EXCEPTION "$LOG" && abort
run p_geluz   90  ./probe geluz    || true; grep -q CARD_EXCEPTION "$LOG" && abort
run p_geluzM  90  ./probe geluzM   || true; grep -q CARD_EXCEPTION "$LOG" && abort
say "--- 探针结果 ---"
grep -a '^\[probe\]' "$LOG" | tail -20 | tee -a "$LOG"

say "--- [2] 768² a: VIS_CARDFFN=0 VIS_HEAP=0 (基线, 期望与第19章落盘字节相同) ---"
run n7a 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=0 VIS_PROF=1 OMP_NUM_THREADS=4 \
    ./vistest.cards run "$MM" /tmp/inp768.f32 768 768 $D/n7/a || true; grep -q CARD_EXCEPTION "$LOG" && abort
say "--- [3] 768² b: VIS_CARDFFN=0 VIS_HEAP=1 (只开 B, 期望与 a 字节相同) ---"
run n7b 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=1 VIS_PROF=1 OMP_NUM_THREADS=4 \
    ./vistest.cards run "$MM" /tmp/inp768.f32 768 768 $D/n7/b || true; grep -q CARD_EXCEPTION "$LOG" && abort
say "--- [4] 768² c: VIS_CARDFFN=1 VIS_HEAP=1 (A全卡 + B) ---"
run n7c 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=1 VIS_HEAP=1 VIS_PROF=1 OMP_NUM_THREADS=4 \
    ./vistest.cards run "$MM" /tmp/inp768.f32 768 768 $D/n7/c || true; grep -q CARD_EXCEPTION "$LOG" && abort
say "--- [5] 768² d: VIS_CARDFFN=2 VIS_HEAP=1 (A半卡: 主机 bias/gelu) ---"
run n7d 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=2 VIS_HEAP=1 VIS_PROF=1 OMP_NUM_THREADS=4 \
    ./vistest.cards run "$MM" /tmp/inp768.f32 768 768 $D/n7/d || true; grep -q CARD_EXCEPTION "$LOG" && abort
say "--- [6] 512² a: VIS_CARDFFN=0 VIS_HEAP=0 ---"
run n1a 600 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=0 VIS_HEAP=0 VIS_PROF=1 OMP_NUM_THREADS=4 \
    ./vistest.cards run "$MM" /tmp/inp512.f32 512 512 $D/n1/a || true; grep -q CARD_EXCEPTION "$LOG" && abort
say "--- [7] 512² c: VIS_CARDFFN=1 VIS_HEAP=1 ---"
run n1c 600 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=1 VIS_HEAP=1 VIS_PROF=1 OMP_NUM_THREADS=4 \
    ./vistest.cards run "$MM" /tmp/inp512.f32 512 512 $D/n1/c || true; grep -q CARD_EXCEPTION "$LOG" && abort

say ""
say "=========== 对拍 1: 768² 本轮 mode0 vs 第19章生产落盘 (证明 padding 布局逐位透明) ==========="
python3 /home/caden/orn_engine/cmp20.py /tmp/vs20/n7/a /tmp/vsw/n7c/new "768 mode0(pre20binary) vs ch19-new(cardA)" 2>&1 | tee -a "$LOG"
say ""
say "=========== 对拍 2: 768² (B)HEAP=1 vs 基线 (证明 (B) 位级不变) ==========="
python3 /home/caden/orn_engine/cmp20.py /tmp/vs20/n7/b /tmp/vs20/n7/a "768 HEAP=1 vs HEAP=0" 2>&1 | tee -a "$LOG"
say ""
say "=========== 对拍 3: 768² (A)全卡 vs 基线 ==========="
python3 /home/caden/orn_engine/cmp20.py /tmp/vs20/n7/c /tmp/vs20/n7/a "768 CARDFFN=1 vs 0" 2>&1 | tee -a "$LOG"
say ""
say "=========== 对拍 4: 768² (A)半卡 vs 基线 ==========="
python3 /home/caden/orn_engine/cmp20.py /tmp/vs20/n7/d /tmp/vs20/n7/a "768 CARDFFN=2 vs 0" 2>&1 | tee -a "$LOG"
say ""
say "=========== 对拍 5: 512² (A)全卡 vs 基线 ==========="
python3 /home/caden/orn_engine/cmp20.py /tmp/vs20/n1/c /tmp/vs20/n1/a "512 CARDFFN=1 vs 0" 2>&1 | tee -a "$LOG"
say ""
say "=========== 对拍 6: 512² 本轮 mode0 vs 第19章生产落盘 ==========="
python3 /home/caden/orn_engine/cmp20.py /tmp/vs20/n1/a /tmp/vsw/n1c/new "512 mode0 vs ch19-new" 2>&1 | tee -a "$LOG"
say ""
say "=========== 耗时 (各档最后一行的 总耗时/prof) ==========="
grep -a -E '完成: |PROF:|PROF2:|图像编码|片上|总耗时|前向:' "$LOG" | tail -60 | tee -a "$LOG"
say ""
say "=== stage20 结束 $(date '+%F %T') | 异常 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log) | reset $(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) ==="
bash window_close.sh >>"$LOG" 2>&1
echo "=== stage20 done ==="
