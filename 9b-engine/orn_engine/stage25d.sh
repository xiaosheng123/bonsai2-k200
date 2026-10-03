#!/bin/bash
# stage25d.sh — 第22轮诊断窗口: 定位 CARDFFN=1 数值分歧到底在【卡上 gelu】还是【卡上 down 投影】
#   mode2 (VIS_CARDFFN=2) = 主机 bias/gelu(与生产档逐位相同) + 卡上 Up 上传 + 卡上分块 down gemm
#     => 若 mode2 的 layer0 也与生产档【位级不同】, 则分歧来自 down 投影 (与 gelu 无关);
#        若 mode2 的 layer0 与生产档位级全同, 则分歧来自卡上 api::gelu。
set -u
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
LOG=/home/caden/orn_engine/stage25d.log
: > "$LOG"
MM=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
D=/tmp/vs25
mkdir -p $D/n768/m2
V=./vistest.cards.new25
say() { echo "$*" | tee -a "$LOG"; }
exc() { grep -ac 'Exception in kernel execution' /var/log/kern.log; }
say "=== stage25d 开始 $(date '+%F %T') | 异常起 = $(exc) ==="
bash window_open25.sh "stage25d-diag" >>"$LOG" 2>&1
say "--- [1] 768² VIS_ONLY=27 CARDFFN=2 (主机 bias/gelu + 卡上 down 投影), 落盘 ---"
bash safe_run.sh -n m2 -t 900 -- env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_CARDFFN=2 VIS_HEAP=1 VIS_HEAPD2H=4 VIS_ONLY=27 VIS_PROF=1 OMP_NUM_THREADS=4 \
    $V run "$MM" /tmp/inp768.f32 768 768 $D/n768/m2 >>"$LOG" 2>&1
RC=$?
say "[stage25d] m2 safe_run rc=$RC (3=卡异常)"
if [ "$RC" -eq 3 ]; then say "!! 卡异常 -> 停手收尾 !!"; bash window_close25.sh >>"$LOG" 2>&1; echo ABORT; exit 3; fi
say "=========== 对拍: mode2 vs 生产档(CARDFFN=0) ==========="
python3 cmp20.py $D/n768/m2 $D/n768/ref "768 mode2(CARDFFN=2) vs ref(CARDFFN=0)" 2>&1 | tee -a "$LOG"
say "=========== 对拍: mode2 vs mode1(全卡) ==========="
python3 cmp20.py $D/n768/m2 $D/n768/card "768 mode2 vs card(mode1)" 2>&1 | tee -a "$LOG"
say "--- 逐层耗时 mode2 ---"
grep -a -E '^\[vis\]   层 (0|20|21|26) |完成: ' "$LOG" | tee -a "$LOG"
say "=== stage25d 结束 $(date '+%F %T') | 异常 = $(exc) | reset $(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) ==="
bash window_close25.sh >>"$LOG" 2>&1
echo "=== stage25d done ==="
