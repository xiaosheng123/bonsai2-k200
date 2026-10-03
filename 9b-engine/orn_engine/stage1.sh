#!/bin/bash
# stage1.sh — 新算子极小尺寸单发验证 (每个算子独立进程; 卡异常立即停手)
set -u
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
LOG=/home/caden/orn_engine/stage1_probe.log
: > "$LOG"

run() {
  echo "########## probe $*  $(date '+%T') ##########" | tee -a "$LOG"
  bash safe_run.sh -n "p_$1" -t 180 -- timeout 170 ./probe "$@" >>"$LOG" 2>&1
  rc=$?
  echo "[stage1] arg='$*' safe_run rc=$rc" | tee -a "$LOG"
  if [ "$rc" -eq 3 ]; then
    echo "[stage1] ★★★ 卡异常 -> 立即停手, 绝不重试, 直接收尾 ★★★" | tee -a "$LOG"
    echo "CARD_EXCEPTION" >> "$LOG"
    return 3
  fi
  if [ "$rc" -eq 2 ]; then echo "[stage1] 前置拒绝(有并发占用?) -> 停手" | tee -a "$LOG"; return 2; fi
  return 0
}

bash /home/caden/orn_engine/window_open.sh "stage1-probe $(date '+%T')" >>"$LOG" 2>&1

for t in ln64 ln1024 ln1152 gelu ew softmax reduce trans; do
  run "$t"; r=$?
  if [ $r -eq 3 ] || [ $r -eq 2 ]; then break; fi
done

if ! grep -q CARD_EXCEPTION "$LOG"; then
  echo "########## 中等尺寸 ##########" | tee -a "$LOG"
  for t in lnM geluF ewM softmaxM; do
    run "$t"; r=$?
    if [ $r -eq 3 ] || [ $r -eq 2 ]; then break; fi
  done
fi

echo "########## 全量尺寸 (仅 LN 1152-全宽) ##########" | tee -a "$LOG"
grep -q CARD_EXCEPTION "$LOG" || { run lnF; }

bash /home/caden/orn_engine/window_close.sh >>"$LOG" 2>&1
echo "=== stage1 done $(date '+%F %T') ===" | tee -a "$LOG"
