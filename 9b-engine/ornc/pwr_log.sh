#!/bin/bash
# =============================================================================
# pwr_log.sh — K200 功耗/温度采样 (第28章 交付)
#   用法: bash pwr_log.sh <秒数> [间隔秒] [标签]
#   例:   bash pwr_log.sh 300 30 idle_with_driver
#   输出: 每行一条 xpu_smi -m 读数 -> $OUT(csv), 结束打印 均值/最大/最小
#   只读: 只调用 xpu_smi(经 /dev/xpu* 读寄存器), 不改任何状态。
# =============================================================================
set -u
DUR=${1:-300}; INT=${2:-30}; LABEL=${3:-unlabeled}
OUT=/home/caden/ornc/pwr_${LABEL}_$(date +%H%M%S).csv
echo "ts,elapsed_s,power_mW_dev0,power_mW_dev1,temp_dev0,temp_dev1,hbm_MB_dev0,freq_dev0,state_dev0" > "$OUT"
T0=$(date +%s); N=0; SP=0; ST0=0; ST1=0; SPM=0; STM=0
while :; do
  TS=$(date '+%F %T'); EL=$(( $(date +%s) - T0 ))
  L0=$(xpu_smi -m 2>/dev/null | sed -n 1p); L1=$(xpu_smi -m 2>/dev/null | sed -n 2p)
  P0=$(echo "$L0"|awk '{print $9}');  P1=$(echo "$L1"|awk '{print $9}')
  T0T=$(echo "$L0"|awk '{print $5}'); T1T=$(echo "$L1"|awk '{print $5}')
  MB=$(echo "$L0"|awk '{print $18}'); F0=$(echo "$L0"|awk '{print $10}')
  ST=$(cat /proc/xpu/dev0/state 2>/dev/null)
  echo "$TS,$EL,$P0,$P1,$T0T,$T1T,$MB,$F0,$ST" >> "$OUT"
  [ -n "${P0:-}" ] && { SP=$((SP+P0)); N=$((N+1)); [ $P0 -gt $SPM ] && SPM=$P0; }
  [ -n "${T0T:-}" ] && { ST0=$((ST0+T0T)); ST1=$((ST1+T1T)); [ $T0T -gt $STM ] && STM=$T0T; }
  [ "$EL" -ge "$DUR" ] && break
  sleep "$INT"
done
echo "=== 标签=$LABEL 文件=$OUT 样本=$N 时长=$(( $(date +%s)-T0 ))s ==="
[ $N -gt 0 ] && echo "功耗 均值=$((SP/N)) mW (=$((SP/N/1000)).$(( (SP/N)%1000/100 )) W)  最大=$SPM mW"
[ $N -gt 0 ] && echo "温度 均值 dev0=$((ST0/N))℃ dev1=$((ST1/N))℃  最大=$STM℃"
