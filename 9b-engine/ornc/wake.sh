#!/bin/bash
# =============================================================================
# wake.sh — K200 唤醒 (第28章)
#   一个动作把卡唤醒: 清休眠标记 -> start.sh -> 轮询 /health 直到 ready -> 打印耗时。
#   用法:  bash /home/caden/ornc/wake.sh          # 唤醒并等待 ready
#          bash /home/caden/ornc/wake.sh --bg     # 后台唤醒, 立即返回(网关 WAITREADY 用)
#   实测暖机(灌 7.4GiB 权重到双芯 HBM) = 70~90s, 用户已接受“刚运行时慢”。
# =============================================================================
set -u
MARK=/home/caden/ornc/.idle_asleep
LOG=/home/caden/ornc/idle_sleep.log
log(){ echo "$(date '+%F %T') [wake] $*" >> "$LOG"; }
T0=$(date +%s)
rm -f "$MARK"
log "☀ 唤醒请求 (标记已清) 开始 start.sh"
if [ "${1:-}" = "--bg" ]; then
  nohup bash -c 'bash /home/caden/ornc/wake.sh' >/dev/null 2>&1 &
  exit 0
fi
bash /home/caden/ornc/start.sh >> "$LOG" 2>&1
# 轮询 ready
for i in $(seq 1 60); do
  H=$(curl -s -m 5 http://127.0.0.1:8090/health 2>/dev/null)
  case "$H" in *'"ready": true'*) EL=$(( $(date +%s) - T0 )); log "✓ 唤醒完成 耗时=${EL}s health=$H"; echo "wake ready in ${EL}s"; exit 0;; esac
  sleep 5
done
EL=$(( $(date +%s) - T0 ))
log "✗ 唤醒超时 ${EL}s (未 ready, 见 serve2.log)"
echo "wake TIMEOUT after ${EL}s"; exit 1
