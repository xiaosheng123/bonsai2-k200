#!/bin/bash
# =============================================================================
# idle_sleep.sh — K200 空闲休眠执行器 (第28章)
#   设计: cron */1 调用; 读 k200_idle_sleep.conf; K200_IDLE_SLEEP_MIN=0 时直接退出。
#   判据: 8090 /health 的 requests 计数 在 N 分钟内不变 => 判为空闲 => 停服务。
#   安全: 不动驱动/PCI(LEVEL=L1); 拿 .cardlock 才动手; 有卡作业立刻放弃。
#   ★ 默认关闭。开启前请先读 /home/caden/ornc/idle_sleep_README.md
# =============================================================================
set -u
CONF=/home/caden/ornc/k200_idle_sleep.conf
[ -f "$CONF" ] || { echo "$(date '+%F %T') [idle_sleep] 缺配置文件 $CONF => 退出"; exit 0; }
. "$CONF"
LOG="${K200_IDLE_SLEEP_LOG:-/home/caden/ornc/idle_sleep.log}"
log(){ echo "$(date '+%F %T') [idle_sleep] $*" >> "$LOG"; }
[ "${K200_IDLE_SLEEP_MIN:-0}" -gt 0 ] 2>/dev/null || exit 0     # 默认关闭

STATE=/home/caden/ornc/.idle_sleep_state
MARK="${K200_IDLE_SLEEP_MARKER:-/home/caden/ornc/.idle_asleep}"
[ -f "$MARK" ] && exit 0                                        # 已在休眠

# 读 /health 的 requests 计数
H=$(curl -s -m 5 "${K200_IDLE_SLEEP_SRC:-http://127.0.0.1:8090/health}" 2>/dev/null)
REQ=$(echo "$H" | grep -o '"requests": *[0-9]*' | grep -o '[0-9]*$')
[ -z "${REQ:-}" ] && { log "health 不可读(服务可能已停) => 不动"; exit 0; }
NOW=$(date +%s)
if [ -f "$STATE" ]; then . "$STATE"; else OLD_REQ=$REQ; OLD_TS=$NOW; fi
if [ "$REQ" != "$OLD_REQ" ]; then
  printf 'OLD_REQ=%s\nOLD_TS=%s\n' "$REQ" "$NOW" > "$STATE"; exit 0
fi
IDLE_MIN=$(( (NOW - OLD_TS) / 60 ))
if [ "$IDLE_MIN" -lt "$K200_IDLE_SLEEP_MIN" ]; then
  printf 'OLD_REQ=%s\nOLD_TS=%s\n' "$OLD_REQ" "$OLD_TS" > "$STATE"; exit 0
fi

# 卡上有作业? 有则不动
if pgrep -f 'safe_run.sh -n (?!orn_serve)' >/dev/null 2>&1 || pgrep -f 'pfbench|visbench' >/dev/null 2>&1; then
  log "卡上有其它作业 => 放弃本轮休眠"; exit 0
fi
# 拿锁(最多等 30s, 拿不到就不动手)
if ! mkdir /home/caden/ornc/.cardlock 2>/dev/null; then log "cardlock 被占 => 放弃本轮"; exit 0; fi
echo "agent=idle_sleep pid=$$" > /home/caden/ornc/.cardlock/owner.txt
trap 'rmdir /home/caden/ornc/.cardlock 2>/dev/null' EXIT
T0=$(date +%s)
touch "$MARK"
bash /home/caden/ornc/stop.sh >> "$LOG" 2>&1
log "☾ 进入休眠 L1(停服务) 空闲=${IDLE_MIN}min 耗时=$(( $(date +%s) - T0 ))s 用 wake.sh 唤醒"
exit 0
