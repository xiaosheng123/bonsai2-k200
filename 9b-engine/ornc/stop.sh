#!/bin/bash
# =============================================================================
# stop.sh — 8090 服务的【唯一停止入口】(与 start.sh 共用同一套命名约定)
#
#   为什么这么写: 上一轮 start 起的是 serve2.py/orn3, 而旧的停止脚本只认
#   serve.py/orn, 结果「停不掉」, 带病引擎继续占卡。所以这里:
#     1) 名字列表与 start.sh 一一对应 (serve2.py / orn3 / safe_run.sh -n orn*)
#     2) 最后用 /proc/<pid>/fd 实测「谁真的开着 /dev/xpu*」兜底 kill -9, 不靠猜
#     3) 只杀进程 —— 绝不 soft_reset, 绝不刷固件, 绝不碰 watchdog
# =============================================================================
set -u
PORT=8090
PAT_SRV='serve[0-9]*\.py'
PAT_SAFE='safe_run\.sh -n orn'
PAT_ENG='orn_engine/orn[0-9]*'

echo "=== stop.sh $(date '+%F %T') ==="

echo "--- 1) 服务进程 ($PAT_SRV) ---"
pkill -f "$PAT_SRV" 2>/dev/null && echo "  已 TERM: $PAT_SRV" || echo "  $PAT_SRV: 无"

echo "--- 2) 安全网包装 + 卡上引擎 ---"
pkill -f "$PAT_SAFE" 2>/dev/null && echo "  已 TERM: $PAT_SAFE" || echo "  $PAT_SAFE: 无"
sleep 1
pkill -9 -f "$PAT_SAFE" 2>/dev/null
pkill -9 -f "$PAT_ENG"  2>/dev/null
sleep 2

echo "--- 3) 兜底: 实测谁还开着 /dev/xpu* -> kill -9 ---"
LEFT=0
for p in /proc/[0-9]*; do
  pid=${p#/proc/}
  [ "$pid" = "$$" ] && continue
  [ "$pid" = "$PPID" ] && continue
  cmd="$(cat "$p/cmdline" 2>/dev/null | tr -d '\0')"
  case "$cmd" in *watchdog.py*) continue ;; esac
  if ls -l "$p/fd" 2>/dev/null | grep -q '/dev/xpu'; then
    echo "  仍持有 /dev/xpu: pid=$pid ${cmd:0:90} => kill -9"
    kill -9 "$pid" 2>/dev/null && LEFT=$((LEFT+1))
  fi
done
sleep 2

echo "--- 4) 复核 ---"
pgrep -a -f "$PAT_SRV" >/dev/null 2>&1 && pgrep -a -f "$PAT_SRV" || echo "  server: 无"
pgrep -a -f "$PAT_ENG" >/dev/null 2>&1 && pgrep -a -f "$PAT_ENG" || echo "  engine: 无"
printf '  dev state: %s / %s\n' "$(cat /proc/xpu/dev0/state 2>/dev/null)" "$(cat /proc/xpu/dev1/state 2>/dev/null)"
if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -qw "$PORT"; then
  echo "  8090: 仍在监听 (异常!)"
else
  echo "  8090: 未监听 (已停)"
fi
echo "  被兜底 kill 的持卡进程数=$LEFT"
echo "  卡异常累计(Exception in kernel execution) = $(grep -ac 'Exception in kernel execution' /var/log/kern.log 2>/dev/null)"
exit 0
