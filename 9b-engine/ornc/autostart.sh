#!/bin/bash
# autostart.sh — 开机自拉起 8090 服务 (这台 VM 会被人重启, 服务必须自己回来)
# 纪律: 只做「检查 + 启动」。状态不对就记日志退出, 绝不 soft_reset、绝不重试刷屏。
# v3 (2026-09-21): 唯一入口统一为 start.sh / stop.sh —— 不再在 autostart 里
#   自己拼 orn3/serve2.py 或退回老版, 免得再出现「起的名字和停的名字不一致」。
LOG=/home/caden/ornc/autostart.log
say() { echo "[autostart $(date '+%F %T')] $*" >> "$LOG"; }

say "=== 开机自拉起开始 (pid=$$) ==="
# 1) 等驱动/设备节点就绪 (最多 180s)
for i in $(seq 1 60); do
  [ -e /dev/xpu0 ] && [ -e /proc/xpu/dev0/state ] && break
  sleep 3
done
if [ ! -e /proc/xpu/dev0/state ]; then say "驱动没上来, 放弃(不重试)"; exit 1; fi

# 2) 两芯必须 RUNNING
S0=$(cat /proc/xpu/dev0/state 2>/dev/null); S1=$(cat /proc/xpu/dev1/state 2>/dev/null)
say "dev0=$S0 dev1=$S1"
if [ "$S0" != "RUNNING" ] || [ "$S1" != "RUNNING" ]; then
  say "状态不是 RUNNING ⇒ 不自作主张 reset, 放弃(等人工处理)"; exit 2
fi

# 3) 起服务: start.sh 自己会 (a) 检测已有健康服务->不动 (b) 残留->调 stop.sh (c) 两芯校验
say "调用唯一入口 start.sh"
bash /home/caden/ornc/start.sh >> "$LOG" 2>&1
RC=$?
say "start.sh rc=$RC"
timeout 6 curl -s http://127.0.0.1:8090/health >> "$LOG" 2>&1; echo >> "$LOG"
say "=== 结束 ==="
exit $RC
