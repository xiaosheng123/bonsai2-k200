#!/bin/bash
# recycle.sh — 干净地停掉 8090 服务 + 卡上残留的 orn 引擎, 并确认两芯 state
# 注意: 引擎被 setsid 放到独立会话, 必须显式按名字杀; 服务的 cmdline 是 "python3 serve.py"(无全路径)
cd /home/caden/ornc 2>/dev/null || exit 1
echo "--- 杀服务 ---"
pkill -f "serve\.py"                  2>/dev/null; echo "  pkill serve=$?"
pkill -f "ornserv\.py"                2>/dev/null
pkill -f "safe_run\.sh -n orn"        2>/dev/null
pkill -9 -f "orn_engine/orn --n"      2>/dev/null
sleep 3
echo "--- 仍持有 /dev/xpu* 的进程 ---"
for p in /proc/[0-9]*; do
  pid=${p#/proc/}
  ls -l "$p/fd" 2>/dev/null | grep -q "/dev/xpu" && echo "  pid=$pid $(tr -d '\0' < $p/cmdline | head -c 70)"
done
echo "--- 残留检查 ---"
pgrep -a -f "serve\.py"                    || echo "  server: 无"
pgrep -a -f "safe_run\.sh -n orn"          || echo "  safe_run: 无"
pgrep -a -f "orn_engine/orn --n"           || echo "  引擎: 无"
echo "--- 设备 state ---"
cat /proc/xpu/dev0/state /proc/xpu/dev1/state
echo "--- 卡异常累计(kern.log) ---"
grep -c "Exception in kernel execution" /var/log/kern.log 2>/dev/null
