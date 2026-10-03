#!/bin/bash
# rescue27.sh — 第27轮: 卡救援 (EDMA -809 / H2D 全失败) + 老二进制离线验证
#   背景: 同机另一代理 00:00:03 的 soft_reset 把卡打进 RECOVERING; 00:02:3x 驱动被 reload;
#         之后 EDMA 报 -809, 任何 H2D 都失败 ⇒ 引擎 1.5s 退出 rc=1 (与我的补丁无关,
#         00:02:57/00:03:29 两次用【老生产件】启动同样失败, 证据在 safe_20260923_00025*.log)。
#   本脚本: 用包内官方工具 /usr/local/xpu-4.33.0/tools/soft_reset <dev> 各芯一次 (不重试),
#           然后用【老生产二进制】离线 load 一次, 验证卡是否恢复 (不碰服务/不装新件)。
set -u
L=/home/caden/ornc/rescue27.log
exec >> "$L" 2>&1
echo "=================================================================="
echo "=== rescue27 start $(date '+%F %T') ==="
echo "--- 前置: state / reset_count ---"
for d in 0 1; do echo "  dev$d state=$(cat /proc/xpu/dev$d/state 2>/dev/null) reset_count=$(cat /proc/xpu/dev$d/reset_count 2>/dev/null)"; done
n=0; for x in /proc/[0-9]*/fd/*; do t=$(readlink "$x" 2>/dev/null); case "$t" in /dev/xpu*) echo "  持卡: $x -> $t"; n=$((n+1));; esac; done
echo "  持卡 fd 数=$n"
echo "  异常累计 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log 2>/dev/null)"
echo "--- soft_reset 0 ---"
timeout 240 /usr/local/xpu-4.33.0/tools/soft_reset 0; echo "  rc=$?"
sleep 2; echo "  dev0 state=$(cat /proc/xpu/dev0/state 2>/dev/null)"
echo "--- soft_reset 1 ---"
timeout 240 /usr/local/xpu-4.33.0/tools/soft_reset 1; echo "  rc=$?"
sleep 2; echo "  dev0=$(cat /proc/xpu/dev0/state 2>/dev/null) dev1=$(cat /proc/xpu/dev1/state 2>/dev/null)"
echo "  reset_count = $(cat /proc/xpu/dev0/reset_count 2>/dev/null) / $(cat /proc/xpu/dev1/reset_count 2>/dev/null)"
echo "  异常累计 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log 2>/dev/null)"
echo "--- dmesg 尾 10 ---"; dmesg | tail -10
echo "=== 卡恢复验证: 老生产件 orn3(6785f070) 离线 load (safe_run -t 50) ==="
cd /home/caden/orn_engine
timeout 150 bash safe_run.sh -n p27rescue -t 50 -- /bin/sh -c 'exec env LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib VIS_CH=96 VIS_CHA=0 VIS_FOLD=1 ./orn3 --n 8 --model /home/caden/orn/Ornith-1.5-9B-Q8_0.gguf < /dev/null'
echo "  safe_run rc=$?"
echo "  --- 本次 safe 日志 ---"
ls -t /home/caden/orn_engine/safe_*.log | head -1 | xargs tail -8
echo "  stderr(.target):"; ls -t /home/caden/orn_engine/safe_*.log.target 2>/dev/null | head -1 | xargs cat 2>/dev/null | tail -5
echo "  异常累计 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log 2>/dev/null)"
echo "  reset_count = $(cat /proc/xpu/dev0/reset_count 2>/dev/null) / $(cat /proc/xpu/dev1/reset_count 2>/dev/null)"
echo "=== rescue27 done $(date '+%F %T') ==="
