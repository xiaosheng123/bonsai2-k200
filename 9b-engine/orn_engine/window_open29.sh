#!/bin/bash
# window_open29.sh <name> — 第24轮卡窗口: 声明占用 + 注掉 svc_guard cron + 停服务 + 证明卡真空
set -u
cd /home/caden/ornc
echo "=== window_open29 $(date '+%F %T') name=$1 ==="
python3 - "$1" <<'PY'
import sys
p = '/home/caden/orn_engine/COORD_NOTE2.txt'
s = "\n" + "-"*70 + "\n"
s += "【占用声明 / CLAIM】(代理: 第24轮 —— MAXT 1024→2048 + 预填充耗时分解/优化)\n"
s += "  目标: 位置上限抬到 2048 (逐处核算 rope/KV/缓冲/主机内存), 预填充不变慢,\n"
s += "        文本 6 条 + 黄金题逐字节不变, 看图三图不退化, 结束卡异常 18 / reset 5-5。\n"
s += "  卡窗口: %s 起; 每个占卡作业走 safe_run.sh -n/-t; 出现任何异常立即停手回滚, 不重试。\n" % sys.argv[1]
s += "  临时产物: orn3_m.cpp / orn3.m29 / pfb_*.json; 生产 orn3 与 orn3.o 不动。\n"
open(p, 'a').write(s)
PY
crontab -l > /home/caden/ornc/crontab.bak.agent29 2>/dev/null
crontab -l 2>/dev/null | sed 's|^\(\*/5 \* \* \* \* /bin/bash /home/caden/ornc/svc_guard.sh\)|#AGENT29# \1|' | crontab -
echo "--- crontab now ---"; crontab -l
bash stop.sh
sleep 3
echo "--- 持卡进程扫描 (必须为空) ---"
n=0
for d in /proc/[0-9]*/fd/*; do t=$(readlink "$d" 2>/dev/null); case "$t" in /dev/xpu*) echo "  持卡: $d -> $t"; n=$((n+1));; esac; done
echo "  持卡 fd 数 = $n"
echo "=== window_open29 done $(date '+%F %T') ==="
