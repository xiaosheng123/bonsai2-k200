#!/bin/bash
# window_open.sh <name> — 临时独占卡的作业窗口: 声明占用 + 注掉 svc_guard cron + 停服务
set -u
cd /home/caden/ornc
echo "=== window_open $(date '+%F %T') name=$1 ==="
# 1) 声明占用
python3 - "$1" <<'PY'
import sys, io, time
p = '/home/caden/orn_engine/COORD_NOTE2.txt'
s = "\n" + "-"*70 + "\n"
s += "【占用声明 / CLAIM】%s (代理 8: 视觉塔激活搬卡 = 卡上折回 + 卡上 LN/GELU)\n" % time.strftime('%Y-%m-%d %H:%M')
s += "  目标: 512² ≤10s / 768² ≤20s, 行为不退化。工作副本 vis.cpp.cards / vistest.cards。\n"
s += "  卡窗口: %s 起; 每个占卡作业走 safe_run.sh; 出现任何异常立即停手回滚, 不重试。\n" % sys.argv[1]
s += "  ★ svc_guard cron 行临时注掉(收尾原样恢复); 卡异常起 17 / reset_count 4-4。\n"
open(p, 'a').write(s)
PY
# 2) 备份 crontab + 注掉 svc_guard
crontab -l > /home/caden/ornc/crontab.bak.agent8 2>/dev/null
crontab -l 2>/dev/null | sed 's|^\(\*/5 \* \* \* \* /bin/bash /home/caden/ornc/svc_guard.sh\)|#AGENT8# \1|' | crontab -
echo "--- crontab now ---"; crontab -l
# 3) 停服务
bash stop.sh
sleep 2
echo "=== window_open done $(date '+%F %T') ==="
