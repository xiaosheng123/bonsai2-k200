#!/bin/bash
# window_open26.sh <name> — 第23轮卡窗口: 声明占用 + 注掉 svc_guard cron + 停服务
set -u
cd /home/caden/ornc
echo "=== window_open26 $(date '+%F %T') name=$1 ==="
python3 - "$1" <<'PY'
import sys, time
p = '/home/caden/orn_engine/COORD_NOTE2.txt'
s = "\n" + "-"*70 + "\n"
s += "【占用声明 / CLAIM】(代理: 第23轮 —— 跨头抽干点变稀 / H2D 目的地乒乓, 收 r21 的 0.66s)\n"
s += "  目标: 位级等价前提下把 mode4 每头抽干改稀; 行为零退化; 文本零退化。\n"
s += "  工作副本: vis.cpp.cards26 -> vistest.cards.new26 / orn3.new26。\n"
s += "  卡窗口: %s 起; 每个占卡作业走 safe_run.sh; 出现任何异常立即停手回滚, 不重试。\n" % sys.argv[1]
s += "  ★ svc_guard cron 行临时注掉(收尾原样恢复); 卡异常起 18 / reset_count 5-5。\n"
open(p, 'a').write(s)
PY
crontab -l > /home/caden/ornc/crontab.bak.agent26 2>/dev/null
crontab -l 2>/dev/null | sed 's|^\(\*/5 \* \* \* \* /bin/bash /home/caden/ornc/svc_guard.sh\)|#AGENT26# \1|' | crontab -
echo "--- crontab now ---"; crontab -l
bash stop.sh
sleep 2
echo "=== window_open26 done $(date '+%F %T') ==="
