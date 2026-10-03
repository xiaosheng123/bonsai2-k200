#!/bin/bash
# window_open25.sh <name> — 第22轮卡窗口: 声明占用 + 注掉 svc_guard cron + 停服务
set -u
cd /home/caden/ornc
echo "=== window_open25 $(date '+%F %T') name=$1 ==="
python3 - "$1" <<'PY'
import sys, time
p = '/home/caden/orn_engine/COORD_NOTE2.txt'
s = "\n" + "-"*70 + "\n"
s += "【占用声明 / CLAIM】%s (代理: 第22轮 —— FFN 全卡路径(VIS_CARDFFN=1)行为审计与上线)\n" % time.strftime('%Y-%m-%d %H:%M')
s += "  目标: 768² 16.17s -> ~12.7s; 行为零退化(表格 9/9 / K200 / 图形颜色位置); 文本零退化。\n"
s += "  工作副本: vis.cpp.cards25 (含 CARDFFN 默认 1) -> vistest.cards.new25 / orn3.new25。\n"
s += "  卡窗口: %s 起; 每个占卡作业走 safe_run.sh; 出现任何异常立即停手回滚, 不重试。\n" % sys.argv[1]
s += "  ★ svc_guard cron 行临时注掉(收尾原样恢复); 卡异常起 18 / reset_count 5-5。\n"
open(p, 'a').write(s)
PY
crontab -l > /home/caden/ornc/crontab.bak.agent25b 2>/dev/null
crontab -l 2>/dev/null | sed 's|^\(\*/5 \* \* \* \* /bin/bash /home/caden/ornc/svc_guard.sh\)|#AGENT25# \1|' | crontab -
echo "--- crontab now ---"; crontab -l
bash stop.sh
sleep 2
echo "=== window_open25 done $(date '+%F %T') ==="
