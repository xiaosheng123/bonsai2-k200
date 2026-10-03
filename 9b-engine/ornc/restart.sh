#!/bin/bash
# restart.sh — stop.sh + start.sh 的唯一组合入口 (别在命令行里手搓 pkill)
set -u
echo "### restart.sh $(date '+%F %T')"
bash /home/caden/ornc/stop.sh
sleep 1
exec bash /home/caden/ornc/start.sh
