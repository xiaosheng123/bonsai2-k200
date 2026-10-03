#!/bin/bash
# run_accept192.sh — 后台跑 6 条验收 (max_tokens=192), 结果落 accept192.log
cd /home/caden/ornc || exit 1
: > /home/caden/ornc/accept192.log
/usr/bin/setsid /usr/bin/nohup python3 /home/caden/ornc/accept6.py 8090 /home/caden/ornc/accept192.log 192 \
    >> /home/caden/ornc/accept192.run.log 2>&1 < /dev/null &
echo "accept pid=$!"
sleep 2
echo "--- 已启动, 当前日志 ---"
tail -3 /home/caden/ornc/accept192.log
