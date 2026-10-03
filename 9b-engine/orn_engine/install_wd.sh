#!/bin/bash
CRON='*/1 * * * * /bin/bash /home/caden/orn_engine/wd_guard.sh >/dev/null 2>&1'
cat > /home/caden/orn_engine/wd_guard.sh <<'GEOF'
#!/bin/bash
if [ -f /tmp/watchdog.pid ] && kill -0 "$(cat /tmp/watchdog.pid)" 2>/dev/null; then exit 0; fi
nohup /usr/bin/python3 /home/caden/orn_engine/watchdog.py >> /home/caden/orn_engine/watchdog.log 2>&1 &
GEOF
chmod +x /home/caden/orn_engine/wd_guard.sh
(crontab -l 2>/dev/null | grep -v -e watchdog -e wd_guard; echo "$CRON") | crontab -
pkill -f watchdog.py 2>/dev/null; rm -f /tmp/watchdog.pid; sleep 1
/bin/bash /home/caden/orn_engine/wd_guard.sh
sleep 3
echo "crontab:"; crontab -l | tail -1
echo "pidfile: $(cat /tmp/watchdog.pid 2>/dev/null)"
echo "进程数: $(pgrep -c -f watchdog.py)"
tail -2 /home/caden/orn_engine/watchdog.log
