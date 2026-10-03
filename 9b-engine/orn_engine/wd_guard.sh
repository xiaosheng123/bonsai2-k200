#!/bin/bash
if [ -f /tmp/watchdog.pid ] && kill -0 "$(cat /tmp/watchdog.pid)" 2>/dev/null; then exit 0; fi
nohup /usr/bin/python3 /home/caden/orn_engine/watchdog.py >> /home/caden/orn_engine/watchdog.log 2>&1 &
