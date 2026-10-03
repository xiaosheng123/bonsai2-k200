#!/bin/bash
# recycle2.sh — 兼容壳: 唯一停止入口已统一为 stop.sh (2026-09-21)
exec bash /home/caden/ornc/stop.sh "$@"
