#!/bin/bash
# 切到 per-row 版 (8.443 tok/s, 黄金逐字一致)
exec bash /home/caden/ornc/stop.sh && cp -a /home/caden/orn_engine/orn3.perrow /home/caden/orn_engine/orn3 && bash /home/caden/ornc/start.sh
