#!/bin/bash
# bl19.sh — 本轮"改前"基线: 8090 三图 + 文本 6 条 (不占卡独占, 服务在线跑)
set -u
O=/home/caden/ornc/accept.pre19
mkdir -p $O
M=/home/caden/ornc/mmtests
A=/home/caden/orn_engine/ask_img.py
LOG=$O/run.log
: > "$LOG"
say() { echo "$*" | tee -a "$LOG"; }

say "=== pre19 baseline $(date '+%F %T') ==="
say "--- 1) 512² 图形/颜色 ---"
python3 $A 8090 $M/img_shapes.png '图里有什么？用中文简短回答，说出形状和颜色。' 96 > $O/v1.log 2>&1
tail -6 $O/v1.log | tee -a "$LOG"
say "--- 2) 512² 文字图 ---"
python3 $A 8090 $M/img_text.png '图中写的文字是什么？请只回答文字内容。' 64 > $O/v2.log 2>&1
tail -6 $O/v2.log | tee -a "$LOG"
say "--- 3) 768² 表格 ---"
python3 $A 8090 $M/big_table.png '这张图是一个表格，请逐行说出每一格的字符。' 320 > $O/v3.log 2>&1
tail -10 $O/v3.log | tee -a "$LOG"
say "--- 4) 文本 6 条 ---"
python3 /home/caden/ornc/accept6.py 8090 $O/accept6.log 64 > /dev/null 2>&1
grep -a -E "^########## Q|^回复原文|^\[[0-9]" $O/accept6.log | tee -a "$LOG"
say "--- 5) 引擎日志耗时 ---"
grep -a -E "图像编码|编码耗时|prompt=" /home/caden/ornc/serve2.log | tail -30 | tee -a "$LOG"
say "=== pre19 done $(date '+%F %T') ==="
