#!/bin/bash
# vacc2.sh <tag> —— 只跑三张图 (用于 CHA 档对比)
P=8090
A=/home/caden/orn_engine/ask_img.py
M=/home/caden/ornc/mmtests
O=/home/caden/ornc/accept.opt/$1
mkdir -p $O
echo "########## $1: 512² 图形 ##########" | tee $O/vision.log
python3 $A $P $M/img_shapes.png '图里有什么？用中文简短回答，说出形状和颜色。' 96 2>&1 | tee -a $O/vision.log
echo "########## $1: 512² K200 文字 ##########" | tee -a $O/vision.log
python3 $A $P $M/img_text.png '图中写的文字是什么？请只回答文字内容。' 64 2>&1 | tee -a $O/vision.log
echo "########## $1: 768² 表格 ##########" | tee -a $O/vision.log
python3 $A $P $M/big_table.png '这张图是一个表格，请逐行说出每一格的字符。' 320 2>&1 | tee -a $O/vision.log
echo "=== vacc2 $1 done $(date '+%F %T') ==="
