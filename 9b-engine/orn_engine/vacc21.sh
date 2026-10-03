#!/bin/bash
# vacc21.sh — 第 21 轮交付验收 (8090 单口): 三张图 + 文本 6 条
set -u
P=8090
A=/home/caden/orn_engine/ask_img.py
M=/home/caden/ornc/mmtests
O=/home/caden/ornc/accept.post21
mkdir -p $O
echo "=== vacc21 开始 $(date '+%F %T') ===" | tee $O/vision.log
echo "########## 1) 512² 图形/颜色 (img_shapes) ##########" | tee -a $O/vision.log
python3 $A $P $M/img_shapes.png '图里有什么？用中文简短回答，说出形状和颜色。' 96 2>&1 | tee -a $O/vision.log
echo "########## 2) 512² 文字图 (img_text / K200) ##########" | tee -a $O/vision.log
python3 $A $P $M/img_text.png '图中写的文字是什么？请只回答文字内容。' 64 2>&1 | tee -a $O/vision.log
echo "########## 3) 768² 表格图 (big_table / 逐格读数) ##########" | tee -a $O/vision.log
python3 $A $P $M/big_table.png '这张图是一个表格，请逐行说出每一格的字符。' 320 2>&1 | tee -a $O/vision.log
echo "########## 4) 文本回归 6 条 (含黄金题「你好」) ##########" | tee -a $O/vision.log
python3 /home/caden/ornc/accept6.py $P $O/accept6.log 64 2>&1 | tail -4 | tee -a $O/vision.log
echo "=== vacc21 done $(date '+%F %T') ===" | tee -a $O/vision.log
