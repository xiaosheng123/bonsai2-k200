#!/bin/bash
# vacc.sh —— 交付验收 (8090 单口): 三张图 + 文本回归
P=8090
A=/home/caden/orn_engine/ask_img.py
M=/home/caden/ornc/mmtests
O=/home/caden/ornc/accept.opt
mkdir -p $O
echo "########## 1) 512² 图形/颜色 ##########" | tee $O/vision.log
python3 $A $P $M/img_shapes.png '图里有什么？用中文简短回答，说出形状和颜色。' 96 2>&1 | tee -a $O/vision.log
echo "########## 2) 512² 文字图 (K200) ##########" | tee -a $O/vision.log
python3 $A $P $M/img_text.png '图中写的文字是什么？请只回答文字内容。' 64 2>&1 | tee -a $O/vision.log
echo "########## 3) 768² 表格图 (逐格读数) ##########" | tee -a $O/vision.log
python3 $A $P $M/big_table.png '这张图是一个表格，请逐行说出每一格的字符。' 320 2>&1 | tee -a $O/vision.log
echo "########## 4) 文本回归 6 条 ##########"
python3 /home/caden/ornc/accept6.py $P $O/accept6.log 64 2>&1 | tail -6
echo "=== vacc done $(date '+%F %T') ==="
