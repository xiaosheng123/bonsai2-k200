#!/bin/bash
# accept23.sh — 第23轮端到端验收 (服务态, 不动卡窗口): 三图 + 文本 6 条
#   生产档未变 (orn3 = 24a92510…), 本脚本只为"现状不改"提供本轮的新鲜证据。
set -u
P=8090
A=/home/caden/orn_engine/ask_img.py
M=/home/caden/ornc/mmtests
O=/home/caden/ornc/accept.post23
BASE=/home/caden/ornc/accept.post22rb
mkdir -p $O
L=$O/vision.log
: > "$L"
echo "=== accept23 开始 $(date '+%F %T') | 引擎 md5 = $(md5sum /home/caden/orn_engine/orn3 | cut -d' ' -f1) ===" | tee -a "$L"
echo "########## 1) 512² 文字图 img_text (期望 K200) ##########" | tee -a "$L"
python3 $A $P $M/img_text.png '图中写的文字是什么？请只回答文字内容。' 32 2>&1 | tee -a "$L"
echo "########## 2) 512² 图形/颜色 img_shapes ##########" | tee -a "$L"
python3 $A $P $M/img_shapes.png '图里有什么？用中文简短回答，说出形状和颜色。' 96 2>&1 | tee -a "$L"
echo "########## 3) ★768² 表格图 big_table (逐格读数, 期望 9/9, max_tokens=400 给足) ##########" | tee -a "$L"
python3 $A $P $M/big_table.png '这张图是一个表格，请逐行说出每一格的字符。' 400 2>&1 | tee -a "$L"
echo "########## 4) 768² 图形/颜色 big_shapes ##########" | tee -a "$L"
python3 $A $P $M/big_shapes.png '图里有什么？用中文简短回答，说出形状和颜色。' 96 2>&1 | tee -a "$L"
echo "########## 5) 768² 文字图 big_text ##########" | tee -a "$L"
python3 $A $P $M/big_text.png '图中写的文字是什么？请只回答文字内容。' 32 2>&1 | tee -a "$L"
echo "########## 6) 文本回归 6 条 (含黄金题「你好」) ##########" | tee -a "$L"
python3 /home/caden/ornc/accept6.py $P $O/accept6.log 64 2>&1 | tail -4 | tee -a "$L"
echo "########## 7) 文本逐字节对拍 (vs $BASE/accept6.log) ##########" | tee -a "$L"
python3 /home/caden/orn_engine/cmp_txt21.py $O/accept6.log $BASE/accept6.log 2>&1 | tee -a "$L"
echo "########## 8) 编码耗时汇总 (从 k200.image_encode 抓) ##########" | tee -a "$L"
grep -a -o "总耗时 [0-9.]*s" "$L" | tee -a "$L"
echo "=== accept23 结束 $(date '+%F %T') ===" | tee -a "$L"
