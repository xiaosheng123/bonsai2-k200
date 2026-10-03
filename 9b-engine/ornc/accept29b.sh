#!/bin/bash
# accept29.sh — 第25轮端到端验收 (服务态, 不动卡窗口): 三图 + 6 条文本 + 逐字节对拍 + 长 prompt + 看图表
set -u
P=8090
A=/home/caden/orn_engine/ask_img.py
M=/home/caden/ornc/mmtests
O=/home/caden/ornc/accept.post25b
BASE=/home/caden/ornc/accept.post23
mkdir -p $O
L=$O/vision.log
: > "$L"
echo "=== accept29 开始 $(date '+%F %T') ===" | tee -a "$L"
echo "引擎 md5 = $(md5sum /home/caden/orn_engine/orn3 | cut -d' ' -f1)" | tee -a "$L"
echo "serve2 md5 = $(md5sum /home/caden/ornc/serve2.py | cut -d' ' -f1)  引擎 MAXT 档 = $(grep -a 'MAXT=' /home/caden/ornc/serve2.log | tail -1)" | tee -a "$L"
echo "########## 0) /health (含 prefill 可见性) ##########" | tee -a "$L"
curl -s --max-time 8 http://127.0.0.1:$P/health | tee -a "$L"; echo | tee -a "$L"

echo "########## 1) 512² 文字图 img_text (期望 K200) ##########" | tee -a "$L"
python3 $A $P $M/img_text.png '图中写的文字是什么？请只回答文字内容。' 32 2>&1 | tee -a "$L"
echo "########## 2) 512² 图形/颜色 img_shapes ##########" | tee -a "$L"
python3 $A $P $M/img_shapes.png '图里有什么？用中文简短回答，说出形状和颜色。' 96 2>&1 | tee -a "$L"
echo "########## 3) ★768² 表格图 big_table (逐格读, 期望 9/9, max_tokens=400) ##########" | tee -a "$L"
python3 $A $P $M/big_table.png '这张图是一个表格，请逐行说出每一格的字符。' 400 2>&1 | tee -a "$L"
echo "########## 4) 768² 图形/颜色 big_shapes ##########" | tee -a "$L"
python3 $A $P $M/big_shapes.png '图里有什么？用中文简短回答，说出形状和颜色。' 96 2>&1 | tee -a "$L"
echo "########## 5) 768² 文字图 big_text (已知既有缺陷档: K20) ##########" | tee -a "$L"
python3 $A $P $M/big_text.png '图中写的文字是什么？请只回答文字内容。' 32 2>&1 | tee -a "$L"

echo "########## 6) 文本回归 6 条 (含黄金题「你好」) ##########" | tee -a "$L"
python3 /home/caden/ornc/accept6.py $P $O/accept6.log 64 2>&1 | tail -3 | tee -a "$L"
echo "########## 7) 文本逐字节对拍 (vs $BASE/accept6.log) ##########" | tee -a "$L"
python3 /home/caden/orn_engine/cmp_txt21.py $O/accept6.log $BASE/accept6.log 2>&1 | tee -a "$L"

echo "########## 8) ★ /home/caden/sdnn/verify21.py (三图 + 黄金题 + 卡状态) ##########" | tee -a "$L"
python3 /home/caden/sdnn/verify21.py 2>&1 | tee -a "$L"

echo "########## 9) ★ 长 prompt 测试 (DSH 形态: 5742 字符/14 条 ⇒ 必须答 7, 且保留条数变多) ##########" | tee -a "$L"
python3 /home/caden/sdnn/longtest.py 2>&1 | tee -a "$L"
echo "--- _TRIM 日志行 (本次) ---" | tee -a "$L"
grep -a 'prompt 过长' /home/caden/ornc/serve2.log | tail -3 | tee -a "$L"

echo "########## 10) 卡状态 / 异常台账 ##########" | tee -a "$L"
echo "异常累计 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log)" | tee -a "$L"
echo "reset_count dev0/1 = $(cat /proc/xpu/dev0/reset_count) / $(cat /proc/xpu/dev1/reset_count)" | tee -a "$L"
echo "state dev0/1 = $(cat /proc/xpu/dev0/state) / $(cat /proc/xpu/dev1/state)" | tee -a "$L"
echo "=== accept29 结束 $(date '+%F %T') ===" | tee -a "$L"
