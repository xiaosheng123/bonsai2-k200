#!/bin/bash
# rollback25.sh — 立即回滚到已验收的生产组合, 并当场复验三图/黄金题 (单次停机 ≤2 分钟)
#   引擎: orn3.pre25.bak  (md5 24a925107d44a9b73174147927d93f78 = 第23章验收版)
#   网关: serve2.rollback.py (用户已验收版 + 预算 600 / K200_PROMPT_BUDGET 可覆盖)
set -u
cd /home/caden/ornc
O=/home/caden/ornc/accept.rb25; mkdir -p $O; L=$O/verify.log; : > "$L"
T0=$(date '+%F %T'); S0=$(date +%s)
echo "=== rollback25 开始 $T0 ===" | tee -a "$L"
echo "--- 回滚前: 在线件 md5 ---" | tee -a "$L"
md5sum /home/caden/orn_engine/orn3 /home/caden/ornc/serve2.py | tee -a "$L"
bash stop.sh 2>&1 | tail -2 | tee -a "$L"
sleep 2
n=0; for d in /proc/[0-9]*/fd/*; do t=$(readlink "$d" 2>/dev/null); case "$t" in /dev/xpu*) n=$((n+1));; esac; done
echo "  停后持卡 fd=$n" | tee -a "$L"
[ "$n" != "0" ] && { echo "★ 卡未真空, 中止" | tee -a "$L"; exit 1; }
cp -p /home/caden/orn_engine/orn3 /home/caden/orn_engine/orn3.exp_25.$(date +%H%M).bak   # 留档实验件
cp /home/caden/orn_engine/orn3.pre25.bak /home/caden/orn_engine/orn3
cp /home/caden/ornc/serve2.rollback.py /home/caden/ornc/serve2.py
echo "--- 回滚后: 生产件 md5 (期望 orn3=24a925107d44a9b73174147927d93f78) ---" | tee -a "$L"
md5sum /home/caden/orn_engine/orn3 /home/caden/ornc/serve2.py | tee -a "$L"
unset K200_PROF
bash start.sh 2>&1 | tail -2 | tee -a "$L"
T1=$(date '+%F %T'); S1=$(date +%s)
echo "=== rollback25 停机窗口: $T0 → $T1 = $((S1-S0))s ===" | tee -a "$L"
curl -s --max-time 8 http://127.0.0.1:8090/health | tee -a "$L"; echo | tee -a "$L"

# ============ 当场复验 (服务态, 不再开窗) ============
P=8090; A=/home/caden/orn_engine/ask_img.py; M=/home/caden/ornc/mmtests
echo "########## R1 文字图 img_text (期望 K200) ##########" | tee -a "$L"
python3 $A $P $M/img_text.png '图中写的文字是什么？请只回答文字内容。' 32 2>&1 | tee -a "$L"
echo "########## R2 768² 表格图 big_table (期望 9/9: A/1/7 B/2/8 C/3/9) ##########" | tee -a "$L"
python3 $A $P $M/big_table.png '这张图是一个表格，请逐行说出每一格的字符。' 400 2>&1 | tee -a "$L"
echo "########## R3 图形/颜色 img_shapes ##########" | tee -a "$L"
python3 $A $P $M/img_shapes.png '图里有什么？用中文简短回答，说出形状和颜色。' 96 2>&1 | tee -a "$L"
echo "########## R4 黄金题逐字节 ##########" | tee -a "$L"
python3 /home/caden/ornc/accept6.py $P $O/accept6.log 64 2>&1 | tail -2 | tee -a "$L"
python3 /home/caden/orn_engine/cmp_txt21.py $O/accept6.log /home/caden/ornc/accept.post23/accept6.log 2>&1 | tee -a "$L"
echo "########## R5 卡台账 ##########" | tee -a "$L"
echo "异常累计 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log)" | tee -a "$L"
echo "reset_count = $(cat /proc/xpu/dev0/reset_count) / $(cat /proc/xpu/dev1/reset_count)" | tee -a "$L"
echo "state = $(cat /proc/xpu/dev0/state) / $(cat /proc/xpu/dev1/state)  crontab 行数 = $(crontab -l | wc -l)" | tee -a "$L"
echo "=== rollback25 结束 $(date '+%F %T') ===" | tee -a "$L"
