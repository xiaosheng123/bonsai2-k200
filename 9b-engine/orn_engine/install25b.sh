#!/bin/bash
# install25b.sh — 第25轮第2次上线窗口 (目标 ≤100s, 一次搞定):
#   tail-keep 截断 + !TOK 分词接口 + ATTBLK 查询分块注意力 + gemv_batch 并行归一化 + 网关真实 token 裁剪
set -u
cd /home/caden/ornc
T0=$(date '+%F %T'); S0=$(date +%s)
echo "=== install25b 开始 $T0 ==="
bash stop.sh 2>&1 | tail -2
sleep 2
n=0; for d in /proc/[0-9]*/fd/*; do t=$(readlink "$d" 2>/dev/null); case "$t" in /dev/xpu*) n=$((n+1));; esac; done
echo "  停后: 持卡 fd=$n serve2=$(pgrep -cf serve2.py)"
[ "$n" != "0" ] && { echo "★ 卡没真空, 中止"; exit 1; }
echo "--- 备份当前在线件 (回滚用) ---"
cp -p /home/caden/orn_engine/orn3 /home/caden/orn_engine/orn3.pre25b.bak
cp -p /home/caden/ornc/serve2.py /home/caden/ornc/serve2.py.bak.pre25b
md5sum /home/caden/orn_engine/orn3.pre25b.bak /home/caden/ornc/serve2.py.bak.pre25b
echo "--- 安装 ---"
cp /home/caden/orn_engine/orn3.m30 /home/caden/orn_engine/orn3
cp /home/caden/ornc/serve2.m29.py /home/caden/ornc/serve2.py
md5sum /home/caden/orn_engine/orn3 /home/caden/ornc/serve2.py
echo "  期望 orn3=fa4d690db7d317d954fe85cba6bc96c0 (数值路径与生产一致: 只多了 MAXT/保尾/!TOK/结束原因/PPROF)  serve2=(m29 真实裁剪版)"
echo "--- 拉起 ---"
export K200_PROF=1
bash start.sh 2>&1 | tail -3
T1=$(date '+%F %T'); S1=$(date +%s)
echo "=== install25b 结束 $T1  停机时长 = $((S1-S0))s ==="
curl -s --max-time 8 http://127.0.0.1:8090/health; echo
grep -a 'MAXT=\|全注意力路径\|rope 查表' /home/caden/ornc/serve2.log | tail -3
