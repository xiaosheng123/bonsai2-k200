#!/bin/bash
# install25.sh — 第25轮上线的【唯一】停机窗口 (目标 ≤100s):
#   stop.sh -> 备份 -> 换引擎(orn3=m29) + 换网关(serve2 预算1700/finish_reason) -> start.sh -> 等 ready
#   全程打印时间戳, 用于如实计算停机时长。
set -u
cd /home/caden/ornc
T0=$(date '+%F %T'); S0=$(date +%s)
echo "=== install25 开始 $T0 ==="
bash stop.sh 2>&1 | tail -3
sleep 2
echo "--- 停后自检: 持卡 fd ---"
n=0; for d in /proc/[0-9]*/fd/*; do t=$(readlink "$d" 2>/dev/null); case "$t" in /dev/xpu*) echo "  HELD $d -> $t"; n=$((n+1));; esac; done
echo "  持卡 fd = $n ; serve2 进程 = $(pgrep -cf 'serve2.py')"
if [ "$n" != "0" ]; then echo "★ 卡没真空, 中止"; exit 1; fi
echo "--- 备份 ---"
cp -p /home/caden/orn_engine/orn3 /home/caden/orn_engine/orn3.pre25.bak
cp -p /home/caden/ornc/serve2.py /home/caden/ornc/serve2.py.bak.pre25
md5sum /home/caden/orn_engine/orn3.pre25.bak /home/caden/ornc/serve2.py.bak.pre25
echo "--- 安装 ---"
cp /home/caden/orn_engine/orn3.m29 /home/caden/orn_engine/orn3
cp /home/caden/ornc/serve2.m29.py /home/caden/ornc/serve2.py
md5sum /home/caden/orn_engine/orn3 /home/caden/ornc/serve2.py
echo "  期望 orn3 = 974b8164e488cd460f7de0d0f8a23649"
echo "--- 拉起服务 (K200_PROF=1: 让引擎把预填充分桶写进日志) ---"
export K200_PROF=1
bash start.sh 2>&1 | tail -4
T1=$(date '+%F %T'); S1=$(date +%s)
echo "=== install25 结束 $T1  停机时长 = $((S1-S0))s ==="
curl -s --max-time 8 http://127.0.0.1:8090/health; echo
grep -a 'MAXT=\|MAXT 档\|rope 查表' /home/caden/ornc/serve2.log | tail -3
