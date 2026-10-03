#!/bin/bash
# regress32.sh — 第32轮 全量回归闸门 (服务态 8090; 文本 + 视觉 + 台账), 一次跑完
set -u
O=/home/caden/ornc/accept.post32
mkdir -p $O
TS(){ date '+%F %T'; }
{
echo "=== regress32 START $(TS) ==="
echo "md5:"; md5sum /home/caden/orn_engine/orn3 /home/caden/ornc/serve2.py
echo "health: $(curl -s -m 8 http://127.0.0.1:8090/health)"
echo "--- 文本: 黄金题 + 长 prompt 5 次 + 6 条验收题 (det_accept26.py) ---"
cd /home/caden/sdnn && nice -n 19 python3 det_accept26.py post32 2>&1 | tail -30
echo "--- 视觉1: 表格图 9/9 (cb_vistab.py) ---"
cd /home/caden/sdnn && nice -n 19 python3 cb_vistab.py 8090 400 2>&1 | tail -12
echo "--- 视觉2: 黄金题 + 表格图 + 形状图 + 换图不复读 (det_myverify.py) ---"
cd /home/caden/sdnn && nice -n 19 python3 det_myverify.py 2>&1 | tail -20
echo "--- 台账 ---"
echo "异常累计 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log)"
echo "reset_count = $(cat /proc/xpu/dev0/reset_count) / $(cat /proc/xpu/dev1/reset_count)"
echo "state = $(cat /proc/xpu/dev0/state) / $(cat /proc/xpu/dev1/state)"
echo "crontab 行数 = $(crontab -l | wc -l)"
echo "NAS ping:"; ping -c 2 -W 2 192.168.66.26 | tail -3
echo "=== regress32 END $(TS) ==="
} 2>&1 | tee $O/regress.log
