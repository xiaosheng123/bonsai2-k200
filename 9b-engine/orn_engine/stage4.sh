#!/bin/bash
# stage4.sh — 安装卡上折回+卡上 softmax 版到 8090, 然后跑完整验收 (三图 + 文本 6 条 + 卡状态)
set -u
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
LOG=/home/caden/orn_engine/stage4.log
: > "$LOG"
say() { echo "$*" | tee -a "$LOG"; }

say "=== stage4 $(date '+%F %T') 安装 ==="
cp -a vis.cpp  vis.cpp.preCARD2.bak
cp -a orn3     orn3.preCARD.bak
cp -a vistest  vistest.preCARD.bak
cp -a /home/caden/ornc/serve2.py /home/caden/ornc/serve2.py.preCARD.bak
md5sum vis.cpp.preCARD2.bak orn3.preCARD.bak >>"$LOG"

bash /home/caden/ornc/stop.sh >>"$LOG" 2>&1
cp -a vis.cpp.cards vis.cpp
bash buildorn3v.sh >>"$LOG" 2>&1
RC=$?
say "[stage4] build rc=$RC"
if [ $RC -ne 0 ]; then say "[stage4] ★ 编译失败 -> 回滚"; cp -a vis.cpp.preCARD2.bak vis.cpp; cp -a orn3.preCARD.bak orn3; bash /home/caden/ornc/start.sh >>"$LOG" 2>&1; exit 1; fi
md5sum orn3 vis.cpp | tee -a "$LOG"

bash /home/caden/ornc/start.sh >>"$LOG" 2>&1
for i in $(seq 1 60); do
  H=$(curl -s --max-time 5 http://127.0.0.1:8090/health 2>/dev/null)
  case "$H" in *'"ready": true'*) say "[stage4] READY: $H"; break;; esac
  sleep 5
done

# 关掉 svc_guard 抢卡窗口 (验收期间)
crontab -l > /home/caden/ornc/crontab.bak.stage4 2>/dev/null
crontab -l 2>/dev/null | sed 's|^\(\*/5 \* \* \* \* /bin/bash /home/caden/ornc/svc_guard.sh\)|#S4# \1|' | crontab -

O=/home/caden/ornc/accept.cards
mkdir -p $O
M=/home/caden/ornc/mmtests
A=/home/caden/orn_engine/ask_img.py
say "=== 1) 512² 图形/颜色 $(date '+%T') ==="
python3 $A 8090 $M/img_shapes.png '图里有什么？用中文简短回答，说出形状和颜色。' 96 > $O/v1.log 2>&1
tail -6 $O/v1.log | tee -a "$LOG"
say "=== 2) 512² 文字图 $(date '+%T') ==="
python3 $A 8090 $M/img_text.png '图中写的文字是什么？请只回答文字内容。' 64 > $O/v2.log 2>&1
tail -6 $O/v2.log | tee -a "$LOG"
say "=== 3) 768² 表格 \((date '+%T') ==="
python3 $A 8090 $M/big_table.png '这张图是一个表格，请逐行说出每一格的字符。' 320 > $O/v3.log 2>&1
tail -10 $O/v3.log | tee -a "$LOG"
say "=== 4) 文本回归 6 条 $(date '+%T') ==="
python3 /home/caden/ornc/accept6.py 8090 $O/accept6.log 64 > /dev/null 2>&1
grep -a -E "^########## Q|^  ->|^  ✓|^\[|^=== " $O/accept6.log | head -40 | tee -a "$LOG"

crontab /home/caden/ornc/crontab.bak.stage4 && say "[stage4] crontab 已恢复"
say "=== 5) 引擎日志里的编码耗时 ==="
grep -a -E "图像 img|图像编码|prompt=|tok/s" /home/caden/ornc/serve2.log | tail -20 | tee -a "$LOG"

say "=== 6) 卡状态 / 服务 ==="
say "异常累计 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log)"
say "reset_count dev0/1 = $(cat /proc/xpu/dev0/reset_count) / $(cat /proc/xpu/dev1/reset_count)"
say "state dev0/1 = $(cat /proc/xpu/dev0/state) / $(cat /proc/xpu/dev1/state)"
say "health = $(curl -s --max-time 6 http://127.0.0.1:8090/health)"
say "=== stage4 done $(date '+%F %T') ==="
