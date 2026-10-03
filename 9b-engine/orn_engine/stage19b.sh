#!/bin/bash
# stage19b.sh — 卡窗口: 修好的 cardA 自检 -> 复测 768²/512² -> 安装到 8090 -> 三图+文本验收
set -u
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
LOG=/home/caden/orn_engine/stage19b.log
: > "$LOG"
MM=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
mkdir -p /tmp/vsw/n1b/o /tmp/vsw/n7b/o /tmp/vsw/n7b/old
say() { echo "$*" | tee -a "$LOG"; }

bash window_open.sh "stage19b-install" >>"$LOG" 2>&1

run() {
  local nm="$1"; local tt="$2"; shift 2
  echo "########## $nm $(date '+%T') ##########" | tee -a "$LOG"
  bash safe_run.sh -n "$nm" -t "$tt" -- "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "[stage19b] $nm safe_run rc=$rc" | tee -a "$LOG"
  if [ "$rc" -eq 3 ]; then echo "CARD_EXCEPTION" >>"$LOG"; return 3; fi
  if [ "$rc" -eq 2 ]; then echo "PRECHECK_REFUSE" >>"$LOG"; return 2; fi
  return 0
}
abort() { bash window_close.sh >>"$LOG" 2>&1; echo ABORT; exit 3; }

# 1) cardA 自检 (A>=0 生产路径)
run caself2  150 ./vistest.cards cardAtest || true
grep -q CARD_EXCEPTION "$LOG" && abort

# 2) 复测 (第二次数据点)
run n768card2 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_PROF=1 OMP_NUM_THREADS=4 \
    ./vistest.cards run "$MM" /tmp/inp768.f32 768 768 /tmp/vsw/n7b/o || true
grep -q CARD_EXCEPTION "$LOG" && abort
run n512card2 600 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_PROF=1 OMP_NUM_THREADS=4 \
    ./vistest.cards run "$MM" /tmp/inp512.f32 512 512 /tmp/vsw/n1b/o || true
grep -q CARD_EXCEPTION "$LOG" && abort
echo "=== cmp768b ===" >>"$LOG"; python3 /home/caden/orn_engine/vscmp2.py /tmp/vsw/n7b 768 o >>"$LOG" 2>&1
echo "=== cmp512b ===" >>"$LOG"; python3 /home/caden/orn_engine/vscmp2.py /tmp/vsw/n1b 512 o >>"$LOG" 2>&1

# 3) 安装到 8090
say "=== 安装 orn3.new19 -> orn3 ($(date '+%T')) ==="
cp -a orn3 orn3.pre19b.bak
cp -a orn3.new19 orn3
cp -a vis.cpp.cards vis.cpp
md5sum orn3 vis.cpp | tee -a "$LOG"
bash /home/caden/ornc/start.sh >>"$LOG" 2>&1
for i in $(seq 1 40); do
  H=$(curl -s --max-time 5 http://127.0.0.1:8090/health 2>/dev/null)
  case "$H" in *'"ready": true'*) say "[stage19b] READY: $H"; break;; esac
  sleep 5
done

# 4) 三图 + 文本 6 条 (验收期间 crontab 已由 window_open 注掉 svc_guard)
O=/home/caden/ornc/accept.post19
mkdir -p $O
M=/home/caden/ornc/mmtests
A=/home/caden/orn_engine/ask_img.py
say "=== 1) 512² 图形/颜色 $(date '+%T') ==="
python3 $A 8090 $M/img_shapes.png '图里有什么？用中文简短回答，说出形状和颜色。' 96 > $O/v1.log 2>&1
tail -6 $O/v1.log | tee -a "$LOG"
say "=== 2) 512² 文字图 $(date '+%T') ==="
python3 $A 8090 $M/img_text.png '图中写的文字是什么？请只回答文字内容。' 64 > $O/v2.log 2>&1
tail -6 $O/v2.log | tee -a "$LOG"
say "=== 3) 768² 表格 $(date '+%T') ==="
python3 $A 8090 $M/big_table.png '这张图是一个表格，请逐行说出每一格的字符。' 320 > $O/v3.log 2>&1
tail -10 $O/v3.log | tee -a "$LOG"
say "=== 4) 文本 6 条 $(date '+%T') ==="
python3 /home/caden/ornc/accept6.py 8090 $O/accept6.log 64 > /dev/null 2>&1
grep -a -E "^########## Q|^回复原文|^\[[0-9]" $O/accept6.log | tee -a "$LOG"

say "=== 5) 引擎日志里的编码耗时 ==="
grep -a -E "图像编码|编码合计|tok/s" /home/caden/ornc/serve2.log | tail -16 | tee -a "$LOG"

say "=== 6) 卡状态 / 服务 ==="
say "异常累计 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log)"
say "reset_count dev0/1 = $(cat /proc/xpu/dev0/reset_count) / $(cat /proc/xpu/dev1/reset_count)"
say "state dev0/1 = $(cat /proc/xpu/dev0/state) / $(cat /proc/xpu/dev1/state)"
say "health = $(curl -s --max-time 6 http://127.0.0.1:8090/health)"

bash window_close.sh >>"$LOG" 2>&1
say "=== stage19b done $(date '+%F %T') ==="
