#!/bin/bash
# stage19c.sh — 卡窗口: 位级复现验证 (卡上归一链 vs 主机路径) -> 安装 -> 三图+文本验收
set -u
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
LOG=/home/caden/orn_engine/stage19c.log
: > "$LOG"
MM=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
rm -rf /tmp/vsw/n7c /tmp/vsw/n1c; mkdir -p /tmp/vsw/n7c/new /tmp/vsw/n7c/old /tmp/vsw/n1c/new
say() { echo "$*" | tee -a "$LOG"; }

bash window_open.sh "stage19c-install" >>"$LOG" 2>&1

run() {
  local nm="$1"; local tt="$2"; shift 2
  echo "########## $nm $(date '+%T') ##########" | tee -a "$LOG"
  bash safe_run.sh -n "$nm" -t "$tt" -- "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "[stage19c] $nm safe_run rc=$rc" | tee -a "$LOG"
  if [ "$rc" -eq 3 ]; then echo "CARD_EXCEPTION" >>"$LOG"; return 3; fi
  if [ "$rc" -eq 2 ]; then echo "PRECHECK_REFUSE" >>"$LOG"; return 2; fi
  return 0
}
abort() { bash window_close.sh >>"$LOG" 2>&1; echo ABORT; exit 3; }

run p_reddim0 90 ./probe reddim0 || true
run p_div2d   90 ./probe div2d   || true
grep -q CARD_EXCEPTION "$LOG" && abort

run caself3 150 ./vistest.cards cardAtest || true
grep -q CARD_EXCEPTION "$LOG" && abort

run n768new 900 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_PROF=1 OMP_NUM_THREADS=4 \
    ./vistest.cards run "$MM" /tmp/inp768.f32 768 768 /tmp/vsw/n7c/new || true
grep -q CARD_EXCEPTION "$LOG" && abort
run n768ref 300 env VIS_CARDA=0 VIS_CH=96 VIS_CHA=0 VIS_PROF=1 OMP_NUM_THREADS=4 \
    ./vistest.cards run "$MM" /tmp/inp768.f32 768 768 /tmp/vsw/n7c/old || true
grep -q CARD_EXCEPTION "$LOG" && abort
run n512new 600 env VIS_CARDA=1 VIS_CH=96 VIS_CHA=0 VIS_PROF=1 OMP_NUM_THREADS=4 \
    ./vistest.cards run "$MM" /tmp/inp512.f32 512 512 /tmp/vsw/n1c/new || true
echo "=== 位级对拍 768 新(卡上归一) vs 老(主机归一, 同一二进制) ===" >>"$LOG"
python3 /tmp/cmp19.py /tmp/vsw/n7c/new /tmp/vsw/n7c/old "768 new vs ref" >>"$LOG" 2>&1
echo "=== 位级对拍 512 本轮二进制 vs 上一版老路径落盘 ===" >>"$LOG"
python3 /tmp/cmp19.py /tmp/vsw/n1c/new /tmp/vsw/n1/old "512 new-binary vs pre19-ref" >>"$LOG" 2>&1
echo "=== 512 vs 真值 ===" >>"$LOG"
python3 /home/caden/orn_engine/vscmp2.py /tmp/vsw/n1c 512 new >>"$LOG" 2>&1
echo "=== 768 vs 真值 ===" >>"$LOG"
python3 /home/caden/orn_engine/vscmp2.py /tmp/vsw/n7c 768 new >>"$LOG" 2>&1

say "=== 安装 orn3.new19 -> orn3 ($(date '+%T')) ==="
cp -a orn3 orn3.pre19c.bak
cp -a orn3.new19 orn3
cp -a vis.cpp.cards vis.cpp
md5sum orn3 vis.cpp | tee -a "$LOG"
bash /home/caden/ornc/start.sh >>"$LOG" 2>&1
for i in $(seq 1 40); do
  H=$(curl -s --max-time 5 http://127.0.0.1:8090/health 2>/dev/null)
  case "$H" in *'"ready": true'*) say "[stage19c] READY: $H"; break;; esac
  sleep 5
done

O=/home/caden/ornc/accept.post19c
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
tail -12 $O/v3.log | tee -a "$LOG"
say "=== 4) 文本 6 条 $(date '+%T') ==="
python3 /home/caden/ornc/accept6.py 8090 $O/accept6.log 64 > /dev/null 2>&1
grep -a -E "^########## Q|^回复原文|^\[[0-9]" $O/accept6.log | tee -a "$LOG"
say "=== 5) 引擎日志里的图像编码耗时 ==="
grep -a -E "图像编码合计|tok/s" /home/caden/ornc/serve2.log | tail -14 | tee -a "$LOG"
say "=== 6) 卡状态 / 服务 ==="
say "异常累计 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log)"
say "reset_count dev0/1 = $(cat /proc/xpu/dev0/reset_count) / $(cat /proc/xpu/dev1/reset_count)"
say "state dev0/1 = $(cat /proc/xpu/dev0/state) / $(cat /proc/xpu/dev1/state)"
say "health = $(curl -s --max-time 6 http://127.0.0.1:8090/health)"

bash window_close.sh >>"$LOG" 2>&1
say "=== stage19c done $(date '+%F %T') ==="
