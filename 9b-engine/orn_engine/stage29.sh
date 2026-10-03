#!/bin/bash
# stage29.sh — 第24轮卡窗口: MAXT 1024→2048 与预填充分解/优化的引擎级对拍
#
#   顺序 (每步都走 safe_run.sh -n/-t; 任何一次卡异常立即停手收尾, 绝不重试):
#     1) pb_base : 生产引擎 orn3 (MAXT=1024) 基线
#     2) pb_new1 : orn3.m29 (MAXT=2048, rope 查表开)
#     3) pb_new2 : orn3.m29 + K200_ROPETAB=0  (A/B: 查表是否位级等价)
#     4) pb_new3 : orn3.m29 复跑 (噪声档)
#   每一步前后: 扫持卡 fd / 卡异常计数 / reset_count; 不合规立刻 abort。
set -u
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
LOG=/home/caden/orn_engine/stage29.log
: > "$LOG"
exc() { grep -ac 'Exception in kernel execution' /var/log/kern.log 2>/dev/null; }
rst() { cat /proc/xpu/dev0/reset_count /proc/xpu/dev1/reset_count 2>/dev/null | tr '\n' '/'; }
say() { echo "$*" | tee -a "$LOG"; }
abort() { say "!!! 停手: $* !!!"; bash window_close29.sh >>"$LOG" 2>&1; echo ABORT; exit 3; }
held() { local n=0; for d in /proc/[0-9]*/fd/*; do t=$(readlink "$d" 2>/dev/null); case "$t" in /dev/xpu*) n=$((n+1));; esac; done; echo $n; }
clean() { pkill -f 'orn_engine/orn3 --n 64' 2>/dev/null; pkill -f 'orn_engine/orn3.m29 --n 64' 2>/dev/null; sleep 2;
          pkill -9 -f 'orn_engine/orn3 --n 64' 2>/dev/null; pkill -9 -f 'orn_engine/orn3.m29 --n 64' 2>/dev/null; sleep 1; }
run() { local nm="$1"; local tt="$2"; shift 2
  echo "########## $nm $(date '+%T') ##########" | tee -a "$LOG"
  bash safe_run.sh -n "$nm" -t "$tt" -- "$@" >>"$LOG" 2>&1
  local rc=$?; echo "[stage29] $nm rc=$rc" | tee -a "$LOG"
  clean
  echo "[stage29] $nm 后: 持卡 fd=$(held) 异常=$(exc) reset=$(rst) state=$(cat /proc/xpu/dev0/state)/$(cat /proc/xpu/dev1/state)" | tee -a "$LOG"
  [ "$(held)" != "0" ] && abort "$nm 结束后仍有进程持卡"
  [ "$(exc)" != "18" ] && abort "$nm 出现卡异常 (计数 $(exc))"
  return 0; }

say "=== stage29 开始 $(date '+%F %T') | 异常起 = $(exc) | reset $(rst) | state $(cat /proc/xpu/dev0/state)/$(cat /proc/xpu/dev1/state) ==="
if [ "$(exc)" != "18" ]; then say "★ 异常基线不是 18 (是 $(exc)) => 停手"; exit 9; fi

bash window_open29.sh "stage29-maxt2048" >>"$LOG" 2>&1
say "停服务后: 持卡 fd = $(held) (必须 0)"

# 主机内存采样 (峰值 RSS): 每 2s 记一次 max
( mx=0; while :; do
    for p in $(pgrep -f 'orn_engine/orn3' 2>/dev/null); do
      r=$(awk '/VmHWM/{print $2}' /proc/$p/status 2>/dev/null || echo 0); [ -n "$r" ] && [ "$r" -gt "$mx" ] 2>/dev/null && mx=$r
    done
    echo "hostmem_max_kB=$mx free=$(free -m | awk '/Mem:/{print $3"/"$2}')" > /home/caden/orn_engine/stage29.mem
    sleep 2
  done ) & MEMPID=$!

run pb_base 1500 python3 pfbench.py /home/caden/orn_engine/orn3     /home/caden/orn_engine/pfb_base.json base1024
run pb_new1 1500 python3 pfbench.py /home/caden/orn_engine/orn3.m29 /home/caden/orn_engine/pfb_new1.json new2048
run pb_new2 1500 env K200_ROPETAB=0 python3 pfbench.py /home/caden/orn_engine/orn3.m29 /home/caden/orn_engine/pfb_new2.json new2048_notab
run pb_new3 1500 python3 pfbench.py /home/caden/orn_engine/orn3.m29 /home/caden/orn_engine/pfb_new3.json new2048_r2

kill $MEMPID 2>/dev/null
say "主机内存峰值: $(cat /home/caden/orn_engine/stage29.mem 2>/dev/null)"
say "=== stage29 结束 $(date '+%F %T') | 异常 = $(exc) | reset $(rst) ==="
bash window_close29.sh >>"$LOG" 2>&1
echo "=== stage29 done ==="
