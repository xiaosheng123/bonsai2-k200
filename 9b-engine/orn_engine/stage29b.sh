#!/bin/bash
# stage29b.sh — 第24/25轮卡窗口 (续): 编译 orn3.m29 + 新引擎三档对拍
#   前置: 窗口已开 (服务已停, svc_guard 已注掉, 卡真空), 基线 pfb_base.json 已由 pb_base 产出。
#   纪律: 每个占卡作业走 safe_run.sh -n/-t; 一次卡异常立即停手。
set -u
cd /home/caden/orn_engine
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
LOG=/home/caden/orn_engine/stage29b.log
: > "$LOG"
exc() { grep -ac 'Exception in kernel execution' /var/log/kern.log 2>/dev/null; }
rst() { cat /proc/xpu/dev0/reset_count /proc/xpu/dev1/reset_count 2>/dev/null | tr '\n' '/'; }
say() { echo "$*" | tee -a "$LOG"; }
abort() { say "!!! 停手: $* !!!"; bash window_close29.sh >>"$LOG" 2>&1; echo ABORT; exit 3; }
held() { local n=0; for d in /proc/[0-9]*/fd/*; do t=$(readlink "$d" 2>/dev/null); case "$t" in /dev/xpu*) n=$((n+1));; esac; done; echo $n; }
clean() { pkill -f 'orn_engine/orn3 --n 64' 2>/dev/null; pkill -f 'orn_engine/orn3.m29 --n 64' 2>/dev/null; sleep 2
          pkill -9 -f 'orn_engine/orn3 --n 64' 2>/dev/null; pkill -9 -f 'orn_engine/orn3.m29 --n 64' 2>/dev/null; sleep 2; }
run() { local nm="$1"; local tt="$2"; shift 2
  echo "########## $nm $(date '+%T') ##########" | tee -a "$LOG"
  bash safe_run.sh -n "$nm" -t "$tt" -- "$@" >>"$LOG" 2>&1
  local rc=$?; echo "[stage29b] $nm rc=$rc" | tee -a "$LOG"
  clean
  echo "[stage29b] $nm 后: 持卡 fd=$(held) 异常=$(exc) reset=$(rst) state=$(cat /proc/xpu/dev0/state)/$(cat /proc/xpu/dev1/state)" | tee -a "$LOG"
  [ "$(held)" != "0" ] && abort "$nm 结束后仍有进程持卡"
  [ "$(exc)" != "18" ] && abort "$nm 出现卡异常 (计数 $(exc))"
  return 0; }

say "=== stage29b 开始 $(date '+%F %T') | 异常 = $(exc) | reset $(rst) | 持卡 fd = $(held) ==="
[ "$(held)" != "0" ] && abort "开工前就有进程持卡"
[ "$(exc)" != "18" ] && abort "开工前异常不是 18"

say "--- 编译 orn3.m29 (主机侧, 卡空闲) ---"
bash build29.sh >>"$LOG" 2>&1 || abort "编译失败"
ls -la orn3.m29 >>"$LOG" 2>&1

run pb_new1 900 python3 pfbench.py /home/caden/orn_engine/orn3.m29 /home/caden/orn_engine/pfb_new1.json new2048
run pb_new2 900 env K200_ROPETAB=0 python3 pfbench.py /home/caden/orn_engine/orn3.m29 /home/caden/orn_engine/pfb_new2.json new2048_notab
run pb_new3 900 python3 pfbench.py /home/caden/orn_engine/orn3.m29 /home/caden/orn_engine/pfb_new3.json new2048_r2

say "=== stage29b 结束 $(date '+%F %T') | 异常 = $(exc) | reset $(rst) ==="
echo "=== stage29b done (窗口仍开着, 等人工收尾) ==="
