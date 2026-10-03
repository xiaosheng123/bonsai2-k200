#!/bin/bash
# probe_lm.sh — 按用户收紧后的硬规矩做一次 LM 容量探测
#   1) 探测期间服务必须停掉 (调用方负责 recycle2.sh, 或本脚本用 -s 自己停)
#   2) 记录 kern.log 字节偏移 -> 过 safe_run -> 立刻比对异常增量
#   3) 立刻验证【宿主 NAS 还活着】: ping 192.168.66.26 + ssh 一次
#   4) 任何一步异常 => 立即停下, 并把服务恢复可用
# 用法: bash probe_lm.sh <LBUF>          (只探这一个值)
set -uo pipefail
LBUF="$1"
BASE=/home/caden/orn_engine/lmp
X=/home/caden/xtdk/xtdk-x86_64
RT=/usr/local/xpu-4.33.0
export LD_LIBRARY_PATH=$RT/lib64:$X/shlib
LOG=/home/caden/orn_engine/lmprobe_history.log
say() { echo "$*" | tee -a "$LOG"; }

OFF_BEFORE=$(stat -c %s /var/log/kern.log 2>/dev/null || echo 0)
say "=============================================================="
say "[$(date '+%F %T')] 探测 LBUF=$LBUF (<<<1,16>>> 单簇)  kern.log 偏移(前)=$OFF_BEFORE"

# 编译这一个变体
cd "$BASE" || exit 1
$X/bin/clang -I$RT/include -DLBUF=$LBUF -std=c++11 -O2 -fno-builtin -o lmp_$LBUF.sec lmp.xpu --xpu-device-only -c > /tmp/lmp_dev_$LBUF.log 2>&1 \
  || { say "  编译(device)失败:"; tail -5 /tmp/lmp_dev_$LBUF.log; exit 1; }
$X/bin/xpu-elfconv lmp_$LBUF.sec lmp_$LBUF.proxy.o $X/bin/clang > /tmp/lmp_elf_$LBUF.log 2>&1 \
  || { say "  elfconv 失败:"; tail -5 /tmp/lmp_elf_$LBUF.log; exit 1; }
$X/bin/clang -I$RT/include -DLBUF=$LBUF -std=c++11 -o lmp_${LBUF}_host.o lmp.xpu --xpu-host-only -fPIC -c > /tmp/lmp_host_$LBUF.log 2>&1 \
  || { say "  编译(host stub)失败:"; tail -5 /tmp/lmp_host_$LBUF.log; exit 1; }
g++ -std=c++11 -O2 -DLBUF=$LBUF -I$RT/include -c lmp.cpp -o lmp_main_$LBUF.o > /tmp/lmp_cpp_$LBUF.log 2>&1 \
  || { say "  编译(driver)失败:"; tail -8 /tmp/lmp_cpp_$LBUF.log; exit 1; }
g++ lmp_main_$LBUF.o lmp_$LBUF.proxy.o lmp_${LBUF}_host.o -o lmp_$LBUF -L$RT/lib64 -lxpurt -L$X/shlib -lxpuapi -lm > /tmp/lmp_link_$LBUF.log 2>&1 \
  || { say "  链接失败:"; tail -8 /tmp/lmp_link_$LBUF.log; exit 1; }

# 过 safe_run (安全网武装)
bash /home/caden/orn_engine/safe_run.sh -n lmprobe_$LBUF -t 120 -- ./lmp_$LBUF 0 1 16
RC=$?
OFF_AFTER=$(stat -c %s /var/log/kern.log 2>/dev/null || echo 0)
INC=$(tail -c +$((OFF_BEFORE+1)) /var/log/kern.log 2>/dev/null | grep -c "Exception in kernel execution" || true)
say "  safe_run rc=$RC   异常增量=$INC (kern.log 偏移 $OFF_BEFORE -> $OFF_AFTER)"

# 宿主 NAS 存活检查
say "  --- 宿主存活检查 192.168.66.26 ---"
if ping -c2 -W2 192.168.66.26 >/dev/null 2>&1; then say "  ping: 通 ✓"; else say "  ping: 不通 ✗✗"; fi
SSHOUT=$(timeout 8 ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5 192.168.66.26 'echo HOST_ALIVE; uptime' 2>&1 | head -3)
say "  ssh: $SSHOUT"

# 设备状态 + 结论
say "  dev0=$(cat /proc/xpu/dev0/state 2>/dev/null) dev1=$(cat /proc/xpu/dev1/state 2>/dev/null)"
if [ "$INC" -gt 0 ]; then
  say "  裁决: LBUF=$LBUF => ✗ 超限 (新异常 $INC 条)"
  if [ "$(cat /proc/xpu/dev0/state 2>/dev/null)" != "RUNNING" ]; then
    say "  设备异常, 执行 soft_reset 0 恢复"; $RT/tools/soft_reset 0; sleep 3
    say "  reset 后 dev0=$(cat /proc/xpu/dev0/state 2>/dev/null)"
  fi
  exit 3
fi
say "  裁决: LBUF=$LBUF => ✓ 可用 (0 新异常)"
exit 0
