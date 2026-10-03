#!/bin/bash
# =============================================================================
# cb_install29.sh — 第29轮 (连续批处理) 单窗口上件 / 自动回滚
#   ★ 一窗一脚本, 开-跑-关一体; 目标 ≤85s, 硬上限 (铁律) ≤120s。
#   ★ 失败 (引擎起不来 / /health 不 ready) ⇒ 自动回滚到生产件并重起。
#   ★ 绝不 soft_reset / 绝不 rmmod / 绝不碰 NAS。
#   用法: bash cb_install29.sh            # 上新件
#         bash cb_install29.sh rollback   # 只用回滚路径 (恢复生产件)
# =============================================================================
set -u
ORN=/home/caden/orn_engine
ORN_DEV=/home/caden/ornc
ACC=/home/caden/ornc/accept.post29
mkdir -p "$ACC"
LOG="$ACC/install.log"
NDEV="$ORN/orn3.cb"
SDEV="$ORN_DEV/serve3.py"
PROD="$ORN/orn3"
SPROD="$ORN_DEV/serve2.py"

tb() { date '+%F %T'; }
say() { echo "[$(tb)] $*" | tee -a "$LOG"; }

EXC0=$(grep -ac "Exception in kernel execution" /var/log/kern.log)
RC0=$(cat /proc/xpu/dev0/reset_count /proc/xpu/dev1/reset_count 2>/dev/null | tr '\n' '/')
MD5P=$(md5sum "$PROD" | awk '{print $1}')
MD5S=$(md5sum "$SPROD" | awk '{print $1}')
MD5N=$(md5sum "$NDEV" | awk '{print $1}')
MD5T=$(md5sum "$SDEV" | awk '{print $1}')

say "=== cb_install29 开始 $(tb) === 模式=${1:-install}"
say "开工台账: 异常=$EXC0  reset_count=$RC0"
say "生产件: engine=$MD5P  gateway=$MD5S"
say "新  件: engine=$MD5N  gateway=$MD5T"

rollback() {
  say "!! 回滚到生产件"
  bash "$ORN_DEV/stop.sh" 2>&1 | tail -3 | tee -a "$LOG"
  if [ -f "$ORN/orn3.pre29.bak" ]; then cp -f "$ORN/orn3.pre29.bak" "$PROD"; fi
  if [ -f "$ORN_DEV/serve2.py.pre29.bak" ]; then cp -f "$ORN_DEV/serve2.py.pre29.bak" "$SPROD"; fi
  bash "$ORN_DEV/start.sh" 2>&1 | tail -4 | tee -a "$LOG"
  for i in $(seq 1 40); do
    sleep 2
    timeout 5 curl -s "http://127.0.0.1:8090/health" 2>/dev/null | grep -q '"ready": true' && { say "回滚后 8090 ready (${i}0s)"; return 0; }
  done
  say "!! 回滚后 8090 仍未 ready —— 需要人工介入"; return 1
}

if [ "${1:-install}" = "rollback" ]; then rollback; exit $?; fi

# ---- 0) 前置检查 (只读) ----
for d in 0 1; do
  st=$(cat /proc/xpu/dev$d/state 2>/dev/null)
  [ "$st" = "RUNNING" ] || { say "拒绝: dev$d state=$st"; exit 2; }
done
[ -x "$NDEV" ] || { say "拒绝: 新引擎 $NDEV 不存在"; exit 2; }
[ -f "$SDEV" ] || { say "拒绝: 新网关 $SDEV 不存在"; exit 2; }
# 除生产服务外不许有别的持卡进程
EXTRA=0
for p in /proc/[0-9]*; do
  pid=${p#/proc/}
  cmd=$(cat "$p/cmdline" 2>/dev/null | tr -d '\0')
  case "$cmd" in *watchdog.py*|*serve[0-9]*.py*|*safe_run.sh\ -n\ orn_serve*|*orn_engine/orn3*) continue ;; esac
  if ls -l "$p/fd" 2>/dev/null | grep -q '/dev/xpu'; then say "  发现别的持卡进程 pid=$pid ${cmd:0:70}"; EXTRA=$((EXTRA+1)); fi
done
[ "$EXTRA" -eq 0 ] || { say "拒绝: 有 $EXTRA 个别的持卡进程"; exit 2; }
say "前置检查通过 (两芯 RUNNING, 无外来持卡进程)"

T_WIN0=$(date +%s)

# ---- 1) 停生产 ----
bash "$ORN_DEV/stop.sh" 2>&1 | tail -4 | tee -a "$LOG"
sleep 1
pgrep -f 'orn_engine/orn3' >/dev/null 2>&1 && { say "拒绝: 引擎进程还在, 不覆盖"; exit 2; }

# ---- 2) 备份 + 换件 ----
cp -a "$PROD"  "$ORN/orn3.pre29.bak"
cp -a "$SPROD" "$ORN_DEV/serve2.py.pre29.bak"
cp -f "$NDEV"  "$PROD"   || { say "cp 引擎失败"; rollback; exit 3; }
cp -f "$SDEV"  "$SPROD"  || { say "cp 网关失败"; rollback; exit 3; }
M1=$(md5sum "$PROD" | awk '{print $1}'); M2=$(md5sum "$SPROD" | awk '{print $1}')
[ "$M1" = "$MD5N" ] && [ "$M2" = "$MD5T" ] || { say "换件后 md5 不符 ($M1/$M2)"; rollback; exit 3; }
say "换件完成并核对 md5 ok"

# ---- 3) 起服务 (含 8091) ----
export K200_SLOTS="${K200_SLOTS:-4}"
export K200_CKPT="${K200_CKPT:-8}"
export K200_PORT2="${K200_PORT2:-8091}"
export K200_SOLO="${K200_SOLO:-1}"
export K200_SOLO2="${K200_SOLO2:-0}"
bash "$ORN_DEV/start.sh" 2>&1 | tail -5 | tee -a "$LOG"
RD=""
for i in $(seq 1 45); do
  sleep 2
  if timeout 5 curl -s "http://127.0.0.1:8090/health" 2>/dev/null | grep -q '"ready": true'; then RD="$((i*2))"; break; fi
done
if [ -z "$RD" ]; then
  say "!! 新件未 ready ⇒ 自动回滚"
  tail -25 "$ORN_DEV/serve2.log" | tee -a "$LOG"
  rollback; exit 3
fi
say "8090 ready (${RD}s 内)"

# ---- 4) 立刻验黄金题 (逐字节) ----
G=$(timeout 120 curl -s -X POST "http://127.0.0.1:8090/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"ornith-1.5-9b-k200","temperature":0,"messages":[{"role":"user","content":"你好"}],"max_tokens":24}' \
    | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null)
EXPECT='你好！有什么我可以帮你的吗？😊'
if [ "$G" = "$EXPECT" ]; then say "★ 黄金题逐字节一致 ✓"; OK=1; else say "!! 黄金题不一致 ✗ got=[$G]"; OK=0; fi
H1=$(timeout 5 curl -s http://127.0.0.1:8091/health 2>/dev/null)
say "8091 /health: ${H1:0:220}"
T_WIN1=$(date +%s)
EXC1=$(grep -ac "Exception in kernel execution" /var/log/kern.log)
RC1=$(cat /proc/xpu/dev0/reset_count /proc/xpu/dev1/reset_count 2>/dev/null | tr '\n' '/')
say "窗口耗时 $((T_WIN1-T_WIN0))s (停生产 → 起服务 → 黄金题, 硬上限 120s)"
say "收工台账: 异常=$EXC1 (起 $EXC0)  reset_count=$RC1 (起 $RC0)"
if [ "$OK" != "1" ]; then say "!! 黄金题不过 ⇒ 回滚"; rollback; exit 4; fi
say "=== cb_install29 完成 (新件在线) $(tb) ==="
exit 0
