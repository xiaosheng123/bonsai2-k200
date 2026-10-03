#!/bin/bash
# =============================================================================
# atn_probe_window.sh — ATN_ON_CARD 阶段①【极小尺寸探针】单窗口执行器
#
#   做四件事, 顺序固定: 抢锁 -> 证明卡真空 -> 停服务 -> 跑探针 -> 起服务 -> 复验台账 -> 释放锁
#
#   ★ 本脚本【不替换任何生产件】: 探针是一个独立二进制 atn_probe, 不加载权重,
#     所以窗口内的"回滚"就是"把服务起回来", 无需 .bak。
#   ★ 停机目标 <=60s, 硬上限 120s (用户铁律; 历史上有一次 416s 违规被当场 500)。
#   ★ 入口守卫 (比 det_install30.sh 更严, 因为今天已有并发上机废卡的先例):
#       1) .cardlock 存在       => 拒绝
#       2) 检测到别的代理的作业 => 拒绝 (acc*.py / verify*.py / det_vis*.py / longtest / mtsession / probe / pfbench / visbench)
#       3) 持卡 fd 数 > 2       => 拒绝 (期望 1 = 只有生产引擎)
#       4) 两芯 state 非 RUNNING => 拒绝
#   ★ 危险用例 (P7d 全掩行 / P9-raw 生除 / P5b max_a=0) 必须显式 --danger 才跑,
#     且永远放在【另一个窗口】; 它们可能触发 FP_DIV0 掉 session。
#
#   用法:
#     bash atn_probe_window.sh --dry            # 只打印计划, 不动作 (语法+逻辑干跑)
#     bash atn_probe_window.sh                  # 跑 safe 用例
#     bash atn_probe_window.sh --danger         # 跑危险用例 (单独窗口, 后果自负前先确认看门狗在跑)
#     bash atn_probe_window.sh --selftest       # 只跑 safe_run.sh --selftest (完全不碰卡)
#
#   退出码: 0=窗口全过  2=前置拒绝  3=卡异常(失败,严禁重试)  4=探针有用例失败  5=窗口超时
# =============================================================================
set -u
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
DD=/home/caden/orn_engine
OO=/home/caden/ornc
PROBE=$DD/atn_probe
JSON=$OO/atn_probe.json
DANGER=0
DRY=0
SELFTEST=0
for a in "$@"; do
  case "$a" in
    --danger) DANGER=1 ;;
    --dry)    DRY=1 ;;
    --selftest) SELFTEST=1 ;;
    *) echo "未知参数: $a"; exit 2 ;;
  esac
done

log(){ echo "[atnwin $(date +%H:%M:%S)] $*"; }
die(){ echo "[atnwin ★拒绝] $*"; exit 2; }

# ------------------------------- --selftest ---------------------------------
if [ "$SELFTEST" = 1 ]; then
  log "自检模式: 只跑 safe_run.sh --selftest (验证安全网闭环, 完全不碰卡)"
  bash $DD/safe_run.sh --selftest
  exit $?
fi

# ------------------------------- --dry --------------------------------------
if [ "$DRY" = 1 ]; then
  cat <<EOF
[atnwin --dry] 计划 (不执行任何动作):
  0) 抢锁:            mkdir $OO/.cardlock
  1) 守卫:            .cardlock 不存在 / 无其它代理作业 / 持卡fd<=2 / 两芯 RUNNING
  2) 台账基线:        grep 异常 / reset_count / state
  3) 停服务:          bash $OO/stop.sh   (含卡真空自检)
  4) 跑探针:          bash $DD/safe_run.sh -n atnprobe -t 120 -- \\
                        $PROBE $( [ "$DANGER" = 1 ] && echo danger || echo safe ) $JSON
  5) 起服务:          bash $OO/start.sh  && curl /health
  6) 复验台账:        异常 18->18 / reset 0/0->0/0 / state RUNNING / NAS ping
  7) 释放锁 + 报窗口秒数 (目标<=60s 硬上限120s)
  danger 模式 = $DANGER
EOF
  exit 0
fi

# --------------------------- 0) 抢卡锁 + 入口守卫 -----------------------------
[ -d $OO/.cardlock ] && die "卡锁被占 ($OO/.cardlock) => 有别的代理在跑, 停手"
mkdir $OO/.cardlock || die "抢锁失败"
lock_release(){ rmdir $OO/.cardlock 2>/dev/null; }
trap 'lock_release' EXIT

OTHERS="$(pgrep -af 'acc[0-9]|verify[0-9]|det_vis|longtest|mtsession|pfbench|visbench|probe21|/atn_probe' 2>/dev/null | grep -v 'atn_probe_window' | grep -v pgrep)"
if [ -n "$OTHERS" ]; then
  log "★ 检测到【别的代理】的作业, 拒绝开窗口:"
  echo "$OTHERS"
  die "并发上机会废卡 => 等对方结束再来"
fi
log "入口守卫: 未发现别的代理作业 ✓"

HOLD=0
for d in /proc/[0-9]*/fd/*; do t=$(readlink "$d" 2>/dev/null); case "$t" in /dev/xpu*) HOLD=$((HOLD+1));; esac; done
log "持卡 fd 数=$HOLD (期望 1 = 只有生产引擎)"
[ "$HOLD" -gt 2 ] && die "卡上有别的作业 (fd=$HOLD)"
for d in 0 1; do
  st=$(cat /proc/xpu/dev$d/state 2>/dev/null)
  log "  dev$d state=$st"
  [ "$st" = "RUNNING" ] || die "dev$d 不是 RUNNING"
done
[ -x "$PROBE" ] || die "缺探针二进制 $PROBE (先跑 build_atn_probe.sh)"

AN0=$(grep -ac "Exception in kernel execution" /var/log/kern.log)
R0="$(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count)"
log "台账基线: 异常=$AN0 (期望 18)  reset=$R0 (期望 0/0)"

# --------------------- 1) 临时注掉 svc_guard (否则它会救活服务) ---------------
CRON_ORIG=$OO/.crontab.atnwin.orig
crontab -l > $CRON_ORIG 2>/dev/null
sed 's#^\(\*/5 .*svc_guard.sh\)#\#ATNWIN \1#' $CRON_ORIG > $OO/.crontab.atnwin
crontab $OO/.crontab.atnwin 2>/dev/null
restore_cron(){ crontab $CRON_ORIG 2>/dev/null; }
log "crontab 行数=$(crontab -l | wc -l) (svc_guard 临时注掉, 收尾恢复)"

# ------------------------------- 2) 停服务 ----------------------------------
T0=$(date +%s)
bash $OO/stop.sh >/dev/null 2>&1
sleep 3
STOP_TS=$(date +%s)
V=0
for d in /proc/[0-9]*/fd/*; do t=$(readlink "$d" 2>/dev/null); case "$t" in /dev/xpu*) log "  ★仍有持卡: $d -> $t"; V=1;; esac; done
[ "$V" = 0 ] && log "✓ 卡真空自检通过 (无进程持 /dev/xpu*)"

# ------------------------------- 3) 跑探针 ----------------------------------
MODE=$( [ "$DANGER" = 1 ] && echo danger || echo safe )
log "跑探针: mode=$MODE"
if [ "$DANGER" = 1 ]; then
  bash $DD/safe_run.sh -n atnprobe -t 120 -- env ATN_DANGER=1 $PROBE $MODE $JSON
else
  bash $DD/safe_run.sh -n atnprobe -t 120 -- $PROBE $MODE $JSON
fi
RC=$?
log "safe_run 退出码=$RC  (0=通过 / 3=卡异常⇒停手 / 4=探针有用例失败)"

# --------------------- 4) 无论结果如何, 先把服务起回来 -----------------------
bash $OO/start.sh >/dev/null 2>&1
sleep 2
restore_cron
READY_TS=$(date +%s)
W=$((READY_TS - T0))
STOPW=$((READY_TS - STOP_TS))
log "停机窗口(停→ready)=${STOPW}s  总窗口=${W}s  (目标<=60s, 硬上限120s)"
[ "$W" -gt 120 ] && log "★★ 窗口超硬上限 120s ⇒ 记账并检讨"
curl -s -m 8 http://127.0.0.1:8090/health; echo

# ------------------------------- 5) 复验台账 ---------------------------------
AN1=$(grep -ac "Exception in kernel execution" /var/log/kern.log)
R1="$(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count)"
S0=$(cat /proc/xpu/dev0/state 2>/dev/null); S1=$(cat /proc/xpu/dev1/state 2>/dev/null)
PING=$(ping -c 2 -W 2 192.168.66.2 >/dev/null 2>&1 && echo OK || echo FAIL)
log "复验: 异常=$AN1 (基线 $AN0)  reset=$R1 (基线 $R0)  两芯=$S0/$S1  NAS ping=$PING"
FAIL=0
[ "$AN1" != "$AN0" ] && { log "★ 异常计数变了 ⇒ 失败"; FAIL=1; }
[ "$R1" != "$R0" ] && { log "★ reset_count 变了 ⇒ 失败"; FAIL=1; }
[ "$S0" = "RUNNING" ] && [ "$S1" = "RUNNING" ] || { log "★ 两芯不是 RUNNING ⇒ 失败"; FAIL=1; }
[ "$PING" = "OK" ] || { log "★ NAS ping 不通 ⇒ 失败"; FAIL=1; }
if [ -f "$JSON" ]; then log "探针 JSON: $JSON"; fi

# ------------------------------- 6) 结论 -------------------------------------
lock_release
if [ "$RC" = 3 ]; then log "→ 结论: 卡异常, 本阶段【失败】, 严禁重试"; exit 3; fi
[ "$FAIL" = 1 ] && { log "→ 结论: 台账退化, 失败"; exit 4; }
[ "$RC" != 0 ] && { log "→ 结论: 探针有用例失败 (见 JSON), 本阶段失败"; exit 4; }
log "→ 结论: 窗口全过 ✓  探针结论见 $JSON"
exit 0
