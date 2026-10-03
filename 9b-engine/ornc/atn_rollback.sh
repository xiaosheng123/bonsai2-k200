#!/bin/bash
# =============================================================================
# atn_rollback.sh — ATN_ON_CARD 通用回滚器 (阶段②~⑥ 每个阶段复制一份, 只改 N)
#
#   幂等: 任何时候可跑, 跑完生产件回到"本阶段之前"的状态。
#   ★ 备份必须含 md5 记录, 验收报告里逐条列出"本阶段回滚点 = md5"。
#   ★ 回滚脚本本身也要包窗口纪律: 先停 → 换回 → 起 → 验健康。
#
#   用法: bash atn_rollback.sh <N> [--dry]
#     例: bash atn_rollback.sh 2      # 回到阶段②之前 (orn3.preATN2.bak / serve2.preATN2.bak)
# =============================================================================
set -u
DD=/home/caden/orn_engine
OO=/home/caden/ornc
N="${1:-}"
DRY="${2:-}"
[ -z "$N" ] && { echo "用法: bash atn_rollback.sh <N> [--dry]"; exit 2; }

ENGBACK=$DD/orn3.preATN$N.bak
GWBAK=$OO/serve2.preATN$N.bak
MD5F=$OO/.md5.preATN$N

log(){ echo "[rollback$N $(date +%H:%M:%S)] $*"; }

if [ "$DRY" = "--dry" ]; then
  cat <<EOF
[rollback$N --dry] 计划:
  0) 抢锁 mkdir $OO/.cardlock
  1) 校验备份存在: $ENGBACK  $GWBAK   且 $MD5F 存在
  2) 停服务 bash $OO/stop.sh
  3) cp -a $ENGBACK -> $DD/orn3 ;  cp -f $GWBAK -> $OO/serve2.py
  4) 起服务 bash $OO/start.sh ; curl /health
  5) 打印回滚后 orn3/serve2 md5 与健康状态
  6) 释放锁
EOF
  exit 0
fi

[ -d $OO/.cardlock ] && { log "卡锁被占 ⇒ 拒绝"; exit 9; }
mkdir $OO/.cardlock || { log "抢锁失败"; exit 9; }
trap 'rmdir $OO/.cardlock 2>/dev/null' EXIT

[ -f "$ENGBACK" ] || { log "FATAL: 缺 $ENGBACK"; exit 2; }
[ -f "$GWBAK" ]   || { log "FATAL: 缺 $GWBAK";   exit 2; }
if [ -f "$MD5F" ]; then log "回滚点记录: $(tr '\n' ' ' < $MD5F)"; else log "★ 警告: 缺 $MD5F (验收报告要求逐条列 md5)"; fi

log "停服务"
bash $OO/stop.sh >/dev/null 2>&1
sleep 3
cp -a "$ENGBACK" $DD/orn3
cp -f "$GWBAK"   $OO/serve2.py
log "已换回: orn3=$(md5sum $DD/orn3 | cut -d' ' -f1)  网关=$(md5sum $OO/serve2.py | cut -d' ' -f1)"
bash $OO/start.sh
sleep 2
curl -s -m 8 http://127.0.0.1:8090/health; echo
log "台账: 异常=$(grep -ac 'Exception in kernel execution' /var/log/kern.log) reset=$(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count) 两芯=$(cat /proc/xpu/dev0/state)/$(cat /proc/xpu/dev1/state)"
log "回滚完成"
