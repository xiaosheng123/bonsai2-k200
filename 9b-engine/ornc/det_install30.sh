#!/bin/bash
# =============================================================================
# det_install30.sh — 第30轮【单窗口上线】(停机目标 ≤120s; 失败自动回滚)
#   上件: 引擎 /home/caden/orn_engine/orn3.exp (aea804…) —— 已修: ①保尾砍头 ④换回生产视觉塔(vis23.o) ③解码速率台账
#         网关 /home/caden/ornc/serve2.py.new30 (新: ⑤工具调用 ①裁剪/空回答/finish_reason ②⑧syscall)
#   备份: orn3.pre30.bak / serve2.pre30.bak; 失败自动回滚这两份并重新起服务
# =============================================================================
set -u
DD=/home/caden/orn_engine
OO=/home/caden/ornc
NEWENG=$DD/orn3.exp
NEWENG_MD5=aea804aa93f9dc4f697658b9d44cb504
OLDENG_MD5=00b0d1a75167ee07cc7b2f7131bab758
T0=$(date +%s)
log(){ echo "[inst30 $(date +%H:%M:%S)] $*"; }
rollback(){
  log "★ 回滚中"
  bash $OO/stop.sh >/dev/null 2>&1; sleep 2
  cp -a $DD/orn3.pre30.bak $DD/orn3
  cp -f $OO/serve2.pre30.bak $OO/serve2.py
  crontab $OO/.crontab.orig30 2>/dev/null
  bash $OO/start.sh
  log "回滚后 orn3=$(md5sum $DD/orn3 | cut -d' ' -f1) 网关=$(md5sum $OO/serve2.py | cut -d' ' -f1)"
  rmdir $OO/.cardlock 2>/dev/null
  exit 4
}

# 0) 抢卡锁 + 证明卡上没有别的作业
if [ -d $OO/.cardlock ]; then echo "FATAL: 卡锁被占 ($OO/.cardlock)"; exit 9; fi
mkdir $OO/.cardlock || { echo "FATAL: 抢锁失败"; exit 9; }
HOLD=0
for d in /proc/[0-9]*/fd/*; do t=$(readlink "$d" 2>/dev/null); case "$t" in /dev/xpu*) HOLD=$((HOLD+1));; esac; done
log "持卡 fd 数=$HOLD (期望 1 = 只有生产引擎)"
[ "$HOLD" -gt 2 ] && { log "FATAL: 卡上有别的作业"; rmdir $OO/.cardlock; exit 9; }
for f in "$NEWENG" "$OO/serve2.py.new30"; do
  [ -f "$f" ] || { log "FATAL: 缺件 $f"; rmdir $OO/.cardlock; exit 2; }
done
[ "$(md5sum $NEWENG | cut -d' ' -f1)" = "$NEWENG_MD5" ] || { log "FATAL: 新引擎 md5 不符"; rmdir $OO/.cardlock; exit 2; }
log "新引擎 md5 OK; 新网关 md5=$(md5sum $OO/serve2.py.new30 | cut -d' ' -f1)"
for d in 0 1; do st=$(cat /proc/xpu/dev$d/state 2>/dev/null); log "dev$d state=$st"; [ "$st" = "RUNNING" ] || { log "FATAL: dev$d 不是 RUNNING"; rmdir $OO/.cardlock; exit 2; }; done

# 1) 备份 + 临时关掉 svc_guard (收尾必恢复); 记录基线台账
crontab -l > $OO/.crontab.orig30
sed 's#^\(\*/5 .*svc_guard.sh\)#\#TMP30 \1#' $OO/.crontab.orig30 > $OO/.crontab.pre30
crontab $OO/.crontab.pre30
log "crontab 行数=$(crontab -l | wc -l) (svc_guard 临时注掉; 收尾恢复)"
cp -a $DD/orn3 $DD/orn3.pre30.bak
cp -f $OO/serve2.py $OO/serve2.pre30.bak
log "备份: orn3.pre30.bak=$(md5sum $DD/orn3.pre30.bak|cut -d' ' -f1) serve2.pre30.bak=$(md5sum $OO/serve2.pre30.bak|cut -d' ' -f1)"
AN0=$(grep -ac "Exception in kernel execution" /var/log/kern.log)
R0="$(cat /proc/xpu/dev0/reset_count 2>/dev/null)/$(cat /proc/xpu/dev1/reset_count 2>/dev/null)"
log "台账基线: 异常=$AN0 reset=$R0"

# 2) 停
bash $OO/stop.sh
sleep 3
if pgrep -f "[s]erve2.py" >/dev/null 2>&1 || pgrep -f "[o]rn3 --n" >/dev/null 2>&1; then
  log "有残留, 强清"; pkill -f "[s]erve2.py"; pkill -f "[o]rn3 --n"; sleep 3
fi
STOP_TS=$(date +%s)
log "停机完成; 卡真空自检:"
V=0; for d in /proc/[0-9]*/fd/*; do t=$(readlink "$d" 2>/dev/null); case "$t" in /dev/xpu*) log "  ★ 仍有持卡: $d -> $t"; V=1;; esac; done
[ "$V" = "0" ] && log "  ✓ 无进程持 /dev/xpu*"

# 3) 换件
cp -f $NEWENG $DD/orn3 || rollback
cp -f $OO/serve2.py.new30 $OO/serve2.py || rollback
log "换件后: orn3=$(md5sum $DD/orn3|cut -d' ' -f1) 网关=$(md5sum $OO/serve2.py|cut -d' ' -f1)"

# 4) 起 + 等 ready
bash $OO/start.sh || rollback
W=$(($(date +%s) - T0))
log "READY; 停机窗口(停→ready) = $(( $(date +%s) - STOP_TS ))s; 总窗口 = ${W}s (铁律 ≤120s)"
curl -s -m 5 http://127.0.0.1:8090/health; echo

# 5) 窗口内冒烟: 黄金题 (逐字节)
G=$(curl -s -m 180 http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":32}')
echo "SMOKE_GOLDEN=$G"
if echo "$G" | grep -q '你好！有什么我可以帮你的吗'; then
  log "✓ 窗口内黄金题通过"
else
  log "✗ 窗口内黄金题失败 ⇒ 回滚"; rollback
fi

# 6) 收尾: 恢复 crontab
crontab $OO/.crontab.orig30
log "crontab 恢复后行数=$(crontab -l|wc -l) (必须 = 3)"
crontab -l
rmdir $OO/.cardlock
log "INSTALL30_OK  orn3=$(md5sum $DD/orn3|cut -d' ' -f1)  gateway=$(md5sum $OO/serve2.py|cut -d' ' -f1)"
