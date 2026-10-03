#!/bin/bash
# grow.sh <BMIN> <BMAX> <LDUMP路径> —— 离线跑同一条 216-token prompt, dump 逐位置 logits
set -u
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
cd /home/caden/orn_engine || exit 1
BM="${1:-16}"; BMX="${2:-8}"; LD="${3:-/tmp/grow.bin}"
T=$(python3 - <<'PY'
u = "在数字化转型过程中，企业需要重新审视自身的组织结构、数据治理能力与业务流程，"
print("请阅读下面这段材料，然后只用一句话总结它的核心观点。材料如下：" + u * 12 + "（编号 GROW）", end="")
PY
)
rm -f "$LD"
echo "[grow] BMIN=$BM BMAX=$BMX LDUMP=$LD 模板字符数=${#T}"
K200_LDUMP="$LD" K200_PROF=1 K200_BMIN="$BM" K200_BMAX="$BMX" K200_TMPL="$T {q}" ./orn3.batch --n 4 --gen "总结"
echo "[grow] rc=$?"
ls -la "$LD"
