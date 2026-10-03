#!/bin/bash
# off_test.sh <BMIN> [--n N]  —— 离线跑一条 ≥256 token 的长 prompt, 测 prefill 耗时
#   用法: BMIN=99999 -> 强制逐 token 路径 (旧行为); BMIN=16 -> 批量路径
set -u
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
cd /home/caden/orn_engine || exit 1
BM="${1:-16}"
NT="${2:-16}"
T=$(python3 - <<'PY'
u = "在数字化转型过程中，企业需要重新审视自身的组织结构、数据治理能力与业务流程，"
print("请阅读下面这段材料，然后只用一句话总结它的核心观点。材料如下：" + u * 12 + "（编号 OFF%s）" % "test", end="")
PY
)
echo "[off_test] BMIN=$BM 模板字符数=${#T}"
K200_PROF=1 K200_BMIN="$BM" K200_BMAX=64 K200_TMPL="$T {q}" ./orn3.batch --n "$NT" --gen "总结"
echo "[off_test] rc=$?"
