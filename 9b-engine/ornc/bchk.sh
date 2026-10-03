#!/bin/bash
# bchk.sh —— 批量 vs 逐 token 数值对拍 (逐 T=2/4/8/16/32/64)
set -u
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
cd /home/caden/orn_engine || exit 1
T=$(python3 - <<'PY'
u = "在数字化转型过程中，企业需要重新审视自身的组织结构、数据治理能力与业务流程，"
print("请阅读下面这段材料，然后只用一句话总结它的核心观点。材料如下：" + u * 12 + "（编号 CHK）", end="")
PY
)
echo "[bchk] 模板字符数=${#T}"
K200_BATCHCHK=1 K200_PROF=1 K200_TMPL="$T {q}" ./orn3.batch --n 1 --gen "总结"
echo "[bchk] rc=$?"
