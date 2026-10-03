#!/bin/bash
# vopt2.sh <name> <inp> <W> <H> <CH/CHA> ...  —— 优化后二进制测量 (带剖析), 环境变量外部给
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
cd /home/caden/orn_engine || exit 1
MM=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
NM=$1; INP=$2; W=$3; H=$4; shift 4
TAG="${VIS_TAG:-$NM}"
echo "=== vopt2 $TAG ${W}x${H} ===  $(date '+%F %T')"
for cfg in "$@"; do
  CH=${cfg%%/*}; CHA=${cfg##*/}
  DD=/tmp/vsw/opt/$TAG/${CH}_${CHA}
  rm -rf "$DD"; mkdir -p "$DD"
  LG=/tmp/vsw/opt/$TAG/${CH}_${CHA}.log
  echo "### $TAG CH=$CH CHA=$CHA  start $(date +%T)"
  env VIS_CH=$CH VIS_CHA=$CHA VIS_PROF=1 OMP_NUM_THREADS=4 \
      bash safe_run.sh -n vo_${TAG}_${CH}_${CHA} -t 900 -- \
      ./vistest run $MM $INP $W $H "$DD" > "$LG" 2>&1
  echo "    rc=$?  $(date +%T)"
  grep -aE "完成:|PROF|WARN|FATAL|判定|量化/折回" "$LG" | tail -5
done
echo "=== vopt2 $TAG done $(date '+%F %T') ==="
