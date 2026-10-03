#!/bin/bash
# vprof.sh <name> <inp> <W> <H> <CH/CHA> ...  —— 带剖析的档位测量
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
cd /home/caden/orn_engine || exit 1
MM=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
NM=$1; INP=$2; W=$3; H=$4; shift 4
echo "=== vprof $NM ${W}x${H} ===  $(date '+%F %T')"
for cfg in "$@"; do
  CH=${cfg%%/*}; CHA=${cfg##*/}
  DD=/tmp/vsw/$NM/${CH}_${CHA}
  mkdir -p "$DD"
  LG=/tmp/vsw/$NM/${CH}_${CHA}.log
  echo "### $NM CH=$CH CHA=$CHA  start $(date +%T)"
  env VIS_CH=$CH VIS_CHA=$CHA VIS_PROF=1 OMP_NUM_THREADS=4 \
      bash safe_run.sh -n vp_${NM}_${CH}_${CHA} -t 900 -- \
      ./vistest run $MM $INP $W $H "$DD" > "$LG" 2>&1
  echo "    rc=$?  $(date +%T)"
  grep -aE "完成:|PROF|FATAL|判定" "$LG" | tail -4
done
echo "=== vprof $NM done $(date '+%F %T') ==="
