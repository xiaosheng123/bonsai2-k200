#!/bin/bash
# vsweep.sh <name> <inp> <W> <H> <CH/CHA> ...   —— CH/CHA 扫档: 时间 + dump
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
cd /home/caden/orn_engine || exit 1
MM=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
NM=$1; INP=$2; W=$3; H=$4; shift 4
echo "=== vsweep $NM ${W}x${H} ===  $(date '+%F %T')  nproc=$(nproc)"
for cfg in "$@"; do
  CH=${cfg%%/*}; CHA=${cfg##*/}
  DD=/tmp/vsw/$NM/${CH}_${CHA}
  rm -rf "$DD"; mkdir -p "$DD"
  LG=/tmp/vsw/$NM/${CH}_${CHA}.log
  echo "### $NM CH=$CH CHA=$CHA  start $(date +%T)"
  env VIS_CH=$CH VIS_CHA=$CHA OMP_NUM_THREADS=4 \
      bash safe_run.sh -n vsw_${NM}_${CH}_${CHA} -t 900 -- \
      ./vistest run $MM $INP $W $H "$DD" > "$LG" 2>&1
  echo "    rc=$?  $(date +%T)"
  grep -aE "完成:|rc=|FATAL|Exception" "$LG" | tail -4
  grep -ac "层 0 " "$LG" >/dev/null
  grep -a "层  0 \|层 26 " "$LG" | tail -2
done
echo "=== vsweep $NM done $(date '+%F %T') ==="
