#!/bin/bash
# vcpu.sh —— 纯主机参考路径 (VIS_CPUGEMM=1, 不占卡): 只量化权重、激活精确
#   用来判定 "权重粒度 vs 激活量化" 谁是误差主因
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
cd /home/caden/orn_engine || exit 1
MM=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
for cfg in "$@"; do
  CH=${cfg%%/*}; CHA=${cfg##*/}
  DD=/tmp/vsw/cpu/${CH}_${CHA}
  rm -rf "$DD"; mkdir -p "$DD"
  echo "### CPU CH=$CH CHA=$CHA start $(date +%T)"
  env VIS_CPUGEMM=1 VIS_CH=$CH VIS_CHA=$CHA OMP_NUM_THREADS=4 \
      ./vistest run $MM /tmp/inp512.f32 512 512 "$DD" > /tmp/vsw/cpu/${CH}_${CHA}.log 2>&1
  echo "    rc=$? $(date +%T)"
  grep -aE "完成:|FATAL" /tmp/vsw/cpu/${CH}_${CHA}.log | tail -2
done
echo "=== vcpu done $(date +%T) ==="
