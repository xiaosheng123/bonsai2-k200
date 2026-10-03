#!/bin/bash
# 逐层真值 (低优先级!): MTMD_DEBUG_TENSORS 把 clip 图里所有具名节点落盘
# ★ 纪律: nice -n 19 + -t 2, 绝不与 8090 抢核
cd /home/caden/llamacpp
export LD_LIBRARY_PATH=/home/caden/llamacpp/build/bin
M=/home/caden/orn/Ornith-1.5-9B-Q4_K_M.gguf
P=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
for n in shapes; do
  D=/tmp/gt/T_$n
  rm -rf $D; mkdir -p $D
  nice -n 19 env MTMD_DEBUG_TENSORS=$D MTMD_DEBUG_EMBEDDINGS=/tmp/gt/embd2_$n.bin \
     timeout 1200 ./build/bin/llama-mtmd-cli \
     -m $M --mmproj $P --image /home/caden/ornc/mmtests/img_$n.png \
     -p "hi" -n 4 -t 2 \
     > /tmp/gt/tens_$n.log 2>&1
  echo "done $n rc=$? files=$(ls $D 2>/dev/null | wc -l) size=$(du -sh $D 2>/dev/null | cut -f1)"
done
echo ALLDONE
