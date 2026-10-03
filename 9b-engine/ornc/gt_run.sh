#!/bin/bash
# 真值: 用 llama.cpp 的 mtmd-cli 跑同一批图片, 导出 image embedding 与回答
 
cd /home/caden/llamacpp
export LD_LIBRARY_PATH=/home/caden/llamacpp/build/bin
mkdir -p /tmp/gt
M=/home/caden/orn/Ornith-1.5-9B-Q4_K_M.gguf
P=/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf
for n in text shapes table; do
  MTMD_DEBUG_EMBEDDINGS=/tmp/gt/embd_$n.bin timeout 900 ./build/bin/llama-mtmd-cli \
     -m $M --mmproj $P --image /home/caden/ornc/mmtests/img_$n.png \
     -p "Describe what you see in this image in one short sentence." -n 64 -t 4 \
     > /tmp/gt/ans_$n.log 2>&1
  echo "done $n rc=$?"
done
ls -la /tmp/gt/
