#!/bin/bash
cd /home/caden/llamacpp
export LD_LIBRARY_PATH=/home/caden/llamacpp/build/bin
mkdir -p /tmp/gt
for n in big_shapes big_text big_table; do
  nice -n 19 env MTMD_DEBUG_EMBEDDINGS=/tmp/gt/e768_$n.bin timeout 1200 ./build/bin/llama-mtmd-cli \
     -m /home/caden/orn/Ornith-1.5-9B-Q4_K_M.gguf --mmproj /home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf \
     --image /home/caden/ornc/mmtests/$n.png \
     -p "What do you see in this image? Answer in one short sentence." -n 64 -t 2 \
     > /tmp/gt/a768_$n.log 2>&1
  echo "done $n rc=$? size=$(stat -c%s /tmp/gt/e768_$n.bin 2>/dev/null)"
done
echo ALLDONE
