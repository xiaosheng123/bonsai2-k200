#!/bin/bash
cd /home/caden/llamacpp
export LD_LIBRARY_PATH=/home/caden/llamacpp/build/bin
for s in 1 2 3 99 0; do
  nice -n 19 env MTMD_DUMP_STAGE=$s MTMD_DEBUG_EMBEDDINGS=/tmp/gt/stage$s.bin \
    timeout 600 ./build/bin/llama-mtmd-cli -m /home/caden/orn/Ornith-1.5-9B-Q4_K_M.gguf \
    --mmproj /home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf --image /home/caden/ornc/mmtests/big_shapes.png \
    -p "hi" -n 1 -t 2 > /tmp/gt/stage$s.log 2>&1
  echo "stage $s rc=$? size=$(stat -c%s /tmp/gt/stage$s.bin 2>/dev/null)"
done
echo ALLDONE
