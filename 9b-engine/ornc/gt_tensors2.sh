#!/bin/bash
cd /home/caden/llamacpp
export LD_LIBRARY_PATH=/home/caden/llamacpp/build/bin
D=/tmp/gt/T2_shapes
rm -rf $D; mkdir -p $D
nice -n 19 env MTMD_DEBUG_TENSORS=$D timeout 1200 ./build/bin/llama-mtmd-cli \
   -m /home/caden/orn/Ornith-1.5-9B-Q4_K_M.gguf --mmproj /home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf \
   --image /home/caden/ornc/mmtests/img_shapes.png -p "hi" -n 4 -t 2 > /tmp/gt/tens2.log 2>&1
echo "rc=$? files=$(ls $D|wc -l) size=$(du -sh $D|cut -f1)"
