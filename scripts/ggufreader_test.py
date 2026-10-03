#!/usr/bin/env python3
# 用 gguf-py GGUFReader 读 output.weight 原始数据, 验证 TQ1_0 解包
import sys, struct
sys.path.insert(0, "/home/caden/bonsai2/fork-src")
try:
    from gguf_reader import GGUFReader
    reader = GGUFReader("/home/caden/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
    print("GGUFReader OK")
    print("tensors:", len(reader.tensors))
    t0 = reader.tensors[0]
    print("tensor0:", t0.name, t0.shape, t0.tensor_type.name if hasattr(t0.tensor_type,'name') else t0.tensor_type)
    # field 数据
    fi = reader.fields
    for k in ["general.architecture","qwen35.block_count","prism.hadamard.block_size","prism.hadamard.transform"]:
        if k in fi:
            fld = fi[k]
            print(f"FIELD {k}: {fld.parts[1] if len(fld.parts)>1 else fld}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("ERR:", e)