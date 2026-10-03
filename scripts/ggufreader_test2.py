#!/usr/bin/env python3
# 用 gguf-py GGUFReader 读 output.weight (importlib 加载带连字符的模块)
import sys, importlib.util

spec = importlib.util.spec_from_file_location("gguf_reader_mod",
    "/home/caden/bonsai2/fork-src/gguf-reader.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
GGUFReader = mod.GGUFReader

reader = GGUFReader("/home/caden/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
print("GGUFReader OK, tensors:", len(reader.tensors))

# 元数据
fi = reader.fields
for k in ["general.architecture","qwen35.block_count","qwen35.embedding_length",
          "qwen35.ssm.state_size","prism.hadamard.block_size","prism.hadamard.transform"]:
    if k in fi:
        fld = fi[k]
        v = fld.parts[-1] if len(fld.parts) > 1 else fld
        try:
            print(f"  {k} = {v}")
        except Exception:
            print(f"  {k} = {v.tolist() if hasattr(v,'tolist') else v}")

# 第一个 tensor
t0 = reader.tensors[0]
print("tensor0:", t0.name, "shape:", t0.shape, "type:", t0.tensor_type)
# 原始字节 (mmap 数据)
import numpy as np
data = t0.data
print("  data nbytes:", data.nbytes, "dtype:", data.dtype)

# 读第一个 block (54 字节) 并解包
pow3 = [1, 3, 9, 27, 81]
raw = bytes(data[:54])
d = struct_val = np.frombuffer(raw[52:54], dtype="<f2")[0]
qs = raw[0:48]
qh = raw[48:52]
vals = []
for j in range(0, 32, 32):
    for n in range(5):
        for m in range(32):
            q = qs[j+m] * pow3[n]
            xi = (int(q) * 3) >> 8
            vals.append((xi - 1) * float(d))
for j in range(32, 48, 16):
    for n in range(5):
        for m in range(16):
            q = qs[j+m] * pow3[n]
            xi = (int(q) * 3) >> 8
            vals.append((xi - 1) * float(d))
for n in range(4):
    for j in range(4):
        q = qh[j] * pow3[n]
        xi = (int(q) * 3) >> 8
        vals.append((xi - 1) * float(d))
print("解出:", len(vals), "值")
from collections import Counter
dist = Counter(round(v/float(d)) for v in vals if abs(d) > 1e-9)
print("分布:", dict(dist))
print("前12:", [round(v,3) for v in vals[:12]])