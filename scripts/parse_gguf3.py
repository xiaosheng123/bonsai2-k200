#!/usr/bin/env python3
# 解析 Bonsai 2 27B GGUF - 完整读取 array + tensor 类型
import struct

f = open("/home/caden/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf", "rb")
f.read(4)
ver, nt, nk = struct.unpack("<IQQ", f.read(20))
print("tensors:", nt, "kv:", nk)

def readstr():
    l = struct.unpack("<Q", f.read(8))[0]
    if l > 1000000:
        raise ValueError(f"bad str len {l}")
    return f.read(l).decode()

TNAME = {0:"u8",1:"i8",2:"u16",3:"i16",4:"u32",5:"i32",6:"f32",7:"bool",
    8:"str",9:"arr",10:"i64",11:"f64",12:"u64"}

def read_field():
    t = struct.unpack("<I", f.read(4))[0]
    if t == 8:
        return readstr()
    elif t == 9:
        atype = struct.unpack("<I", f.read(4))[0]
        alen = struct.unpack("<Q", f.read(8))[0]
        arr = []
        for _ in range(alen):
            if atype == 8:
                arr.append(readstr())
            elif atype == 4: arr.append(round(struct.unpack("<f", f.read(4))[0],5))
            elif atype == 5: arr.append(struct.unpack("<i", f.read(4))[0])
            elif atype == 6: arr.append(struct.unpack("<I", f.read(4))[0])
            elif atype == 7: arr.append(f.read(1)[0] != 0)
            elif atype == 1: arr.append(f.read(1)[0])
            elif atype == 3: arr.append(struct.unpack("<h", f.read(2))[0])
            else: arr.append(f"?{atype}")
        return arr
    elif t == 0: return f.read(1)[0]
    elif t == 1: return struct.unpack("<b", f.read(1))[0]
    elif t == 2: return struct.unpack("<H", f.read(2))[0]
    elif t == 3: return struct.unpack("<h", f.read(2))[0]
    elif t == 4: return struct.unpack("<I", f.read(4))[0]
    elif t == 5: return struct.unpack("<i", f.read(4))[0]
    elif t == 6: return struct.unpack("<f", f.read(4))[0]
    elif t == 7: return f.read(1)[0] != 0
    elif t == 10: return struct.unpack("<q", f.read(8))[0]
    elif t == 11: return struct.unpack("<d", f.read(8))[0]
    elif t == 12: return struct.unpack("<Q", f.read(8))[0]
    else: raise ValueError(f"unknown t={t}")

# 读 KV
kvs = {}
for i in range(nk):
    k = readstr()
    v = read_field()
    kvs[k] = v

print("=== 关键 KV ===")
for k in ["general.architecture","qwen35.block_count","qwen35.embedding_length",
          "qwen35.ssm.state_size","qwen35.ssm.group_count","qwen35.ssm.inner_size",
          "qwen35.full_attention_interval","qwen35.attention.head_count",
          "qwen35.attention.key_length","prism.hadamard.block_size",
          "prism.hadamard.transform","prism.hadamard.axis","prism.hadamard.weight_names"]:
    v = kvs.get(k)
    if isinstance(v, list):
        print(f"  {k} = arr({len(v)})")
        if k.endswith("weight_names") and v:
            print(f"    [0..5]: {v[:6]}")
            print(f"    [...]: {v[-2:]}")
    else:
        print(f"  {k} = {v}")

print("=== tensor 类型统计 ===")
# GGUF type = 0x10000 + ggml_type; ggml.h: 0=F32,1=F16,2=Q4_0,... 34=TQ1_0,35=TQ2_0
tcount = {}
for i in range(nt):
    name = readstr()
    dn = struct.unpack("<I", f.read(4))[0]
    dims = struct.unpack(f"<{dn}Q", f.read(8*dn))
    tt = struct.unpack("<I", f.read(4))[0]
    off = struct.unpack("<Q", f.read(8))[0]
    # 打印原始 type 值分布
    tcount[tt] = tcount.get(tt, 0) + 1

print("  (原始 type 值, 数量):")
for t, c in sorted(tcount.items()):
    print(f"    0x{t:x} ({t}): {c}")
f.close()