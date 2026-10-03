#!/usr/bin/env python3
# 解包第一个 TQ1_0 tensor (基于成功解析 KV 的完整 read_field)
import struct
from collections import Counter

f = open("/home/caden/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf", "rb")
f.read(4)
ver, nt, nk = struct.unpack("<IQQ", f.read(20))
print("tensors:", nt, "kv:", nk)

def readstr():
    l = struct.unpack("<Q", f.read(8))[0]
    if l > 1000000:
        raise ValueError(f"bad str len {l}")
    return f.read(l).decode()

def read_field():
    t = struct.unpack("<I", f.read(4))[0]
    if t == 8:
        return readstr()
    elif t == 9:
        atype = struct.unpack("<I", f.read(4))[0]
        alen = struct.unpack("<Q", f.read(8))[0]
        arr = []
        for _ in range(alen):
            if atype == 8: arr.append(readstr())
            elif atype == 4: arr.append(struct.unpack("<f", f.read(4))[0])
            elif atype == 5: arr.append(struct.unpack("<i", f.read(4))[0])
            elif atype == 6: arr.append(struct.unpack("<I", f.read(4))[0])
            elif atype == 7: arr.append(f.read(1)[0] != 0)
            elif atype == 1: arr.append(f.read(1)[0])
            elif atype == 2: arr.append(struct.unpack("<H", f.read(2))[0])
            elif atype == 3: arr.append(struct.unpack("<h", f.read(2))[0])
            elif atype == 10: arr.append(struct.unpack("<q", f.read(8))[0])
            else: arr.append(0)
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

for i in range(nk):
    readstr(); read_field()

# tensor infos
tensor_infos = []
for i in range(nt):
    name = readstr()
    dn = struct.unpack("<I", f.read(4))[0]
    dims = struct.unpack(f"<{dn}Q", f.read(8*dn))
    tt = struct.unpack("<I", f.read(4))[0]
    off = struct.unpack("<Q", f.read(8))[0]
    tensor_infos.append((name, dims, tt, off))

tq = [t for t in tensor_infos if t[2] == 143]
print("TQ1_0 数量:", len(tq), "of", nt)
for t in tq[:6]:
    print(f"  {t[0]} dims={t[1]} off={t[3]}")

name, dims, tt, off = tq[0]
print(f"\n解包 {name} dims={dims}")
# GGUF tensor 数据的偏移: 在文件里 off 是相对数据区起点的偏移
# 数据区起点 = 读完所有 tensor info 后的位置
data_start = f.tell()
f.seek(data_start + off)
QK_K = 256
blk = f.read(54)
# block: qs[48] qh[4] half_d[2]
d = struct.unpack("<e", blk[52:54])[0]
print(f"  scale d = {d}")
qs = blk[0:48]
qh = blk[48:52]
pow3 = [1, 3, 9, 27, 81]
vals = []
for j in range(0, 48 - 48 % 32, 32):
    for n in range(5):
        for m in range(32):
            q = qs[j + m] * pow3[n]
            xi = (q * 3) >> 8
            vals.append((xi - 1) * d)
for j in range(48 - 48 % 32, 48, 16):
    for n in range(5):
        for m in range(16):
            q = qs[j + m] * pow3[n]
            xi = (q * 3) >> 8
            vals.append((xi - 1) * d)
for n in range(4):
    for j in range(4):
        q = qh[j] * pow3[n]
        xi = (q * 3) >> 8
        vals.append((xi - 1) * d)
print(f"  解出 {len(vals)} 值 (应=256)")
dist = Counter(round(v/d) for v in vals if abs(d) > 1e-9)
print(f"  分布: {dict(dist)}")
ok = all(abs(round(v/d) - v/d) < 1e-4 for v in vals if abs(d) > 1e-9)
print(f"  全为 ±d/0: {ok}")
print(f"  前12: {[round(v,3) for v in vals[:12]]}")
f.close()