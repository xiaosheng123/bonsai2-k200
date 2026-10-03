#!/usr/bin/env python3
# 最终验证: 找到正确的 tensor 数据偏移并解包 output.weight
import struct
from collections import Counter

f = open("/home/caden/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf", "rb")
f.read(4)
ver, nt, nk = struct.unpack("<IQQ", f.read(20))

def readstr():
    l = struct.unpack("<Q", f.read(8))[0]
    if l > 1000000: raise ValueError(f"bad {l}")
    return f.read(l).decode()

def read_field():
    t = struct.unpack("<I", f.read(4))[0]
    if t == 8: return readstr()
    elif t == 9:
        atype = struct.unpack("<I", f.read(4))[0]
        alen = struct.unpack("<Q", f.read(8))[0]
        for _ in range(alen):
            if atype == 8: readstr()
            elif atype == 4: f.read(4)
            elif atype == 5: f.read(4)
            elif atype == 6: f.read(4)
            elif atype == 7: f.read(1)
            elif atype == 1: f.read(1)
            elif atype == 2: f.read(2)
            elif atype == 3: f.read(2)
            elif atype == 10: f.read(8)
            else: f.read(1)
        return None
    elif t == 0: return f.read(1)[0]
    elif t == 1: return f.read(1)[0]
    elif t == 2: return f.read(2)[0]
    elif t == 3: return f.read(2)[0]
    elif t == 4: return f.read(4)[0]
    elif t == 5: return f.read(4)[0]
    elif t == 6: return f.read(4)[0]
    elif t == 7: return f.read(1)[0]
    elif t == 10: return f.read(8)[0]
    elif t == 11: return f.read(8)[0]
    elif t == 12: return f.read(8)[0]
    else: raise ValueError(f"t={t}")

for i in range(nk):
    readstr(); read_field()

# tensor infos
infos = []
for i in range(nt):
    name = readstr()
    dn = struct.unpack("<I", f.read(4))[0]
    dims = struct.unpack(f"<{dn}Q", f.read(8*dn))
    tt = struct.unpack("<I", f.read(4))[0]
    off = struct.unpack("<Q", f.read(8))[0]
    infos.append((name, dims, tt, off))

end_infos = f.tell()
print("tensor infos 结束位置:", end_infos, "align32:", end_infos % 32)

pow3 = [1, 3, 9, 27, 81]
def dequant(blk):
    d = struct.unpack("<e", blk[52:54])[0]
    qs = blk[0:48]; qh = blk[48:52]
    vals = []
    for j in range(0, 32, 32):
        for n in range(5):
            for m in range(32):
                q = (qs[j+m] * pow3[n]) & 0xFF
                xi = (int(q) * 3) >> 8
                vals.append((xi-1)*d)
    for j in range(32, 48, 16):
        for n in range(5):
            for m in range(16):
                q = (qs[j+m] * pow3[n]) & 0xFF
                xi = (int(q) * 3) >> 8
                vals.append((xi-1)*d)
    for n in range(4):
        for j in range(4):
            q = (qh[j] * pow3[n]) & 0xFF
            xi = (int(q) * 3) >> 8
            vals.append((xi-1)*d)
    return d, vals

# output.weight 是第一个 TQ1_0 (off=0)
name, dims, tt, off = infos[0]
print(f"tensor0: {name} dims={dims} type={tt} off={off}")

# 候选数据起点: end_infos 对齐 32
for base in [end_infos, (end_infos + 31) // 32 * 32]:
    f.seek(base + off)
    blk = f.read(54)
    d, vals = dequant(blk)
    bad = sum(1 for v in vals if abs(round(v/d) - v/d) > 1e-4)
    dist = Counter(round(v/d) for v in vals if abs(d) > 1e-9)
    print(f"base={base}: d={d} 非纯三元={bad} 分布={dict(list(dist.items())[:6])}")
    if bad == 0:
        print("  ==> 正确偏移!")
f.close()