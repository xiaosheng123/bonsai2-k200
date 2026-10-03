#!/usr/bin/env python3
# 诊断: TQ1_0 解包值异常 - 验证 tensor 数据偏移
import struct

f = open("/home/caden/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf", "rb")
f.read(4)
ver, nt, nk = struct.unpack("<IQQ", f.read(20))

def readstr():
    l = struct.unpack("<Q", f.read(8))[0]
    if l > 1000000:
        raise ValueError(f"bad {l}")
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
data_start = f.tell()
print("data_start:", data_start)

# 读第一个 tensor info (output.weight)
name = readstr()
dn = struct.unpack("<I", f.read(4))[0]
dims = struct.unpack(f"<{dn}Q", f.read(8*dn))
tt = struct.unpack("<I", f.read(4))[0]
off = struct.unpack("<Q", f.read(8))[0]
print("first tensor:", name, dims, "type", tt, "off", off)

# 方法1: data_start + off
f.seek(data_start + off)
blk1 = f.read(54)
# 方法2: 直接用 off
f.seek(off)
blk2 = f.read(54)
print("blk1 == blk2:", blk1 == blk2)
print("blk1[:12]:", blk1[:12].hex())
print("blk2[:12]:", blk2[:12].hex())

# 用 blk2 (绝对 off) 解包
pow3 = [1, 3, 9, 27, 81]
def dequant(blk):
    d = struct.unpack("<e", blk[52:54])[0]
    qs = blk[0:48]; qh = blk[48:52]
    vals = []
    for j in range(0, 32, 32):
        for n in range(5):
            for m in range(32):
                q = qs[j+m] * pow3[n]
                xi = (q * 3) >> 8
                vals.append((xi-1)*d)
    for j in range(32, 48, 16):
        for n in range(5):
            for m in range(16):
                q = qs[j+m] * pow3[n]
                xi = (q * 3) >> 8
                vals.append((xi-1)*d)
    for n in range(4):
        for j in range(4):
            q = qh[j] * pow3[n]
            xi = (q * 3) >> 8
            vals.append((xi-1)*d)
    return d, vals

for tag, blk in [("data_start+off", blk1), ("abs off", blk2)]:
    d, vals = dequant(blk)
    bad = [v for v in vals if abs(round(v/d)-v/d) > 1e-4]
    print(f"{tag}: d={d} vals={len(vals)} 非纯三元={len(bad)}")

# 打印 blk2 的 qs 前 8 字节和对应解出值
d, vals = dequant(blk2)
print("qs[:8]:", blk2[0:8].hex())
print("vals[:16]:", [round(v/d,2) for v in vals[:16]])
f.close()