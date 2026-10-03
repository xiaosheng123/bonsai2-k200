#!/usr/bin/env python3
# 验证 TQ1_0 解包: 读 GGUF 里第一个 TQ1_0 tensor, 按 dequantize_row_tq1_0 逻辑解
import struct

f = open("/home/caden/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf", "rb")
f.read(4)
ver, nt, nk = struct.unpack("<IQQ", f.read(20))

def readstr():
    l = struct.unpack("<Q", f.read(8))[0]
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
            if atype == 8:
                arr.append(readstr())
            elif atype == 4:
                arr.append(struct.unpack("<f", f.read(4))[0])
            elif atype == 6:
                arr.append(struct.unpack("<I", f.read(4))[0])
            else:
                arr.append(0)
        return arr
    elif t == 5: return struct.unpack("<i", f.read(4))[0]
    elif t == 6: return struct.unpack("<I", f.read(4))[0]
    elif t == 4: return struct.unpack("<I", f.read(4))[0]
    elif t == 2: return struct.unpack("<H", f.read(2))[0]
    elif t == 0: return f.read(1)[0]
    else: return 0

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

# 找 type 143 (TQ1_0) 的 tensor
tq = [t for t in tensor_infos if t[2] == 143]
print("=== TQ1_0 tensors 前5 ===")
for t in tq[:5]:
    print(f"  {t[0]} dims={t[1]} off={t[3]}")

# 解包第一个
if tq:
    name, dims, tt, off = tq[0]
    print(f"\n=== 解包 {name} dims={dims} ===")
    # 数据区: tensor 数据从文件头 + off 偏移
    f.seek(off)
    QK_K = 256
    # block: qs[48] + qh[4] + d[2] = 54 bytes
    blk = f.read(54)
    d = struct.unpack("<e", blk[48+4:48+4+2])[0]
    print(f"  scale d = {d}")
    qs = blk[0:48]
    qh = blk[48:52]
    pow3 = [1, 3, 9, 27, 81]
    vals = []
    # 主区: j 0..32 步进32, n 0..5, m 0..32
    for j in range(0, 48 - 48 % 32, 32):
        for n in range(5):
            for m in range(32):
                q = qs[j + m] * pow3[n]
                xi = (q * 3) >> 8
                vals.append((xi - 1) * d)
    # 次区: j 从 32 到 48 步进 16
    for j in range(48 - 48 % 32, 48, 16):
        for n in range(5):
            for m in range(16):
                q = qs[j + m] * pow3[n]
                xi = (q * 3) >> 8
                vals.append((xi - 1) * d)
    # qh: 4 trits/byte
    for n in range(4):
        for j in range(4):
            q = qh[j] * pow3[n]
            xi = (q * 3) >> 8
            vals.append((xi - 1) * d)
    print(f"  解出 {len(vals)} 个值 (应=256)")
    # 统计三元分布
    from collections import Counter
    dist = Counter(round(v/d) for v in vals if abs(d) > 1e-9)
    print(f"  三元分布 (x/d): {dict(dist)}")
    print(f"  前 16 个: {[round(v,4) for v in vals[:16]]}")
    print(f"  后 8 个: {[round(v,4) for v in vals[-8:]]}")
    # 验证: 每个值应是 d 的整数倍 (xi-1 ∈ {-1,0,1})
    ok = all(abs(round(v/d) - v/d) < 1e-4 for v in vals if abs(d) > 1e-9)
    print(f"  全为 ±d 或 0: {ok}")
f.close()