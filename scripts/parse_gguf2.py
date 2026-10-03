#!/usr/bin/env python3
# 解析 Bonsai 2 27B PTQ1_0 GGUF (修正: GGUF v3 value type 是 u32)
import struct

f = open("/home/caden/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf", "rb")
magic = f.read(4)
ver, nt, nk = struct.unpack("<IQQ", f.read(20))
print("magic:", magic, "ver:", ver, "tensors:", nt, "kv:", nk)

def readstr():
    l = struct.unpack("<Q", f.read(8))[0]
    if l > 1000000:
        raise ValueError(f"bad str len {l}")
    return f.read(l).decode()

TNAME = {0:"u8",1:"i8",2:"u16",3:"i16",4:"u32",5:"i32",6:"f32",7:"bool",
    8:"str",9:"arr",10:"i64",11:"f64",12:"u64"}
QTYPE = {0x10000:"F32",0x10001:"F16",0x10002:"Q4_0",0x10003:"Q4_1",0x10004:"Q5_0",
    0x10005:"Q5_1",0x10006:"Q8_0",0x10007:"Q8_1",0x10008:"Q2_K",0x10009:"Q3_K",
    0x1000A:"Q4_K",0x1000B:"Q5_K",0x1000C:"Q6_K",0x1000D:"Q8_K",0x1000E:"IQ2_XXS",
    0x1000F:"IQ2_XS",0x10010:"IQ3_XXS",0x10011:"IQ1_S",0x10012:"IQ4_NL",0x10013:"IQ3_S",
    0x10014:"IQ2_S",0x10015:"IQ4_XS",0x10016:"I8",0x10017:"I16",0x10018:"I32",
    0x10019:"I64",0x1001A:"F64",0x1001B:"IQ1_M",0x1001C:"BF16",
    0x10020:"TQ1_0",0x10021:"TQ2_0"}

kvs = {}
for i in range(nk):
    k = readstr()
    t = struct.unpack("<I", f.read(4))[0]   # GGUF v3: type 是 u32!
    tn = TNAME.get(t, f"unk{t}")
    if t == 8:
        v = readstr()
    elif t == 9:
        atype = struct.unpack("<I", f.read(4))[0]
        alen = struct.unpack("<Q", f.read(8))[0]
        arr = []
        for _ in range(min(alen, 25)):
            if atype == 8: arr.append(readstr())
            elif atype == 4: arr.append(round(struct.unpack("<f", f.read(4))[0],5))
            elif atype == 5: arr.append(struct.unpack("<i", f.read(4))[0])
            elif atype == 6: arr.append(struct.unpack("<I", f.read(4))[0])
            elif atype == 7: arr.append(f.read(1)[0] != 0)
            elif atype == 1: arr.append(f.read(1)[0])
            elif atype == 3: arr.append(struct.unpack("<h", f.read(2))[0])
            elif atype == 2: arr.append(struct.unpack("<H", f.read(2))[0])
            else: arr.append(f"?{atype}")
        v = (f"arr[{atype}]({alen})", arr)
    elif t == 0: v = f.read(1)[0]
    elif t == 1: v = struct.unpack("<b", f.read(1))[0]
    elif t == 2: v = struct.unpack("<H", f.read(2))[0]
    elif t == 3: v = struct.unpack("<h", f.read(2))[0]
    elif t == 4: v = struct.unpack("<I", f.read(4))[0]
    elif t == 5: v = struct.unpack("<i", f.read(4))[0]
    elif t == 6: v = struct.unpack("<f", f.read(4))[0]
    elif t == 7: v = f.read(1)[0] != 0
    elif t == 10: v = struct.unpack("<q", f.read(8))[0]
    elif t == 11: v = struct.unpack("<d", f.read(8))[0]
    elif t == 12: v = struct.unpack("<Q", f.read(8))[0]
    else:
        raise ValueError(f"unknown t={t} key={k}")
    s = str(v)
    if len(s) > 130: s = s[:130] + "..."
    print(f"  [{i}] {tn} {k} = {s}")
    kvs[k] = v

print("=== tensors (全部 851, 前 40 + 特殊类型) ===")
tcount = {}
for i in range(nt):
    name = readstr()
    dn = struct.unpack("<I", f.read(4))[0]
    dims = struct.unpack(f"<{dn}Q", f.read(8*dn))
    tt = struct.unpack("<I", f.read(4))[0]
    off = struct.unpack("<Q", f.read(8))[0]
    tn = QTYPE.get(tt, f"UNK_{tt:#x}_{tt}")
    tcount[tn] = tcount.get(tn, 0) + 1
    if i < 40 or "UNK" in tn:
        print(f"  [{i}] {name} dims={dims} type={tn}")
print("=== 类型统计 ===")
for t, c in sorted(tcount.items()):
    print(f"  {t}: {c}")
f.close()