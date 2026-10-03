#!/usr/bin/env python3
# 解析 Bonsai 2 27B PTQ1_0 GGUF: 逐KV打印, 错误保护
import struct, sys

f = open("/home/caden/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf", "rb")
magic = f.read(4)
ver, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
print("magic:", magic, "ver:", ver, "tensors:", n_tensors, "kv:", n_kv)

def readstr():
    l = struct.unpack("<Q", f.read(8))[0]
    if l > 1000000:
        raise ValueError(f"bad str len {l}")
    return f.read(l).decode()

GGUF_TYPE = {0:"u8",1:"i8",2:"u16",3:"i16",4:"u32",5:"i32",6:"f32",7:"bool",
    8:"str",9:"arr",10:"i64",11:"f64",12:"u64"}

kvs = {}
try:
    for i in range(n_kv):
        k = readstr()
        t = f.read(1)[0]
        tname = GGUF_TYPE.get(t, f"unk{t}")
        if t == 8:
            v = readstr()
        elif t == 9:
            atype = f.read(1)[0]
            alen = struct.unpack("<Q", f.read(8))[0]
            arr = []
            for _ in range(min(alen, 30)):
                if atype == 8: arr.append(readstr())
                elif atype == 4: arr.append(round(struct.unpack("<f", f.read(4))[0],5))
                elif atype == 5: arr.append(struct.unpack("<i", f.read(4))[0])
                elif atype == 6: arr.append(struct.unpack("<I", f.read(4))[0])
                elif atype == 7: arr.append(f.read(1)[0] != 0)
                elif atype == 1: arr.append(f.read(1)[0])
                elif atype == 3: arr.append(struct.unpack("<h", f.read(2))[0])
                elif atype == 2: arr.append(struct.unpack("<H", f.read(2))[0])
                elif atype == 10: arr.append(struct.unpack("<q", f.read(8))[0])
                elif atype == 12: arr.append(struct.unpack("<Q", f.read(8))[0])
                else:
                    arr.append(f"?{atype}")
                    break
            v = (f"arr[{atype}]", alen, arr)
        elif t == 0: v = f.read(1)[0]
        elif t == 1: v = f.read(1)[0]
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
            raise ValueError(f"unknown t={t} at key={k}")
        s = str(v)
        if len(s) > 110: s = s[:110] + "..."
        print(f"  [{i}] {tname} {k} = {s}")
        kvs[k] = v
except Exception as e:
    print("PARSE_STOP at KV", len(kvs), ":", e)
    print("file pos:", f.tell())

print("=== tensor 区 (前40个) ===")
TNAME = {0x10000:"F32",0x10001:"F16",0x10002:"Q4_0",0x10006:"Q8_0",
    0x10008:"Q2_K",0x1000E:"IQ2_XXS",0x10020:"TQ1_0",0x10021:"TQ2_0"}
try:
    for i in range(min(n_tensors, 45)):
        name = readstr()
        dn = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack(f"<{dn}Q", f.read(8*dn))
        tt = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        tn = TNAME.get(tt, f"UNK_{tt:#x}")
        print(f"  [{i}] {name} dims={dims} type={tn}")
except Exception as e:
    print("TENSOR_STOP:", e)
f.close()