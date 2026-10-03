#!/usr/bin/env python3
# 调试 GGUF KV0 解析过程
import struct

f = open("/home/caden/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf", "rb")
print("pos0:", f.tell())
magic = f.read(4)
ver, nt, nk = struct.unpack("<IQQ", f.read(20))
print("magic:", magic, "ver:", ver, "nt:", nt, "nk:", nk, "pos:", f.tell())

def readstr():
    p = f.tell()
    raw = f.read(8)
    l = struct.unpack("<Q", raw)[0]
    print("  readstr len@", p, "raw:", raw.hex(), "len:", l)
    if l > 1000000:
        raise ValueError(f"bad str len {l}")
    s = f.read(l).decode()
    print("  readstr str@", p+8, "len:", l, "val:", s[:40])
    return s

try:
    k = readstr()
    t = f.read(1)[0]
    print("  key:", k, "type:", t, "pos:", f.tell())
    if t == 8:
        v = readstr()
        print("  value:", v)
except Exception as e:
    print("ERR:", e, "pos:", f.tell())
f.close()