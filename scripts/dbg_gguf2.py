#!/usr/bin/env python3
# 精确 dump GGUF 头部字节 (每个字节带绝对位置)
f = open("/home/caden/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf", "rb")
d = f.read(160)
# 打印每一行的绝对偏移
for base in range(0, 128, 16):
    row = d[base:base+16]
    pos_hex = []
    for j, b in enumerate(row):
        pos = base + j
        pos_hex.append(f"{pos:02d}:{b:02x}")
    print(" ".join(pos_hex))
# 同时读关键字段
import struct
f.seek(0)
magic = f.read(4)
ver = struct.unpack("<I", f.read(4))[0]
nt = struct.unpack("<Q", f.read(8))[0]
nk = struct.unpack("<Q", f.read(8))[0]
print("ver:", ver, "nt:", nt, "nk:", nk, "pos after header:", f.tell())
# 手动解析 KV0
print("--- KV0 手动解析 ---")
p = f.tell()
raw = f.read(8)
klen = struct.unpack("<Q", raw)[0]
print("pos", p, "keylen raw:", raw.hex(), "=", klen)
key = f.read(klen)
print("key:", key)
t = f.read(1)[0]
print("type:", t, "pos:", f.tell())
raw = f.read(8)
vlen = struct.unpack("<Q", raw)[0]
print("vlen raw:", raw.hex(), "=", vlen, "pos:", f.tell())
if vlen < 50:
    print("value:", f.read(vlen))
f.close()