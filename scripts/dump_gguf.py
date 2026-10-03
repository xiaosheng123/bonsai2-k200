#!/usr/bin/env python3
# dump Bonsai GGUF 头部
f = open("/home/caden/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf", "rb")
d = f.read(192)
for i in range(0, 128, 16):
    hexs = " ".join(f"{b:02x}" for b in d[i:i+16])
    chars = "".join(chr(b) if 32 <= b < 127 else "." for b in d[i:i+16])
    print(f"{i:5d}: {hexs}  |{chars}|")
f.close()