#!/usr/bin/env python3
import struct
f = open('/home/caden/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf', 'rb')
f.read(4)
ver, nt, nk = struct.unpack('<IQQ', f.read(20))
def rs():
    l = struct.unpack('<Q', f.read(8))[0]
    return f.read(l).decode()
def rf():
    t = struct.unpack('<I', f.read(4))[0]
    if t == 8:
        return rs()
    if t == 9:
        at = struct.unpack('<I', f.read(4))[0]
        al = struct.unpack('<Q', f.read(8))[0]
        a = []
        for _ in range(al):
            if at == 8: a.append(rs())
            elif at in (0, 1, 7): a.append(f.read(1))
            elif at in (2, 3): a.append(f.read(2))
            elif at in (4, 5, 6): a.append(struct.unpack('<i', f.read(4))[0])
            elif at in (10, 11, 12): a.append(f.read(8))
            else: a.append(f.read(4))
        return a
    if t in (0, 1, 7): return f.read(1)
    if t in (2, 3): return f.read(2)
    if t in (4, 5, 6): return struct.unpack('<I', f.read(4))[0]
    if t in (10, 11, 12): return f.read(8)
    raise ValueError(t)
kvs = {}
for i in range(nk):
    k = rs()
    v = rf()
    kvs[k] = v
for k in sorted(kvs):
    if k.startswith('prism') or 'hadamard' in k or 'sign' in k:
        v = kvs[k]
        if isinstance(v, list):
            print(f'{k}: list len={len(v)}')
            if len(v) > 0 and isinstance(v[0], bytes):
                print(f'  first bytes: {v[0][:8].hex()}')
            elif len(v) <= 12:
                print(f'  {v}')
        else:
            print(f'{k}: {v}')