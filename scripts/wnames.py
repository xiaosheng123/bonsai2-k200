#!/usr/bin/env python3
import struct, collections
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
            a.append(rs() if at == 8 else f.read(4))
        return a
    if t in (0, 1, 7):
        return f.read(1)
    if t in (2, 3):
        return f.read(2)
    if t in (4, 5, 6):
        return f.read(4)
    if t in (10, 11, 12):
        return f.read(8)
    raise ValueError(t)
kvs = {}
for i in range(nk):
    k = rs()
    v = rf()
    kvs[k] = v
w = kvs.get('prism.hadamard.weight_names', [])
print('count:', len(w))
c = collections.Counter()
for x in w:
    if '.blk.' in x:
        c['blk.' + x.split('.')[2]] += 1
    else:
        c[x] += 1
for k, v in sorted(c.items()):
    print(f'  {k}: {v}')
print('--- all names ---')
for x in w:
    print(' ', x)