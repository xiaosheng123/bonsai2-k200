import numpy as np, os, struct, glob
D = '/tmp/gt/T2_shapes'
seen = {}
for f in sorted(glob.glob(D + '/*.bin')):
    b = open(f, 'rb').read(20)
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b)
    sz = os.path.getsize(f)
    key = (t, ne0, ne1, ne2, ne3, sz)
    nm = os.path.basename(f)
    seen.setdefault(key, []).append(nm)
print('n files', len(glob.glob(D+'/*.bin')))
for k, v in sorted(seen.items(), key=lambda x: -len(x[1])):
    print(k, 'x%d' % len(v), v[:3])
