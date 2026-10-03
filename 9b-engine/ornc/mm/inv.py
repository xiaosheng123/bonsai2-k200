import numpy as np
from PIL import Image
import ggufload, math

P = '/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf'
g = ggufload.GGUF(P)
print('version', g.version, 'n_tensors', g.n_tensors, 'n_kv', g.n_kv)
for k in sorted(g.kv):
    if k.startswith('general.') or k.startswith('clip.') or k.startswith('mmproj'):
        print(' ', k, '=', g.kv[k])
print('data_off', g.data_off, 'file size', __import__('os').path.getsize(P))
tot = 0
for n, t in g.tensors.items():
    ne = t['ne']
    tot += int(np.prod(ne)) * ggufload.NBYTES[t['type']]
print('sum bytes', tot)
names = sorted(g.tensors)
print('n names', len(names))
for n in names[:12]:
    print(' ', n, g.tensors[n]['ne'], 'type', g.tensors[n]['type'])
print('--- mm.* / post_ln / patch / pos ---')
for n in names:
    if n.startswith('mm.') or 'post_ln' in n or 'patch' in n or 'position' in n or n.startswith('v.blk.0.'):
        print(' ', n, g.tensors[n]['ne'], 'type', g.tensors[n]['type'])
