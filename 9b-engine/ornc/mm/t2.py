"""诊断: 用合成索引张量走 A/B 两条 cont 语义, 得到 (x,y) 排列, 与经验排列对比"""
import numpy as np, struct, os
import ggufload, gg

G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
D = '/tmp/gt/T2_shapes'
ow = oh = 32
NB = ow//2

def rd(name):
    b = open(os.path.join(D, name), 'rb').read()
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b[:20])
    return np.frombuffer(b[20:], dtype='<f4').reshape(ne1, ne0).T

# 经验排列 (从 pos emb 推)
pe = rd('inp_pos_emb.bin'); pb = rd('patch_bias.bin')
diff = pe - pb
posw = G.np('v.position_embd.weight').astype(np.float32).reshape(48, 48, 1152)
grid = gg.VT(G).interp_align_corners(posw, 48, ow, ow)
emp = np.zeros((ow*ow, 2), np.int64)
for t in range(ow*ow):
    d2 = ((grid - diff[:, t][None, None, :])**2).sum(axis=2)
    i = int(np.argmin(d2)); emp[t] = (i//ow, i % ow)
print('经验排列 t=0..23:', [tuple(emp[t]) for t in range(24)])
print('经验排列 t=0,16,32,48,64,128,256,512,1023:', [tuple(emp[t]) for t in (0,16,32,48,64,128,256,512,1023)])

# 合成: conv[x,y,c] = x*1000+y  (独立于 c)
syn = np.empty((ow, oh, 1152), np.float32)
for x in range(ow):
    for y in range(oh):
        syn[x, y, :] = x*1000.0 + y
conv = gg.mk(np.ascontiguousarray(syn).ravel(), (ow, oh, 1152, 1))

for tag, fn in (('A', gg.cont), ('B', gg.contB)):
    t1 = fn(gg.permute(conv, 1, 2, 0, 3), (1152*2, ow//2, oh, 1))
    t2 = gg.cont(t1, (1152*2, ow//2, 2, oh//2))
    t3 = gg.permute(t2, 0, 2, 1, 3)
    t4 = gg.cont(t3, (1152, ow*oh, 1))
    out = gg.mat(t4)[:, :, 0, 0]
    code = out[0, :]
    xy = np.stack([np.floor(code/1000).astype(np.int64), (code % 1000).astype(np.int64)], 1)
    n_match = int((xy == emp).all(axis=1).sum())
    print('cand %s  与经验排列一致 %d/%d' % (tag, n_match, ow*ow))
    print('   cand t=0..23:', [tuple(v) for v in xy[:24]])
