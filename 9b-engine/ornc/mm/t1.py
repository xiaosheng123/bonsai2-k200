"""经验反推 token->patch 排列 + 用 inp_pos_emb-patch_bias 验证 bilinear 插值"""
import numpy as np, struct, os, sys
from PIL import Image
import ggufload, gg

G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
IMG = '/home/caden/ornc/mmtests/img_shapes.png'
D = '/tmp/gt/T2_shapes'


def load_dump(name):
    b = open(os.path.join(D, name), 'rb').read()
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b[:20])
    return np.frombuffer(b[20:], dtype='<f4'), (ne0, ne1, ne2, ne3)


def rd(name):
    a, ne = load_dump(name)
    return a.reshape(ne[1], ne[0]).T          # (ne0, ne1)

pb = rd('patch_bias.bin')
pe = rd('inp_pos_emb.bin')
diff = pe - pb
print('diff shape', diff.shape, 'norms', np.linalg.norm(diff, axis=0)[:5])

posw = G.np('v.position_embd.weight').astype(np.float32).reshape(48, 48, 1152)  # [p= x+48y] -> [x,y,c]
nside = 48
ow = 32
grid = gg.VT(G).interp_align_corners(posw, nside, ow, ow)   # (32,32,1152) [x,y,c]
print('grid', grid.shape)

# 每个 token 找最近的 (x,y)
mapxy = np.full((ow*ow, 2), -1, np.int64)
errs = []
for t in range(diff.shape[1]):
    v = diff[:, t]
    d2 = ((grid - v[None, None, :])**2).sum(axis=2)
    i = int(np.argmin(d2))
    x, y = i // ow, i % ow
    mapxy[t] = (x, y)
    errs.append(np.sqrt(d2[x, y])/(np.linalg.norm(v)+1e-30))
errs = np.array(errs)
print('bilinear 匹配: 中位相对误差 %.3e  最大 %.3e  (接近 0 => 插值实现对)' % (np.median(errs), errs.max()))
print('前 20 个 token 的 (x,y):', [tuple(mapxy[t]) for t in range(20)])
uniq = set(map(tuple, mapxy))
print('不同位置数 =', len(uniq), ' 应为', ow*ow)

# 检验假设 t = ((yb*NB+xb)*2+dy)*2+dx, (x,y)=(2xb+dx, 2yb+dy)
NB = ow//2
ok = 0
for t in range(ow*ow):
    yb, xb, dy, dx = (t//16) % NB, (t//4) % NB, (t//2) % 2, t % 2
    x, y = 2*xb+dx, 2*yb+dy
    if tuple(mapxy[t]) == (x, y):
        ok += 1
print('假设 t=((yb*NB+xb)*2+dy)*2+dx 匹配 %d/%d' % (ok, ow*ow))

# 用 mapxy 直接把 conv 输出拼成 token 序, 与 patch_bias 对比
im = Image.open(IMG).convert('RGB')
a = np.asarray(im, np.uint8).astype(np.float32)
H, W = a.shape[0], a.shape[1]
pix = np.ascontiguousarray(np.transpose((a/255.0 - 0.5)/0.5, (1, 0, 2))).ravel().copy()
inp_raw = gg.mk(pix, (W, H, 3, 1))
kern = gg.mat(gg.mk(G.np('v.patch_embd.weight').ravel().copy(), (16, 16, 3, 1152))) + \
       gg.mat(gg.mk(G.np('v.patch_embd.weight.1').ravel().copy(), (16, 16, 3, 1152)))
x3 = gg.mat(inp_raw)[:, :, :, 0]
ow2, oh2 = W//16, H//16
cols = np.empty((ow2*oh2, 3*16*16), np.float32)
for i in range(16):
    for j in range(16):
        blk = x3[i:i+ow2*16:16, j:j+oh2*16:16, :]
        for c2 in range(3):
            cols[:, c2*256 + j*16 + i] = blk[:, :, c2].reshape(-1)
conv = (cols @ kern.reshape(3*256, 1152)).reshape(ow2, oh2, 1152)
bias = G.np('v.patch_embd.bias').astype(np.float32)
mine = np.empty((1152, ow*ow), np.float32)
for t in range(ow*ow):
    x, y = mapxy[t]
    mine[:, t] = conv[x, y] + bias
np.save('/tmp/gt_patchbias_mine.npy', mine)
def relrms(A, B):
    A = A.astype(np.float64); B = B.astype(np.float64)
    return 100.0*np.sqrt(((A-B)**2).sum()/max((B*B).sum(), 1e-30))
print('relrms(conv+empirical map, patch_bias) = %.4f%%' % relrms(mine, pb))
