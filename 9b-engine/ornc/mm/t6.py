"""干净版: 正确构建 pos 网格, 定出真值 (x,y) 排列, 并与 MM_PLAN 公式 / 候选链路比较"""
import numpy as np, struct, os
from PIL import Image
import ggufload, gg

G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
D = '/tmp/gt/T2_shapes'
ow = oh = 32
N = ow*oh
NB = ow//2


def rd(name):
    b = open(os.path.join(D, name), 'rb').read()
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b[:20])
    return np.frombuffer(b[20:], dtype='<f4').reshape(ne1, ne0).T


pb = rd('patch_bias.bin'); pe = rd('inp_pos_emb.bin')
bias = G.np('v.patch_embd.bias').astype(np.float32)
ref = pb - bias[:, None]
diff = pe - pb

# --- 正确的 pos 网格: tab[i,j,c], i = p%48 (x), j = p//48 (y) ---
A3 = G.np('v.position_embd.weight').astype(np.float32).reshape(1152, 48, 48)  # [c, j, i]
tab = np.ascontiguousarray(A3.transpose(2, 1, 0))                            # [i, j, c]
grid = gg.VT(G).interp_align_corners(tab, 48, ow, oh)                        # [x, y, c]
print('grid shape', grid.shape)
P = np.zeros(N, np.int64)
err = []
for t in range(N):
    d2 = ((grid - diff[:, t][None, None, :])**2).sum(axis=2)
    i = int(np.argmin(d2)); P[t] = i
    err.append(np.sqrt(d2.ravel()[i])/(np.linalg.norm(diff[:, t])+1e-30))
print('pos-emb 匹配: 中位 %.2e 最大 %.2e' % (np.median(err), max(err)))
posmap = np.stack([P % ow, P//ow], 1)     # (x,y)  true map from pos embedding
print('P t=0..31:', [tuple(v) for v in posmap[:32]])

# --- 从 conv 侧独立验证同一排列 ---
im = Image.open('/home/caden/ornc/mmtests/img_shapes.png').convert('RGB')
a = np.asarray(im, np.uint8).astype(np.float32)
H, W = a.shape[0], a.shape[1]
pix = np.ascontiguousarray(np.transpose((a/255.0 - 0.5)/0.5, (1, 0, 2))).ravel().copy()
x3 = pix.reshape(W, H, 3)
K = (gg.mat(gg.mk(G.np('v.patch_embd.weight').ravel().copy(), (16, 16, 3, 1152))) +
     gg.mat(gg.mk(G.np('v.patch_embd.weight.1').ravel().copy(), (16, 16, 3, 1152))))
conv = np.zeros((ow, oh, 1152), np.float32)
for i in range(16):
    for j in range(16):
        conv += x3[i::16, j::16, :] @ K[i, j]
V = (conv + bias).reshape(N, 1152)
Vn = V/np.maximum(np.linalg.norm(V, axis=1), 1e-30)[:, None]
Rn = ref/np.maximum(np.linalg.norm(ref, axis=0), 1e-30)[None, :]
C = Vn @ Rn
Q = C.argmax(axis=0)
convmap = np.stack([Q % ow, Q//ow], 1)
print('conv 匹配 与 pos 排列一致 %d/%d' % ((convmap == posmap).all(axis=1).sum(), N))

# --- MM_PLAN 公式 ---
mm = np.zeros((N, 2), np.int64)
for t in range(N):
    yb, xb, dy, dx = (t//4) % NB, (t//4)//NB % NB, (t//2) % 2, t % 2
    yb = (t//(4*NB)) % NB; xb = (t//4) % NB
    mm[t] = (2*xb+dx, 2*yb+dy)
print('MM_PLAN 公式 与真值一致 %d/%d' % ((mm == posmap).all(axis=1).sum(), N))
# 交换 x/y 与 bit 顺序的变体
for name, f in [
    ('t=(y*NB+x)*4+dy*2+dx  (dx fastest)', lambda t: ((t//4) % NB, (t//(4*NB)) % NB, (t//2) % 2, t % 2)),
    ('t=(y*NB+x)*4+dx*2+dy  (dy fastest)', lambda t: ((t//4) % NB, (t//(4*NB)) % NB, t % 2, (t//2) % 2)),
    ('t=(x*NB+y)*4+dy*2+dx', lambda t: ((t//(4*NB)) % NB, (t//4) % NB, (t//2) % 2, t % 2)),
    ('t=(x*NB+y)*4+dx*2+dy', lambda t: ((t//(4*NB)) % NB, (t//4) % NB, t % 2, (t//2) % 2)),
]:
    m = np.zeros((N, 2), np.int64)
    for t in range(N):
        xb, yb, dy, dx = f(t)
        m[t] = (2*xb+dx, 2*yb+dy)
    print('  %-38s 一致 %4d/%d' % (name, (m == posmap).all(axis=1).sum(), N))
