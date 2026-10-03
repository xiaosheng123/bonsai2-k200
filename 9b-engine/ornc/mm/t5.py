"""定排列: 用真值 patch_bias 暴力反推每个 token 的 (x,y), 与候选链路的排列逐一对照"""
import numpy as np, struct, os
from PIL import Image
import ggufload, gg

G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
D = '/tmp/gt/T2_shapes'
ow = oh = 32
N = ow*oh


def rd(name):
    b = open(os.path.join(D, name), 'rb').read()
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b[:20])
    return np.frombuffer(b[20:], dtype='<f4').reshape(ne1, ne0).T


pb = rd('patch_bias.bin'); pe = rd('inp_pos_emb.bin')
bias = G.np('v.patch_embd.bias').astype(np.float32)
ref = pb - bias[:, None]

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
V = (conv + bias).reshape(N, 1152)          # V[x + 32*y] 索引; 我的数组序 (x,y)

# --- 真值排列 T ---
T = np.zeros(N, np.int64)
for t in range(N):
    C = V @ ref[:, t]
    T[t] = int(np.argmax(C))
print('T[0:32]  (x = T%%32, y = T//32):')
print('  ', ['(%d,%d)' % (T[t] % ow, T[t]//ow) for t in range(32)])

# --- 候选 A 排列 ---
syn = np.zeros((ow, oh, 1152), np.float32)
for x in range(ow):
    for y in range(oh):
        syn[x, y] = x*1000.0 + y
t1 = gg.cont(gg.permute(gg.mk(syn.ravel().copy(), (ow, oh, 1152, 1)), 1, 2, 0, 3), (2304, ow//2, oh, 1))
t2 = gg.cont(t1, (2304, ow//2, 2, oh//2))
t3 = gg.permute(t2, 0, 2, 1, 3)
out = gg.mat(gg.cont(t3, (1152, N, 1)))[:, :, 0, 0]
code = out[0, :].astype(np.float64)
A = np.round(code/1000).astype(np.int64)*32 + np.round(code % 1000).astype(np.int64)
print('A[0:32]:', ['(%d,%d)' % (A[t] % ow, A[t]//ow) for t in range(32)])
print('A == T ? %d/%d' % ((A == T).sum(), N))
# 是否 T = "A 的 x/y 互换"
Asw = np.array([( (A[t]//ow)*32 + (A[t] % ow) ) for t in range(N)])
print('A(x/y互换) == T ? %d/%d' % ((Asw == T).sum(), N))

# --- pos-emb 经验排列 emp (我的 grid 标注) ---
posw = G.np('v.position_embd.weight').astype(np.float32).reshape(48, 48, 1152)
grid = gg.VT(G).interp_align_corners(posw, 48, ow, oh)
diff = pe - pb
E = np.zeros(N, np.int64)
for t in range(N):
    d2 = ((grid - diff[:, t][None, None, :])**2).sum(axis=2)
    E[t] = int(np.argmin(d2))       # idx = ix + 32*iy  with ix=dim0
print('E[0:32]:', ['(%d,%d)' % (E[t] % ow, E[t]//ow) for t in range(32)])
print('E == T ? %d/%d' % ((E == T).sum(), N))
gridT = np.ascontiguousarray(grid.transpose(1, 0, 2))
E2 = np.zeros(N, np.int64)
for t in range(N):
    d2 = ((gridT - diff[:, t][None, None, :])**2).sum(axis=2)
    E2[t] = int(np.argmin(d2))
print('E(transposed grid) == T ? %d/%d' % ((E2 == T).sum(), N))
print('A == E2 ? %d/%d' % ((A == E2).sum(), N))
