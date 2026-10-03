"""用已验证的链路从 pos 侧定出排列, 再单独检查 conv 值"""
import numpy as np, struct, os, sys
import ggufload, gg

G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
IMG = '/home/caden/ornc/mmtests/img_shapes.png'
D = '/tmp/gt/T2_shapes'


def rd(name):
    b = open(os.path.join(D, name), 'rb').read()
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b[:20])
    return np.frombuffer(b[20:], dtype='<f4').reshape(ne1, ne0).T


vt = gg.VT(G)
conv_part, pos_part = vt.head(IMG)
pb = rd('patch_bias.bin'); pe = rd('inp_pos_emb.bin')
bias = G.np('v.patch_embd.bias').astype(np.float32)
grid = vt.pos_grid(32, 32)                # (32,32,1152) [x,y,c]
Gf = gg.flat_of(grid)

# 每个 token 用 pos_part 定位置
MAP = np.zeros((1024, 2), np.int64)
err = []
for t in range(1024):
    v = pos_part[:, t]
    d2 = ((grid - v[None, None, :])**2).sum(axis=2)
    i = int(np.argmin(d2)); MAP[t] = (i // 32, i % 32)
    err.append(np.sqrt(d2.ravel()[i])/(np.linalg.norm(v)+1e-30))
print('pos 定位置: 中位误差 %.2e 最大 %.2e' % (np.median(err), max(err)))
print('t=0..19 (x,y):', [tuple(int(z) for z in MAP[t]) for t in range(20)])

# conv 单独检查
pix, W, H = vt.load_img(IMG)
inp_raw = gg.mk(pix, (W, H, 3, 1))
W0 = gg.mkarr(G.np('v.patch_embd.weight')); W1 = gg.mkarr(G.np('v.patch_embd.weight.1'))
conv = vt.conv_patch(inp_raw, W0, W1, bias, 32, 32)
C = gg.mat(conv)[:, :, :, 0]              # (32,32,1152)
mine = np.stack([C[MAP[t][0], MAP[t][1]] for t in range(1024)], 1)
print('relrms(conv[map]+bias, patch_bias) = %.4f%%' % gg.relrms(mine, pb))
print('max|conv| = %.5f  max|patch_bias-bias| = %.5f' % (np.abs(C).max(), np.abs(pb-bias[:, None]).max()))
# conv_part 反推
mine2 = np.stack([conv_part[:, t] for t in range(1024)], 1)
print('relrms(chain(conv), conv[map]) = %.4f%%' % gg.relrms(conv_part, mine))
# 单 token 直查
for t in (0, 1, 2, 100, 500, 1023):
    x, y = MAP[t]
    print(' t=%4d map=(%2d,%2d) conv|x-y|最大差 %.5f' % (t, x, y, np.abs(C[x, y] - (pb[:, t]-bias)).max()))
