"""诊断 conv 值: 用经验(位置)排列对齐, 逐个变体比对"""
import numpy as np, struct, os
from PIL import Image
import ggufload, gg

G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
D = '/tmp/gt/T2_shapes'
ow = oh = 32


def rd(name):
    b = open(os.path.join(D, name), 'rb').read()
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b[:20])
    return np.frombuffer(b[20:], dtype='<f4').reshape(ne1, ne0).T


def relrms(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    return 100.0*np.sqrt(((A-B)**2).sum()/max((B*B).sum(), 1e-30))


pb = rd('patch_bias.bin'); pe = rd('inp_pos_emb.bin')
bias = G.np('v.patch_embd.bias').astype(np.float32)
ref = pb - bias[:, None]
diff = pe - pb

im = Image.open('/home/caden/ornc/mmtests/img_shapes.png').convert('RGB')
a = np.asarray(im, np.uint8).astype(np.float32)
H, W = a.shape[0], a.shape[1]
pix = np.ascontiguousarray(np.transpose((a/255.0 - 0.5)/0.5, (1, 0, 2))).ravel().copy()
x3 = pix.reshape(W, H, 3)
K = (gg.mat(gg.mk(G.np('v.patch_embd.weight').ravel().copy(), (16, 16, 3, 1152))) +
     gg.mat(gg.mk(G.np('v.patch_embd.weight.1').ravel().copy(), (16, 16, 3, 1152))))

# 经验位置排列
posw = G.np('v.position_embd.weight').astype(np.float32).reshape(48, 48, 1152)
grid = gg.VT(G).interp_align_corners(posw, 48, ow, ow)
emp = np.zeros((ow*ow, 2), np.int64)
for t in range(ow*ow):
    d2 = ((grid - diff[:, t][None, None, :])**2).sum(axis=2)
    i = int(np.argmin(d2)); emp[t] = (i//ow, i % ow)

refpos = ref.copy()
print('ref 与 conv 的绝对值多重集: ref[sum,sum2]=%.6f,%.6f  conv[..]=%.6f,%.6f' % (
    np.abs(ref).sum(), (ref*ref).sum(), 0, 0))

variants = {}
# V1 标准
v = np.zeros((ow, oh, 1152), np.float32)
for i in range(16):
    for j in range(16):
        v += x3[i::16, j::16, :] @ K[i, j]
variants['V1 std'] = v
# V2 核转置 (kp[i,j]=K[j,i])
v = np.zeros((ow, oh, 1152), np.float32)
for i in range(16):
    for j in range(16):
        v += x3[i::16, j::16, :] @ K[j, i]
variants['V2 kern^T'] = v
# V3 输入 x/y 交换 (即 x3T[H,W,3]) -> 输出 (oh,ow)
x3T = np.ascontiguousarray(pix.reshape(W, H, 3).transpose(1, 0, 2))
v = np.zeros((ow, oh, 1152), np.float32)
for i in range(16):
    for j in range(16):
        v += x3T[i::16, j::16, :] @ K[i, j]
variants['V3 in^T'] = v

for nm, v in variants.items():
    m = np.empty((1152, ow*oh), np.float32)
    for t in range(ow*oh):
        x, y = emp[t]
        m[:, t] = v[x, y] + bias
    print('%-12s 用经验位置排列 relrms=%.4f%%' % (nm, relrms(m, pb)))
    # 多重集
    print('              sorted|.| 前5: %s vs ref %s' % (
        np.sort(np.abs(v[:, :, :]).ravel())[-5:], np.sort(np.abs(ref).ravel())[-5:]))
    # 逐 token 与 ref 的最大相关
    C = v.reshape(ow*oh, 1152) @ ref          # (1024, 1024)
    nn = np.linalg.norm(v.reshape(ow*oh, 1152), axis=1)*np.linalg.norm(ref, axis=0)
    C = C/np.maximum(nn[:, None], 1e-30)
    best = C.max(axis=0)
    print('              每 token 最佳相关系数: 中位 %.5f 最小 %.5f' % (np.median(best), best.min()))
