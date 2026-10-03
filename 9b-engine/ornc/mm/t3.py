"""穷举 cont 语义候选, 对真值 dump 找唯一解释"""
import numpy as np, struct, os, itertools
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


pb = rd('patch_bias.bin'); bias = G.np('v.patch_embd.bias').astype(np.float32)
ref = pb - bias[:, None]

# ---- 真值 conv (im2col 序与其他序都试) ----
from PIL import Image
im = Image.open('/home/caden/ornc/mmtests/img_shapes.png').convert('RGB')
a = np.asarray(im, np.uint8).astype(np.float32)
H, W = a.shape[0], a.shape[1]
pix = np.ascontiguousarray(np.transpose((a/255.0 - 0.5)/0.5, (1, 0, 2))).ravel().copy()
x3 = pix.reshape(W, H, 3)
K = (gg.mat(gg.mk(G.np('v.patch_embd.weight').ravel().copy(), (16, 16, 3, 1152))) +
     gg.mat(gg.mk(G.np('v.patch_embd.weight.1').ravel().copy(), (16, 16, 3, 1152))))


def conv_naive():
    """直接法, 无 im2col 序歧义: conv[ox,oy,oc] = sum K[i,j,c,oc]*x[16ox+i,16oy+j,c]"""
    out = np.zeros((ow, oh, 1152), np.float32)
    for i in range(16):
        for j in range(16):
            blk = x3[i::16, j::16, :]                   # (ow,oh,3)
            out += blk @ K[i, j]                        # (ow,oh,3)@(3,1152)
    return out


conv = conv_naive()
print('conv 值域', conv.min(), conv.max())

# 经验排列
pe = rd('inp_pos_emb.bin'); diff = pe - pb
posw = G.np('v.position_embd.weight').astype(np.float32).reshape(48, 48, 1152)
grid = gg.VT(G).interp_align_corners(posw, 48, ow, ow)
emp = np.zeros((ow*ow, 2), np.int64)
for t in range(ow*ow):
    d2 = ((grid - diff[:, t][None, None, :])**2).sum(axis=2)
    i = int(np.argmin(d2)); emp[t] = (i//ow, i % ow)


def chainA(tens, fn):
    t1 = fn(gg.permute(tens, 1, 2, 0, 3), (2304, ow//2, oh, 1))
    t2 = gg.cont(t1, (2304, ow//2, 2, oh//2))
    t3 = gg.permute(t2, 0, 2, 1, 3)
    return gg.mat(gg.cont(t3, (1152, ow*oh, 1)))[:, :, 0, 0]


def cand_map(name, mk_tens, fn):
    out = chainA(mk_tens(), fn)
    R = relrms(out, ref)
    # 与经验排列一致性 (用合成码)
    syn = np.empty((ow, oh, 1152), np.float32)
    for x in range(ow):
        for y in range(oh):
            syn[x, y, :] = x*1000.0+y
    out2 = chainA(mk_tens(syn), fn)
    code = out2[0, :]
    xy = np.stack([np.round(code/1000).astype(np.int64), (code % 1000).astype(np.int64)], 1)
    nm = int((xy == emp).all(axis=1).sum())
    print('%-28s relrms=%8.4f%%  排列一致 %4d/%d' % (name, R, nm, ow*ow))
    return R, nm


def defA(syn=None):
    v = conv if syn is None else syn
    return gg.mk(np.ascontiguousarray(v).ravel(), (ow, oh, 1152, 1))


def defT(syn=None):
    v = conv if syn is None else syn
    return gg.mk(np.ascontiguousarray(v.transpose(1, 0, 2)).ravel(), (oh, ow, 1152, 1))


def defC(syn=None):
    # 候选 C: cont 退化为原始字节序
    v = conv if syn is None else syn
    a2 = np.ascontiguousarray(v).ravel().reshape(2304, ow//2, oh, 1)
    a3 = a2.reshape(2304, ow//2, 2, oh//2)
    t3 = gg.permute(gg.mk(a3.ravel(), (2304, ow//2, 2, oh//2)), 0, 2, 1, 3)
    return gg.cont(t3, (1152, ow*oh, 1))


cand_map('A: cont=逻辑线性序', defA, gg.cont)
cand_map('T: A + conv 转置 (dim0=oh)', defT, gg.cont)
cand_map('C: cont=原始字节序', defC, lambda t, ne: t)
