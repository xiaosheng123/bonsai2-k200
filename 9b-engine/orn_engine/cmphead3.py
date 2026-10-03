import numpy as np, struct, sys
sys.path.insert(0, '/home/caden/ornc/mm')
import ggufload, gg
VD = '/tmp/vd512'
G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')


def rr(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    return 100.0*np.sqrt(((A-B)**2).sum()/max((B*B).sum(), 1e-30))


def rd(name):
    b = open('/tmp/gt/T2_shapes/' + name, 'rb').read()
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b[:20])
    return np.frombuffer(b[20:], dtype='<f4').reshape(ne1, ne0).T


Pb = rd('patch_bias.bin')
bias = G.np('v.patch_embd.bias').astype(np.float32)
conv_c = np.fromfile(VD+'/conv.f32', dtype='<f4').reshape(1024, 1152)
grid_c = np.fromfile(VD+'/gridi.f32', dtype='<f4').reshape(1024, 1152)
mw = np.fromfile(VD+'/mapw.f32', dtype='<f4').astype(int)
mh = np.fromfile(VD+'/maph.f32', dtype='<f4').astype(int)

# numpy 参考 conv (已验证)
im = gg.VT(G).load_img('/home/caden/ornc/mmtests/img_shapes.png')
pix, W, H = im
x3 = gg.from_flat(pix, (W, H, 3, 1))[:, :, :, 0]
K = gg.from_flat(G.flat('v.patch_embd.weight'), (16, 16, 3, 1152)) + \
    gg.from_flat(G.flat('v.patch_embd.weight.1'), (16, 16, 3, 1152))
cn = np.zeros((32, 32, 1152), np.float32)
for i in range(16):
    for j in range(16):
        cn += x3[i::16, j::16, :] @ K[i, j]
print('numpy conv ok', cn.shape, cn.max())
# numpy conv 平面: (ox,oy) -> 行 = ox*32+oy
cn_flat = cn.reshape(1024, 1152)
sel = np.arange(32)[:, None] + 32*np.arange(32)[None, :]      # cn 行 = ox*32+oy

for name, cf in (('C++行=ox+32oy', conv_c), ('C++行=ox*32+oy(转置还原)', conv_c[np.argsort(sel.ravel()) if False else (np.arange(32)[None, :] + 32*np.arange(32)[:, None]).ravel()])):
    m = np.stack([cf[mw[t] + 32*mh[t]] for t in range(1024)], 1)
    print('%-28s conv[map]+bias vs patch_bias = %.4f%%' % (name, rr(m + bias[:, None], Pb)))
    m2 = np.stack([cf[mw[t]*32 + mh[t]] for t in range(1024)], 1)
    print('%-28s conv[map*32]  +bias vs patch_bias = %.4f%%' % (name, rr(m2 + bias[:, None], Pb)))
# numpy conv 直接 map
mn = np.stack([cn_flat[mw[t]*32+mh[t]] for t in range(1024)], 1)
print('numpy conv[map ox*32+oy]+bias vs patch_bias = %.4f%%' % rr(mn + bias[:, None], Pb))
mn2 = np.stack([cn_flat[mw[t]+32*mh[t]] for t in range(1024)], 1)
print('numpy conv[map ox+32oy]+bias  vs patch_bias = %.4f%%' % rr(mn2 + bias[:, None], Pb))
# C++ conv 与 numpy conv 逐元素 (两种展平都试)
print('C++conv vs numpyconv 若同序(行=ox*32+oy假设): %.4f%%' % rr(conv_c.reshape(32,32,1152).transpose(1,0,2).reshape(1024,1152), cn_flat))
print('C++conv vs numpyconv 直接: %.4f%%' % rr(conv_c, cn_flat))
d = conv_c.reshape(1024, 1152) - cn_flat
print('C++conv - numpyconv 的 |.|均值 %.5f, conv |.|均值 %.5f' % (np.abs(d).mean(), np.abs(cn_flat).mean()))
print('C++conv 直方图范围', conv_c.min(), conv_c.max(), ' numpy', cn_flat.min(), cn_flat.max())
