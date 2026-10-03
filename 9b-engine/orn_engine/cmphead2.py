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


Pb = rd('patch_bias.bin'); pe = rd('inp_pos_emb.bin')
bias = G.np('v.patch_embd.bias').astype(np.float32)
conv_c = np.fromfile(VD+'/conv.f32', dtype='<f4').reshape(1024, 1152)
grid_c = np.fromfile(VD+'/gridi.f32', dtype='<f4').reshape(32*32, 1152)
inp_c = np.fromfile(VD+'/inp.f32', dtype='<f4').reshape(1024, 1152).T
mw = np.fromfile(VD+'/mapw.f32', dtype='<f4').astype(int)
mh = np.fromfile(VD+'/maph.f32', dtype='<f4').astype(int)
idx = mw + 32*mh                     # C++ conv 行约定
mine = np.stack([conv_c[idx[t]] for t in range(1024)], 1)
gridg = np.stack([grid_c[idx[t]] for t in range(1024)], 1)
print('★ C++ conv[map]+bias vs 真值 patch_bias        = %.4f%%' % rr(mine + bias[:, None], Pb))
print('★ C++ grid[map]      vs 真值 (inp-pos-pb)      = %.4f%%' % rr(gridg, pe - Pb))
print('★ C++ inp[map 自洽]  vs conv+grid+bias          = %.4f%%' % rr(inp_c, mine + gridg + bias[:, None]))
print('★ C++ inp            vs 真值 inp_pos_emb        = %.4f%%' % rr(inp_c, pe))
# numpy 参考侧
vt = gg.VT(G)
conv_n = gg.mat(vt.conv_patch(gg.mk(*vt.load_img('/home/caden/ornc/mmtests/img_shapes.png')), 
                              gg.mkarr(G.np('v.patch_embd.weight')), gg.mkarr(G.np('v.patch_embd.weight.1')), bias, 32, 32))[:, :, :, 0].reshape(1024, 1152)
# numpy conv 行 = ox*32+oy  -> 取同一 patch
mine_n = np.stack([conv_n[mw[t]*32 + mh[t]] for t in range(1024)], 1)
print('   (对照) numpy conv[map]+bias vs patch_bias     = %.4f%%' % rr(mine_n + bias[:, None], Pb))
print('   (对照) C++ conv vs numpy conv (同 patch 序)   = %.4f%%' % rr(
    np.stack([conv_c[idx[t]] for t in range(1024)], 1), mine_n))
