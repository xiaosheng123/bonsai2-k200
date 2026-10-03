import numpy as np, sys, os
sys.path.insert(0,'/home/caden/ornc/mm')
import ggufload, gg
VD = '/tmp/vd512'
G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
def rr(A,B):
    A=np.asarray(A,np.float64);B=np.asarray(B,np.float64)
    return 100.0*np.sqrt(((A-B)**2).sum()/max((B*B).sum(),1e-30))
def rd(name):
    b=open('/tmp/gt/T2_shapes/'+name,'rb').read()
    t,ne0,ne1,ne2,ne3=__import__('struct').unpack('<5i',b[:20])
    return np.frombuffer(b[20:],dtype='<f4').reshape(ne1,ne0).T
# C++ 侧
conv_c = np.fromfile(VD+'/conv.f32',dtype='<f4').reshape(1024,1152)     # [m][oc], m = ox+32*oy
grid_c = np.fromfile(VD+'/gridi.f32',dtype='<f4').reshape(32*32,1152)
inp_c  = np.fromfile(VD+'/inp.f32',dtype='<f4').reshape(1024,1152).T
mw = np.fromfile(VD+'/mapw.f32',dtype='<f4').astype(int)
mh = np.fromfile(VD+'/maph.f32',dtype='<f4').astype(int)
# numpy 参考侧
vt = gg.VT(G)
W0 = gg.mkarr(G.np('v.patch_embd.weight')); W1 = gg.mkarr(G.np('v.patch_embd.weight.1'))
bias = G.np('v.patch_embd.bias').astype(np.float32)
pix,W,H = vt.load_img('/home/caden/ornc/mmtests/img_shapes.png')
conv_n = gg.mat(vt.conv_patch(gg.mk(pix,(W,H,3,1)),W0,W1,bias,32,32))[:,:,:,0].reshape(1024,1152)
grid_n = vt.pos_grid(32,32).reshape(1024,1152)
idx = (np.arange(32)[:,None] + 32*np.arange(32)[None,:]).ravel()   # conv 行 m = ox + 32*oy
print('conv  C++ vs numpy(浮点) relrms = %.4f%%' % rr(conv_c[idx], conv_n))
print('inp  C++ vs 真值 inp_pos_emb relrms = %.4f%%' % rr(inp_c, pe))
print('grid  C++ vs numpy relrms = %.4f%%' % rr(grid_c, grid_n))
# 排列: 用 numpy 的链验证 mapw/maph
# 由 numpy 网格直接反查每个 token 的位置 (精确)
Pb = rd('patch_bias.bin'); pe = rd('inp_pos_emb.bin')
diff = pe - Pb
M2 = np.zeros((1024,2),np.int64)
for t in range(1024):
    d2 = ((vt.pos_grid(32,32) - diff[:,t][None,None,:])**2).sum(axis=2)
    i = int(np.argmin(d2)); M2[t] = (i//32, i%32)
print('排列一致 (C++ vs numpy 反查): %d/1024' % int(((M2[:,0]==mw)&(M2[:,1]==mh)).sum()))
print('  期望 t=0..7:', [tuple(M2[t]) for t in range(8)], ' C++:', list(zip(mw[:8],mh[:8])))
# inp 一致性
g1 = np.stack([conv_c[mw[t]+32*mh[t]] + bias for t in range(1024)],1)
g2 = np.stack([grid_c[mw[t]+32*mh[t]] for t in range(1024)],1)
print('inp C++ vs (conv+grid+bias) relrms = %.4f%%' % rr(inp_c, g1+g2))

# 若用 numpy 的 map 重排 conv+grid
h1 = np.stack([conv_n[M2[t][0]+32*M2[t][1]] for t in range(1024)],1)
print('numpy conv[正确map]+bias vs 真值 patch_bias = %.4f%%' % rr(h1+bias[:,None], Pb))
