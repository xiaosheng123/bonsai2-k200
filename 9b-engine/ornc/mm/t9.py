import numpy as np, struct, os
import ggufload, gg
G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
D = '/tmp/gt/T2_shapes'

def rrd(name):
    b = open(os.path.join(D, name), 'rb').read()
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b[:20])
    return np.frombuffer(b[20:], dtype='<f4').reshape(ne1, ne0).T

X = rrd('inp_pos_emb.bin').astype(np.float32)
lnw = G.f32('v.blk.0.ln1.weight'); lnb = G.np('v.blk.0.ln1.bias').astype(np.float32)
mean = X.mean(axis=0, dtype=np.float32)
d = X - mean
var = (d*d).mean(axis=0, dtype=np.float32)
ys = (d*(1.0/np.sqrt(var+1e-6)).astype(np.float32))*lnw[:, None]
ref = rrd('norm_w-0.bin')
print('ys[:5,0]', ys[:5, 0])
print('ref[:5,0]', ref[:5, 0])
print('ys max', np.abs(ys).max(), 'ref max', np.abs(ref).max())
print('relrms', gg.relrms(ys, ref))
# 是否整体比例?
r = (ref*ys).sum()/(ys*ys).sum()
print('最佳比例 %.5f  relrms(ys*r,ref)=%.4f%%' % (r, gg.relrms(ys*r, ref)))
# 是否只是通道顺序不同 -> 逐 token 匹配
for t in (0, 5):
    C = ys[:, t][:, None]*ref[:, t][None, :]
    n = np.linalg.norm(ys[:, t])*np.linalg.norm(ref[:, t])
    best = (C/n).max()
    print('token %d 最佳通道相关系数 %.6f' % (t, best))
# 若 ref 是 (x-mean)/std 不带 affine
ys2 = d*(1.0/np.sqrt(var+1e-6)).astype(np.float32)
print('relrms(无 affine, ref) = %.4f%%' % gg.relrms(ys2, ref))
print('relrms(带 affine+bias, ref) = %.4f%%' % gg.relrms(ys2*lnw[:, None]+lnb[:, None], ref))
# 也许 ref 是别的层的输入?
for nm in ('inp_pos_emb.bin', 'ffn_inp_normed-0.bin'):
    print('relrms(ref, %s) = %.4f%%' % (nm, gg.relrms(ref, rrd(nm))))
# 逐列比较 ref 的第 0 列是否与 ys 的某列相同
best = (ref[:, 0][:, None]*ys).sum(0)/np.maximum(np.linalg.norm(ref[:, 0])*np.linalg.norm(ys, axis=0), 1e-30)
print('ref 第0列与 ys 各列相关最大 %.6f (token %d)' % (best.max(), int(best.argmax())))
