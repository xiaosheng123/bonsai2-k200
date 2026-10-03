"""隔离 LN1+qkv: 用 Qcur-0 的原始视图 dump 直接对拍 Q (pre-rope)"""
import numpy as np, struct, os
import ggufload, gg
G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
D = '/tmp/gt/T2_shapes'
NE = 1152

def rrd(name):
    b = open(os.path.join(D, name), 'rb').read()
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b[:20])
    return np.frombuffer(b[20:], dtype='<f4').reshape(ne1, ne0).T

def raw(name):
    b = open(os.path.join(D, name), 'rb').read()
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b[:20])
    return np.frombuffer(b[20:], dtype='<f4'), (ne0, ne1, ne2, ne3)

X = rrd('inp_pos_emb.bin').astype(np.float32)
lnw = G.f32('v.blk.0.ln1.weight'); lnb = G.np('v.blk.0.ln1.bias').astype(np.float32)
mean = X.mean(axis=0, dtype=np.float32); d = X - mean
var = (d*d).mean(axis=0, dtype=np.float32)
Y = ((d*(1.0/np.sqrt(var+1e-6)).astype(np.float32))*lnw[:, None] + lnb[:, None]).astype(np.float32)
qkv = (Y.T @ G.f32('v.blk.0.attn_qkv.weight').T).astype(np.float32) + G.np('v.blk.0.attn_qkv.bias').astype(np.float32)
Qpre = qkv[:, :NE].T
f, ne = raw('Qcur-0.bin')
print('Qcur-0 dump ne', ne, 'len', len(f), 'tokens', len(f)//3456)
# 每 token 的 Q 段 = 该 token 的 3456 块的前 1152
nt = len(f)//3456
refQ = f[:nt*3456].reshape(nt, 3456)[:, :NE].T
Qpre = Qpre[:, :nt]
print('比较 token 数', nt)
print('relrms(Qpre_mine, Qcur-0 前1152段) = %.4f%%' % gg.relrms(Qpre, refQ))
# 若通道顺序不同? 逐 token 相关
for t in (0, 1, 100):
    n = np.linalg.norm(Qpre[:, t])*np.linalg.norm(refQ[:, t])
    print('  token %d 相关 %.6f' % (t, float((Qpre[:, t]*refQ[:, t]).sum()/max(n, 1e-30))))
# norm_w-0 是什么?
nw = rrd('norm_w-0.bin')
print('relrms(带bias Y, norm_w-0) = %.4f%%' % gg.relrms(Y, nw))
print('relrms(无bias, norm_w-0)   = %.4f%%' % gg.relrms(Y-lnb[:, None], nw))
print('relrms(LN 无affine, nw)   = %.4f%%' % gg.relrms(d*(1.0/np.sqrt(var+1e-6)).astype(np.float32), nw))
print('relrms(bias, nw)          = %.4f%%  (bias 本身)' % gg.relrms(np.repeat(lnb[:, None], 1024, 1), nw))
Pb = rrd('patch_bias.bin')
for nm, Z in (('patch_bias', Pb), ('inp_pos_emb', X), ('inp_pos_emb+patch_bias', X+Pb)):
    m = Z.mean(axis=0, dtype=np.float32); dd = Z - m
    vv = (dd*dd).mean(axis=0, dtype=np.float32)
    Zy = ((dd*(1.0/np.sqrt(vv+1e-6)).astype(np.float32))*lnw[:, None] + lnb[:, None]).astype(np.float32)
    print('relrms(LN1(%s), nw) = %.4f%%' % (nm, gg.relrms(Zy, nw)))
