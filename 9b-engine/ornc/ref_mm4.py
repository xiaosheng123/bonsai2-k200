"""ref_mm4.py — 逐层"新鲜误差"(从真值输入单独跑一层) vs "自由跑", 判定 3.72% 是混沌放大还是系统性算子差异
   输入直接取真值 inp_pos_emb (= 层0输入), 不涉及头部/排列。"""
import numpy as np, struct, os, sys, time
sys.path.insert(0, '/home/caden/ornc/mm')
import ggufload
G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
D = '/tmp/gt/T2_shapes'
NL, NH, DH, NE, FFN, EPS = 27, 16, 72, 1152, 4304, 1e-6
SQ = np.float32(1.0/np.sqrt(72.0))

def rr(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    return 100.0*np.sqrt(((A-B)**2).sum()/max((B*B).sum(), 1e-30))

def rrd(name):
    b = open(os.path.join(D, name), 'rb').read()
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b[:20])
    n = ne0*ne1*ne2*ne3
    return np.frombuffer(b[20:20+4*n], dtype='<f4').reshape(ne1, ne0).T

def LN(X, il, which, eps=EPS):
    w = G.f32('v.blk.%d.%s.weight' % (il, which)); b = G.np('v.blk.%d.%s.bias' % (il, which)).astype(np.float32)
    mean = X.mean(axis=0, dtype=np.float32); d = X - mean
    var = (d*d).mean(axis=0, dtype=np.float32)
    ys = (d*(1.0/np.sqrt(var+eps)).astype(np.float32))*w[:, None]
    return (ys + b[:, None]).astype(np.float32)

def rope(Q, p_row, p_col):
    ts = 10000.0**(-2.0/36.0)
    th = (ts**np.arange(36, dtype=np.float64)).astype(np.float32)
    out = Q.copy()
    T = Q.shape[1]
    for h in range(NH):
        v = Q[h*DH:(h+1)*DH, :]
        a = v[:36, :]; b = v[36:, :]
        ca = np.empty((36, T), np.float32); sa = np.empty((36, T), np.float32)
        for ic in range(36):
            p = p_row if ic < 18 else p_col
            e = ic if ic < 18 else ic-18
            theta = (p*th[e]).astype(np.float32)
            ca[ic] = np.cos(theta); sa[ic] = np.sin(theta)
        out[h*DH:h*DH+36, :] = (a*ca - b*sa).astype(np.float32)
        out[h*DH+36:(h+1)*DH, :] = (a*sa + b*ca).astype(np.float32)
    return out

p_row = np.load('/tmp/pos_row.npy').astype(np.float32)
p_col = np.load('/tmp/pos_col.npy').astype(np.float32)

def layer(X, il):
    Y = LN(X, il, 'ln1')
    qkv = (Y.T @ G.f32('v.blk.%d.attn_qkv.weight' % il).T).astype(np.float32) + G.np('v.blk.%d.attn_qkv.bias' % il).astype(np.float32)
    Qb = qkv[:, :NE].T.astype(np.float32); Kb = qkv[:, NE:2*NE].T.astype(np.float32); Vb = qkv[:, 2*NE:].T.astype(np.float32)
    Qr = rope(Qb, p_row, p_col); Kr = rope(Kb, p_row, p_col)
    T = X.shape[1]
    O = np.empty((NE, T), np.float32)
    for h in range(NH):
        q = Qr[h*DH:(h+1)*DH]; k = Kr[h*DH:(h+1)*DH]; v = Vb[h*DH:(h+1)*DH]
        S = (k.T @ q).astype(np.float32)*SQ
        mx = S.max(axis=0)
        E = np.exp((S - mx[None, :]).astype(np.float32))
        Sm = (E/E.sum(axis=0, dtype=np.float32)[None, :]).astype(np.float32)
        O[h*DH:(h+1)*DH, :] = (v @ Sm).astype(np.float32)
    Wo = G.f32('v.blk.%d.attn_out.weight' % il); bo = G.np('v.blk.%d.attn_out.bias' % il).astype(np.float32)
    Ao = (Wo @ O + bo[:, None]).astype(np.float32)
    X = (X + Ao).astype(np.float32)
    Y2 = LN(X, il, 'ln2')
    up = (Y2.T @ G.f32('v.blk.%d.ffn_up.weight' % il).T).astype(np.float32) + G.np('v.blk.%d.ffn_up.bias' % il).astype(np.float32)
    g = (0.5*up*(1.0 + np.tanh(np.float32(np.sqrt(2.0/np.pi))*(up + 0.044715*up**3)))).astype(np.float32)
    dn = (g @ G.f32('v.blk.%d.ffn_down.weight' % il).T).astype(np.float32) + G.np('v.blk.%d.ffn_down.bias' % il).astype(np.float32)
    X = (X + dn.T).astype(np.float32)
    return X, Ao, Y2

t0 = time.time()
X = rrd('inp_pos_emb.bin')            # 真值层0输入
XT = X.copy()
print('%-5s %-12s %-12s %-12s %-12s' % ('layer', '新鲜(层输出)', '自由跑(层输出)', '新鲜(attn_out)', '新鲜(LN2)'), flush=True)
for il in range(NL):
    tp = rrd('layer_out-%d.bin' % il)
    ta = rrd('attn_out-%d.bin' % il)
    tl = rrd('ffn_inp_normed-%d.bin' % il)
    Xf, Ao_f, Y2_f = layer(XT, il)      # 从真值输入跑这一层 (新鲜)
    X, _, _ = layer(X, il)              # 自由跑
    print('%-5d %-12.5f %-12.5f %-12.5f %-12.5f' % (il, rr(Xf, tp), rr(X, tp), rr(Ao_f, ta), rr(Y2_f, tl)), flush=True)
    XT = tp                             # 下一层的真值输入
print('总用时 %.1fs' % (time.time()-t0), flush=True)
