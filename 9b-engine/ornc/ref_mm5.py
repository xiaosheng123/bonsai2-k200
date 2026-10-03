"""ref_mm5.py — 假设验证: llama.cpp 对 bf16 权重做 matmul 时会把激活转成 BF16 (vec_dot_type=BF16)。
   若在激活上复刻同一 BF16 舍入, 逐层"新鲜误差"应从 ~0.08% 塌到 ~1e-5。"""
import numpy as np, struct, os, sys, time
sys.path.insert(0, '/home/caden/ornc/mm')
import ggufload
G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
D = '/tmp/gt/T2_shapes'
NL, NH, DH, NE, EPS = 27, 16, 72, 1152, 1e-6
SQ = np.float32(1.0/np.sqrt(72.0))
BF = int(os.environ.get('ENV_BF16', '1'))
ONLY = int(os.environ.get('NLAYER', '3'))

def to_bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((u >> 16) & 1).astype(np.uint32) + np.uint32(0x7FFF)
    return ((u + r) & np.uint32(0xFFFF0000)).view(np.float32)

def mm(W, x):          # ggml_mul_mat(W, x): src1 转 BF16 (仅当 W 是 bf16 权重)
    return (to_bf16(x) if BF else x) @ W.T

def rr(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    return 100.0*np.sqrt(((A-B)**2).sum()/max((B*B).sum(), 1e-30))

def rrd(name):
    b = open(os.path.join(D, name), 'rb').read()
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b[:20])
    return np.frombuffer(b[20:20+4*ne0*ne1*ne2*ne3], dtype='<f4').reshape(ne1, ne0).T

def LN(X, il, which, eps=EPS):
    w = G.f32('v.blk.%d.%s.weight' % (il, which)); b = G.np('v.blk.%d.%s.bias' % (il, which)).astype(np.float32)
    mean = X.mean(axis=0, dtype=np.float32); d = X - mean
    var = (d*d).mean(axis=0, dtype=np.float32)
    ys = (d*(1.0/np.sqrt(var+eps)).astype(np.float32))*w[:, None]
    return (ys + b[:, None]).astype(np.float32)

def rope(Q, p_row, p_col):
    th = (10000.0**(-2.0/36.0))**np.arange(36, dtype=np.float64)
    out = Q.copy(); T = Q.shape[1]
    for h in range(NH):
        a = Q[h*DH:h*DH+36, :]; b = Q[h*DH+36:(h+1)*DH, :]
        ca = np.empty((36, T), np.float32); sa = np.empty((36, T), np.float32)
        for ic in range(36):
            p = p_row if ic < 18 else p_col
            e = ic if ic < 18 else ic-18
            theta = (p*np.float32(th[e])).astype(np.float32)
            ca[ic] = np.cos(theta); sa[ic] = np.sin(theta)
        out[h*DH:h*DH+36, :] = (a*ca - b*sa).astype(np.float32)
        out[h*DH+36:(h+1)*DH, :] = (a*sa + b*ca).astype(np.float32)
    return out

p_row = np.load('/tmp/pos_row.npy').astype(np.float32)
p_col = np.load('/tmp/pos_col.npy').astype(np.float32)

def layer(X, il):
    Y = LN(X, il, 'ln1')
    qkv = mm(G.f32('v.blk.%d.attn_qkv.weight' % il), Y.T).astype(np.float32) + G.np('v.blk.%d.attn_qkv.bias' % il).astype(np.float32)
    Qb = qkv[:, :NE].T.astype(np.float32); Kb = qkv[:, NE:2*NE].T.astype(np.float32); Vb = qkv[:, 2*NE:].T.astype(np.float32)
    Qr = rope(Qb, p_row, p_col); Kr = rope(Kb, p_row, p_col)
    T = X.shape[1]; O = np.empty((NE, T), np.float32)
    for h in range(NH):
        q = Qr[h*DH:(h+1)*DH]; k = Kr[h*DH:(h+1)*DH]; v = Vb[h*DH:(h+1)*DH]
        S = (k.T @ q).astype(np.float32)*SQ
        E = np.exp((S - S.max(axis=0)[None, :]).astype(np.float32))
        Sm = (E/E.sum(axis=0, dtype=np.float32)[None, :]).astype(np.float32)
        O[h*DH:(h+1)*DH, :] = (v @ Sm).astype(np.float32)
    Ao = (mm(G.f32('v.blk.%d.attn_out.weight' % il), O.T).astype(np.float32) + G.np('v.blk.%d.attn_out.bias' % il).astype(np.float32)).T
    X = (X + Ao).astype(np.float32)
    Y2 = LN(X, il, 'ln2')
    up = mm(G.f32('v.blk.%d.ffn_up.weight' % il), Y2.T).astype(np.float32) + G.np('v.blk.%d.ffn_up.bias' % il).astype(np.float32)
    g = (0.5*up*(1.0 + np.tanh(np.float32(np.sqrt(2.0/np.pi))*(up + 0.044715*up**3)))).astype(np.float32)
    dn = mm(G.f32('v.blk.%d.ffn_down.weight' % il), g).astype(np.float32) + G.np('v.blk.%d.ffn_down.bias' % il).astype(np.float32)
    X = (X + dn.T).astype(np.float32)
    return X, Ao, Y2

print('ENV_BF16 =', BF, flush=True)
XT = rrd('inp_pos_emb.bin'); X = XT.copy()
print('%-5s %-14s %-14s %-14s' % ('layer', '新鲜(层输出)', '自由跑(层输出)', '新鲜(attn)'), flush=True)
for il in range(ONLY):
    tp = rrd('layer_out-%d.bin' % il); ta = rrd('attn_out-%d.bin' % il)
    Xf, Aof, _ = layer(XT, il)
    X, _, _ = layer(X, il)
    print('%-5d %-14.6f %-14.6f %-14.6f' % (il, rr(Xf, tp), rr(X, tp), rr(Aof, ta)), flush=True)
    XT = tp
