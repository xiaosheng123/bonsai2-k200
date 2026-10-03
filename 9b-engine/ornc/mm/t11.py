"""搜 RoPE 变体: 用已验证的 Q (pre-rope) 与 Qcur_rope-0 真值对拍"""
import numpy as np, struct, os, itertools
import ggufload, gg
G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
D = '/tmp/gt/T2_shapes'
NE = 1152; NH = 16; DH = 72

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
Qpre = np.ascontiguousarray(qkv[:, :NE].T)          # (1152, 1024) 行 = h*72+d

# 位置: 从 pos 网格反查 (已验证)
vt = gg.VT(G); grid = vt.pos_grid(32, 32)
Pb = rrd('patch_bias.bin')
P = np.zeros((1024, 2), np.int64)
for t in range(1024):
    v = X[:, t] - Pb[:, t]
    d2 = ((grid - v[None, None, :])**2).sum(axis=2)
    i = int(np.argmin(d2)); P[t] = (i//32, i % 32)
prow = P[:, 1].astype(np.float64); pcol = P[:, 0].astype(np.float64)

f, ne = raw('Qcur_rope-0.bin')
ref = f.reshape(1024, NE).T          # 连续 [72,16,1024]: flat = d + 72h + 1152t


def rope_variant(Q, mode, ts_pow, pair, pswap, sign):
    ts = 10000.0**(-ts_pow)
    out = Q.copy().astype(np.float32)
    T = Q.shape[1]
    th = (ts**np.arange(36, dtype=np.float64)).astype(np.float64)
    p1 = pcol if pswap else prow
    p2 = prow if pswap else pcol
    sec = 18
    for h in range(NH):
        blk = Q[h*DH:(h+1)*DH, :]
        if pair == 'half':
            a = blk[:36, :]; b = blk[36:, :]
            oa = np.empty_like(a); ob = np.empty_like(b)
        else:
            a = blk[0::2, :]; b = blk[1::2, :]
            oa = np.empty_like(a); ob = np.empty_like(b)
        for ic in range(36):
            if mode == 'sect':
                p = p1 if ic < sec else p2
                ph = ic if ic < sec else ic - sec
            elif mode == 'row':
                p = prow; ph = ic
            elif mode == 'col':
                p = pcol; ph = ic
            theta = (p*th[ph]).astype(np.float32)
            c = np.cos(theta); s = np.sin(theta)*sign
            oa[ic] = a[ic]*c - b[ic]*s
            ob[ic] = a[ic]*s + b[ic]*c
        if pair == 'half':
            out[h*DH:h*DH+36, :] = oa; out[h*DH+36:(h+1)*DH, :] = ob
        else:
            idx = np.arange(h*DH, h*DH+DH)
            out[idx[0::2], :] = oa; out[idx[1::2], :] = ob
    return out


best = []
for mode in ('sect', 'row', 'col'):
    for ts_pow in (2.0/36.0, 2.0/72.0):
        for pair in ('half', 'interleaved'):
            for pswap in (False, True):
                for sign in (1.0, -1.0):
                    o = rope_variant(Qpre, mode, ts_pow, pair, pswap, sign)
                    r = gg.relrms(o, ref)
                    best.append((r, mode, ts_pow, pair, pswap, sign))
best.sort()
for b in best[:10]:
    print('relrms=%8.4f%%  mode=%s ts=2/%.0f pair=%s pswap=%s sign=%+0.0f' % (
        b[0], b[1], 1.0/b[2] if b[2] else 0, b[3], b[4], b[5]))
print('Qpre 与 ref 的原始 relrms = %.4f%%' % gg.relrms(Qpre, ref))
