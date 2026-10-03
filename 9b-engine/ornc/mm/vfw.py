"""视觉塔参考前向 (numpy, 主机): 逐层与真值桩 dump 对拍"""
import numpy as np, struct, os, sys, time
import ggufload, gg

G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
D = os.environ.get('GTDIR', '/tmp/gt/T2_shapes')
NL = 27
NH = 16; DH = 72; NE = 1152; FFN = 4304; EPS = 1e-6
SQ = 1.0/np.sqrt(72.0)


def rd(name):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        return None
    b = open(p, 'rb').read()
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b[:20])
    return np.frombuffer(b[20:], dtype='<f4'), (ne0, ne1, ne2, ne3)


def rrd(name):
    r = rd(name)
    if r is None:
        return None
    f, ne = r
    return f.reshape(ne[1], ne[0]).T          # (ne0, ne1)


def W(name):
    return G.f32(name)


def LN(X, w, b, eps=EPS, affine_bias=True):
    """X:(1152,T)"""
    mean = X.mean(axis=0, dtype=np.float32)
    d = X - mean
    var = (d*d).mean(axis=0, dtype=np.float32)
    s = (1.0/np.sqrt(var + eps)).astype(np.float32)
    y = d*s
    ys = y*w[:, None]
    if affine_bias:
        return (ys + b[:, None]).astype(np.float32), ys.astype(np.float32)
    return ys.astype(np.float32), ys.astype(np.float32)


def rope(Q, p_t, p_h):
    """Q:(1152=72*16, T)  ->  旋转后的同形状"""
    ts = 10000.0**(-2.0/36.0)
    out = Q.copy()
    for h in range(NH):
        v = Q[h*DH:(h+1)*DH, :]                     # (72, T)
        x0 = v[:36, :]; x1 = v[36:, :]
        th = ts**np.arange(36, dtype=np.float64)
        cos = np.empty((36, Q.shape[1]), np.float32); sin = np.empty_like(cos)
        for ic in range(36):
            p = (p_t if ic < 18 else p_h).astype(np.float64)
            t = (p*th[ic]).astype(np.float32)
            cos[ic] = np.cos(t); sin[ic] = np.sin(t)
        out[h*DH:h*DH+36, :] = (x0*cos - x1*sin).astype(np.float32)
        out[h*DH+36:(h+1)*DH, :] = (x0*sin + x1*cos).astype(np.float32)
    return out


def attn(Qb, Kb, Vb, Wo, bo):
    T = Qb.shape[1]
    O = np.empty((NE, T), np.float32)
    for h in range(NH):
        q = Qb[h*DH:(h+1)*DH, :]                    # (72,T)
        k = Kb[h*DH:(h+1)*DH, :]
        v = Vb[h*DH:(h+1)*DH, :]
        S = (k.T @ q).astype(np.float32)*np.float32(SQ)     # (kt, qt)
        mx = S.max(axis=0)
        E = np.exp((S - mx[None, :]).astype(np.float32))
        Sm = (E/E.sum(axis=0, dtype=np.float32)[None, :]).astype(np.float32)
        O[h*DH:(h+1)*DH, :] = (v @ Sm).astype(np.float32)
    return (Wo @ O + bo[:, None]).astype(np.float32)


def main():
    Wqkv = lambda i: W('v.blk.%d.attn_qkv.weight' % i)
    Wqkvb = lambda i: G.np('v.blk.%d.attn_qkv.bias' % i).astype(np.float32)
    X = rrd('inp_pos_emb.bin')
    if X is None:
        print('缺 inp_pos_emb 真值'); return
    X = X.astype(np.float32)
    print('输入 inp_pos_emb', X.shape)
    # token 位置 (x,y) -> p_t = y, p_h = x
    yy, xx = np.divmod(np.arange(1024), 32)
    # 从 pos 侧定位置 (已验证): 用 P 表直接反查
    vt = gg.VT(G)
    grid = vt.pos_grid(32, 32)
    pg = gg.flat_of(grid)
    P = np.zeros((1024, 2), np.int64)
    for t in range(1024):
        v = X[:, t] - rrd('patch_bias.bin')[:, t]
        d2 = ((grid - v[None, None, :])**2).sum(axis=2)
        i = int(np.argmin(d2)); P[t] = (i//32, i % 32)
    p_t = P[:, 1].astype(np.float32); p_h = P[:, 0].astype(np.float32)   # 行=y, 列=x
    res = {}
    t0 = time.time()
    for il in range(NL):
        lnw = W('v.blk.%d.ln1.weight' % il); lnb = G.np('v.blk.%d.ln1.bias' % il).astype(np.float32)
        Y, ys = LN(X, lnw, lnb)
        ref = rrd('norm_w-%d.bin' % il)
        if ref is not None:
            res['ln1_%d' % il] = gg.relrms(ys, ref)
        qkv = (Y.T @ W('v.blk.%d.attn_qkv.weight' % il).T).astype(np.float32) + Wqkvb(il)
        Qb = qkv[:, :NE].T.astype(np.float32)
        Kb = qkv[:, NE:2*NE].T.astype(np.float32)
        Vb = qkv[:, 2*NE:].T.astype(np.float32)
        Qr = rope(Qb, p_t, p_h)
        Kr = rope(Kb, p_t, p_h)
        ref = rd('Qcur_rope-%d.bin' % il)
        if ref is not None:
            A = ref[0].reshape(1024, NE).T
            res['qrope_%d' % il] = gg.relrms(Qr, A)
        Ao = attn(Qr, Kr, Vb, W('v.blk.%d.attn_out.weight' % il),
                  G.np('v.blk.%d.attn_out.bias' % il).astype(np.float32))
        ref = rrd('attn_out-%d.bin' % il)
        if ref is not None:
            res['attn_%d' % il] = gg.relrms(Ao, ref)
        X = (X + Ao).astype(np.float32)
        lnw2 = W('v.blk.%d.ln2.weight' % il); lnb2 = G.np('v.blk.%d.ln2.bias' % il).astype(np.float32)
        Y2, _ = LN(X, lnw2, lnb2)
        ref = rrd('ffn_inp_normed-%d.bin' % il)
        if ref is not None:
            res['ln2_%d' % il] = gg.relrms(Y2, ref)
        up = (Y2.T @ W('v.blk.%d.ffn_up.weight' % il).T).astype(np.float32) + \
             G.np('v.blk.%d.ffn_up.bias' % il).astype(np.float32)
        g = (0.5*up*(1.0 + np.tanh(np.sqrt(2.0/np.pi)*(up + 0.044715*up**3)))).astype(np.float32)
        dn = (g @ W('v.blk.%d.ffn_down.weight' % il).T).astype(np.float32) + \
             G.np('v.blk.%d.ffn_down.bias' % il).astype(np.float32)
        X = (X + dn.T).astype(np.float32)
        ref = rrd('layer_out-%d.bin' % il)
        if ref is not None:
            res['L%d_out' % il] = gg.relrms(X, ref)
        if il == 0 or il == NL-1:
            print('  layer %d 用时 %.1fs' % (il, time.time()-t0))
    for k in sorted(res, key=lambda s: (s.split('_')[-1], s)):
        print('  %-12s %.4f%%' % (k, res[k]))
    print('总用时 %.1fs' % (time.time()-t0))
    np.save('/tmp/ref_layerout.npy', X)


main()
