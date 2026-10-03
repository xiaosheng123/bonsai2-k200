"""视觉塔参考前向 v2: 逐阶段与真值桩对拍 (含 merger)"""
import numpy as np, struct, os, sys, time
import ggufload, gg

G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
D = os.environ.get('GTDIR', '/tmp/gt/T2_shapes')
NL = 27; NH = 16; DH = 72; NE = 1152; FFN = 4304; EPS = 1e-6
SQ = np.float32(1.0/np.sqrt(72.0))
Wc = lambda nm: G.f32(nm)
Bc = lambda nm: G.np(nm).astype(np.float32)


def rne(name):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        return None, None
    b = open(p, 'rb').read()
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b[:20])
    return np.frombuffer(b[20:], dtype='<f4'), (ne0, ne1, ne2, ne3)


def rrd(name):                       # 逻辑 [ne0, ne1]
    f, ne = rne(name)
    if f is None:
        return None
    return f.reshape(ne[1], ne[0]).T


def rcont(name):                     # 已连续张量: 逻辑 (i0,i1,..) 数组
    f, ne = rne(name)
    if f is None:
        return None
    n = len(ne)
    r = tuple(range(n-1, -1, -1))
    cut = int(np.prod(ne))
    return f[:cut].reshape(tuple(ne[::-1])).transpose(r)


def LN(X, il, which):
    w = Wc('v.blk.%d.%s.weight' % (il, which)); b = Bc('v.blk.%d.%s.bias' % (il, which))
    mean = X.mean(axis=0, dtype=np.float32)
    d = X - mean
    var = (d*d).mean(axis=0, dtype=np.float32)
    ys = (d*(1.0/np.sqrt(var+EPS)).astype(np.float32))*w[:, None]
    return (ys + b[:, None]).astype(np.float32), ys.astype(np.float32)


def rope(Q, p_row, p_col, NH_, DH_):
    ts = 10000.0**(-2.0/36.0)
    th = (ts**np.arange(36, dtype=np.float64)).astype(np.float32)
    out = Q.copy()
    for h in range(NH_):
        v = Q[h*DH_:(h+1)*DH_, :]
        a = v[:36, :]; b = v[36:, :]
        ca = np.empty((36, Q.shape[1]), np.float32)
        sa = np.empty_like(ca)
        for ic in range(36):
            p = p_row if ic < 18 else p_col
            e = ic if ic < 18 else ic-18
            theta = (p*th[e]).astype(np.float32)
            ca[ic] = np.cos(theta); sa[ic] = np.sin(theta)
        out[h*DH_:h*DH_+36, :] = (a*ca - b*sa).astype(np.float32)
        out[h*DH_+36:(h+1)*DH_, :] = (a*sa + b*ca).astype(np.float32)
    return out


def attn(Qb, Kb, Vb, Wo, bo):
    T = Qb.shape[1]
    O = np.empty((NE, T), np.float32)
    for h in range(NH):
        q = Qb[h*DH:(h+1)*DH]; k = Kb[h*DH:(h+1)*DH]; v = Vb[h*DH:(h+1)*DH]
        S = (k.T @ q).astype(np.float32)*SQ
        mx = S.max(axis=0)
        E = np.exp((S - mx[None, :]).astype(np.float32))
        Sm = (E/E.sum(axis=0, dtype=np.float32)[None, :]).astype(np.float32)
        O[h*DH:(h+1)*DH, :] = (v @ Sm).astype(np.float32)
    return (Wo @ O + bo[:, None]).astype(np.float32)


def main():
    Xin = sys.argv[1] if len(sys.argv) > 1 else None
    if Xin:
        a = np.ascontiguousarray(np.load(Xin))
        X = a['d'] if a.dtype.names else a
        print('输入 X', X.shape)
    else:
        X = rrd('inp_pos_emb.bin').astype(np.float32)
        print('输入 inp_pos_emb', X.shape)
    # token 位置 (行/列) —— 用 pos 网格反查
    vt = gg.VT(G); grid = vt.pos_grid(32, 32)
    Pb = rrd('patch_bias.bin')
    P = np.zeros((1024, 2), np.int64)
    for t in range(1024):
        v = X[:, t] - Pb[:, t]
        d2 = ((grid - v[None, None, :])**2).sum(axis=2)
        i = int(np.argmin(d2)); P[t] = (i//32, i % 32)
    p_row = P[:, 1].astype(np.float32); p_col = P[:, 0].astype(np.float32)
    np.save('/tmp/pos_row.npy', p_row); np.save('/tmp/pos_col.npy', p_col)
    res = {}
    t0 = time.time()
    for il in range(NL):
        Y, ys = LN(X, il, 'ln1')
        qkv = (Y.T @ Wc('v.blk.%d.attn_qkv.weight' % il).T).astype(np.float32) + Bc('v.blk.%d.attn_qkv.bias' % il)
        rqf, rqne = rne('Qcur-%d.bin' % il)
        if rqf is not None:
            nt = len(rqf)//3456
            Pq = rqf[:nt*3456].reshape(nt, 3456)[:, :NE].T
            res.setdefault('Qpre', []).append(gg.relrms(qkv[:, :NE].T[:, :nt], Pq))
        Qb = qkv[:, :NE].T.astype(np.float32)
        Kb = qkv[:, NE:2*NE].T.astype(np.float32)
        Vb = qkv[:, 2*NE:].T.astype(np.float32)
        Qr = rope(Qb, p_row, p_col, NH, DH)
        Kr = rope(Kb, p_row, p_col, NH, DH)
        rq = rcont('Qcur_rope-%d.bin' % il)
        if rq is not None:
            res.setdefault('Qrope', []).append(gg.relrms(Qr, np.ascontiguousarray(rq.transpose(1,0,2,3)).reshape(NE, 1024)))
        Ao = attn(Qr, Kr, Vb, Wc('v.blk.%d.attn_out.weight' % il), Bc('v.blk.%d.attn_out.bias' % il))
        r = rrd('attn_out-%d.bin' % il)
        if r is not None:
            res.setdefault('attn', []).append(gg.relrms(Ao, r))
        X = (X + Ao).astype(np.float32)
        Y2, _ = LN(X, il, 'ln2')
        r = rrd('ffn_inp_normed-%d.bin' % il)
        if r is not None:
            res.setdefault('ln2', []).append(gg.relrms(Y2, r))
        up = (Y2.T @ Wc('v.blk.%d.ffn_up.weight' % il).T).astype(np.float32) + Bc('v.blk.%d.ffn_up.bias' % il)
        g = (0.5*up*(1.0 + np.tanh(np.float32(np.sqrt(2.0/np.pi))*(up + 0.044715*up**3)))).astype(np.float32)
        dn = (g @ Wc('v.blk.%d.ffn_down.weight' % il).T).astype(np.float32) + Bc('v.blk.%d.ffn_down.bias' % il)
        X = (X + dn.T).astype(np.float32)
        r = rrd('layer_out-%d.bin' % il)
        if r is not None:
            res.setdefault('layer_out', []).append(gg.relrms(X, r))
        if il in (0, 26):
            print('  layer %d 累计 %.1fs' % (il, time.time()-t0))
    np.save('/tmp/ref_layerout.npy', X)
    # post_ln
    pw = G.np('v.post_ln.weight').astype(np.float32); pb = G.np('v.post_ln.bias').astype(np.float32)
    mean = X.mean(axis=0, dtype=np.float32); d = X - mean
    var = (d*d).mean(axis=0, dtype=np.float32)
    Xp = ((d*(1.0/np.sqrt(var+EPS)).astype(np.float32))*pw[:, None] + pb[:, None]).astype(np.float32)
    # merger
    E = Xp.T.reshape(256, 4608)
    h = (E @ Wc('mm.0.weight').T).astype(np.float32) + Bc('mm.0.bias')
    h = (0.5*h*(1.0 + np.tanh(np.float32(np.sqrt(2.0/np.pi))*(h + 0.044715*h**3)))).astype(np.float32)
    emb = (h @ Wc('mm.2.weight').T).astype(np.float32) + Bc('mm.2.bias')
    print('emb', emb.shape)
    EP = os.environ.get('EMB', '/tmp/gt/embd2_shapes.bin')
    if os.path.exists(EP):
        bb = open(EP, 'rb').read()
        nt, nd = struct.unpack('<2i', bb[:8])
        e = np.frombuffer(bb[8:], dtype='<f4').reshape(nt, nd)
        print('真值 embedding: %d tok x %d' % (nt, nd))
        print('★ relrms(最终图像 embedding) = %.4f%%' % gg.relrms(emb, e))
        np.save('/tmp/ref_emb.npy', emb)
    else:
        print('无真值 embedding 文件', EP)
    for k, v in sorted(res.items()):
        v = np.array(v)
        print('%-10s  中位 %8.4f%%  最大 %8.4f%%  (n=%d, 逐层: %s)' % (
            k, np.median(v), v.max(), len(v), ' '.join('%.2f' % x for x in v)))
    print('总用时 %.1fs' % (time.time()-t0))


main()
