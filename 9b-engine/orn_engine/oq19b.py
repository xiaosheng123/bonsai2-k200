#!/usr/bin/env python3
# oq19b.py — 修正版: l0_sm0.f32 是 softmax D2H【之前】落盘的 (全 0), 真数据在 l0_sm1.f32 (h=1 时落盘的是 h=0 的 softmax)
import numpy as np, os

VNE, VDH = 1152, 72

def chunks_unif(T, want):
    for d in range(min(want, T), 7, -1):
        if T % d == 0: return d
    return T

def relrms(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return 100.0 * np.sqrt(((a - b) ** 2).sum() / max((b * b).sum(), 1e-30))

def emul(Sm, Vt, CH, mode):
    T, D = Vt.shape
    VtT = np.ascontiguousarray(Vt.T)
    cs, nc = [], 0
    if CH <= 0 or CH > T: CH = T
    o = 0
    while o < T:
        n = min(CH, T - o); cs.append(n); o += n
    nc = len(cs); koff = np.cumsum([0] + cs[:-1])
    Bq, rs = [], []
    for c in range(nc):
        blk = VtT[:, koff[c]:koff[c] + cs[c]]
        s = np.maximum(np.abs(blk).max(axis=1) / 127.0, 1e-8)
        Bq.append(np.clip(np.rint(blk / s[:, None]), -127, 127)); rs.append(s)
    mxfull = np.abs(Sm).max(axis=1)
    out = np.zeros((T, D))
    if mode == 'perchunk':
        for c in range(nc):
            Ac = Sm[:, koff[c]:koff[c] + cs[c]]
            mx = np.abs(Ac).max(axis=1); mc = mx.max()
            An = Ac * (mc / np.maximum(mx, 1e-30))[:, None]
            maxa = np.abs(An).max()
            Ad = np.clip(np.rint(An * 127.0 / maxa), -127, 127) * (maxa / 127.0)
            out += (Ad * (np.maximum(mx, 1e-30) / mc)[:, None]) @ (Bq[c] * rs[c][:, None]).T
    else:
        An_all = Sm / np.maximum(mxfull, 1e-30)[:, None]
        for c in range(nc):
            An = An_all[:, koff[c]:koff[c] + cs[c]]
            maxa = np.abs(An).max()
            Ad = np.clip(np.rint(An * 127.0 / maxa), -127, 127) * (maxa / 127.0)
            out += (Ad * mxfull[:, None]) @ (Bq[c] * rs[c][:, None]).T
    return out

for tag, base, T in [('512²', '/tmp/vsw/c1/o', 1024), ('768²', '/tmp/vsw/c7/o', 2304)]:
    print('=' * 80)
    S0 = np.fromfile(os.path.join(base, 'l0_sm0.f32'), dtype='<f4')
    S1 = np.fromfile(os.path.join(base, 'l0_sm1.f32'), dtype='<f4')
    print('%s  l0_sm0 |max|=%.3e (≈0 => 落盘在 softmax 之前, 是陈旧缓冲)' % (tag, np.abs(S0).max()))
    Sm = S1.reshape(T, T).astype(np.float64)          # ★ h=0 的真实 softmax 输出
    Vb = np.fromfile(os.path.join(base, 'l0_vb.f32'), dtype='<f4').reshape(T, VNE).T
    for head in (0, 1, 2, 5):
        Vt = Vb[head*VDH:(head+1)*VDH, :].T.astype(np.float64)
        exact = Sm @ Vt
        rm = Sm.max(axis=1)
        print('--- head %d : |exact|rms=%.4f sm 行 max 中位=%.4e min=%.4e p95=%.4e   行和=%.6f' %
              (head, np.sqrt((exact**2).mean()), np.median(rm), rm.min(), np.percentile(rm, 95), Sm.sum(axis=1).mean()))
        u = chunks_unif(T, 96)
        for name, kw in [('现方案 A逐块 CH=96 (老路径)', dict(CH=96, mode='perchunk')),
                         ('新方案 A逐块 CH=%d (整除) + B整除分块' % u, dict(CH=u, mode='perchunk')),
                         ('新方案 A整行 + B分块 CH=%d' % u, dict(CH=u, mode='fullrow')),
                         ('A整行 + B整块 nc=1', dict(CH=0, mode='fullrow'))]:
            print('      %-38s relrms = %8.4f%%' % (name, relrms(emul(Sm, Vt, **kw), exact)))
