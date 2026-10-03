"""量化误差账本 v3: 现状 / 分块+行归一(需主机折回) / 分块+全局scale+alpha折回(卡上累加)"""
import numpy as np, sys
sys.path.insert(0, '/home/caden/ornc/mm')
import ggufload
G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')


def rr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return 100.0*np.sqrt(((a-b)**2).sum()/max((b*b).sum(), 1e-30))


def qrow(W):
    am = np.abs(W).max(axis=1)
    s = np.maximum(am, 1e-30)/127.0
    return np.clip(np.rint(W/s[:, None]), -127, 127), s


def sim(A, W, ch, mode):
    N, K = W.shape
    q0, rs0 = qrow(W)
    ref = A @ (q0*rs0[:, None]).T
    out = np.zeros((A.shape[0], N))
    if mode == 'now':
        mx = np.abs(A).max(axis=1); M0 = float(mx.max())
        f = M0/np.maximum(mx, 1e-30); Ap = A*f[:, None]
        step = np.abs(Ap).max()/127.0
        Aq = np.clip(np.rint(Ap/step), -127, 127)
        return (Aq*step) @ q0.T*(mx/M0)[:, None]*rs0[None, :], ref
    rng = list(range(0, K, ch)) if ch else [0]
    for i in rng:
        ci = np.arange(i, min(i+ch, K))
        Ac = A[:, ci]; Wc = W[:, ci]
        qc, rsc = qrow(Wc)
        mxc = np.abs(Ac).max(axis=1)
        if mode == 'chunkrow':
            Mc = float(mxc.max())
            f = Mc/np.maximum(mxc, 1e-30); Ap = Ac*f[:, None]
            step = np.abs(Ap).max()/127.0
            Aq = np.clip(np.rint(Ap/step), -127, 127)
            out += (Aq*step) @ qc.T*(mxc/Mc)[:, None]*rsc[None, :]
        else:   # chunkglob: 全局 scale + alpha 折回 => 卡上 beta=1 可直接累加
            step = np.abs(Ac).max()/127.0
            Aq = np.clip(np.rint(Ac/step), -127, 127)
            out += (step*Aq) @ qc.T*rsc[None, :]
    return out, ref


d = np.fromfile('/tmp/vh512/ln2_0.f32', dtype='<f4').reshape(1024, 1152).astype(np.float64)
A = d
for name in ('ffn_up', 'attn_out', 'ffn_down'):
    W = np.ascontiguousarray(G.np('v.blk.0.%s.weight' % name).astype(np.float64).T)   # [out,in]
    o, ref = sim(A, W, 0, 'now')
    print('\n== %s %s ==' % (name, W.shape))
    print('  现状 (整块激活+行归一)              = %.4f%%' % rr(o, ref))
    for ch in (96, 144, 288, 576):
        o, _ = sim(A, W, ch, 'chunkrow')
        r1 = rr(o, ref)
        o, _ = sim(A, W, ch, 'chunkglob')
        r2 = rr(o, ref)
        print('  CH=%4d  分块+行归一(主机折回) = %6.4f%%   分块+全局scale+alpha(卡上累加) = %6.4f%%' % (ch, r1, r2))
