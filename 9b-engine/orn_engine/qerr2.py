"""量化误差账本 v2: 正确折叠 scale"""
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
    q = np.clip(np.rint(W/s[:, None]), -127, 127)
    return q, s


def sim(A, W, ch=0, fold='host'):
    """返回 (输出, 参考). ch=0 表示不分块。fold: host=每块折回(需主机累加), dev=卡上 beta=1 累加"""
    N, K = W.shape
    M0 = float(np.abs(A).max())
    mx = np.abs(A).max(axis=1)
    q0, rs0 = qrow(W)
    ref = A @ (q0*rs0[:, None]).T
    out = np.zeros((A.shape[0], N))
    rng = range(0, K, ch) if ch else [0]
    for i in rng:
        kc = K if not ch else min(ch, K-i)
        ci = np.arange(i, i+kc)
        Ac = A[:, ci]
        Wc = W[:, ci]
        qc, rsc = qrow(Wc)
        mxc = np.abs(Ac).max(axis=1)
        Mc = float(M0 if not ch else np.abs(Ac*(M0/np.maximum(mx, 1e-30))[:, None]).max())
        f = Mc/np.maximum(mxc, 1e-30)
        Ap = Ac*f[:, None]
        step = np.abs(Ap).max()/127.0
        Aq = np.clip(np.rint(Ap/step), -127, 127)
        part = (Aq*step) @ qc.T
        if fold == 'host':
            out += part*(mxc/Mc)[:, None]*rsc[None, :]
        else:
            out += part
    if fold == 'dev':
        # 卡上累加后统一折回 (只用第 0 块的行归一化与最后一块的权重 scale 作近似说明)
        out = out*np.ones((A.shape[0], 1))*rs0[None, :]
    return out, ref


d = np.fromfile('/tmp/vh512/ln2_0.f32', dtype='<f4').reshape(1024, 1152).astype(np.float64)
A = d
print('A 行内 max/median 中位 %.1f' % np.median(np.abs(A).max(axis=1)/np.median(np.abs(A), axis=1)))
for name in ('ffn_up', 'attn_out'):
    W = G.f32('v.blk.0.%s.weight' % name).astype(np.float64)
    o0, ref = sim(A, W, 0)
    print('\n== %s ==' % name)
    print('  现状(整块激活量化+每行权重)      = %.4f%%' % rr(o0, ref))
    for ch in (144, 288, 576):
        o, _ = sim(A, W, ch, 'host')
        print('  K分块 CH=%4d 主机每块折回       = %.4f%%' % (ch, rr(o, ref)))
        o, _ = sim(A, W, ch, 'dev')
        print('  K分块 CH=%4d 卡上beta=1累加     = %.4f%%' % (ch, rr(o, ref)))
