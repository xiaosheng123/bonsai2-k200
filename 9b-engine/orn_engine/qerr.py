"""量化误差账本: 用真值 dump 的激活 + 真权重, 模拟官方 gemm_int8 的量化, 量化各来源"""
import numpy as np, sys, os
sys.path.insert(0, '/home/caden/ornc/mm')
import ggufload
G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')


def rr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return 100.0*np.sqrt(((a-b)**2).sum()/max((b*b).sum(), 1e-30))


def qrow(W):
    am = np.abs(W).max(axis=1)
    s = am/127.0
    q = np.clip(np.rint(W/s[:, None]), -127, 127)
    return q, s


def op_sim(A, q, rs, M0=None, chunk=0):
    """模拟官方: 激活按 per-tensor(整块) max 量化到 int8, 权重 int8; 返回 (折回后输出, 激活量化误差)"""
    if chunk == 0:
        chunks = [np.arange(A.shape[1])]
    else:
        chunks = [np.arange(i, min(i+chunk, A.shape[1])) for i in range(0, A.shape[1], chunk)]
    out = np.zeros((A.shape[0], q.shape[0]), np.float64)
    for ci in chunks:
        Ac = A[:, ci]
        mx = np.abs(Ac).max(axis=1)
        M = float(M0) if M0 else float(mx.max())
        f = M/np.maximum(mx, 1e-30)
        Ap = Ac*f[:, None]
        step = np.abs(Ap).max()/127.0
        Aq = np.clip(np.rint(Ap/step), -127, 127)
        out += (Aq*step @ q[:, ci].T)
    return out


# A = 真值 ln2_0 (CPU 模式 dump, 即精确参考激活)
d = np.fromfile('/tmp/vh512/ln2_0.f32', dtype='<f4').reshape(1024, 1152)  # [t][k] (i + 1152*t)
A = d.astype(np.float64)
print('A: shape', A.shape, 'max %.3f' % np.abs(A).max())
mx = np.abs(A).max(axis=1)
print('每行 max/中位比: 中位 %.2f 最大 %.2f' % (np.median(mx/np.median(np.abs(A), axis=1)), (mx/np.median(np.abs(A), axis=1)).max()))
print('行内 max/median|a| (离群程度): 中位 %.2f' % np.median(mx/np.median(np.abs(A), axis=1)))

for name, shape in (('ffn_up', (4304, 1152)), ('attn_out', (1152, 1152))):
    W = G.f32('v.blk.0.%s.weight' % name).astype(np.float64)
    q, rs = qrow(W)
    ref = A @ (q*rs[:, None]).T
    exact = A @ W.T
    print('\n== %s %s ==' % (name, shape))
    print('  权重 int8 每行量化误差      = %.4f%%' % rr(ref, exact))
    o = op_sim(A, q, rs)
    print('  + 激活 int8 整块量化 (现状)  = %.4f%%' % rr(o, ref))
    for ch in (128, 288, 576):
        o = op_sim(A, q, rs, chunk=ch)
        print('  + 激活 int8 K分块 CH=%4d    = %.4f%%' % (ch, rr(o, ref)))
    # 只激活分块, 权重同块
    Wb = W.copy()
    for ch in (288, 576):
        # 权重也按同一 K 块分别量化 (每行每块一个 scale)
        o = np.zeros((A.shape[0], q.shape[0]))
        for i in range(0, A.shape[1], ch):
            ci = np.arange(i, min(i+ch, A.shape[1]))
            Wc = W[:, ci]
            qc, sc = qrow(Wc)
            o += op_sim(A[:, ci], qc, sc, chunk=ch)
        print('  权重+激活都按 K 分块 CH=%4d = %.4f%%' % (ch, rr(o, exact)))
