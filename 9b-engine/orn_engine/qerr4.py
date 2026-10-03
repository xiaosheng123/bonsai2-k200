"""账本 v4: 可行的分块方案 = 行归一(全行 max) + 每块自己的 max_a + 卡上 beta=1 累加 + 统一折回"""
import numpy as np, sys
sys.path.insert(0, '/home/caden/ornc/mm')
import ggufload
G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')


def rr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return 100.0*np.sqrt(((a-b)**2).sum()/max((b*b).sum(), 1e-30))


def qrow_full(W):
    am = np.abs(W).max(axis=1)
    s = np.maximum(am, 1e-30)/127.0
    return np.clip(np.rint(W/s[:, None]), -127, 127), s


def sim(A, W, CH):
    N, K = W.shape
    q, rs = qrow_full(W)                     # ★ 权重: 全行一个 scale (分块共用, 才能一次折回)
    exact = A @ W.T
    ref = A @ (q*rs[:, None]).T
    mx = np.abs(A).max(axis=1); M0 = float(mx.max())
    f = M0/np.maximum(mx, 1e-30)             # 行归一化 (全行 max, 与是否分块无关)
    Ap = A*f[:, None]
    out = np.zeros((A.shape[0], N))
    KC = K//CH
    for c in range(CH):
        ci = np.arange(c*KC, (c+1)*KC) if CH > 1 else np.arange(K)
        Ac = Ap[:, ci]
        step = np.abs(Ac).max()/127.0        # 本块的 max_a
        Aq = np.clip(np.rint(Ac/step), -127, 127)
        out += (step*Aq) @ q[:, ci].T        # 卡上 beta=1 累加
    out = out*(mx/M0)[:, None]*rs[None, :]   # 统一折回
    return out, ref, exact


d = np.fromfile('/tmp/vh512/ln2_0.f32', dtype='<f4').reshape(1024, 1152).astype(np.float64)
A = d
for name in ('ffn_up', 'attn_out', 'attn_qkv'):
    Wt = G.np('v.blk.0.%s.weight' % name).astype(np.float64)   # ne=[in,out]
    W = np.ascontiguousarray(Wt.T.reshape(Wt.shape[1], Wt.shape[0]))  # [out,in]
    print('\n== %s %s ==' % (name, W.shape))
    for CH in (1, 2, 4, 8, 16):
        o, ref, exact = sim(A, W, CH)
        print('  CH=%2d  vs int8参考 = %6.4f%%   vs fp32权重精确值 = %6.4f%%' % (CH, rr(o, ref), rr(o, exact)))
