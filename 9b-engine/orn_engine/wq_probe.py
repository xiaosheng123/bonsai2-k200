"""wq_probe.py —— 权重/激活量化粒度对 ffn_up 单次 gemm 的影响 (真实 mmproj 权重 + 真实激活)
用法: python3 wq_probe.py <mmproj> <act.f32> <T> [tensor]
  act.f32 = 逐阶段 dump 的 [VNE,T] 布局 (index = i + VNE*t)  -> A[t,i]
"""
import numpy as np, struct, sys

MM = sys.argv[1]; ACT = sys.argv[2]; T = int(sys.argv[3])
TEN = sys.argv[4] if len(sys.argv) > 4 else 'v.blk.0.ffn_up.weight'
VNE = 1152

def relrms(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    return 100.0 * np.sqrt(((A - B) ** 2).sum() / max((B * B).sum(), 1e-30))

# --- 极简 GGUF 读取 (只要 F32/BF16/F16) ---
f = open(MM, 'rb')
magic, ver = struct.unpack('<II', f.read(8))
nt, nkv = struct.unpack('<QQ', f.read(16))
def skip(t):
    if t in (0,1,7): return 1
    if t in (2,3): return 2
    if t in (4,5,6): return 4
    if t in (10,11,12): return 8
    if t == 8:
        n, = struct.unpack('<Q', f.read(8)); f.seek(n, 1); return 0
    if t == 9:
        et, n = struct.unpack('<IQ', f.read(12))
        for _ in range(n): skip(et)
        return 0
    raise Exception('kv type %d' % t)
align = 32
for _ in range(nkv):
    kl, = struct.unpack('<Q', f.read(8)); k = f.read(kl).decode()
    t, = struct.unpack('<I', f.read(4))
    if k == 'general.alignment' and t == 4:
        align, = struct.unpack('<I', f.read(4))
    else:
        skip(t)
ts = {}
for _ in range(nt):
    nl, = struct.unpack('<Q', f.read(8)); nm = f.read(nl).decode()
    nd, = struct.unpack('<I', f.read(4))
    ne = struct.unpack('<%dQ' % nd, f.read(8*nd))
    ty, off = struct.unpack('<IQ', f.read(12))
    ts[nm] = (ne, ty, off)
p = f.tell(); dstart = (p + align - 1)//align*align
ne, ty, off = ts[TEN]
N, K = int(ne[0]), int(ne[1])
f.seek(dstart + off)
nel = N*K
if ty == 0:
    W = np.frombuffer(f.read(4*nel), dtype='<f4').reshape(N, K).astype(np.float64)
elif ty in (1, 30):
    u = np.frombuffer(f.read(2*nel), dtype='<u2').astype(np.uint32) << 16
    W = u.view('<f4').reshape(N, K).astype(np.float64)
else:
    raise Exception('type %d' % ty)
print('张量 %s  N=%d K=%d type=%d' % (TEN, N, K, ty))

A = np.fromfile(ACT, dtype='<f4').reshape(VNE, T).T.astype(np.float64)   # [T,VNE]
print('激活 %s  A=%s' % (ACT, A.shape))

def q_perrow(W, CH):
    """每行每 CH 块一个 scale (当前卡上做法)"""
    Q = np.zeros_like(W); rs = np.zeros((W.shape[0], 0))
    out = np.zeros_like(W); scales = []
    for k0 in range(0, K, CH):
        cs = min(CH, K - k0)
        blk = W[:, k0:k0+cs]
        s = np.abs(blk).max(axis=1, keepdims=True) / 127.0
        s = np.where(s <= 0, 1e-30, s)
        out[:, k0:k0+cs] = np.rint(blk/s)*s
        scales.append(s)
    return out

def q_pertensor(W, CH):
    out = np.zeros_like(W)
    for k0 in range(0, K, CH):
        cs = min(CH, K - k0)
        blk = W[:, k0:k0+cs]
        s = np.abs(blk).max()/127.0
        if s <= 0: s = 1e-30
        out[:, k0:k0+cs] = np.rint(blk/s)*s
    return out

def act_pertensor(A, CH):
    out = np.zeros_like(A)
    for k0 in range(0, K, CH):
        cs = min(CH, K - k0)
        blk = A[:, k0:k0+cs]
        s = np.abs(blk).max()/127.0
        if s <= 0: s = 1e-30
        out[:, k0:k0+cs] = np.rint(blk/s)*s
    return out

def act_perrow(A, CH):
    """逐块把每行归一到同一下的 Mc, 再按单标量整块量化 (= 卡上内部行为)"""
    out = np.zeros_like(A)
    for k0 in range(0, K, CH):
        cs = min(CH, K - k0)
        blk = A[:, k0:k0+cs].copy()
        mx = np.abs(blk).max(axis=1, keepdims=True)
        mc = max(mx.max(), 1e-30)
        g = mc/np.where(mx <= 1e-30, 1e-30, mx)
        blk = blk*g
        s = np.abs(blk).max()/127.0
        if s <= 0: s = 1e-30
        out[:, k0:k0+cs] = np.rint(blk/s)*s/g      # 折回
    return out

exact = A @ W.T
for name, Wa, Aa in [
    ('权重 per-row 全 K (0/0 档)',            q_perrow(W, K),   A),
    ('权重 per-row 96 (CH=96 现档)',          q_perrow(W, 96),  A),
    ('权重 per-row 24',                       q_perrow(W, 24),  A),
    ('权重 per-tensor 96',                    q_pertensor(W, 96), A),
    ('权重 per-tensor 24',                    q_pertensor(W, 24), A),
    ('权重 per-tensor 96 + 激活 per-tensor 96', q_pertensor(W, 96), act_pertensor(A, 96)),
    ('权重 per-tensor 24 + 激活 per-tensor 24', q_pertensor(W, 24), act_pertensor(A, 24)),
    ('权重 per-row 96 + 激活 行归一 96 (=现档)', q_perrow(W, 96),  act_perrow(A, 96)),
    ('权重 per-tensor 96 + 激活 行归一 96',    q_pertensor(W, 96), act_perrow(A, 96)),
    ('权重 per-tensor 24 + 激活 行归一 24',    q_pertensor(W, 24), act_perrow(A, 24)),
    ('权重 per-row 全K + 激活 行归一 全K',     q_perrow(W, K),   act_perrow(A, K)),
]:
    y = Aa @ Wa.T
    print('  %-40s relrms = %7.4f%%' % (name, relrms(y, exact)))
