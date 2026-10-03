#!/usr/bin/env python3
# ref_mm3.py — 逐层定位: 自由跑 vs "从真值输入单独跑一层"
# 目的: 判断 llama.cpp 真值本身是否含逐层噪声(即我们能否 < 它)
import numpy as np, gguf, os, sys, struct, time

MMPROJ = "/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf"
IMG    = sys.argv[1] if len(sys.argv) > 1 else "/home/caden/ornc/mmtests/img_shapes.png"
GTD    = sys.argv[2] if len(sys.argv) > 2 else "/tmp/gt/T2_shapes"
GTF    = sys.argv[3] if len(sys.argv) > 3 else "/tmp/gt/embd2_shapes.bin"

def rr(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    return 100.0*np.sqrt(((A-B)**2).sum()/max((B*B).sum(), 1e-30))

def rd(name):
    p = os.path.join(GTD, name)
    b = open(p, 'rb').read()
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b[:20])
    return np.frombuffer(b[20:], dtype='<f4').reshape(ne1, ne0).T.copy()   # [VNE, T]

def rdt(name):   # [T, VNE]
    return np.ascontiguousarray(rd(name).T)

t0 = time.time()
r = gguf.GGUFReader(MMPROJ)
TD = {t.name: t for t in r.tensors}
def T(n):
    t = TD[n]; a = np.asarray(t.data)
    if a.dtype in (np.uint8, np.int8): a = a.view(np.uint16)
    if a.dtype == np.uint16: a = (a.astype(np.uint32) << 16).view(np.float32)
    return np.ascontiguousarray(a.reshape(tuple(reversed(list(t.shape)))).astype(np.float32))

W0 = T("v.patch_embd.weight"); W1 = T("v.patch_embd.weight.1")
PBI = T("v.patch_embd.bias").reshape(-1)
POS = T("v.position_embd.weight")
from PIL import Image
im = np.asarray(Image.open(IMG).convert("RGB"), dtype=np.float32)
ny, nx, _ = im.shape
im = (im/255.0 - 0.5)/0.5
gh, gw = ny//16, nx//16
X = np.stack([im[:, :, c].T for c in range(3)], axis=2)
print("grid %dx%d = %d patches %.1fs" % (gw, gh, gw*gh, time.time()-t0), flush=True)
K = 3*256
P = np.zeros((gw*gh, K), dtype=np.float32)
for ky in range(16):
    for kx in range(16):
        blk = X[kx::16, ky::16, :]
        for c in range(3): P[:, c*256 + ky*16 + kx] = blk[:, :, c].reshape(-1)
Wm = np.zeros((1152, K), dtype=np.float32)
for ky in range(16):
    for kx in range(16):
        for c in range(3): Wm[:, c*256 + ky*16 + kx] = W0[:, c, ky, kx] + W1[:, c, ky, kx]
C = P @ Wm.T
del P, Wm, W0, W1, X
NP = gw*gh; nbx, nby = gw//2, gh//2
srcidx = np.zeros(NP, dtype=np.int64); rows = np.zeros(NP, dtype=np.int64); cols = np.zeros(NP, dtype=np.int64)
for yb in range(nby):
    for xb in range(nbx):
        for dy in range(2):
            for dx in range(2):
                t = ((yb*nbx + xb)*2 + dy)*2 + dx
                x, y = xb*2+dx, yb*2+dy
                srcidx[t] = x + gw*y; rows[t] = y; cols[t] = x
x = C[srcidx] + PBI[None, :]
del C
x = x + POS[srcidx]
print("head done %.1fs" % (time.time()-t0), flush=True)
print("  head vs 真值 inp_pos_emb : %.6f%%" % rr(x, rdt("inp_pos_emb.bin")), flush=True)

DH, NH_, HD = 1152, 16, 72
t_scale = 10000.0 ** (-2.0/HD)
icc = np.arange(36)
expnt = (icc % 18).astype(np.float64)
theta = np.where(icc < 18, rows[:, None], cols[:, None]) * np.power(t_scale, expnt)[None, :]
cos_t = np.cos(theta).astype(np.float32); sin_t = np.sin(theta).astype(np.float32)

def lnorm(v, w, b, eps=1e-6):
    mu = v.mean(axis=-1, keepdims=True); var = v.var(axis=-1, keepdims=True)
    return (v-mu)/np.sqrt(var+eps)*w + b

def gelu(v):
    return 0.5*v*(1.0+np.tanh(0.7978845608028654*v*(1.0+0.044715*v*v)))

def layer(x, il, dump=None):
    """x: [T, VNE] -> x'  ; 返回 (x', 可选中间量)"""
    pf = "v.blk.%d." % il
    ln1w, ln1b = T(pf+"ln1.weight"), T(pf+"ln1.bias")
    ln2w, ln2b = T(pf+"ln2.weight"), T(pf+"ln2.bias")
    qkv_w, qkv_b = T(pf+"attn_qkv.weight"), T(pf+"attn_qkv.bias")
    o_w, o_b = T(pf+"attn_out.weight"), T(pf+"attn_out.bias")
    up_w, up_b = T(pf+"ffn_up.weight"), T(pf+"ffn_up.bias")
    dn_w, dn_b = T(pf+"ffn_down.weight"), T(pf+"ffn_down.bias")
    h = lnorm(x, ln1w, ln1b)
    qkv = h @ qkv_w.T + qkv_b
    del h, qkv_w, qkv_b
    Q = qkv[:, :1152].reshape(NP, NH_, HD).copy()
    Kk = qkv[:, 1152:2304].reshape(NP, NH_, HD).copy()
    V = qkv[:, 2304:].reshape(NP, NH_, HD)
    del qkv
    if dump is not None: dump['Q'] = np.ascontiguousarray(Q.transpose(1, 2, 0))   # 供 rope 对拍
    for A_ in (Q, Kk):
        lo = A_[:, :, :36].copy(); hi = A_[:, :, 36:].copy()
        A_[:, :, :36] = lo*cos_t[:, None, :] - hi*sin_t[:, None, :]
        A_[:, :, 36:] = lo*sin_t[:, None, :] + hi*cos_t[:, None, :]
    if dump is not None: dump['Qr'] = np.ascontiguousarray(Q.transpose(1, 2, 0))
    obuf = np.empty((NP, 1152), dtype=np.float32)
    for hh in range(NH_):
        sc = (Q[:, hh, :] @ Kk[:, hh, :].T) * (1.0/np.sqrt(HD))
        sc -= sc.max(axis=1, keepdims=True)
        np.exp(sc, out=sc); sc /= sc.sum(axis=1, keepdims=True)
        obuf[:, hh*HD:(hh+1)*HD] = sc @ V[:, hh, :]
    del Q, Kk, V
    ao = obuf @ o_w.T + o_b
    x = x + ao
    if dump is not None: dump['attn_out'] = ao.copy()
    h = lnorm(x, ln2w, ln2b)
    if dump is not None: dump['ln2'] = h.copy()
    f = gelu(h @ up_w.T + up_b)
    del h, up_w, up_b
    d = f @ dn_w.T + dn_b
    x = x + d
    del f, d, up_w, dn_w, ln1w, ln1b, ln2w, ln2b
    return x

xt = rdt("inp_pos_emb.bin")                 # 真值输入 (T, VNE)
print("\n%-6s %-14s %-14s" % ("layer", "自由跑 vs 真值", "单层(真值输入)"), flush=True)
for il in range(27):
    x = layer(x, il)
    tp = rdt("layer_out-%d.bin" % il)
    r1 = rr(x, tp)
    x1 = layer(xt, il)                     # 从真值输入单独跑这一层
    r2 = rr(x1, tp)
    print("%-6d %-14.5f %-14.5f" % (il, r1, r2), flush=True)
    xt = tp                                # 下一层的真值输入
    if il == 0:
        # rope 对拍
        d = {}
        layer(rdt("inp_pos_emb.bin"), 0, d)
        qr = rd("Qcur_rope-0.bin")
        print("  [chk] Qcur_rope-0 真值形状 %s ; 我们 %s" % (qr.shape, d['Qr'].shape), flush=True)
        print("  [chk] pre-rope Q vs Qcur-0 = %.5f%%" % rr(d['Q'].reshape(1152, -1), rd("Qcur-0.bin")), flush=True)
        print("  [chk] post-rope Q vs Qcur_rope-0 = %.5f%%" % rr(d['Qr'].reshape(1152, -1), qr), flush=True)
print("\n自由跑最终 embedding vs 真值 = %.4f%%" % rr(x, np.load('/tmp/ref_emb.npy')) if os.path.exists('/tmp/ref_emb.npy') else "", flush=True)
print("total %.1fs" % (time.time()-t0), flush=True)
