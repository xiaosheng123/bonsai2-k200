#!/usr/bin/env python3
# 视觉塔 numpy 参考实现 (mmproj = clip arch, projector_type=qwen3vl_merger)
# 目标: 与 llama.cpp 落盘的 image embeddings 对拍 (relrms)
import numpy as np, gguf, os, sys, struct, time

MMPROJ = "/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf"
IMG    = sys.argv[1] if len(sys.argv) > 1 else "/home/caden/ornc/mmtests/big_shapes.png"
GTF    = sys.argv[2] if len(sys.argv) > 2 else "/tmp/gt/e768_big_shapes.bin"
ORDER  = sys.argv[3] if len(sys.argv) > 3 else "A"

t0 = time.time()
r = gguf.GGUFReader(MMPROJ)
TD = {t.name: t for t in r.tensors}
def T(n):
    t = TD[n]; a = np.asarray(t.data)
    if a.dtype in (np.uint8, np.int8):
        a = a.view(np.uint16)
    if a.dtype == np.uint16:
        a = (a.astype(np.uint32) << 16).view(np.float32)
    out = a.reshape(tuple(reversed(list(t.shape)))).astype(np.float32)
    return np.ascontiguousarray(out)

W0 = T("v.patch_embd.weight"); W1 = T("v.patch_embd.weight.1")
PBI = T("v.patch_embd.bias").reshape(-1)
POS = T("v.position_embd.weight")

from PIL import Image
im = np.asarray(Image.open(IMG).convert("RGB"), dtype=np.float32)
ny, nx, _ = im.shape
im = (im/255.0 - 0.5)/0.5
gh, gw = ny//16, nx//16
X = np.stack([im[:, :, c].T for c in range(3)], axis=2)     # X[x,y,c]
print("grid %dx%d = %d patches  %.1fs" % (gw, gh, gw*gh, time.time()-t0), flush=True)

K = 3*256
P = np.zeros((gw*gh, K), dtype=np.float32)
for ky in range(16):
    for kx in range(16):
        blk = X[kx::16, ky::16, :]
        for c in range(3):
            P[:, c*256 + ky*16 + kx] = blk[:, :, c].reshape(-1)
Wm = np.zeros((1152, K), dtype=np.float32)
for ky in range(16):
    for kx in range(16):
        for c in range(3):
            Wm[:, c*256 + ky*16 + kx] = W0[:, c, ky, kx] + W1[:, c, ky, kx]
C = P @ Wm.T
del P, Wm, W0, W1, X
print("conv done %.1fs" % (time.time()-t0), flush=True)

NP = gw*gh
nbx, nby = gw//2, gh//2
srcidx = np.zeros(NP, dtype=np.int64); rows = np.zeros(NP, dtype=np.int64); cols = np.zeros(NP, dtype=np.int64)
if ORDER == "A":      # 合并块顺序 (yb, xb, dy, dx)
    for yb in range(nby):
        for xb in range(nbx):
            for dy in range(2):
                for dx in range(2):
                    t = ((yb*nbx + xb)*2 + dy)*2 + dx
                    x, y = xb*2+dx, yb*2+dy
                    srcidx[t] = x + gw*y; rows[t] = y; cols[t] = x
elif ORDER == "B":
    srcidx = np.arange(NP); rows = np.repeat(np.arange(gh), gw); cols = np.tile(np.arange(gw), gh)
elif ORDER == "C":    # 合并块顺序, 但块内 (dx,dy) 交换
    for yb in range(nby):
        for xb in range(nbx):
            for dy in range(2):
                for dx in range(2):
                    t = ((yb*nbx + xb)*2 + dy)*2 + dx
                    t2 = ((yb*nbx + xb)*2 + dx)*2 + dy
                    x, y = xb*2+dx, yb*2+dy
                    srcidx[t2] = x + gw*y; rows[t2] = y; cols[t2] = x
else:
    raise SystemExit("order?")

x = C[srcidx] + PBI[None, :]
del C
x = x + POS[srcidx]
print("patch+pos %.1fs  std=%.4f" % (time.time()-t0, x.std()), flush=True)

DH, NH_, HD = 1152, 16, 72
t_scale = 10000.0 ** (-2.0/HD)
icc = np.arange(36)
expnt = (icc % 18).astype(np.float64)   # indep_sects: 每 section 重置 theta
theta = np.where(icc < 18, rows[:, None], cols[:, None]) * np.power(t_scale, expnt)[None, :]
cos_t = np.cos(theta).astype(np.float32); sin_t = np.sin(theta).astype(np.float32)

def lnorm(v, w, b, eps=1e-6):
    mu = v.mean(axis=-1, keepdims=True)
    var = v.var(axis=-1, keepdims=True)
    return (v-mu)/np.sqrt(var+eps)*w + b

def gelu(v):
    return 0.5*v*(1.0+np.tanh(0.7978845608028654*v*(1.0+0.044715*v*v)))

for il in range(27):
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
    Q  = qkv[:, :1152].reshape(NP, NH_, HD).copy()
    Kk = qkv[:, 1152:2304].reshape(NP, NH_, HD).copy()
    V  = qkv[:, 2304:].reshape(NP, NH_, HD)
    del qkv
    for A_ in (Q, Kk):
        lo = A_[:, :, :36].copy(); hi = A_[:, :, 36:].copy()
        A_[:, :, :36] = lo*cos_t[:, None, :] - hi*sin_t[:, None, :]
        A_[:, :, 36:] = lo*sin_t[:, None, :] + hi*cos_t[:, None, :]
    obuf = np.empty((NP, 1152), dtype=np.float32)
    for hh in range(NH_):
        sc = (Q[:, hh, :] @ Kk[:, hh, :].T) * (1.0/np.sqrt(HD))
        sc -= sc.max(axis=1, keepdims=True)
        np.exp(sc, out=sc)
        sc /= sc.sum(axis=1, keepdims=True)
        obuf[:, hh*HD:(hh+1)*HD] = sc @ V[:, hh, :]
    del Q, Kk, V
    x = x + (obuf @ o_w.T + o_b)
    del obuf, o_w, o_b
    h = lnorm(x, ln2w, ln2b)
    f = gelu(h @ up_w.T + up_b)
    del h, up_w, up_b
    x = x + (f @ dn_w.T + dn_b)
    del f, dn_w, dn_b, ln1w, ln1b, ln2w, ln2b
    print("layer %d done %.1fs std=%.4f" % (il, time.time()-t0, x.std()), flush=True)

PLN_W, PLN_B = T("v.post_ln.weight"), T("v.post_ln.bias")
x = lnorm(x, PLN_W, PLN_B)
g = NP//4
emb = np.empty((g, 4608), dtype=np.float32)
for r4 in range(4):
    emb[:, r4*1152:(r4+1)*1152] = x[4*np.arange(g)+r4]
del x
MM0_W, MM0_B = T("mm.0.weight"), T("mm.0.bias")
MM2_W, MM2_B = T("mm.2.weight"), T("mm.2.bias")
emb = gelu(emb @ MM0_W.T + MM0_B)
out = emb @ MM2_W.T + MM2_B
print("OUT", out.shape, "mean %.5f std %.5f" % (out.mean(), out.std()), flush=True)

with open(GTF, "rb") as f:
    ntok, nemb = struct.unpack("<2i", f.read(8))
    gt = np.frombuffer(f.read(), dtype=np.float32).reshape(ntok, nemb)
print("GT", gt.shape, "mean %.5f std %.5f" % (gt.mean(), gt.std()), flush=True)
if gt.shape == out.shape:
    dd = out-gt
    print("★★ relrms = %.6f%%  maxabs=%.5f" % (100*np.sqrt((dd**2).mean())/np.sqrt((gt**2).mean()), np.abs(dd).max()), flush=True)
    print("逐token relrms 前8:", ["%.3f%%" % (100*np.sqrt(((out[i]-gt[i])**2).mean())/np.sqrt((gt[i]**2).mean())) for i in range(8)], flush=True)
else:
    print("形状不符", out.shape, gt.shape)
print("total %.1fs" % (time.time()-t0), flush=True)
