#!/usr/bin/env python3
# 侦察 2: patch-embed(im2col+gemm) + spatial-merge 置换的真实语义 (用 gguf 包读权重)
import numpy as np, struct, os, gguf

MMPROJ = "/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf"
DUMP   = "/tmp/gt/T2_shapes"
IMG    = "/home/caden/ornc/mmtests/img_shapes.png"
NX = NY = 32
NEMB = 1152
NPOS = NX * NY
NB, NTB = NX//2, NY//2

def load_dump(name):
    """返回 [ne0, ne1] 的 2D f32 数组"""
    with open(os.path.join(DUMP, name + ".bin"), "rb") as f:
        ty, ne0, ne1, ne2, ne3 = struct.unpack("<5i", f.read(20))
        raw = f.read(ne0*ne1*ne2*ne3*4)
    a = np.frombuffer(raw, dtype=np.float32)
    return a.reshape(ne3, ne2, ne1, ne0)[0, 0].T   # [ne0, ne1]

r = gguf.GGUFReader(MMPROJ)
TD = {t.name: t for t in r.tensors}
def T(name):
    t = TD[name]
    ne = list(t.shape)
    a = t.data
    if a.dtype == np.uint16:      # BF16
        a = (a.astype(np.uint32) << 16).view(np.float32)
    a = a.astype(np.float32)
    return a.reshape(tuple(reversed(ne)))   # [ne_{n-1},...,ne0]

W0 = T("v.patch_embd.weight")      # (1152,3,16,16) [cout,cin,ky,kx]
W1 = T("v.patch_embd.weight.1")
PB = T("v.patch_embd.bias").reshape(-1)
POS = T("v.position_embd.weight")  # (2304,1152)
print("W0", W0.shape, "mean %.6f std %.6f" % (W0.mean(), W0.std()), flush=True)
print("PB", PB.shape, "mean %.6f std %.6f" % (PB.mean(), PB.std()), flush=True)
print("POS", POS.shape, "mean %.6f std %.6f" % (POS.mean(), POS.std()), flush=True)

from PIL import Image
im = np.asarray(Image.open(IMG).convert("RGB"), dtype=np.float32)
im = (im/255.0 - 0.5)/0.5
X = np.stack([im[:, :, c].T for c in range(3)], axis=2)   # X[x,y,c]
print("X", X.shape, "mean %.4f std %.4f" % (X.mean(), X.std()), flush=True)

K = 3*256
P = np.zeros((NX*NY, K), dtype=np.float32)
for ky in range(16):
    for kx in range(16):
        blk = X[kx::16, ky::16, :]           # (32,32,3)
        for c in range(3):
            P[:, c*256 + ky*16 + kx] = blk[:, :, c].reshape(-1)
Wm = np.zeros((NEMB, K), dtype=np.float32)
for ky in range(16):
    for kx in range(16):
        for c in range(3):
            Wm[:, c*256 + ky*16 + kx] = (W0[:, c, ky, kx] + W1[:, c, ky, kx])
C = P @ Wm.T                            # (1024, 1152) patch = x + 32*y
print("C", C.shape, "mean %.5f std %.5f" % (C.mean(), C.std()), flush=True)

pb = load_dump("patch_bias")            # [1152, 1024]
dpe = load_dump("inp_pos_emb") - pb
print("pb", pb.shape, "mean %.5f std %.5f" % (pb.mean(), pb.std()), flush=True)
Gv = pb - PB[:, None]
print("Gv(去bias)", Gv.shape, "mean %.6f std %.6f" % (Gv.mean(), Gv.std()), flush=True)

# ---- 候选 A: t=((yb*NB+xb)*2+dy)*2+dx, 通道=cout
candA = np.zeros((NEMB, NPOS), dtype=np.float32)
for yb in range(NTB):
    for xb in range(NB):
        for dy in range(2):
            for dx in range(2):
                t = ((yb*NB + xb)*2 + dy)*2 + dx
                candA[:, t] = C[xb*2+dx + NX*(yb*2+dy), :]
dd = np.abs(candA - Gv)
print("candA: max|d|=%.6g  relrms=%.6g" % (dd.max(), np.sqrt((dd**2).mean())/np.sqrt((Gv**2).mean())), flush=True)

# ---- 候选 A2: 同样顺序但 (dx,dy) 交换
candA2 = np.zeros((NEMB, NPOS), dtype=np.float32)
for yb in range(NTB):
    for xb in range(NB):
        for dy in range(2):
            for dx in range(2):
                t = ((yb*NB + xb)*2 + dy)*2 + dx
                candA2[:, t] = C[xb*2+dy + NX*(yb*2+dx), :]
dd2 = np.abs(candA2 - Gv)
print("candA2: max|d|=%.6g relrms=%.6g" % (dd2.max(), np.sqrt((dd2**2).mean())/np.sqrt((Gv**2).mean())), flush=True)

np.save("/tmp/gt/Cconv.npy", C)
np.save("/tmp/gt/pb.npy", pb)
np.save("/tmp/gt/dpe.npy", dpe)
np.save("/tmp/gt/POS.npy", POS)
print("saved", flush=True)

# ---- 反查真实置换 ----
flatC = C.reshape(-1)
lut = {}
for i in range(flatC.size):
    v = flatC[i].item()
    if v not in lut: lut[v] = i
print("--- 反查 ---", flush=True)
for t in [0,1,2,3,4,5,6,7,8,16,32,17,256,1023]:
    for c in [0,1,2,3,1151]:
        v = Gv[c, t].item()
        idx = lut.get(v)
        if idx is None:
            print("  F[c=%4d,t=%4d] MISS" % (c,t)); continue
        pi, cout = idx % NPOS, idx // NPOS
        print("  F[c=%4d,t=%4d] <- C(x=%2d,y=%2d,out=%4d)" % (c, t, pi % NX, pi // NX, cout))
