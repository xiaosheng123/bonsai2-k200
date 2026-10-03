#!/usr/bin/env python3
# an20.py — 分析 (B) HEAP 路径 l0_ao 错在哪 (按 head 分块统计)
import numpy as np, sys
VNE, VDH, VNH = 1152, 72, 16

a = np.fromfile(sys.argv[1], dtype='<f4').astype(np.float64)   # 新 (heap)
b = np.fromfile(sys.argv[2], dtype='<f4').astype(np.float64)   # 参 (ch19)
T = a.size // VNE
print("T=%d 元素=%d" % (T, a.size))
print("整体: 位不同=%d/%d relrms=%.4f%%" % ((a != b).sum(), a.size,
      100*np.sqrt(((a-b)**2).sum()/(b*b).sum())))
# Ao[h*VDH+d + VNE*t]
for h in range(VNH):
    idx = (h*VDH + np.arange(VDH)[:, None] + VNE*np.arange(T)[None, :]).ravel()
    da, db = a[idx], b[idx]
    nd = int((da != db).sum())
    print("  head %2d: 位不同 %8d/%8d  relrms=%10.4f%%  maxabs=%.4e" %
          (h, nd, da.size, 100*np.sqrt(((da-db)**2).sum()/(db*db).sum()), np.abs(da-db).max()))
# t 半区 (chip) 统计
Mh = (T+1)//2
for p in (0, 1):
    t0, t1 = p*Mh, min(p*Mh+Mh, T)
    idx = (np.arange(VNH)[:, None, None]*VDH + np.arange(VDH)[None, :, None]
           + VNE*np.arange(t0, t1)[None, None, :]).ravel()
    da, db = a[idx], b[idx]
    print("  chip%d (t=%d..%d): 位不同 %d/%d" % (p, t0, t1-1, int((da != db).sum()), da.size))
