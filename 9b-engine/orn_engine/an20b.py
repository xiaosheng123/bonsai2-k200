#!/usr/bin/env python3
# an20b.py — (B) 路径 l0_ao 错块的来源: 是否与别的 head / 别的层 / 别的半区 相同
import numpy as np
VNE, VDH, VNH = 1152, 72, 16
a = np.fromfile("/tmp/vs20/n7/b/l0_ao.f32", dtype='<f4').astype(np.float64)
b = np.fromfile("/tmp/vsw/n7c/new/l0_ao.f32", dtype='<f4').astype(np.float64)
T = a.size // VNE; Mh = (T + 1) // 2
print("T=%d Mh=%d" % (T, Mh))

def blk(arr, h, t0, t1):
    idx = (h * VDH + np.arange(VDH)[:, None] + VNE * np.arange(t0, t1)[None, :]).ravel()
    return arr[idx]

# 1) 芯片 1 的错块是否等于【同 head 的芯片 0 块】或【别的 head 的芯片 1 块】
for h in [0, 2, 6]:
    A = blk(a, h, Mh, min(2 * Mh, T))
    same_self0 = int((A != blk(b, h, 0, Mh)).sum())
    print("head %2d chip1: vs 参同head-chip0 位不同=%d | 数值样例 a=%.5f %.5f  b=%.5f %.5f"
          % (h, same_self0, A[0], A[1], blk(b, h, Mh, min(2*Mh, T))[0], blk(b, h, Mh, min(2*Mh, T))[1]))
    best = None
    for h2 in range(VNH):
        d = int((A != blk(b, h2, Mh, min(2 * Mh, T))).sum())
        if best is None or d < best[1]: best = (h2, d)
    print("     最接近的参 head = %d (位不同 %d/%d)" % (best[0], best[1], A.size))
    # 与参里所有 t 位置 (整行) 比对: 是否只是 t 偏移
    row = np.fromfile("/tmp/vsw/n7c/new/l0_ao.f32", dtype='<f4').astype(np.float64)
    print("     a 的该块 是否 = 参里其它层/无关区域? a[0:4]=%s b[0:4]=%s" %
          (np.array2string(A[:4], precision=6), np.array2string(blk(b, h, Mh, min(2*Mh, T))[:4], precision=6)))
# 2) 是否只是 NaN/0 之类
print("a 有限性: nan=%d inf=%d | b 同位置 nan=%d" %
      (int(np.isnan(a).sum()), int(np.isinf(a).sum()), int(np.isnan(b).sum())))
