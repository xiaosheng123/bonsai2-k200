#!/usr/bin/env python3
# cmp_dumps.py a.bin b.bin —— 逐位置 logits 对拍 (按 (pos,tok) 对齐): argmax/top-k/relrms/maxabs
import struct, sys
import numpy as np

def load(path):
    recs = {}
    with open(path, "rb") as f:
        while True:
            h = f.read(12)
            if len(h) < 12: break
            pos, tok, sz = struct.unpack("<iii", h)
            data = f.read(4 * sz)
            if len(data) < 4 * sz: break
            recs[(pos, tok)] = np.frombuffer(data, dtype=np.float32)
    return recs

A = load(sys.argv[1]); B = load(sys.argv[2])
print("A(%s) 记录 %d 条; B(%s) 记录 %d 条" % (sys.argv[1], len(A), sys.argv[2], len(B)))
common = sorted(set(A) & set(B))
print("可对齐 (pos,tok) 相同 = %d :" % len(common))
same_arg = 0
for k in common:
    a, b = A[k], B[k]
    pa, pb = int(a.argmax()), int(b.argmax())
    eq = (pa == pb)
    same_arg += eq
    d = a.astype(np.float64) - b.astype(np.float64)
    relrms = 100.0 * np.sqrt((d * d).sum() / ((b.astype(np.float64) ** 2).sum() + 1e-300))
    # 去直流后 (本引擎 logits 有巨大公共直流偏置, 见 PROGRESS 12.4)
    da = a.astype(np.float64) - a.mean(); db = b.astype(np.float64) - b.mean()
    dd = da - db
    relrms_ac = 100.0 * np.sqrt((dd * dd).sum() / ((db * db).sum() + 1e-300))
    ta = np.argsort(a)[-5:][::-1]; tb = np.argsort(b)[-5:][::-1]
    print("  pos=%4d tok=%-8d argmax A=%-7d B=%-7d 一致=%s | A_top1=%.4f B_top1=%.4f | relrms=%.5f%% 去直流=%.5f%% maxabs=%.5f"
          % (k[0], k[1], pa, pb, eq, a[pa], b[pb], relrms, relrms_ac, np.abs(d).max()))
    print("        A top5=%s" % (list(map(int, ta)),))
    print("        B top5=%s" % (list(map(int, tb)),))
    if not eq:
        print("        ★ 差异: A_top2=%d(%.4f)  B_top2=%d(%.4f)" % (int(ta[1]), a[ta[1]], int(tb[1]), b[tb[1]]))
print("=== argmax 一致 %d/%d ===" % (same_arg, len(common)))
