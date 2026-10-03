"""vscmp2.py <dumpbase> <W> [baseline_tag] — 各档最终 embedding relrms + 与基线一致性 (VM 上跑)"""
import numpy as np, struct, os, sys, glob

base = sys.argv[1]; W = int(sys.argv[2])
bt = sys.argv[3] if len(sys.argv) > 3 else '96_24'
gt = '/tmp/gt/embd2_shapes.bin' if W == 512 else '/tmp/gt/e768_big_shapes.bin'

def relrms(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    return 100.0 * np.sqrt(((A - B) ** 2).sum() / max((B * B).sum(), 1e-30))

def emb(p):
    b = open(p, 'rb').read(); nt, nd = struct.unpack('<2i', b[:8])
    return np.frombuffer(b[8:], dtype='<f4').reshape(nt, nd).astype(np.float64)

E = {}
for d in sorted(glob.glob(os.path.join(base, '*', 'emb.bin'))):
    E[os.path.basename(os.path.dirname(d))] = emb(d)
G = emb(gt) if os.path.exists(gt) else None
print('真值 = %s  档数=%d  %s' % (gt, len(E), ' '.join(sorted(E))))
print('%-14s %12s %14s %12s' % ('档(CH_CHA)', 'vs真值%', 'vs基线%', '位不同数'))
ref = E.get(bt)
for t in sorted(E):
    a = E[t]
    d1 = relrms(a, G) if G is not None else float('nan')
    d2 = relrms(a, ref) if ref is not None else float('nan')
    nd = int((a != ref).sum()) if ref is not None else -1
    print('%-14s %12.4f %14.4f %12d' % (t, d1, d2, nd))
print()
ks = sorted(E)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        print('  %-10s vs %-10s : %8.4f%%  位不同 %d' % (ks[i], ks[j], relrms(E[ks[i]], E[ks[j]]), int((E[ks[i]] != E[ks[j]]).sum())))
