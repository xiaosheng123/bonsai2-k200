#!/usr/bin/env python3
# cmp19.py — 用两档的落盘逐阶段对拍: 找第一处数值分歧
import numpy as np, struct, os, sys

def relrms(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return 100.0 * np.sqrt(((a - b) ** 2).sum() / max((b * b).sum(), 1e-30))

ORDER = ['l0_qb.f32', 'l0_kb.f32', 'l0_vb.f32', 'l0_qt0.f32', 'l0_sm0.f32', 'l0_sm1.f32',
         'l0_ao.f32', 'l0_o.f32', 'l0_up.f32', 'l0_qt0.f32', 'ln2_0.f32', 'layer0.f32',
         'layer26.f32', 'emb.bin']

def emb(p):
    b = open(p, 'rb').read(); nt, nd = struct.unpack('<2i', b[:8])
    return np.frombuffer(b[8:], dtype='<f4').reshape(nt, nd).astype(np.float64)

def main(newd, oldd, tag):
    print('=' * 76)
    print('%s   新=%s' % (tag, newd))
    seen = set()
    for f in ORDER:
        if f in seen: continue
        seen.add(f)
        pn, po = os.path.join(newd, f), os.path.join(oldd, f)
        if not (os.path.exists(pn) and os.path.exists(po)):
            print('  %-14s 缺' % f); continue
        bn, bo = open(pn, 'rb').read(), open(po, 'rb').read()
        if f.endswith('.bin'):
            a, b = emb(pn), emb(po)
        else:
            a = np.frombuffer(bn, dtype='<f4').astype(np.float64)
            b = np.frombuffer(bo, dtype='<f4').astype(np.float64)
        same = (bn == bo)
        nd = int((np.asarray(a) != np.asarray(b)).sum())
        print('  %-14s %-8s relrms=%9.4f%%  maxabs=%.3e  位不同=%d/%d' %
              (f, 'byte同' if same else 'byte异', relrms(a, b), np.abs(a - b).max(), nd, a.size))

main(sys.argv[1], sys.argv[2], sys.argv[3])
