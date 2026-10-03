#!/usr/bin/env python3
# cmp20.py — 逐阶段对拍 (带尺寸检查: FFN padding 后 l0_up 宽度从 4304 变 4320)
import numpy as np, struct, os, sys

def relrms(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return 100.0 * np.sqrt(((a - b) ** 2).sum() / max((b * b).sum(), 1e-30))

ORDER = ['l0_qb.f32', 'l0_kb.f32', 'l0_vb.f32', 'l0_qt0.f32', 'l0_sm0.f32', 'l0_sm1.f32',
         'l0_ao.f32', 'l0_o.f32', 'l0_up.f32', 'ln2_0.f32', 'layer0.f32',
         'layer26.f32', 'emb.bin']

def emb(p):
    b = open(p, 'rb').read(); nt, nd = struct.unpack('<2i', b[:8])
    return np.frombuffer(b[8:], dtype='<f4').reshape(nt, nd).astype(np.float64)

def main(newd, oldd, tag):
    print('=' * 78)
    print('%s   新=%s   参=%s' % (tag, newd, oldd))
    seen = set()
    nbyte = nsame = ndiff = 0
    for f in ORDER:
        if f in seen: continue
        seen.add(f)
        pn, po = os.path.join(newd, f), os.path.join(oldd, f)
        if not (os.path.exists(pn) and os.path.exists(po)):
            print('  %-14s 缺(%s)' % (f, 'new' if not os.path.exists(pn) else 'old')); continue
        bn, bo = open(pn, 'rb').read(), open(po, 'rb').read()
        if len(bn) != len(bo):
            a = np.frombuffer(bn, dtype='<f4'); b = np.frombuffer(bo, dtype='<f4')
            print('  %-14s 尺寸不同 new=%d old=%d (元素 %d vs %d)' % (f, len(bn), len(bo), a.size, b.size))
            continue
        if f.endswith('.bin'):
            a, b = emb(pn), emb(po)
        else:
            a = np.frombuffer(bn, dtype='<f4').astype(np.float64)
            b = np.frombuffer(bo, dtype='<f4').astype(np.float64)
        same = (bn == bo)
        nd = int((np.asarray(a) != np.asarray(b)).sum())
        nbyte += 1
        if same: nsame += 1
        else: ndiff += 1
        print('  %-14s %-8s relrms=%11.5f%%  maxabs=%.3e  位不同=%d/%d' %
              (f, '字节同' if same else '字节异', relrms(a, b), np.abs(a - b).max(), nd, a.size))
    print('  ---- 汇总: 可比文件 %d 个, 字节全同 %d, 有差 %d ----' % (nbyte, nsame, ndiff))

main(sys.argv[1], sys.argv[2], sys.argv[3])
