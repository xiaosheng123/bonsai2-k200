"""cmp_stages.py <dumpdir> [T] [EMB] [GTD] — 逐阶段对拍(真值=llama.cpp 桩)"""
import numpy as np, struct, os, sys
VD = sys.argv[1] if len(sys.argv) > 1 else '/tmp/f512ch'
T = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
EP = sys.argv[3] if len(sys.argv) > 3 else '/tmp/gt/embd2_shapes.bin'
D = sys.argv[4] if len(sys.argv) > 4 else '/tmp/gt/T2_shapes'

def rr(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    return 100.0*np.sqrt(((A-B)**2).sum()/max((B*B).sum(), 1e-30))

def rd(name):
    p = os.path.join(D, name)
    if not os.path.exists(p): return None
    b = open(p, 'rb').read()
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b[:20])
    return np.frombuffer(b[20:20+4*ne0*ne1*ne2*ne3], dtype='<f4').reshape(ne1, ne0).T

def vf(name):
    p = os.path.join(VD, name)
    if not os.path.exists(p): return None
    return np.fromfile(p, dtype='<f4').reshape(T, 1152).T

rows = []
x = vf('inp.f32'); r = rd('inp_pos_emb.bin')
if x is not None and r is not None: rows.append(('head (conv+pos+bias)', rr(x, r)))
for il in list(range(0, 6)) + [26]:
    x = vf('layer%d.f32' % il); r = rd('layer_out-%d.bin' % il)
    if x is not None and r is not None: rows.append(('layer %d' % il, rr(x, r)))
x = vf('postln.f32'); r = rd('norm_w-27.bin')
if x is not None and r is not None: rows.append(('post_ln', rr(x, r)))
for k, v in rows: print('  %-22s relrms = %.4f%%' % (k, v))
vp = os.path.join(VD, 'emb.bin')
if os.path.exists(vp) and os.path.exists(EP):
    b = open(vp, 'rb').read(); nt, nd = struct.unpack('<2i', b[:8])
    mine = np.frombuffer(b[8:], dtype='<f4').reshape(nt, nd)
    g = open(EP, 'rb').read(); gt, gd = struct.unpack('<2i', g[:8])
    ref = np.frombuffer(g[8:], dtype='<f4').reshape(gt, gd)
    print('  %-22s relrms = %.4f%%   (mine %dx%d vs 真值 %dx%d)' % ('最终 embedding', rr(mine, ref), nt, nd, gt, gd))
    for nm in ('/tmp/ref_emb.npy',):
        if os.path.exists(nm):
            r2 = np.load(nm)
            print('   (对照) numpy 参考 vs 真值 = %.4f%% ; 本实现 vs numpy 参考 = %.4f%%' % (rr(r2, ref), rr(mine, r2)))
