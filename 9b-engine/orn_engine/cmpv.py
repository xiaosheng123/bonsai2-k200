"""对拍: vistest 的 dump (vs llama.cpp 真值桩)"""
import numpy as np, struct, os, sys
D = os.environ.get('GTDIR', '/tmp/gt/T2_shapes')
VD = sys.argv[1] if len(sys.argv) > 1 else '/tmp/vd512'


def relrms(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    return 100.0*np.sqrt(((A-B)**2).sum()/max((B*B).sum(), 1e-30))


def rd(name, base=D):
    p = os.path.join(base, name)
    if not os.path.exists(p):
        return None
    b = open(p, 'rb').read()
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b[:20])
    return np.frombuffer(b[20:], dtype='<f4').reshape(ne1, ne0).T


def vf(name, T=1024):
    p = os.path.join(VD, name)
    if not os.path.exists(p):
        print('  缺 %s' % p); return None
    return np.fromfile(p, dtype='<f4').reshape(T, 1152).T   # 存的是 [VNE, T] ggml 序


pairs = [('inp.f32', 'inp_pos_emb.bin'), ('layer0.f32', 'layer_out-0.bin')]
for a, b in pairs:
    x = vf(a)
    r = rd(b)
    if x is not None and r is not None:
        print('%-14s vs %-20s relrms = %.4f%%' % (a, b, relrms(x, r)))
# postln / 最终 embedding
if os.path.exists(os.path.join(VD, 'postln.f32')):
    x = vf('postln.f32')
    r = rd('norm_w-27.bin')
    if r is not None:
        print('postln.f32     vs norm_w-27        relrms = %.4f%%  (dump 名存疑, 仅参考)' % relrms(x, r))
EP = os.environ.get('EMB', '/tmp/gt/embd2_shapes.bin')
vp = os.path.join(VD, 'emb.bin')
if os.path.exists(vp) and os.path.exists(EP):
    bb = open(vp, 'rb').read(); nt, nd = struct.unpack('<2i', bb[:8])
    mine = np.frombuffer(bb[8:], dtype='<f4').reshape(nt, nd)
    gb = open(EP, 'rb').read(); gt, gd = struct.unpack('<2i', gb[:8])
    ref = np.frombuffer(gb[8:], dtype='<f4').reshape(gt, gd)
    print('★ 最终图像 embedding %s: relrms = %.4f%% (mine %dx%d vs 真值 %dx%d)' % (
        vp, relrms(mine, ref), nt, nd, gt, gd))
    # 参考: numpy 参考实现对真值
    if os.path.exists('/tmp/ref_emb.npy'):
        rr = np.load('/tmp/ref_emb.npy')
        print('   (对照) numpy 参考实现对真值 = %.4f%%; 本实现对 numpy 参考 = %.4f%%' % (
            relrms(rr, ref), relrms(mine, rr)))
