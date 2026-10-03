"""阶段0对拍: conv -> 空间合并重排 -> +patch_bias / +pos_emb  vs 真值桩 dump"""
import numpy as np, struct, sys, os
from PIL import Image
import ggufload, gg

G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
IMG = sys.argv[1] if len(sys.argv) > 1 else '/home/caden/ornc/mmtests/img_shapes.png'
DUMPD = sys.argv[2] if len(sys.argv) > 2 else '/tmp/gt/T2_shapes'


def load_dump(name):
    b = open(os.path.join(DUMPD, name), 'rb').read()
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b[:20])
    a = np.frombuffer(b[20:], dtype='<f4')
    return a, (ne0, ne1, ne2, ne3)


def relrms(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return 100.0*np.sqrt(np.sum((a-b)**2)/max(np.sum(b*b), 1e-30))


def load_img(path):
    im = Image.open(path).convert('RGB')
    a = np.asarray(im, np.uint8).astype(np.float32)
    H, W = a.shape[0], a.shape[1]
    v = (a/255.0 - 0.5)/0.5
    x = np.ascontiguousarray(np.transpose(v, (1, 0, 2)))   # (W,H,3) -> flat x + W*y + W*H*c
    return x.ravel().copy(), W, H


def tensor(name):
    return gg.mk(G.np(name).ravel().copy(), G.tensors[name]['ne'])


pix, W, H = load_img(IMG)
ow, oh = W//16, H//16
print('img %s  W=%d H=%d  patches=%dx%d' % (IMG, W, H, ow, oh))

inp_raw = gg.mk(pix, (W, H, 3, 1))
kern = gg.mat(tensor('v.patch_embd.weight')) + gg.mat(tensor('v.patch_embd.weight.1'))
x3 = gg.mat(inp_raw)[:, :, :, 0]
kw = kh = 16; cin = 3
cols = np.empty((ow*oh, cin*kh*kw), np.float32)
for i in range(kw):
    for j in range(kh):
        blk = x3[i:i+ow*kw:kw, j:j+oh*kh:kh, :]
        for c2 in range(cin):
            cols[:, c2*kh*kw + j*kw + i] = blk[:, :, c2].reshape(-1)
Wm = kern.reshape(cin*kh*kw, 1152)
y = (cols @ Wm).astype(np.float32).reshape(ow, oh, 1152)
conv = gg.mk(np.ascontiguousarray(y).ravel(), (ow, oh, 1152, 1))
print('conv out ne', conv['ne'])

pb = G.np('v.patch_embd.bias').astype(np.float32)
ref, rne = load_dump('patch_bias.bin')
ref = ref.reshape(rne[1], rne[0]).T          # (n_embd, n_patches): flat = i0 + ne0*i1
print('ref patch_bias', rne, ref.shape)

REF = {}
for tag, fn in (('A', gg.cont), ('B', gg.contB)):
    t1 = fn(gg.permute(conv, 1, 2, 0, 3), (1152*2, ow//2, oh, 1))
    t2 = gg.cont(t1, (1152*2, ow//2, 2, oh//2))
    t3 = gg.permute(t2, 0, 2, 1, 3)
    t4 = gg.cont(t3, (1152, ow*oh, 1))
    out = gg.mat(t4)[:, :, 0, 0] + pb[:, None]
    REF[tag] = out
    print('cand %s: shape %s relrms(patch_bias)=%.4f%%' % (tag, out.shape, relrms(out, ref)))

# 位置编码: 512 -> 需要 bilinear(48->32) align_corners; 768 -> 直接用
posw = G.np('v.position_embd.weight').astype(np.float32)     # [2304, 1152]  (ne=[1152,2304])
nside = int(round(1152**0.5 * 0)) or int(np.sqrt(2304))
tab = posw.reshape(2304, 1152)     # index p = x + 48*y   (ne0=1152 最快)
vt = gg.VT(G)
if W//16 == nside:
    grid = tab.reshape(nside, nside, 1152).astype(np.float32)   # [x,y,c]
    print('pos emb: 尺寸吻合, 不做插值')
else:
    grid = vt.interp_align_corners(tab.reshape(nside, nside, 1152).astype(np.float32), nside, ow, oh)
    print('pos emb: bilinear 48->%d align_corners' % ow)
refp, rpne = load_dump('inp_pos_emb.bin')
refp = refp.reshape(rpne[1], rpne[0]).T
for tag, fn in (('A', gg.cont), ('B', gg.contB)):
    pg = gg.mk(np.ascontiguousarray(grid).ravel(), (ow, oh, 1152, 1))
    t1 = fn(gg.permute(pg, 1, 2, 0, 3), (1152*2, ow//2, oh, 1))
    t2 = gg.cont(t1, (1152*2, ow//2, 2, oh//2))
    t3 = gg.permute(t2, 0, 2, 1, 3)
    t4 = gg.cont(t3, (1152, ow*oh, 1))
    pm = gg.mat(t4)[:, :, 0, 0]
    print('cand %s: relrms(pos_part) = %.4f%%' % (tag, relrms(pm, refp - ref)))
    print('cand %s: relrms(inp_pos_emb 直接) = %.4f%%' % (tag, relrms(REF[tag] + pm, refp)))
