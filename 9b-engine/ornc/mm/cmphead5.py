import numpy as np, sys
sys.path.insert(0, '/home/caden/ornc/mm')
import ggufload, gg
G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
VD = '/tmp/vd512'
pix, W, H = gg.VT(G).load_img('/home/caden/ornc/mmtests/img_shapes.png')
x3 = gg.from_flat(pix, (W, H, 3, 1))[:, :, :, 0]
print('x3 shape', x3.shape)
print('x3[0,0,:]', x3[0, 0, :], 'x3[16,0,:]', x3[16, 0, :], 'x3[0,16,:]', x3[0, 16, :], 'x3[64,64,:]', x3[64, 64, :])
K = gg.from_flat(G.flat('v.patch_embd.weight'), (16, 16, 3, 1152)) + \
    gg.from_flat(G.flat('v.patch_embd.weight.1'), (16, 16, 3, 1152))
cn = np.zeros((32, 32, 1152), np.float32)
for i in range(16):
    for j in range(16):
        cn += x3[i::16, j::16, :] @ K[i, j]
conv_c = np.fromfile(VD+'/conv.f32', dtype='<f4').reshape(1024, 1152)
print('cn[0,0,:3]', cn[0, 0, :3], 'cn[1,0,:3]', cn[1, 0, :3], 'cn[0,1,:3]', cn[0, 1, :3])
for m in (0, 1, 32):
    v = conv_c[m][:3]
    # 在 cn 里找最接近的 patch
    d = ((cn[:, :, :3] - v[None, None, :])**2).sum(axis=2)
    i = np.unravel_index(np.argmin(d), (32, 32))
    print('C++ m=%4d v=%s  -> 最接近 patch (ox=%d, oy=%d) 距离=%.6f' % (m, np.round(v, 5), i[0], i[1], d[i]))
# 全体: 对每个 m 找最佳 patch 并统计
best = np.zeros((1024, 2), int)
for m in range(1024):
    d = ((cn - conv_c[m][None, None, :])**2).sum(axis=2)
    i = np.unravel_index(np.argmin(d), (32, 32))
    best[m] = i
print('m 与最佳 patch 的一致性: (ox==m%%32 且 oy==m//32) -> %d/1024' % int(((best[:, 0] == np.arange(1024) % 32) & (best[:, 1] == np.arange(1024)//32)).sum()))
print('前 12 个 m 的最佳 patch:', [tuple(best[m]) for m in range(12)])
print('m=32,33,64 的最佳 patch:', [tuple(best[m]) for m in (32, 33, 64)])
