import numpy as np, struct, sys
sys.path.insert(0, '/home/caden/ornc/mm')
import ggufload, gg
G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
VD = '/tmp/vd512'
im = gg.VT(G).load_img('/home/caden/ornc/mmtests/img_shapes.png')
pix, W, H = im
x3 = gg.from_flat(pix, (W, H, 3, 1))[:, :, :, 0]
W0 = gg.from_flat(G.flat('v.patch_embd.weight'), (16, 16, 3, 1152))
W1 = gg.from_flat(G.flat('v.patch_embd.weight.1'), (16, 16, 3, 1152))
K = W0 + W1
conv_c = np.fromfile(VD+'/conv.f32', dtype='<f4').reshape(1024, 1152)
print('C++ conv[0][:6]  =', conv_c[0][:6])
print('C++ conv[1][:6]  =', conv_c[1][:6])
print('C++ conv[32][:6] =', conv_c[32][:6])


def dot(ox, oy, oc):
    s = 0.0
    for i in range(16):
        for j in range(16):
            for c in range(3):
                s += x3[ox*16+i, oy*16+j, c]*K[i, j, c, oc]
    return s


print('手工 (ox=0,oy=0,oc)  =', [round(dot(0, 0, oc), 5) for oc in range(6)])
print('手工 (ox=1,oy=0,oc)  =', [round(dot(1, 0, oc), 5) for oc in range(6)])
print('手工 (ox=0,oy=1,oc)  =', [round(dot(0, 1, oc), 5) for oc in range(6)])
print('手工 W0 only (0,0)   =', [round(sum(x3[i, j, c]*W0[i, j, c, oc] for i in range(16) for j in range(16) for c in range(3)), 5) for oc in range(3)])
print('手工 W1 only (0,0)   =', [round(sum(x3[i, j, c]*W1[i, j, c, oc] for i in range(16) for j in range(16) for c in range(3)), 5) for oc in range(3)])
# 直接按 C++ 的 im2col 公式算
def dot_cpp(m, oc):
    ox, oy = m % 32, m // 32
    s = 0.0
    for ic in range(3):
        for kh in range(16):
            for kw in range(16):
                kk = ic*256 + kh*16 + kw
                s += x3[ox*16+kw, oy*16+kh, ic] * K.reshape(768, 1152)[kk, oc]
    return s


print('C++公式 (m=0,oc0..2) =', [round(dot_cpp(0, oc), 5) for oc in range(3)])
print('C++公式 (m=1,oc0..2) =', [round(dot_cpp(1, oc), 5) for oc in range(3)])
print('C++公式 (m=32,oc0..2)=', [round(dot_cpp(32, oc), 5) for oc in range(3)])
