"""阶段0: conv -> 合并重排 -> patch_bias / inp_pos_emb 对拍 (修正版)"""
import numpy as np, struct, os, sys
import ggufload, gg

G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
IMG = sys.argv[1] if len(sys.argv) > 1 else '/home/caden/ornc/mmtests/img_shapes.png'
D = sys.argv[2] if len(sys.argv) > 2 else '/tmp/gt/T2_shapes'


def rd(name):
    b = open(os.path.join(D, name), 'rb').read()
    t, ne0, ne1, ne2, ne3 = struct.unpack('<5i', b[:20])
    return np.frombuffer(b[20:], dtype='<f4').reshape(ne1, ne0).T, (ne0, ne1)


vt = gg.VT(G)
conv_part, pos_part = vt.head(IMG)
pb, pne = rd('patch_bias.bin')
pe, _ = rd('inp_pos_emb.bin')
bias = G.np('v.patch_embd.bias').astype(np.float32)

print('conv_part', conv_part.shape, 'ref', pb.shape)
print('relrms(chain(conv)+bias, patch_bias)      = %.4f%%' % gg.relrms(conv_part + bias[:, None], pb))
print('relrms(pos_part, inp_pos_emb-patch_bias)  = %.4f%%' % gg.relrms(pos_part, pe - pb))
print('relrms(chain(conv)+bias+pos_part, inp_pos_emb) = %.4f%%' % gg.relrms(conv_part + bias[:, None] + pos_part, pe))
# (x,y) 排列
yy, xx = np.divmod(np.arange(1024), 32)
print('t=0..19 (x,y):', [(int(xx[t]), int(yy[t])) for t in range(20)])
