import numpy as np, struct, os
def rr(A,B):
    A=np.asarray(A,np.float64);B=np.asarray(B,np.float64)
    return 100.0*np.sqrt(((A-B)**2).sum()/max((B*B).sum(),1e-30))
def rd(name):
    b=open('/tmp/gt/T2_shapes/'+name,'rb').read()
    t,ne0,ne1,ne2,ne3=struct.unpack('<5i',b[:20])
    return np.frombuffer(b[20:],dtype='<f4').reshape(ne1,ne0).T
print('=== 512x512 (256 token) 卡上 vs llama.cpp 真值 ===')
x=np.fromfile('/tmp/f512/inp.f32',dtype='<f4').reshape(1024,1152).T
print('  patch+pos 输入        relrms = %.4f%%' % rr(x, rd('inp_pos_emb.bin')))
x=np.fromfile('/tmp/f512/layer0.f32',dtype='<f4').reshape(1024,1152).T
print('  层0 输出              relrms = %.4f%%' % rr(x, rd('layer_out-0.bin')))
b=open('/tmp/f512/emb.bin','rb').read(); nt,nd=struct.unpack('<2i',b[:8])
mine=np.frombuffer(b[8:],dtype='<f4').reshape(nt,nd)
g=open('/tmp/gt/embd2_shapes.bin','rb').read(); gt,gd=struct.unpack('<2i',g[:8])
ref=np.frombuffer(g[8:],dtype='<f4').reshape(gt,gd)
print('  ★ 最终图像 embedding  relrms = %.4f%%  (%dx%d)' % (rr(mine,ref),nt,nd))
print('=== 768x768 (576 token) 卡上 vs llama.cpp 真值 ===')
b=open('/tmp/f768/emb.bin','rb').read(); nt,nd=struct.unpack('<2i',b[:8])
mine=np.frombuffer(b[8:],dtype='<f4').reshape(nt,nd)
g=open('/tmp/gt/e768_big_shapes.bin','rb').read(); gt,gd=struct.unpack('<2i',g[:8])
ref=np.frombuffer(g[8:],dtype='<f4').reshape(gt,gd)
print('  ★ 最终图像 embedding  relrms = %.4f%%  (%dx%d vs 真值 %dx%d)' % (rr(mine,ref),nt,nd,gt,gd))
