import numpy as np, sys, subprocess, os
sys.path.insert(0,'/home/caden/ornc/mm')
import ggufload, gg
G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
W=H=512
K = (gg.from_flat(G.flat('v.patch_embd.weight'),(16,16,3,1152))+gg.from_flat(G.flat('v.patch_embd.weight.1'),(16,16,3,1152))).reshape(768,1152)
def run(delta):
    a=np.zeros((W,H,3),np.float32)
    for (x,y,c) in delta: a[x,y,c]=1.0
    a.astype('<f4').tofile('/tmp/delta.f32')
    os.makedirs('/tmp/vdd',exist_ok=True)
    r=subprocess.run(['./vistest','run','/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf','/tmp/delta.f32','512','512','/tmp/vdd'],
        cwd='/home/caden/orn_engine',capture_output=True,text=True,env=dict(os.environ,VIS_CPUGEMM='1',VIS_ONLY='0',OMP_NUM_THREADS='4'))
    assert 'OK' in r.stdout, r.stdout[-200:]+r.stderr[-200:]
    conv=np.fromfile('/tmp/vdd/conv.f32',dtype='<f4').reshape(1024,1152)
    cols=np.fromfile('/tmp/vdd/cols.f32',dtype='<f4')
    return conv, cols
for (x,y,c,tag) in [(0,0,0,'p(0,0)'),(16,0,0,'p(1,0)'),(0,16,0,'p(0,1)'),(0,0,1,'c1'),(1,2,2,'off')]:
    conv,cols=run([(x,y,c)])
    nz=np.where(np.abs(conv).sum(axis=1)>1e-6)[0]
    cz=np.where(np.abs(cols)>1e-6)[0]
    print('%-8s delta=(%2d,%2d,c%d): conv 非零行 %s 值[:3] %s | cols 非零位置 %s' % (
        tag,x,y,c,list(nz[:4]), np.round(conv[nz[0]][:3],5) if len(nz) else '-', list(cz[:4])))
    if len(nz): print('         conv 行 %s 前3值: %s' % (list(nz[:3]), [np.round(conv[m][:3],5) for m in nz[:3]]))
print('cols 形状 =', cols.size, '= 1024*768?', cols.size==1024*768)
