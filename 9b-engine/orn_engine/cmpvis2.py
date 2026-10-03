import numpy as np, sys
A = sys.argv[1]; B = sys.argv[2]
def rr(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    return 100.0*np.sqrt(((a-b)**2).sum()/max((b*b).sum(),1e-30))
import os
for nm in sorted(os.listdir(A)):
    pa, pb = os.path.join(A,nm), os.path.join(B,nm)
    if not os.path.exists(pb): continue
    x=np.fromfile(pa,dtype='<f4'); y=np.fromfile(pb,dtype='<f4')
    if x.size != y.size: print('%-14s 尺寸不同 %d vs %d' % (nm,x.size,y.size)); continue
    print('%-14s relrms(卡 vs 主机参考) = %8.4f%%' % (nm, rr(x,y)))
