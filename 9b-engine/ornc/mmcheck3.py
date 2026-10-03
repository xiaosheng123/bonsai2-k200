import numpy as np
C=np.load("/tmp/gt/Cconv.npy")     # (1024,1152) patch=x+32*y
pb=np.load("/tmp/gt/pb.npy")       # (1152,1024)
import gguf
r=gguf.GGUFReader("/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf")
TD={t.name:t for t in r.tensors}
def T(n):
    t=TD[n]; a=np.asarray(t.data)
    return a.reshape(tuple(reversed(list(t.shape)))).astype(np.float32)
PB=T("v.patch_embd.bias").reshape(-1)
Gv=pb-PB[:,None]
Cf=C.reshape(-1)
tot=0; ok=0
recs=[]
for t in [0,1,2,3,4,5,6,7,8,9,10,15,16,17,31,32,33,256,257,512,1023]:
    for c in [0,1,2,3,4,1150,1151]:
        v=Gv[c,t]
        d=np.abs(Cf-v); i=int(np.argmin(d))
        if d[i] < 1e-3*max(1e-6,abs(v)):
            ok+=1
            recs.append((c,t,i%1024,i//1024))
        else:
            recs.append((c,t,-1,-1))
        tot+=1
print("命中 %d/%d (tol 0.1%%)" % (ok,tot))
for c,t,pi,cout in recs[:60]:
    if pi<0: print("  F[c=%4d,t=%4d] MISS" % (c,t)); continue
    print("  F[c=%4d,t=%4d] <- C(x=%2d,y=%2d,cout=%4d)" % (c,t,pi%32,pi//32,cout))
