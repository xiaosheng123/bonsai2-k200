import numpy as np, sys, json
D='/home/caden/orn_engine/nc/'
def load(p):
    b=open(p,'rb').read(); off=0; recs=[]
    while off < len(b):
        pos,tk,sz = np.frombuffer(b,dtype='<i4',count=3,offset=off)
        off+=12
        lg=np.frombuffer(b,dtype='<f4',count=int(sz),offset=off).copy(); off+=4*int(sz)
        recs.append((int(pos),int(tk),lg))
    return recs
A=load(D+sys.argv[1]); B=load(D+sys.argv[2])
print("参考位置数=%d  新位置数=%d" % (len(A),len(B)))
print("%-4s %-8s %-10s %-10s %-12s %-12s %s" % ("pos","tok","relrms%","maxabs","ref_argmax","new_argmax","argmax同"))
worst=0; allsame=True
for (pa,ta,la),(pb,tb,lb) in zip(A,B):
    assert pa==pb and ta==tb and la.size==lb.size
    d=np.abs(la.astype(np.float64)-lb.astype(np.float64))
    rr=100.0*np.sqrt(np.sum(d*d)/np.sum(la.astype(np.float64)**2))
    am_a=la.argmax(); am_b=lb.argmax()
    same = am_a==am_b
    allsame &= bool(same)
    worst=max(worst,rr)
    print("%-4d %-8d %-10.5f %-10.4g %-12d %-12d %s" % (pa,ta,rr,d.max(),am_a,am_b,"✓" if same else "✗"))
print()
print("=== 逐位置 logits 对拍: 最大 relrms = %.5f%% | argmax 全部一致 = %s ===" % (worst, allsame))
json.dump(dict(positions=len(A),max_relrms_pct=worst,argmax_all_same=bool(allsame)),
          open(D+'logits_cmp.json','w'),indent=1)
