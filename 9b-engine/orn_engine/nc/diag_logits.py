import numpy as np, sys
D='/home/caden/orn_engine/nc/'
def load(p):
    b=open(p,'rb').read(); off=0; recs=[]
    while off < len(b):
        pos,tk,sz = np.frombuffer(b,dtype='<i4',count=3,offset=off); off+=12
        lg=np.frombuffer(b,dtype='<f4',count=int(sz),offset=off).copy(); off+=4*int(sz)
        recs.append((int(pos),int(tk),lg))
    return recs
A=load(D+sys.argv[1]); B=load(D+sys.argv[2])
for (pa,ta,la),(pb,tb,lb) in zip(A,B):
    a=la.astype(np.float64); b=lb.astype(np.float64)
    d=b-a
    ta_=np.argsort(-a)[:5]; tb_=np.argsort(-b)[:5]
    print("=== pos %d (tok %d) ===" % (pa,ta))
    print("  ref: rms=%.4f max=%.4f mean=%.4f   new: rms=%.4f max=%.4f mean=%.4f" %
          (np.sqrt(np.mean(a*a)), a.max(), a.mean(), np.sqrt(np.mean(b*b)), b.max(), b.mean()))
    print("  corr=%.6f  bestfit_scale=<a,b>/<a,a>=%.6f  relrms_after_scale=%.5f%%" % (
        np.corrcoef(a,b)[0,1], np.dot(a,b)/np.dot(a,a),
        100*np.sqrt(np.sum((b*np.dot(a,b)/np.dot(a,a)-a)**2)/np.sum(a*a))))
    print("  ref top5 idx %s val %s" % (list(ta_), np.round(a[ta_],4).tolist()))
    print("  new top5 idx %s val %s" % (list(tb_), np.round(b[tb_],4).tolist()))
    print("  |d|>0.5 的元素数=%d / %d ; |d|max=%.4f ; rms(d)=%.4f" %
          (int(np.sum(np.abs(d)>0.5)), a.size, np.max(np.abs(d)), np.sqrt(np.mean(d*d))))
    print("  90%% 分位 |ref|=%.4f ; |new|=%.4f" % (np.percentile(np.abs(a),90), np.percentile(np.abs(b),90)))
    if pa>=10: break
