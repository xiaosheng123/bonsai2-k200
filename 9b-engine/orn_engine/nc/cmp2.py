import numpy as np, sys, json
D='/home/caden/orn_engine/nc/'
def load(p):
    b=open(p,'rb').read(); off=0; recs=[]
    while off < len(b):
        pos,tk,sz = np.frombuffer(b,dtype='<i4',count=3,offset=off); off+=12
        lg=np.frombuffer(b,dtype='<f4',count=int(sz),offset=off).copy(); off+=4*int(sz)
        recs.append((int(pos),int(tk),lg))
    return recs
A=load(D+sys.argv[1]); B=load(D+sys.argv[2])
print("%-4s %-8s %-12s %-14s %-10s %-10s %s" % ("pos","tok","relrms(原样)","relrms(去DC)","DC_ref","DC_new","argmax"))
mx=0; mxc=0; ns=0
for (pa,ta,la),(pb,tb,lb) in zip(A,B):
    a=la.astype(np.float64); b=lb.astype(np.float64)
    # 原样 relrms (被巨大直流偏置主导)
    rr=100*np.sqrt(np.sum((b-a)**2)/np.sum(a*a))
    # ★ 去掉各自直流后的 relrms —— 这才是"影响决策"的那部分误差
    ac=a-a.mean(); bc=b-b.mean()
    rrc=100*np.sqrt(np.sum((bc-ac)**2)/np.sum(ac*ac))
    same = a.argmax()==b.argmax(); ns += same
    mx=max(mx,rr); mxc=max(mxc,rrc)
    print("%-4d %-8d %-12.4f %-14.4f %-10.4f %-10.4f %s" % (pa,ta,rr,rrc,a.mean(),b.mean(),"✓" if same else "✗"))
print()
print("=== 逐位置 logits 对拍 (参考=改前的 orn3ref / 新=gemm_int8) ===")
print("  原样 relrms 最大 = %.4f%%   <-- 被 logits 的巨大直流偏置 (-1.3~-11.2) 主导" % mx)
print("  去直流 relrms 最大 = %.4f%%  <-- 真正影响 argmax 的误差" % mxc)
print("  argmax 一致 = %d/%d" % (ns,len(A)))
json.dump(dict(positions=len(A), relrms_raw_max=mx, relrms_dc_removed_max=mxc, argmax_same="%d/%d"%(ns,len(A))),
          open(D+'logits_cmp.json','w'),indent=1)
