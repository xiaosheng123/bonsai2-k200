import numpy as np
C=np.load("/tmp/gt/Cconv.npy"); pb=np.load("/tmp/gt/pb.npy")
Gv=pb  # 先不扣 bias
a=np.sort(C.reshape(-1)); b=np.sort(Gv.reshape(-1))
print("C range %.4f..%.4f  pb range %.4f..%.4f" % (a[0],a[-1],b[0],b[-1]))
for q in (0.01,0.25,0.5,0.75,0.99):
    print("q=%.2f  C=%.5f  pb=%.5f  ratio=%.4f" % (q, np.quantile(a,q), np.quantile(b,q), np.quantile(b,q)/(np.quantile(a,q)+1e-30)))
print("std ratio", b.std()/a.std())
# 相关系数 (同一索引)
print("corr(same idx) = %.5f" % np.corrcoef(C.reshape(-1), Gv.reshape(-1))[0,1])
