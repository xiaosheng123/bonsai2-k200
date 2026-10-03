import numpy as np, json
D='/home/caden/orn_engine/nc/'
M,N=256,4096
Wf=np.fromfile(D+'Wf.bin',dtype='<f4').reshape(M,N)
x =np.fromfile(D+'x.bin',dtype='<f4')
sr=np.fromfile(D+'srow.bin',dtype='<f4')
Q =np.fromfile(D+'Q.bin',dtype='<i1').reshape(M,N).astype(np.int64)
yi=np.fromfile(D+'yi.bin',dtype='<f4')
yii=np.fromfile(D+'yii.bin',dtype='<f4')
yref=np.fromfile(D+'yref_cpp.bin',dtype='<f8')

Wf=Wf.astype(np.float64); x=x.astype(np.float64); sr=sr.astype(np.float64)
yref_np = Wf @ x                      # 纯 numpy double 参考 (未量化权重)
print("yref_cpp vs yref_np  maxabs = %.3e  (确认落盘参考一致)" % np.max(np.abs(yref-yref_np)))

def relrms(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    return 100.0*np.sqrt(np.sum((a-b)**2)/np.sum(b**2))

# 激活 per-tensor int8 (模拟官方算子内部对 f32 A 的量化)
xmax=np.max(np.abs(x)); xs=xmax/127.0
xaq=np.clip(np.rint(x/xs),-127,127)
AQ = xaq*xs

# 布局判定: gemm_int8 的 C 应该 == A·Q^T (行优先 B[out][in], trans_b=1); 再乘每行 scale
C_pred = AQ @ Q.T
y_ii_pred = sr*C_pred
print("布局+scale 判定: relrms(yii , srow*(AQ@Q.T)) = %.6f%%  maxabs=%.3e" % (relrms(yii,y_ii_pred), np.max(np.abs(yii-y_ii_pred))))
print("           若无每行 scale: relrms(yii , AQ@Q.T)  = %.3f%%" % relrms(yii, C_pred))

print()
print("--- 三方 relrms (numpy double 参考, <1%% 要求) ---")
print("(i)  kq8v2    vs 参考 : %.5f%%" % relrms(yi, yref_np))
print("(ii) gemm_int8 vs 参考 : %.5f%%" % relrms(yii, yref_np))
print("(i) vs (ii)            : %.5f%%" % relrms(yi, yii))
print()
# 误差来源分解
print("--- 误差分解 (与真值比, 只看激活量化这一项贡献) ---")
y_wq_only = sr*(Wf@Q.T*0)  # placeholder
w_deq = (Q*sr[:,None])     # 行 int8 解量化权重
print("权重行 int8 单独(理想激活 double) : %.5f%%" % relrms(w_deq@x, yref_np))
print("激活 per-tensor int8 单独         : %.5f%%" % relrms(Wf@AQ, yref_np))
print("两者叠加 (=gemm_int8 实际)        : %.5f%%" % relrms(w_deq@AQ, yref_np))
print("x 的 max/rms = %.3f  (per-tensor int8 的步长/rms = %.5f)" % (xmax/np.sqrt(np.mean(x*x)), (xmax/127)/np.sqrt(np.mean(x*x))))
res=dict(yref_cpp_vs_np=float(np.max(np.abs(yref-yref_np))),
         layout_relrms=float(relrms(yii,y_ii_pred)),
         rms_i=float(relrms(yi,yref_np)), rms_ii=float(relrms(yii,yref_np)))
open(D+'numpy_check.json','w').write(json.dumps(res,indent=1))
print("saved", D+'numpy_check.json')
