import numpy as np
pb=np.load("/tmp/gt/pb.npy")
print("pb shape",pb.shape)
print("col0 == col1 ?", np.allclose(pb[:,0],pb[:,1]), " max|col0-col100|", np.abs(pb[:,0]-pb[:,100]).max())
print("col0 stats mean %.6f std %.6f" % (pb[:,0].mean(), pb[:,0].std()))
print("col100 stats mean %.6f std %.6f" % (pb[:,100].mean(), pb[:,100].std()))
print("cov along t: mean over c of std over t =", pb.std(axis=1).mean())
print("first 8 of col0:", pb[:8,0])
print("first 8 of col100:", pb[:8,100])
# 是不是每一列都是同一个 1152 向量?
u=np.unique(np.round(pb,7), axis=1)
print("unique columns:", u.shape[1])
