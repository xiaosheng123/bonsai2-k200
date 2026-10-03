import numpy as np, sys, subprocess, os
sys.path.insert(0, '/home/caden/ornc/mm')
import ggufload, gg
G = ggufload.GGUF('/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf')
W = H = 512
W0 = gg.from_flat(G.flat('v.patch_embd.weight'), (16, 16, 3, 1152))
W1 = gg.from_flat(G.flat('v.patch_embd.weight.1'), (16, 16, 3, 1152))
K = (W0 + W1).reshape(768, 1152)          # [k, oc]


def run(delta, tag):
    a = np.zeros((W, H, 3), np.float32)
    for (x, y, c) in delta:
        a[x, y, c] = 1.0
    a.astype('<f4').tofile('/tmp/delta.f32')
    os.makedirs('/tmp/vdd', exist_ok=True)
    out = subprocess.run(['./vistest', 'run', '/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf',
                          '/tmp/delta.f32', '512', '512', '/tmp/vdd'],
                         cwd='/home/caden/orn_engine', capture_output=True, text=True,
                         env=dict(os.environ, VIS_CPUGEMM='1', VIS_ONLY='0', OMP_NUM_THREADS='4'))
    if 'OK' not in out.stdout:
        print('run fail', out.stdout[-300:], out.stderr[-300:]); return None
    return np.fromfile('/tmp/vdd/conv.f32', dtype='<f4').reshape(1024, 1152)


# delta (0,0,0): patch (0,0) 的 kw=0,kh=0,ic=0
c = run([(0, 0, 0)], 'p0')
print('delta(0,0,c0): conv[0][:4] =', c[0][:4])
print('    期望 W[k=0][oc0..3]     =', K[0][:4]/max(abs(K[0][:4]).max(), 1e-9)*max(abs(c[0][:4]).max(), 1e-9))
print('    期望 K[0][:4] (未量化)  =', K[0][:4])
print('    归一化后 C++/K 比值    =', (c[0][:4]/np.where(np.abs(K[0][:4]) > 1e-9, K[0][:4], 1e-9)))
# 找出该 delta 影响到的哪些 m
nz = np.where(np.abs(c).sum(axis=1) > 1e-6)[0]
print('    非零 m 数 =', len(nz), '前 6 个 =', nz[:6])
# delta (16,0,0): kw=0,kh=0,ic=0 的 patch (1,0)
c2 = run([(16, 0, 0)], 'p1')
nz2 = np.where(np.abs(c2).sum(axis=1) > 1e-6)[0]
print('delta(16,0,0): 非零 m =', nz2[:6], ' conv[m][:3] =', c2[nz2[0]][:3] if len(nz2) else None)
print('    期望 K[0][:3] =', K[0][:3])
# delta (0,16,0): patch (0,1)
c3 = run([(0, 16, 0)], 'p2')
nz3 = np.where(np.abs(c3).sum(axis=1) > 1e-6)[0]
print('delta(0,16,0): 非零 m =', nz3[:6], ' conv[m][:3] =', c3[nz3[0]][:3] if len(nz3) else None)
print('    期望 K[k=16][:3] =', K[16][:3])
# delta (0,0,1): ic=1 -> k = 1*256 = 256
c4 = run([(0, 0, 1)], 'p3')
nz4 = np.where(np.abs(c4).sum(axis=1) > 1e-6)[0]
print('delta(0,0,c1): 非零 m =', nz4[:6], ' conv[m][:3] =', c4[nz4[0]][:3] if len(nz4) else None)
print('    期望 K[256][:3] =', K[256][:3])
