#!/usr/bin/env python3
# an25.py — 第22轮附加证据: ①文本 6 条原文逐字节对比 ②l0_up 前 4304 列数值差异 (定位分歧起点)
import numpy as np, os, sys, re

def lines(p):
    out = []
    for ln in open(p, encoding='utf-8', errors='replace'):
        if ln.startswith('回复原文:'):
            out.append(ln.rstrip('\n'))
    return out

print('=' * 78)
print('① 文本 6 条逐字节对比 (CARDFFN=1 交付档 vs 生产档)')
A = '/home/caden/ornc/accept.post22/accept6.log'
for B in ['/home/caden/ornc/accept.post21/accept6.log', '/home/caden/ornc/accept.post19c/accept6.log']:
    if not os.path.exists(B):
        print('  缺 %s' % B); continue
    a, b = lines(A), lines(B)
    same = sum(1 for x, y in zip(a, b) if x == y)
    print('  %s  vs  %s :  条数 %d/%d, 逐字节相同 %d/%d' % (os.path.basename(A), os.path.basename(B), len(a), len(b), same, min(len(a), len(b))))
    for i, (x, y) in enumerate(zip(a, b), 1):
        if x != y:
            print('    Q%d 不同:' % i)
            print('      新: %s' % x[:200])
            print('      旧: %s' % y[:200])

print('=' * 78)
print('② l0_up 数值差异 (卡上 bias+gelu 档 [T][4320] 的前 4304 列 vs 主机档 [T][4304])')
pc = '/tmp/vs25/n768/card/l0_up.f32'
pr = '/tmp/vs25/n768/ref/l0_up.f32'
if os.path.exists(pc) and os.path.exists(pr):
    T = 2304
    a = np.fromfile(pc, dtype='<f4').reshape(T, 4320)
    b = np.fromfile(pr, dtype='<f4').reshape(T, 4304)
    a2 = a[:, :4304]
    d = (a2.astype(np.float64) - b.astype(np.float64))
    rel = 100.0 * np.sqrt((d ** 2).sum() / max((b.astype(np.float64) ** 2).sum(), 1e-30))
    nd = int((a2 != b).sum())
    print('  前 4304 列: relrms=%.5f%%  maxabs=%.3e  位不同=%d/%d (%.4f%%)' % (rel, np.abs(d).max(), nd, a2.size, 100.0 * nd / a2.size))
    print('  尾部 16 列 (padding, 应为 0): maxabs=%.3e  非零=%d' % (np.abs(a[:, 4304:]).max(), int((a[:, 4304:] != 0).sum())))
    print('  (对照) 差值分布: p50=%.3e p99=%.3e' % (np.percentile(np.abs(d), 50), np.percentile(np.abs(d), 99)))
else:
    print('  缺文件: %s / %s' % (pc, pr))

print('=' * 78)
print('③ 逐阶段 relrms 汇总 (CARDFFN=1 vs CARDFFN=0, 768²/512²)')
for tag, newd, oldd in [('768', '/tmp/vs25/n768/card', '/tmp/vs25/n768/ref'),
                        ('512', '/tmp/vs25/n512/card', '/tmp/vs25/n512/ref')]:
    print('  --- %s² ---' % tag)
    for f, stride_n, stride_o in [('layer0.f32', 1152, 1152), ('layer26.f32', 1152, 1152)]:
        pn, po = os.path.join(newd, f), os.path.join(oldd, f)
        if os.path.exists(pn) and os.path.exists(po):
            x = np.fromfile(pn, dtype='<f4').astype(np.float64); y = np.fromfile(po, dtype='<f4').astype(np.float64)
            r = 100.0 * np.sqrt(((x - y) ** 2).sum() / max((y * y).sum(), 1e-30))
            print('    %-12s relrms=%9.5f%%  maxabs=%.3e' % (f, r, np.abs(x - y).max()))
