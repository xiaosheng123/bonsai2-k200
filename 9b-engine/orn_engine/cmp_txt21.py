#!/usr/bin/env python3
# cmp_txt21.py — 比对两份 accept6 日志的"回复原文"是否逐字节一致
import sys, re
def parse(p):
    out, cur = [], None
    for ln in open(p, encoding='utf-8', errors='replace'):
        m = re.match(r'^回复原文: (.*)$', ln.rstrip('\n'))
        if m: out.append(m.group(1))
    return out
a = parse(sys.argv[1]); b = parse(sys.argv[2])
print("参=%s (%d 条)   新=%s (%d 条)" % (sys.argv[2], len(b), sys.argv[1], len(a)))
n = min(len(a), len(b))
same = 0
for i in range(n):
    if a[i] == b[i]:
        same += 1; print("  Q%d: 逐字节一致 ✓ %s" % (i + 1, a[i][:60]))
    else:
        print("  Q%d: ✗ 不同\n      新=%r\n      参=%r" % (i + 1, a[i][:120], b[i][:120]))
print("汇总: %d/%d 逐字节一致" % (same, n))
