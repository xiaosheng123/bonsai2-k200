#!/usr/bin/env python3
# p25.py — 第 22 轮源码补丁: 在 ch21 交付源码 (vis.cpp.cards23) 上改两处, 生成 vis.cpp.cards25
#   ① selftest eps 源缓冲尺寸修复 (ch21 只在 cards24 里修过, 主线 vis.cpp 还是坏的 -> 让源码自洽)
#   ② g_vcardffn 默认值 0 -> 1 (FFN 全卡路径上线; VIS_CARDFFN=0 仍是回退开关)
#   每处替换都断言"命中 1 次", 失败即退出, 不写文件。
import sys
SRC = '/home/caden/orn_engine/vis.cpp.cards23'
DST = '/home/caden/orn_engine/vis.cpp.cards25'
s = open(SRC, encoding='utf-8').read()
orig_len = len(s)

def sub(old, new, tag):
    global s
    n = s.count(old)
    assert n == 1, '[%s] 命中 %d 次 (期望 1)' % (tag, n)
    s = s.replace(old, new)
    print('  [%s] OK (1 次)' % tag)

sub('//   VIS_CARDFFN: 0=关(默认)  1=(A)全卡: ffn_up 留卡 + 卡上 bias/gelu + 卡上 max|A| 归一 + 分块 gemm',
    '//   VIS_CARDFFN: 0=关(回退档) 1=(A)全卡: ffn_up 留卡 + 卡上 bias/gelu + 卡上 max|A| 归一 + 分块 gemm (★第22轮起默认)',
    '注释: 默认值说明')

sub('    g_vcardffn = getenv("VIS_CARDFFN") ? atoi(getenv("VIS_CARDFFN")) : 0;',
    '    g_vcardffn = getenv("VIS_CARDFFN") ? atoi(getenv("VIS_CARDFFN")) : 1;   // ★ 第22轮: 默认 1 = FFN 全卡 (三图/文本/对拍验收通过后上线; VIS_CARDFFN=0 回退)',
    'g_vcardffn 默认 1')

sub('          std::vector<float> epsv(1 << 15, 1e-30f);',
    '          std::vector<float> epsv((size_t)1 << 16, 1e-30f);   // ★ 第21轮修复: 这里要 1<<18 字节 = 65536 个 float (原写成 1<<15 个 -> 源缓冲不足, H2D 直接失败)',
    'selftest epsv 尺寸')

open(DST, 'w', encoding='utf-8').write(s)
print('P25_OK 写入 %s  字节 %d -> %d' % (DST, orig_len, len(s)))
