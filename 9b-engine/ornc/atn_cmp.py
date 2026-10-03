#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
atn_cmp.py — ATN_ON_CARD 逐层数值对拍台 (读 K200_ATN_DUMP 落盘的两份 dump)

背景: 阶段②/③/④ 都要回答"卡上这条算子链算出来的东西, 与主机路径差多少"。
      引擎在 K200_ATN_DUMP=<dir> 时, 每层把中间量按【原始 float32 二进制】落盘:
          L<l>_scores.f32  打分 [16, pos+1]
          L<l>_probs.f32   softmax 输出 [16, pos+1]
          L<l>_ao.f32      PV 输出 (gate 之前) [16, 256]
          L<l>_Kc.f32      本 token 写进卡上 KV 的 K 行 [1024]
          L<l>_Vc.f32      同上 V 行 [1024]
      主机档 (K200_ATN=0) 与卡上档 (K200_ATN=1/2) 各跑一次同一个 prompt, 落两个目录, 然后用本工具比。

判据 (ATN_ON_CARD.md §5.3):
  Kc/Vc  必须【逐位相同】(KV 写入无错位)          => relrms == 0
  scores/probs                                   => relrms <= 1e-5  (0.001%)
  ao                                             => relrms <= 1e-3  (0.1%)
  output token ids 与主机档一致                   (不一致 => 降级判"内容正确", 见 §4.4)

用法:
  python3 atn_cmp.py <host_dir> <card_dir> [--layers 3,7,11] [--thr-ao 1e-3] [--thr-soft 1e-5]
  python3 atn_cmp.py --selftest          # 自检(不碰卡, 人工构造两份 dump)
"""
import os
import sys
import struct
import math
import tempfile
import argparse

KINDS = ["scores", "probs", "ao", "Kc", "Vc"]
# 每类的默认阈值 (相对 RMS 百分比, 与 C++ 侧 relrms 同口径: 100*||d||/||ref||)
THR_DEFAULT = {"scores": 1e-3, "probs": 1e-3, "ao": 1e-1, "Kc": 0.0, "Vc": 0.0}


def read_f32(path):
    """读原始 float32 二进制。文件不存在 => None"""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        raw = f.read()
    n = len(raw) // 4
    return list(struct.unpack("<%df" % n, raw[: n * 4]))


def write_f32(path, vals):
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % len(vals), *vals))


def relrms(a, b):
    """与 C++ 侧同口径: 100 * sqrt(sum(d^2)/sum(b^2))"""
    se = 0.0
    s2 = 0.0
    for x, y in zip(a, b):
        d = float(x) - float(y)
        se += d * d
        s2 += float(y) * float(y)
    return 100.0 * math.sqrt(se / (s2 if s2 > 0 else 1e-30))


def maxabs(a, b):
    return max((abs(float(x) - float(y)) for x, y in zip(a, b)), default=0.0)


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1


def cmp_dir(host_dir, card_dir, layers, thr_ao, thr_soft):
    thr = dict(THR_DEFAULT)
    thr["ao"] = thr_ao
    thr["scores"] = thr_soft
    thr["probs"] = thr_soft
    rows = []
    n_layer_found = 0
    for l in layers:
        got_any = False
        for kind in KINDS:
            fn = "L%d_%s.f32" % (l, kind)
            h = read_f32(os.path.join(host_dir, fn))
            c = read_f32(os.path.join(card_dir, fn))
            if h is None and c is None:
                continue
            got_any = True
            if h is None or c is None:
                rows.append((l, kind, -1, None, None, None, "缺一侧 (%s)" % ("宿主缺" if h is None else "卡上缺")))
                continue
            if len(h) != len(c):
                rows.append((l, kind, len(h), None, None, None, "长度不同 (host=%d card=%d)" % (len(h), len(c))))
                continue
            rr = relrms(c, h)
            ma = maxabs(c, h)
            fd = first_diff(c, h)
            limit = thr.get(kind, 1e-3)
            ok = rr <= limit if limit > 0 else (fd == -1)
            tag = "OK" if ok else "★FAIL"
            note = "%s (阈值 %s)" % (tag, "要求逐位相同" if limit == 0 else ("%.4g%%" % limit))
            rows.append((l, kind, len(h), rr, ma, fd, note))
        if got_any:
            n_layer_found += 1
    return rows, n_layer_found


def count_fail(rows):
    """统计不达标/异常项数 (与 print_table 同口径)"""
    n = 0
    for r in rows:
        note = r[6]
        if r[3] is None or "FAIL" in note or "不同" in note or "缺" in note:
            n += 1
    return n


def print_table(rows, host_dir, card_dir):
    print("=" * 118)
    print("逐层对拍:  host = %s" % host_dir)
    print("           card = %s" % card_dir)
    print("=" * 118)
    print("%-7s %-8s %10s %14s %13s %11s  %s" % ("layer", "kind", "n", "relrms(%)", "maxabs", "首个差异", "判据"))
    print("-" * 118)
    nfail = 0
    for (l, kind, n, rr, ma, fd, note) in rows:
        if rr is None:
            print("%-7s %-8s %10s %14s %13s %11s  %s" % (l, kind, "-", "-", "-", "-", note))
            nfail += 1
            continue
        if "FAIL" in note or "不同" in note or "缺" in note:
            nfail += 1
        print("%-7d %-8s %10d %14.6g %13.4g %11s  %s" % (l, kind, n, rr, ma, "无" if fd < 0 else str(fd), note))
    print("-" * 118)
    print("结论: %d 项; 不达标/异常 %d 项" % (len(rows), nfail))
    return nfail


def selftest():
    """人工构造两份 dump 自测: (1) 完全相同 => 必须报 0; (2) 注入 1e-6 相对扰动 => 必须报出正确 relrms。"""
    print("[selftest] atn_cmp.py 自检 (不碰卡)")
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        n_scores, n_ao = 16 * 64, 16 * 256
        base_scores = [math.sin(i * 0.01) * 2.0 for i in range(n_scores)]
        base_ao = [math.cos(i * 0.02) * 0.5 for i in range(n_ao)]
        base_k = [float(i % 7) * 0.25 for i in range(1024)]
        for d in (d1, d2):
            write_f32(os.path.join(d, "L3_scores.f32"), base_scores)
            write_f32(os.path.join(d, "L3_probs.f32"), base_scores)
            write_f32(os.path.join(d, "L3_ao.f32"), base_ao)
            write_f32(os.path.join(d, "L3_Kc.f32"), base_k)
            write_f32(os.path.join(d, "L3_Vc.f32"), base_k)
        # 情形 1: 两份完全相同 => relrms 应全 0, 0 项失败
        rows, nf = cmp_dir(d1, d2, [3], 1e-3, 1e-5)
        bad1 = sum(1 for r in rows if r[3] is not None and r[3] != 0.0)
        print("[selftest] 情形1 完全相同: 非零 relrms 项=%d (期望 0); 失败项=%d (期望 0)" % (bad1, count_fail(rows)))
        assert bad1 == 0, "情形1 应全 0"
        assert count_fail(rows) == 0, "情形1 不应有失败项"
        assert nf == 1, "情形1 应覆盖 1 层"
        # 情形 2: 对 scores 注入 1e-6 相对扰动
        pert = [v * (1.0 + 1e-6) for v in base_scores]
        write_f32(os.path.join(d2, "L3_scores.f32"), pert)
        write_f32(os.path.join(d2, "L3_Kc.f32"), [base_k[0] + 1e-9] + base_k[1:])
        rows, nf = cmp_dir(d1, d2, [3], 1e-3, 1e-5)
        r_scores = [r for r in rows if r[1] == "scores"][0]
        r_kc = [r for r in rows if r[1] == "Kc"][0]
        exp = 100.0 * 1e-6 / math.sqrt(1.0)  # ~1e-4 %
        print("[selftest] 情形2 注入 1e-6 相对扰动: scores relrms=%.6g%% (期望≈%.6g%%), maxabs=%.3g"
              % (r_scores[3], exp, r_scores[4]))
        print("[selftest] 情形2 KV 注入 1e-9 绝对差异: Kc relrms=%.6g%% 首个差异=idx %d (期望 != -1 且判 FAIL)"
              % (r_kc[3], r_kc[5]))
        ok = (bad1 == 0 and abs(r_scores[3]) < 1e-2 and r_kc[5] == 0 and "FAIL" in r_kc[6])
        print("[selftest] 结论: %s" % ("PASS 对拍台可信" if ok else "★ FAIL 对拍台不可信"))
        return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("host_dir", nargs="?")
    ap.add_argument("card_dir", nargs="?")
    ap.add_argument("--layers", default="3,7,11,15,19,23,27,31")
    ap.add_argument("--thr-ao", type=float, default=1e-1)
    ap.add_argument("--thr-soft", type=float, default=1e-3)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.host_dir:
        return selftest()
    layers = [int(x) for x in a.layers.split(",") if x.strip()]
    rows, nlf = cmp_dir(a.host_dir, a.card_dir, layers, a.thr_ao, a.thr_soft)
    if not rows:
        print("★ 两份 dump 里都没有任何 L<layer>_<kind>.f32 => 检查 K200_ATN_DUMP 是否生效 / 层号是否正确")
        return 2
    nfail = print_table(rows, a.host_dir, a.card_dir)
    print("覆盖层数 = %d" % nlf)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
