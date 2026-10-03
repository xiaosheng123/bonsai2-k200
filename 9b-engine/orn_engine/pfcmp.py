#!/usr/bin/env python3
# pfcmp.py —— 对拍两份 pfbench JSON: (a) 文本逐字节 (b) 预填充 ms/tok (c) 保留条数
#   用法: pfcmp.py <基线.json> <新版.json> [<新版2.json> ...]
import json, sys

def load(p):
    with open(p) as f:
        return json.load(f)

def itm(D):
    return {i["name"]: i for i in D["items"]}

def main():
    A = load(sys.argv[1]); Bs = [load(p) for p in sys.argv[2:]]
    print("基线 %-28s md5=%s tag=%s ready=%s 启动=%.1fs" % (sys.argv[1], A["engine_md5"], A["tag"], A["ready"], A["startup_s"]))
    for p, B in zip(sys.argv[2:], Bs):
        print("新版 %-28s md5=%s tag=%s ready=%s 启动=%.1fs" % (p, B["engine_md5"], B["tag"], B["ready"], B["startup_s"]))
    am = A.get("meta", {})
    print("\n=== 网关 _TRIM 保留条数 / 估算 tok ===")
    print("  原始会话: %s 条 / %s 字符 (两次运行应一致)" % (am.get("n14"), am.get("chars14")))
    print("  基线(预算900):  保留 %s 条 / 估 %s tok" % (am.get("kept900"), am.get("est900")))
    for B in Bs:
        bm = B.get("meta", {})
        print("  %s(预算1700): 保留 %s 条 / 估 %s tok" % (B["tag"], bm.get("kept1700"), bm.get("est1700")))

    print("\n=== 文本逐字节 (raw = __TOK__ 拼出的原始 token 字节) ===")
    ai = itm(A)
    for B in Bs:
        bi = itm(B)
        for n in ai:
            if n not in bi: continue
            a = ai[n].get("raw") or ""; b = bi[n].get("raw") or ""
            print("  %-14s %s | 基线 %3d 字节 %-40r | 新版 %3d 字节 %-40r"
                  % (n, "一致 ✓" if a == b else "★不同✗", len(a), a[:34], len(b), b[:34]))

    print("\n=== 预填充 (引擎自报 prompt=N tok … ms/tok) ===")
    print("  %-14s %-16s %-12s %-12s %-10s %-10s" % ("名字", "prompt_tok(基/新)", "ms/tok(基)", "ms/tok(新)", "wall(基)", "wall(新)"))
    for n in ai:
        a = ai[n]; cells = []
        for B in Bs:
            b = itm(B).get(n, {}); cells.append(b)
        for b in cells:
            print("  %-14s %-16s %-12s %-12s %-10s %-10s"
                  % (n, "%s / %s" % (a.get("prompt_tok"), b.get("prompt_tok")),
                     a.get("ms_per_tok"), b.get("ms_per_tok"),
                     round(a.get("wall", 0), 1), round(b.get("wall", 0), 1)))

    print("\n=== 预填充耗时分解 (PPROF 原文) ===")
    for nm, D in [("基线", A)] + [("新版%d" % (i + 1), b) for i, b in enumerate(Bs)]:
        for it in D["items"]:
            for L in it.get("pprof", []):
                print("  [%s] %s" % (nm, L))

main()
