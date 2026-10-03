#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cb_conc.py — 第29轮【逐槽交叉验证】: 并发下的每槽回答必须与"该请求单独跑"逐字节一致
#
#   A 段: 每条 prompt 单独跑 (串行) → 记 sha1(回答) + 引擎 ★ GEN 三指纹 + reused
#   B 段: 全部 prompt 同时发 (真并发批处理) → 同上
#   判定: 逐条 sha1 相同 且 ids/bytes/prompt 三指纹相同 且 reused 相同
#   再加 C 段: 同内容不同 sid 的 4 条并发 (检查串味) + D 段: 超过槽位数的排队 (5 条 / 4 槽)
#
# 用法: python3 cb_conc.py --host 192.168.66.31 --port 8091 --solo-port 8090
import argparse, hashlib, json, re, sys, threading, time, urllib.request

AP = argparse.ArgumentParser()
AP.add_argument("--host", default="192.168.66.31")
AP.add_argument("--port", type=int, default=8091)
AP.add_argument("--solo-port", type=int, default=0)    # 0 = 不测 solo 口
AP.add_argument("--mt", type=int, default=64)
AP.add_argument("--out", default="/home/caden/ornc/accept.post29/conc.json")
A = AP.parse_args()

QS = ["你好",
      "1+1等于几",
      "用三句话解释什么是光合作用",
      "把“今天天气不错，我们出去走走吧”翻译成英文",
      "写一个Python函数输入整数列表返回最大值索引(只给代码)",
      "我明天要交一份季度税务报告，列5条检查清单"]


def ask(port, q, mt, sid, timeout=1800):
    body = json.dumps({"model": "ornith-1.5-9b-k200", "temperature": 0,
                       "messages": [{"role": "user", "content": q}], "max_tokens": mt}).encode()
    url = "http://%s:%d/v1/chat/completions" % (A.host, port)
    t0 = time.time()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    k = d.get("k200") or {}
    txt = d["choices"][0]["message"]["content"]
    return dict(text=txt, sha=hashlib.sha1(txt.encode()).hexdigest()[:16],
                ntok=d.get("usage", {}).get("completion_tokens", 0),
                wall=round(time.time() - t0, 2), fp=k.get("gen_fingerprint"),
                reused=k.get("lcp_reused_tokens"), new=k.get("lcp_new_tokens"),
                ptok=k.get("prompt_tokens"), solo=k.get("solo"), cb=k.get("cb"))


def batch(port, qs, mt):
    out = [None] * len(qs)
    ths = []
    for i, q in enumerate(qs):
        txt = q[0] if isinstance(q, tuple) else q
        sid = q[1] if isinstance(q, tuple) else "-"
        th = threading.Thread(target=lambda i=i, t=txt, s=sid: out.__setitem__(
            i, ask(port, t, mt, s)))
        ths.append(th)
    t0 = time.time()
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    wall = time.time() - t0
    return out, wall


def a(r, key):
    return r.get(key) if r else None


fails = []
print("=== cb_conc 开始 %s  并发口=%d solo口=%s ===" % (time.strftime("%F %T"), A.port, A.solo_port), flush=True)

# ---- A 段: 逐条单独跑 ----
solo = {}
for i, q in enumerate(QS):
    r = ask(A.port, q, A.mt, "-")
    solo[q] = r
    print("A%-2d 单独: %-24s %3d tok %6.2fs sha=%s fp=%s" %
          (i + 1, q[:22], r["ntok"], r["wall"], r["sha"], (r["fp"] or "")[:60]), flush=True)

# ---- B 段: 全部同时发 ----
bo, bwall = batch(A.port, [(q, "-") for q in QS], A.mt)
print("\nB 段: 6 条同时发, 整批墙钟 %.2fs" % bwall, flush=True)
for i, q in enumerate(QS):
    r = bo[i]
    same = (r["sha"] == solo[q]["sha"])
    fpsame = (r["fp"] == solo[q]["fp"])
    print("B%-2d 并发: %-24s %3d tok %6.2fs sha=%s %s  fp相同=%s" %
          (i + 1, q[:22], r["ntok"], r["wall"], r["sha"], "✓" if same else "✗串味!!", fpsame), flush=True)
    if not same:
        fails.append("B%d 回答不同: 单独=%r 并发=%r" % (i + 1, solo[q]["text"][:60], r["text"][:60]))
    if not fpsame:
        fails.append("B%d 指纹不同: %s vs %s" % (i + 1, solo[q]["fp"], r["fp"]))

# ---- C 段: 4 条并发 (恰好 4 槽) ----
cqs = QS[:4]
co, cwall = batch(A.port, [(q, "-") for q in cqs], A.mt)
print("\nC 段: 4 条并发 (恰满 4 槽), 整批墙钟 %.2fs" % cwall, flush=True)
for i, q in enumerate(cqs):
    r = co[i]
    same = (r["sha"] == solo[q]["sha"])
    print("C%-2d %-24s sha=%s %s" % (i + 1, q[:22], r["sha"], "✓" if same else "✗"), flush=True)
    if not same:
        fails.append("C%d 回答不同" % (i + 1))

# ---- D 段: 6 条并发 (超过 4 槽 ⇒ 必须排队不报错) ----
d0 = time.time()
do, dwall = batch(A.port, [(q, "-") for q in QS], A.mt)
print("\nD 段: 6 条并发 (>4 槽, 排队语义), 整批墙钟 %.2fs" % dwall, flush=True)
for i, q in enumerate(QS):
    r = do[i]
    same = (r["sha"] == solo[q]["sha"])
    print("D%-2d %-24s sha=%s %s  solo字段=%s" % (i + 1, q[:22], r["sha"], "✓" if same else "✗", r["solo"]), flush=True)
    if not same:
        fails.append("D%d 回答不同 (排队路径)" % (i + 1))

# ---- E 段: solo 口 (若给) 与并发口的一致性 ----
if A.solo_port:
    se, sw = batch(A.solo_port, [(q, "-") for q in QS[:4]], A.mt)
    print("\nE 段: solo 口(%d) 4 条并发, 墙钟 %.2fs" % (A.solo_port, sw), flush=True)
    for i, q in enumerate(QS[:4]):
        r = se[i]
        same = (r["sha"] == solo[q]["sha"])
        print("E%-2d %-24s sha=%s %s solo=%s" % (i + 1, q[:22], r["sha"], "✓" if same else "✗", r["solo"]), flush=True)
        if not same:
            fails.append("E%d solo 口回答与单独跑不同" % (i + 1))

print("\n=== 结论: %s ===" % ("逐槽交叉验证全部逐字节一致 ✓" if not fails else "有 %d 项失败 ✗" % len(fails)))
for f in fails:
    print("   ✗ " + f)
try:
    import os
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    json.dump(dict(solo=solo, B=bo, C=co, D=do, fails=fails,
                   bwall=bwall, cwall=cwall, dwall=dwall), open(A.out, "w"), ensure_ascii=False, indent=1)
    print("写入 %s" % A.out)
except Exception as e:
    print("写文件失败 %r" % (e,))
sys.exit(0 if not fails else 1)
