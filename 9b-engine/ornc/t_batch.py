#!/usr/bin/env python3
# t_batch.py — prefill 批量化 A/B 测试 (只经 8090 唯一接口; 不重试)
import json, time, sys, urllib.request, urllib.error

PORT = 8090
TAG  = sys.argv[1] if len(sys.argv) > 1 else "x"
MAXC = int(sys.argv[2]) if len(sys.argv) > 2 else 16
NONCE = sys.argv[3] if len(sys.argv) > 3 else TAG
OUT  = "/home/caden/ornc/t_batch_%s.log" % TAG
GOLDEN = "你好！有什么我可以帮你的吗？😊"

def log(s):
    print(s, flush=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")

def ask(q, mx=MAXC, timeout=2400):
    body = json.dumps({"model": "ornith-1.5-9b-k200", "temperature": 0,
                       "messages": [{"role": "user", "content": q}], "max_tokens": mx}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    dt = time.time() - t0
    return d["choices"][0]["message"]["content"], d.get("usage", {}), dt, d.get("k200", {})

def mklong(nchar, nonce):
    s = "请阅读下面这段材料，然后只用一句话总结它的核心观点。材料如下："
    unit = "在数字化转型过程中，企业需要重新审视自身的组织结构、数据治理能力与业务流程，"
    while len(s) < nchar:
        s += unit
    s += "（编号 %s）" % nonce
    return s

def main():
    log("=== t_batch TAG=%s %s ===" % (TAG, time.strftime("%F %T")))
    txt, u, dt, k = ask("你好", 16)
    log("[GOLDEN] 逐字节一致=%s 原文=%r" % (txt == GOLDEN, txt))
    for nch, nm in ((500, NONCE + "A"), (1000, NONCE + "B")):
        q = mklong(nch, nm)
        txt, u, dt, k = ask(q, 16)
        log("[LONG %d字 nonce=%s] 客户端墙钟=%.2fs prompt_tokens=%s completion=%s" %
            (nch, nm, dt, u.get("prompt_tokens"), u.get("completion_tokens")))
        log("     回复=%r" % txt[:150])
    log("=== done %s ===" % time.strftime("%F %T"))

main()
