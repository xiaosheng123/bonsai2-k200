#!/usr/bin/env python3
# orn_accept.py —— 经 8090 依次问 6 条验收题, 记录原文/耗时/tok/s。不做重试, 不做 soft_reset。
import json, time, sys, urllib.request, urllib.error

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
OUT  = sys.argv[2] if len(sys.argv) > 2 else "/home/caden/orn_engine/accept.log"
MX   = int(sys.argv[3]) if len(sys.argv) > 3 else 64
QS = ["你好",
      "1+1等于几",
      "用三句话解释什么是光合作用",
      "把“今天天气不错，我们出去走走吧”翻译成英文",
      "写一个Python函数输入整数列表返回最大值索引(只给代码)",
      "我明天要交一份季度税务报告，列5条检查清单"]

def log(s):
    print(s, flush=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")

def ask(q, mx=MX, timeout=2400):
    body = json.dumps({"model": "ornith-1.5-9b-k200", "temperature": 0,
                       "messages": [{"role": "user", "content": q}], "max_tokens": mx}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    dt = time.time() - t0
    return d["choices"][0]["message"]["content"], d.get("usage", {}).get("completion_tokens", -1), dt, d.get("k200", {})

def main():
    log("=== Ornith-1.5-9B Q8_0 @K200 六条验收 (8090, max_tokens=%d) 开始 %s ===" % (MX, time.strftime("%F %T")))
    ok_n = 0
    for i, q in enumerate(QS, 1):
        log("\n########## Q%d: %s" % (i, q))
        try:
            txt, n, dt, k = ask(q)
        except urllib.error.HTTPError as e:
            log("  ✗ HTTP %s: %s" % (e.code, e.read().decode("utf-8", "replace")[:300]))
            log("  -> 按规矩不重试, 直接如实记录")
            continue
        except Exception as e:
            log("  ✗ 异常: %r" % (e,)); continue
        log("回复原文: %s" % txt)
        log("[%d tok, %.1fs, %s tok/s, server_k200=%s]" % (n, dt, round(n / dt, 3) if dt else "-", k))
        if txt.strip():
            ok_n += 1
    log("\n=== 完成: %d/6 条拿到非空回复 %s ===" % (ok_n, time.strftime("%F %T")))

main()
