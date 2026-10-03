#!/usr/bin/env python3
# kv_demo2.py — 同会话续算验证: 第 2 轮的 assistant 文本必须与第 1 轮真实生成的一字不差
import json, time, urllib.request
def ask(msgs, mt=8, tag=""):
    body = json.dumps({"messages": msgs, "max_tokens": mt}).encode()
    req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); d = json.loads(urllib.request.urlopen(req, timeout=1800).read().decode()); w = time.time()-t0
    k = d.get("k200", {})
    print("[%s] wall=%.2fs sid=%s" % (tag, w, k.get("sid")))
    print("     engine: %s" % k.get("engine_prefill"))
    print("     reply : %r" % d["choices"][0]["message"]["content"][:60])
    return d
a = ask([{"role":"user","content":"你好"}], 8, "turn1")
reply = a["choices"][0]["message"]["content"]
print("     (第1轮回复 %r 将原样作为第2轮 history)" % reply)
ask([{"role":"user","content":"你好"},{"role":"assistant","content":reply},{"role":"user","content":"1+1等于几"}], 8, "turn2")
