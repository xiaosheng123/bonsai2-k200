#!/usr/bin/env python3
# kv_demo.py — 经 8090 验证 KV/状态复用: 同一 prompt 连发两次, 第二次 prefill 应≈0
import json, time, urllib.request

def ask(msgs, mt=8, tag=""):
    body = json.dumps({"messages": msgs, "max_tokens": mt}).encode()
    req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read().decode())
    w = time.time() - t0
    k = d.get("k200", {})
    print("[%s] wall=%.2fs gen_secs=%s tok/s=%s sid=%s" % (tag, w, k.get("gen_secs"), k.get("tok_per_s"), k.get("sid")))
    print("     engine: %s" % k.get("engine_prefill"))
    print("     reply : %r" % d["choices"][0]["message"]["content"][:80])
    return d, w

if __name__ == "__main__":
    print("=== 1) 第一次 (全量 prefill) ===")
    ask([{"role": "user", "content": "你好"}], 8, "req1")
    print("=== 2) 同一 prompt 再来一次 (期望: 快照命中, prefill≈0) ===")
    ask([{"role": "user", "content": "你好"}], 8, "req2")
    print("=== 3) 多轮续聊 (期望: 会话续算, 只 prefill 新增 token) ===")
    ask([{"role": "user", "content": "你好"},
         {"role": "assistant", "content": "你好！有什么我可以帮你的吗？😊"},
         {"role": "user", "content": "1+1等于几"}], 8, "req3")
