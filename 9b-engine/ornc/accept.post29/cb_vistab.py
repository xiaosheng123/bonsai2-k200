#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cb_vistab.py — 只测【表格图】(第29轮: 判定"表格读数退化"是不是本轮引入的)
#   用法: python3 cb_vistab.py <port> [max_tokens]
import base64, hashlib, json, re, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
MT = int(sys.argv[2]) if len(sys.argv) > 2 else 400
B = "http://127.0.0.1:%d/v1/chat/completions" % PORT
TAB9 = ["A", "1", "7", "B", "2", "8", "C", "3", "9"]
P = "/home/caden/ornc/mmtests/big_table.png"
Q = "这张图是一个表格，请逐行说出每一格的字符。"


def ask(path, q, mt=400, timeout=900):
    b = base64.b64encode(open(path, "rb").read()).decode()
    body = json.dumps({"model": "ornith-1.5-9b-k200", "max_tokens": mt, "temperature": 0,
                       "messages": [{"role": "user", "content": [
                           {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b}},
                           {"type": "text", "text": q}]}]}).encode()
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(urllib.request.Request(
        B, data=body, headers={"Content-Type": "application/json"}), timeout=timeout).read().decode())
    k = d.get("k200") or {}
    t = d["choices"][0]["message"]["content"]
    return t, k, time.time() - t0


t, k, w = ask(P, Q, MT)
cells = re.findall(r"第\s*([1-3])\s*格\s*[:：]\s*([^\n\s*|])", t)
seq = [c[1] for c in cells][:9]
n9 = sum(1 for i in range(min(9, len(seq))) if seq[i] == TAB9[i])
print("=== cb_vistab port=%d (md5: %s) ===" % (PORT, k.get("engine_prefill", "")[:0]))
print("表格图回答 sha1=%s wall=%.1fs vision_tokens=%s prompt_tokens=%s" %
      (hashlib.sha1(t.encode()).hexdigest()[:12], w, k.get("vision_tokens"), k.get("prompt_tokens")))
print("原文: %s" % t.replace("\n", " | ")[:400])
print("按'第N格：X'顺序解析: %s  命中 %d/9" % (seq, n9))
print("9/9 = %s" % (seq == TAB9))
