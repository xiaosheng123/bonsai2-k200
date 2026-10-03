#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cb_mt.py — 第29轮: 多轮会话前缀复用 (LCP) 在【多槽位批处理路径】上的验收
#   任务书判据: 多轮第 2 轮起预填充 ≤3s (复用 tok > 0), 且输出确定 (同轮重发逐字节同)。
#   用法: python3 cb_mt.py <port> [out.json]
import hashlib, json, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8091
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/caden/ornc/accept.post29/mt.json"
B = "http://127.0.0.1:%d/v1/chat/completions" % PORT

SYS = "你是编码助手。项目规范：所有函数必须有类型注解；提交前跑测试；不要改公共接口。"
HIST = [("背景资料：本项目的接口约定与部署约束如下。", "已了解项目规范。"),
        ("环境：Ubuntu 20.04，4 核，8G 内存。", "收到。"),
        ("注意：所有输出必须中文。", "明白。")]
QS = ["只回答一个数字：5 + 6 等于几？", "只回答一个数字：7 + 8 等于几？",
      "只回答一个数字：9 + 10 等于几？", "只回答一个数字：11 + 12 等于几？",
      "只回答一个数字：13 + 14 等于几？", "只回答一个数字：15 + 16 等于几？"]


def ask(msgs, mt=16, timeout=900):
    body = json.dumps({"model": "ornith-1.5-9b-k200", "temperature": 0,
                       "messages": msgs, "max_tokens": mt}).encode()
    t0 = time.time()
    req = urllib.request.Request(B, data=body, headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    k = d.get("k200") or {}
    t = d["choices"][0]["message"]["content"]
    return dict(text=t, sha=hashlib.sha1(t.encode()).hexdigest()[:12], wall=round(time.time() - t0, 2),
                ptok=k.get("prompt_tokens"), reused=k.get("lcp_reused_tokens"),
                new=k.get("lcp_new_tokens"), prefill=k.get("prefill_secs"),
                fp=k.get("gen_fingerprint"))


msgs = [{"role": "system", "content": SYS * 4}]
for u, a in HIST:
    msgs.append({"role": "user", "content": u * 4})
    msgs.append({"role": "assistant", "content": a})
rows = []
print("=== cb_mt (port %d) 开始 %s ===" % (PORT, time.strftime("%F %T")), flush=True)
for i, q in enumerate(QS):
    m = list(msgs) + [{"role": "user", "content": q}]
    r = ask(m, 16)
    rows.append(dict(rnd=i + 1, q=q, **r))
    print("第%d轮 prompt=%-5s 复用=%-5s 新增=%-5s 预填充=%-7ss 墙钟=%-6ss 答=%r" %
          (i + 1, r["ptok"], r["reused"], r["new"], r["prefill"], r["wall"], r["text"][:20]), flush=True)
# 第 2 轮重发一次 (确定性)
m2 = list(msgs) + [{"role": "user", "content": QS[1]}]
r2 = ask(m2, 16)
same = (r2["sha"] == rows[1]["sha"])
print("重发第2轮: 逐字节一致=%s (%s vs %s)" % (same, r2["sha"], rows[1]["sha"]), flush=True)
bad = [r for r in rows[1:] if (r["prefill"] or 99) > 3.0 or (r["reused"] or 0) <= 0]
ok = (not bad) and same
print("=== 判定: 第2~6轮 预填充≤3s 且 复用>0 => %s ; 重发确定 => %s ; 总: %s ===" %
      ("✓" if not bad else "✗ %d 轮不合格" % len(bad), "✓" if same else "✗", "PASS ✓" if ok else "FAIL ✗"))
try:
    json.dump(dict(rows=rows, resend_same=same, ok=ok), open(OUT, "w"), ensure_ascii=False, indent=1)
    print("写入 %s" % OUT)
except Exception as e:
    print("写文件失败 %r" % (e,))
sys.exit(0 if ok else 1)
