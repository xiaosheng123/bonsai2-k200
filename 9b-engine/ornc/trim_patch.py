#!/usr/bin/env python3
# TRIM 补丁: 引擎 MAXT=1024 且【丢尾】(orn3.cpp:1688 ids.resize(i)) ⇒ 长 prompt 把用户最新问题丢掉
# ⇒ 模型吐 EOS ⇒ 0 token ⇒ 前端"空响应"。这里在网关侧裁中间回合, 保 头部system + 最近消息尾。
P = "/home/caden/ornc/serve2.py"
s = open(P, encoding="utf-8").read()
if "_TRIM" in s:
    print("already patched"); raise SystemExit(0)

anchor = "def build_prompt(messages, vision=None):"
assert s.count(anchor) == 1, "anchor=%d" % s.count(anchor)

helper = '''def _clen(m):
    c = m.get("content")
    if isinstance(c, list):
        return sum(len(x.get("text") or "") for x in c if isinstance(x, dict) and x.get("type") == "text")
    return len(c or "")

def _TRIM(msgs, budget=2600):
    """★ 引擎 MAXT=1024 且截断时【丢尾】(orn3.cpp:1688 `ids.resize(i)`),
       长 prompt 会把用户最后的问题丢掉 ⇒ 模型吐 EOS ⇒ 0 token ⇒ 前端"空响应"。
       这里在网关侧按保守字符预算(中文≈1字/token, 取 2.6 字符/token 偏保守)裁掉中间回合,
       保证【system 头 + 最近若干条(含最后一条问题)】留下。"""
    if not msgs:
        return msgs
    total = sum(_clen(m) for m in msgs)
    if total <= budget:
        return msgs
    out = [msgs[0]] if msgs[0].get("role") == "system" else []
    used = sum(_clen(m) for m in out)
    tail = [msgs[-1]]                     # ★ 最后一条(通常是用户当前问题)无条件保留
    used += _clen(msgs[-1])
    i = len(msgs) - 2
    while i >= len(out) and used + _clen(msgs[i]) <= budget:
        tail.insert(0, msgs[i]); used += _clen(msgs[i]); i -= 1
    out = out + tail
    if len(out) < len(msgs):
        log("prompt 过长: %d 条/%d 字符 ⇒ 裁到 %d 条/%d 字符 (保 system 头 + 最近消息尾, 引擎上限 MAXT=1024 tok)"
            % (len(msgs), total, len(out), used))
    return out

'''
s = s.replace(anchor, helper + anchor, 1)

n = s.count('req.get("messages")')
s = s.replace('req.get("messages")', '_TRIM(req.get("messages"))')
print("已包裹 req.get(\"messages\") 处数 = %d" % n)

open(P, "w", encoding="utf-8").write(s)
print("patched ok bytes=%d" % len(s))
