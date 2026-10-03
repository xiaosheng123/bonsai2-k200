#!/usr/bin/env python3
# TRIM2: 预算改成"中文感知"的 token 估算 (CJK 1字=1token, 非CJK 4字符=1token), 预算 900 token
P = "/home/caden/ornc/serve2.py"
s = open(P, encoding="utf-8").read()
if "EST_TOK" in s:
    print("already patched"); raise SystemExit(0)

i0 = s.index("def _clen(m):")
i1 = s.index("def build_prompt(messages, vision=None):")
new = '''def _est_tok(t):
    """★ EST_TOK: 中文感知的 token 估算 (CJK 1 字 ≈ 1 token, 非 CJK 4 字符 ≈ 1 token)。
       之前用 2.6 字符/token 的英文口径估中文, 导致"裁完仍超标" ⇒ 引擎仍丢尾 ⇒ 仍然空回复。"""
    n = 0
    for ch in t:
        o = ord(ch)
        if 0x2E80 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7FF or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF:
            n += 1
        else:
            n += 0.25
    return n

def _clen(m):
    c = m.get("content")
    if isinstance(c, list):
        c = "".join((x.get("text") or "") for x in c if isinstance(x, dict) and x.get("type") == "text")
    return _est_tok(c or "")

def _TRIM(msgs, budget=900):
    """★ 引擎 MAXT=1024 且截断时【丢尾】(orn3.cpp:1688 `ids.resize(i)`): 长 prompt 会把用户最后
       的问题丢掉, 且截断点会把 ChatML 最后一条消息切成半截(无 <|im_end|>) ⇒ 模型当场吐 EOS
       ⇒ 生成 0 token ⇒ 前端显示"空响应"。
       这里在网关侧按 token 估算裁掉中间回合, 保证【system 头 + 最近若干条(含最后一条问题)】
       留下, 且总估算 ≤ budget(默认 900, 给引擎 MAXT=1024 留余量)。"""
    if not msgs:
        return msgs
    total = sum(_clen(m) for m in msgs)
    if total <= budget:
        return msgs
    out = [msgs[0]] if msgs[0].get("role") == "system" else []
    used = sum(_clen(m) for m in out)
    if used > budget * 0.5:                # 光 system 就占了半预算 ⇒ system 也砍到一半
        out = [dict(msgs[0], content=(msgs[0].get("content") or "")[:int(budget * 0.5 * 2)])]
        used = _clen(out[0])
    tail = [msgs[-1]]                      # ★ 最后一条(用户当前问题)无条件保留
    used += _clen(msgs[-1])
    i = len(msgs) - 2
    while i >= len(out) and used + _clen(msgs[i]) <= budget:
        tail.insert(0, msgs[i]); used += _clen(msgs[i]); i -= 1
    out = out + tail
    if len(out) < len(msgs):
        log("prompt 过长: %d 条/估 %d tok ⇒ 裁到 %d 条/估 %d tok (保 system 头 + 最近消息尾; 引擎上限 MAXT=1024)"
            % (len(msgs), total, len(out), used))
    return out

'''
s = s[:i0] + new + s[i1:]
open(P, "w", encoding="utf-8").write(s)
print("patched ok bytes=%d" % len(s))
