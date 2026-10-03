#!/usr/bin/env python3
# TRIM3: 裁 system 时收在句子边界 (避免半句话把小模型带偏); 并优先保留最后一条完整消息
P = "/home/caden/ornc/serve2.py"
s = open(P, encoding="utf-8").read()
if "SYS_CUT" in s:
    print("already patched"); raise SystemExit(0)

old = '''    if used > budget * 0.5:                # 光 system 就占了半预算 ⇒ system 也砍到一半
        out = [dict(msgs[0], content=(msgs[0].get("content") or "")[:int(budget * 0.5 * 2)])]
        used = _clen(out[0])'''
new = '''    if used > budget * 0.5:                # 光 system 就占了半预算 ⇒ system 也砍
        # ★ SYS_CUT: 砍到句子边界为止 (半句话结尾会把模型带偏 —— 实测问"3+4"答"3")
        sc = (msgs[0].get("content") or "")
        lim = int(budget * 0.35)
        acc, cut = 0, len(sc)
        for idx, ch in enumerate(sc):
            acc += _est_tok(ch)
            if acc > lim:
                cut = idx
                break
        head = sc[:cut]
        for sep in ("。", "！", "？", "\\n", "；", ". ", "! ", "? "):
            k = head.rfind(sep)
            if k > len(head) * 0.5:
                head = head[:k + len(sep)]
                break
        if head.rstrip() and not head.rstrip().endswith(("。", "！", "？", ".", "!", "?")):
            head = head.rstrip() + "。"
        out = [dict(msgs[0], content=head)]
        used = _clen(out[0])'''
assert s.count(old) == 1, "count=%d" % s.count(old)
s = s.replace(old, new)
open(P, "w", encoding="utf-8").write(s)
print("patched ok bytes=%d" % len(s))
