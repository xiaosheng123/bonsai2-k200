#!/usr/bin/env python3
# MAXTOK 补丁: 网关把 max_tokens 夹到 64 (NGEN) 导致"回复一点就断"
#   serve2.py:471  if mt <= 0 or mt > 512: mt = int(NGEN)   ← NGEN=64 ✗
#   start.sh:36    export K200_NGEN=64
# 改为: 只在 <=0 时用默认(默认抬到 512), 上限放到 8192(由引擎按上下文边界自行停止)
P = "/home/caden/ornc/serve2.py"
s = open(P, encoding="utf-8").read()
if "MAXTOK_FIX" in s:
    print("already patched"); raise SystemExit(0)

old = '''        if mt <= 0 or mt > 512: mt = int(NGEN)'''
new = '''        if mt <= 0: mt = int(NGEN)
        # ★ MAXTOK_FIX: 原来 `mt > 512 ⇒ 夹到 NGEN(64)` 会把客户端请求(如 4096)砍成 64 token
        #   ⇒ 症状"回复一点儿就断了"。这里只设一个很宽的上限, 真正的边界交给引擎
        #   (引擎在 pos < MAXT 处自行停止)。可用 K200_MAXTOK 调整。
        _cap = int(os.environ.get("K200_MAXTOK", "8192"))
        if mt > _cap: mt = _cap'''
assert s.count(old) == 1, "count=%d" % s.count(old)
s = s.replace(old, new)
open(P, "w", encoding="utf-8").write(s)
print("serve2.py patched ok")

Q = "/home/caden/ornc/start.sh"
t = open(Q, encoding="utf-8").read()
if "K200_NGEN=64" in t:
    t = t.replace("K200_NGEN=64", "K200_NGEN=512   # ★ 默认生成上限(原来 64 太小, 客户端未指定 max_tokens 时只回 64 token)")
    open(Q, "w", encoding="utf-8").write(t)
    print("start.sh: K200_NGEN 64 -> 512")
else:
    print("start.sh: K200_NGEN 已经是别的值, 未改")
