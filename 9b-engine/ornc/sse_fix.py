import py_compile
P = "/home/caden/ornc/serve2.py"
s = open(P, encoding="utf-8").read()
old = '            def cb(tb):\n                c = tb.decode("utf-8", "ignore")\n'
new = '            def cb(tb):\n                c = tb if isinstance(tb, str) else tb.decode("utf-8", "ignore")\n'
if old in s:
    s = s.replace(old, new, 1)
    open(P, "w", encoding="utf-8").write(s)
    py_compile.compile(P, doraise=True)
    print("[fix] cb 现在同时接受 str/bytes, 语法 OK")
else:
    print("[fix] 没找到目标行 (可能已修过)")
