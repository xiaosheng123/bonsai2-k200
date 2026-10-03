import json
import urllib.request

body = {"model": "ornith-1.5-9b-k200",
        "messages": [{"role": "user", "content": "2+3=? 只给数字"}],
        "max_tokens": 16}
req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(req, timeout=120))
c = d["choices"][0]["message"]["content"]
print("SHORT content=", repr(c), "| expect '5' ->", "5" in c)
cb = d.get("k200", {}).get("cb", {})
print("SHORT cb=", cb)
