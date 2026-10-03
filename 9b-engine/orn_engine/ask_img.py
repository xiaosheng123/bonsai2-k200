#!/usr/bin/env python3
# 图问答 (经 8090): 图片 Base64 塞进 content 数组
import base64, json, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
IMG = sys.argv[2] if len(sys.argv) > 2 else '/home/caden/ornc/mmtests/img_shapes.png'
Q = sys.argv[3] if len(sys.argv) > 3 else '图里有什么？用中文简短回答，说出形状和颜色。'
MX = int(sys.argv[4]) if len(sys.argv) > 4 else 96

raw = open(IMG, 'rb').read()
url = 'data:image/png;base64,' + base64.b64encode(raw).decode()
body = {"model": "ornith-1.5-9b-k200", "max_tokens": MX,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": Q}]}]}
req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    d = json.loads(urllib.request.urlopen(req, timeout=3600).read())
except urllib.error.HTTPError as e:
    print("HTTP %d: %s" % (e.code, e.read()[:500].decode('utf-8', 'replace'))); sys.exit(1)
dt = time.time() - t0
print("图 = %s" % IMG)
print("问 = %s" % Q)
print("答原文 = %r" % d["choices"][0]["message"]["content"])
print("k200 = %s" % json.dumps(d.get("k200", {}), ensure_ascii=False))
print("usage = %s" % json.dumps(d.get("usage", {}), ensure_ascii=False))
print("墙钟 %.2fs" % dt)
