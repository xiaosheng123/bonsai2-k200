#!/usr/bin/env python3
# mmtest_multi.py — 多图 + http URL 两种请求路径的实测 (经 8090)
import base64, json, sys, time, urllib.request

PORT = 8090
def data_url(p):
    return 'data:image/png;base64,' + base64.b64encode(open(p, 'rb').read()).decode()

def ask(content, tag, mx=128):
    body = {"model": "ornith-1.5-9b-k200", "max_tokens": mx,
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    except urllib.error.HTTPError as e:
        print("[%s] HTTP %d: %s" % (tag, e.code, e.read()[:400].decode('utf-8', 'replace')))
        return
    print("[%s] 墙钟 %.2fs" % (tag, time.time() - t0))
    print("[%s] 答原文 = %r" % (tag, d["choices"][0]["message"]["content"]))
    print("[%s] k200 = %s" % (tag, json.dumps(d.get("k200", {}), ensure_ascii=False)))
    print("[%s] usage = %s" % (tag, json.dumps(d.get("usage", {}), ensure_ascii=False)))

mode = sys.argv[1]
M = '/home/caden/ornc/mmtests/'
if mode == 'multi':
    content = [{"type": "image_url", "image_url": {"url": data_url(M + 'img_shapes.png')}},
               {"type": "image_url", "image_url": {"url": data_url(M + 'img_text.png')}},
               {"type": "text", "text": "这是两张图。第一张图里有什么图形？第二张图里的文字是什么？请分别回答。"}]
    ask(content, 'multi-2img')
elif mode == 'http':
    content = [{"type": "image_url", "image_url": {"url": "http://127.0.0.1:8099/big_text.png"}},
               {"type": "text", "text": "请读出图片里的文字（大写字母和数字），只回答文字本身。"}]
    ask(content, 'http-url')
elif mode == 'shapes512':
    content = [{"type": "image_url", "image_url": {"url": data_url(M + 'img_shapes.png')}},
               {"type": "text", "text": "图里有什么？用中文简短回答，说出形状和颜色。"}]
    ask(content, 'shapes512')
