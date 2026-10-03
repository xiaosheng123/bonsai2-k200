#!/usr/bin/env python3
import urllib.request, json, time

def get(url, timeout=60):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except Exception as e:
            print('retry', attempt, repr(e)[:80], flush=True)
            time.sleep(5)
    return None, b''

# 1. PrismML-Eng 组织仓库列表
code, body = get('https://api.github.com/orgs/PrismML-Eng/repos?per_page=100')
if code == 200:
    d = json.loads(body)
    print('=== PrismML-Eng repos ===')
    for r in d:
        print(' ', r['name'], '|', (r.get('description') or '')[:80])
else:
    print('org repos fail', code)

# 2. llama.cpp fork 根目录文件
code, body = get('https://api.github.com/repos/PrismML-Eng/llama.cpp/contents/')
if code == 200:
    d = json.loads(body)
    print('=== llama.cpp fork root ===')
    for r in d:
        print(' ', r['type'], r['name'])
else:
    print('root fail', code)
