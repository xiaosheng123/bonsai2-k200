#!/usr/bin/env python3
import urllib.request, time, sys
urls = [
    'https://cdn.jsdelivr.net/gh/PrismML-Eng/llama.cpp@prism-b10709-9a9394a/src/llama-graph.cpp',
    'https://raw.githack.com/PrismML-Eng/llama.cpp/prism-b10709-9a9394a/src/llama-graph.cpp',
]
dst = '/home/caden/bonsai2/rel_src/llama-graph.cpp'
for url in urls:
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            b = urllib.request.urlopen(req, timeout=180).read()
            if len(b) < 50000:
                print('too small?', len(b), flush=True)
                continue
            open(dst, 'wb').write(b)
            print('OK', len(b), url, flush=True)
            sys.exit(0)
        except Exception as e:
            print('retry', attempt, url[:50], repr(e)[:60], flush=True)
            time.sleep(10)
print('FAILED')