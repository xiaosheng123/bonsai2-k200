#!/usr/bin/env python3
import urllib.request, time
url = 'https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/prism-b10709-9a9394a/src/llama-graph.cpp'
dst = '/home/caden/bonsai2/rel_src/llama-graph.cpp'
for attempt in range(10):
    try:
        b = urllib.request.urlopen(url, timeout=300).read()
        open(dst, 'wb').write(b)
        print('OK', len(b), flush=True)
        break
    except Exception as e:
        print('retry', attempt, repr(e)[:60], flush=True)
        time.sleep(12)
print('DONE')