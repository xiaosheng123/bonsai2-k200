#!/usr/bin/env python3
import urllib.request, time, sys
for base_name, url in [
    ('rel_fwht.cu', 'https://cdn.jsdelivr.net/gh/PrismML-Eng/llama.cpp@prism-b10709-9a9394a/ggml/src/ggml-cuda/fwht.cu'),
    ('rel_fwht.cuh', 'https://cdn.jsdelivr.net/gh/PrismML-Eng/llama.cpp@prism-b10709-9a9394a/ggml/src/ggml-cuda/fwht.cuh'),
]:
    for attempt in range(5):
        try:
            b = urllib.request.urlopen(url, timeout=120).read()
            open('/home/caden/bonsai2/rel_src/' + base_name, 'wb').write(b)
            print('OK', len(b), base_name, flush=True)
            break
        except Exception as e:
            print('retry', attempt, base_name, repr(e)[:50], flush=True)
            time.sleep(8)
print('DONE')