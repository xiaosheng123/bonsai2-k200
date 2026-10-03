#!/usr/bin/env python3
import urllib.request, time
BASE = 'https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/prism-b10709-9a9394a/'
files = [
    'ggml/src/ggml-cpu/ggml-cpu.c',
    'ggml/src/ggml-cpu/ggml-cpu.cu',
    'src/llama-graph.cpp',
    'src/llama-model.cpp',
    'ggml/src/ggml-quants.c',
    'ggml/src/ggml-quants.h',
]
for p in files:
    u = BASE + p
    dst = '/home/caden/bonsai2/rel_src/' + p.split('/')[-1]
    import os
    os.makedirs('/home/caden/bonsai2/rel_src', exist_ok=True)
    for attempt in range(4):
        try:
            b = urllib.request.urlopen(u, timeout=120).read()
            open(dst, 'wb').write(b)
            print('OK', len(b), p)
            break
        except Exception as e:
            print('retry', attempt, p, repr(e)[:60], flush=True)
            time.sleep(8)