#!/usr/bin/env python3
# 从 release 分支拉 CPU FWHT + model 变换源码 (直接 raw, 3 次重试)
import urllib.request, time, os
os.makedirs('/home/caden/bonsai2/rel_src', exist_ok=True)
files = {
    'ggml/src/ggml-cpu/ggml-cpu.c': 'ggml-cpu.c',
    'src/llama-model.cpp': 'llama-model.cpp',
    'src/llama-graph.cpp': 'llama-graph.cpp',
}
BASE = 'https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/prism-b10709-9a9394a/'
for src, dst in files.items():
    u = BASE + src
    for attempt in range(5):
        try:
            b = urllib.request.urlopen(u, timeout=180).read()
            open('/home/caden/bonsai2/rel_src/' + dst, 'wb').write(b)
            print('OK', len(b), dst, flush=True)
            break
        except Exception as e:
            print('retry', attempt, dst, repr(e)[:60], flush=True)
            time.sleep(10)
print('DONE')