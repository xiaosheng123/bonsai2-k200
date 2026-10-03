#!/usr/bin/env python3
import urllib.request, json
b = urllib.request.urlopen('https://api.github.com/repos/PrismML-Eng/llama.cpp/git/trees/prism-b10709-9a9394a?recursive=1', timeout=50).read()
d = json.loads(b)
hits = [x['path'] for x in d['tree'] if x['type'] == 'blob' and ('fwht' in x['path'].lower() or 'hadamard' in x['path'].lower())]
print('total blobs:', len(d['tree']))
for p in hits:
    print(' ', p)