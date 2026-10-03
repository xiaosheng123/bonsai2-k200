#!/usr/bin/env python3
import urllib.request, json
b = urllib.request.urlopen('https://api.github.com/repos/PrismML-Eng/llama.cpp/git/trees/prism-b10709-9a9394a?recursive=1', timeout=60).read()
d = json.loads(b)
for x in d['tree']:
    if x['type'] == 'blob' and 'ggml-cpu' in x['path']:
        print(x['path'])