#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/caden/bonsai2/fork-src')
from gguf_reader import GGUFReader
r = GGUFReader('/home/caden/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf')
for k, f in r.fields.items():
    if 'hadamard' in k or 'prism' in k:
        print(k, '=', f.data)
