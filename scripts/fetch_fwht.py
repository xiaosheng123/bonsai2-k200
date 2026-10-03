#!/usr/bin/env python3
# 下载 fwht.cuh + 搜索 hadamard 相关文件
import urllib.request

def get(url, timeout=40):
    try:
        req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except Exception as e:
        return None, str(e).encode()

BASE = "https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/master"
files = [
    ("ggml/src/ggml-cuda/fwht.cuh", "/home/caden/bonsai2/fork-src/ggml_cuda_fwht.cuh"),
    ("ggml/src/ggml-metal/ggml-metal-impl.m", "/home/caden/bonsai2/fork-src/ggml_metal.m"),
]
for src, dst in files:
    code, body = get(f"{BASE}/{src}")
    if code == 200:
        open(dst, "wb").write(body)
        print(f"OK {len(body):>8} {src}")
    else:
        print(f"FAIL({code}) {src}")
print("DONE")