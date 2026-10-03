#!/usr/bin/env python3
# 下载 fwht.cu (CUDA FWHT 实现)
import urllib.request

def get(url, timeout=40):
    try:
        req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except Exception as e:
        return None, str(e).encode()

BASE = "https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/master"
code, body = get(f"{BASE}/ggml/src/ggml-cuda/fwht.cu")
if code == 200:
    open("/home/caden/bonsai2/fork-src/ggml_cuda_fwht.cu", "wb").write(body)
    print(f"OK fwht.cu {len(body)} bytes")
else:
    print("FAIL", code, body[:100] if body else b'')