#!/usr/bin/env python3
# 下载 llama-graph.cpp (计算图构建, hadamard hint 设置处)
import urllib.request

def get(url, timeout=60):
    try:
        req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except Exception as e:
        return None, str(e).encode()

BASE = "https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/master"
for src, dst in [
    ("src/llama-graph.cpp", "src_llama-graph.cpp"),
    ("src/llama-graph.h", "src_llama-graph.h"),
    ("src/llama-cparams.h", "src_llama-cparams.h"),
]:
    code, body = get(f"{BASE}/{src}")
    if code == 200:
        open(f"/home/caden/bonsai2/fork-src/{dst}", "wb").write(body)
        print(f"OK {len(body):>8} {src}")
    else:
        print(f"FAIL({code}) {src}")
print("DONE")