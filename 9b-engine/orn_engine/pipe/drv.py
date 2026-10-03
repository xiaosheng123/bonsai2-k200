#!/usr/bin/env python3
# drv.py — 卡独占离线驱动: 一次装载权重, 跑多条请求, 收引擎原始 stdout (含 K200_PROF 行)
#   用法: python3 drv.py <引擎二进制> [max_tokens] <q1> [q2] ...
import sys, os, subprocess, base64, time

MODEL = "/home/caden/orn/Ornith-1.5-9B-Q8_0.gguf"
TMPL = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

eng = sys.argv[1]
mt = int(sys.argv[2])
qs = sys.argv[3:]

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = "/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib"
env.setdefault("K200_PROF", "1")

p = subprocess.Popen([eng, "--n", str(mt), "--model", MODEL],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)

def send(q):
    prompt = TMPL.replace("{q}", q)
    b64 = base64.b64encode(prompt.encode("utf-8", "ignore")).decode("ascii")
    p.stdin.write("@%d@b64:%s\n" % (mt, b64))
    p.stdin.flush()

results = []
for q in qs:
    t0 = time.time()
    send(q)
    toks, started = [], False
    while True:
        line = p.stdout.readline()
        if line == "":
            print("[drv] 引擎提前退出!", flush=True); break
        line = line.rstrip("\n")
        print(line, flush=True)
        if line.startswith("__BEGIN__"):
            started = True
        elif line.startswith("__TOK__ "):
            toks.append(bytes.fromhex(line.split(" ", 1)[1]).decode("utf-8", "replace"))
        elif line.startswith("__END__"):
            break
    dt = time.time() - t0
    txt = "".join(toks)
    results.append((q, txt, len(toks), dt))
    print("[drv] ★ 请求 %r 生成 %d tok 墙钟 %.2fs = %.3f tok/s | 原文=%r" %
          (q, len(toks), dt, len(toks) / dt if dt else 0, txt), flush=True)

p.stdin.close()
try:
    p.wait(timeout=20)
except Exception:
    p.kill()

print("\n================ 汇总 ================")
for q, txt, n, dt in results:
    print("Q=%-14r tok=%3d wall=%6.2fs  %7.3f tok/s  原文=%r" % (q, n, dt, n / dt if dt else 0, txt))
