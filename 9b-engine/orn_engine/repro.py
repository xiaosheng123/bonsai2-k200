#!/usr/bin/env python3
# repro.py — 复刻 ornserve.py 的 Popen 链 (stdbuf + safe_run + env + orn), 观察引擎存活与首个真实回复
import subprocess, time, threading, sys

B64 = "PHxpbV9zdGFydHw+dXNlcgoxKzHnrYnkuo7lh6A8fGltX2VuZHw+Cjx8aW1fc3RhcnR8PmFzc2lzdGFudAo8dGhpbms+Cgo8L3RoaW5rPgoK"
LDP = "/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib"
MODE = sys.argv[1] if len(sys.argv) > 1 else "stdbuf"

cmd = []
if MODE == "stdbuf":
    cmd += ["stdbuf", "-oL"]
cmd += ["/home/caden/orn_engine/safe_run.sh", "-n", "repro_" + MODE, "-t", "600", "--",
        "env", "LD_LIBRARY_PATH=" + LDP, "/home/caden/orn_engine/orn",
        "--n", "8", "--model", "/home/caden/orn/Ornith-1.5-9B-Q8_0.gguf"]
print("CMD:", " ".join(cmd), flush=True)
t0 = time.time()
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, bufsize=0)

END = [False]
def rd():
    for line in p.stdout:
        L = line.decode("utf-8", "replace").rstrip()
        print("[%6.1fs] %s" % (time.time() - t0, L), flush=True)
        if L == "__END__":
            END[0] = True
threading.Thread(target=rd, daemon=True).start()

time.sleep(40)
print("=== 40s 后引擎存活? rc=%s ===" % p.poll(), flush=True)
if p.poll() is None:
    try:
        p.stdin.write(("@" + "8" + "@b64:" + B64 + "\n").encode())
        p.stdin.flush()
        print("=== 已喂入请求, 等回复 ===", flush=True)
    except Exception as e:
        print("写 stdin 失败: %r" % e, flush=True)
    for i in range(8):
        time.sleep(15)
        if END[0]:
            print("=== 已收到 __END__, 本轮结束 (引擎仍活) ===", flush=True)
            break
        if p.poll() is not None:
            print("=== 引擎已退出 rc=%s (第%d次检查) ===" % (p.poll(), i + 1), flush=True)
            break
        print("=== %ds: 引擎仍在 (rc=None) ===" % (15 * (i + 1)), flush=True)
    else:
        print("=== 360s 到, 引擎仍在, 主动收尾 ===", flush=True)
p.terminate()
time.sleep(2)
if p.poll() is None: p.kill()
print("=== 总耗时 %.1fs, 最终 rc=%s ===" % (time.time() - t0, p.poll()), flush=True)
