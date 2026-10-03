#!/usr/bin/env python3
# t_fd.py — 看 safe_run 下目标的 fd0 到底是什么 (不碰卡)
import subprocess, time, sys

INNER = ("echo '--- FD0 ---'; readlink /proc/self/fd/0; echo '--- FD LIST ---'; "
         "ls -l /proc/self/fd; echo '--- READ ---'; head -c 40 /dev/stdin | od -c | head -3; echo READ_RC=$?")

cases = {
  "A_direct": ["/bin/sh", "-c", INNER],
  "B_safe":   ["/home/caden/orn_engine/safe_run.sh", "-n", "fdtest", "-t", "20", "--", "/bin/sh", "-c", INNER],
  "C_safe_no_mon": ["/home/caden/orn_engine/safe_run.sh", "-n", "fdtest2", "-t", "20", "--", "/bin/sh", "-c", INNER],
}
which = sys.argv[1] if len(sys.argv) > 1 else "B_safe"
p = subprocess.Popen(cases[which], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, bufsize=0)
time.sleep(1.0)
try:
    p.stdin.write(b"HELLO-WORLD-1234567890\n"); p.stdin.flush()
except Exception as e:
    print("write fail", e)
time.sleep(4)
if p.poll() is None:
    print("== 进程仍活, kill ==")
    p.kill()
out = p.stdout.read().decode("utf-8", "replace")
print("===== %s 输出 =====" % which)
print(out[-2500:])
