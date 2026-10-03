#!/usr/bin/env python3
# t_stdin.py — 弄清 Python Popen 链下 stdin 是否被 safe_run 破坏 (不碰卡, 秒级)
import subprocess, time

CASES = [
    ("direct       ", ["python3", "-c", "import sys;d=sys.stdin.read();print('GOT',len(d))"]),
    ("safe_run     ", ["/home/caden/orn_engine/safe_run.sh", "-n", "t_stdin", "-t", "30", "--",
                       "python3", "-c", "import sys;d=sys.stdin.read();print('GOT',len(d))"]),
    ("safe_run+sh  ", ["/home/caden/orn_engine/safe_run.sh", "-n", "t_stdin2", "-t", "30", "--",
                       "/bin/sh", "-c", "cat > /tmp/t_stdin_cat.out; echo CAT_DONE"]),
]
for label, cmd in CASES:
    t0 = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, bufsize=0)
    time.sleep(6)
    rc5 = p.poll()
    if rc5 is None:
        # 没收到 EOF -> 说明 stdin 管道正常保持打开
        p.stdin.write(b"PING\n"); p.stdin.flush()
        time.sleep(2)
        rc7 = p.poll()
        print("%s 6s后rc=%s(PIPE 保持打开✓) 喂一行后rc=%s" % (label, rc5, rc7))
        p.kill()
    else:
        out = p.stdout.read().decode("utf-8", "replace")[:200].replace("\n", " | ")
        print("%s 6s后rc=%s(已退出 => stdin 被当成 EOF ✗) out=%s" % (label, rc5, out))
    time.sleep(0.5)
