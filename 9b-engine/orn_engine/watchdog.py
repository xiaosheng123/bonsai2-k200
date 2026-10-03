#!/usr/bin/env python3
# K200 safety net v3: on any card exception, kill (a) all /dev/xpu* holders and
# (b) any short-lived test binary whose exe path is under /home/caden/ (allowlist excepted).
import re, os, time, signal, subprocess
PAT = re.compile(r"Exception in kernel execution|exception token=|session error|DMA aborted|INTC_STATUS01")
LOG = "/home/caden/orn_engine/watchdog.log"
ALLOW = ("orn3", "serve2.py", "watchdog.py", "safe_run.sh", "wd_guard.sh")
def w(s):
    open(LOG, "a", buffering=1).write(time.strftime("[%F %T] ") + s + "\n")
def holders():
    out = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        fdd = "/proc/%s/fd" % pid
        try:
            for fd in os.listdir(fdd):
                try:
                    t = os.readlink(os.path.join(fdd, fd))
                except Exception:
                    continue
                if t.startswith("/dev/xpu"):
                    out.add(int(pid)); break
        except Exception:
            continue
    return out
def homecaden_tests():
    out = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            exe = os.readlink("/proc/%s/exe" % pid)
        except Exception:
            continue
        base = os.path.basename(exe)
        if exe.startswith("/home/caden/") and not any(a in base or a in exe for a in ALLOW):
            out.add(int(pid))
    return out
open("/tmp/watchdog.pid", "w").write(str(os.getpid()))
w("watchdog v3 started pid=%d (xpu holders + short-lived /home/caden tests)" % os.getpid())
p = subprocess.Popen(["tail", "-F", "-n", "0", "/var/log/kern.log"],
                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, errors="replace")
for line in p.stdout:
    if not PAT.search(line):
        continue
    w("CARD-FAULT: " + line.strip()[:170])
    n = 0
    for pid in (holders() | homecaden_tests()):
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGKILL); n += 1
            w("   killed pid=%d" % pid)
        except Exception:
            pass
    w("   killed %d procs" % n)
