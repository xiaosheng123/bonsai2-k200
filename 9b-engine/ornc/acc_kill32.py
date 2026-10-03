#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# acc_kill32.py — 第32轮 ③ 验收: 引擎对"客户端中断"健壮
#   A) 发一个长请求(max_tokens=400), 3 秒后【直接关掉客户端 socket】(模拟客户端被 kill)
#      ⇒ 网关必须 !SDEL 取消引擎侧槽; 紧接着的"你好" 必须 <=3s 正常回答 (无需重启 ✗)
#   B) 直接往命令 FIFO 里注入【半截帧】(无换行) 再写一条完整命令
#      ⇒ 引擎必须重同步/丢弃半截帧, 且服务照常 (旧行为: 新命令被静默吞掉 ⇒ 请求干等)
#   C) 注入一条 len= 与实际字节数不符的 !SADD ⇒ 必须 __SREJ__ 拒收, 不入槽
# =============================================================================
import hashlib, json, os, socket, subprocess, threading, time, urllib.request

HOST, PORT = "127.0.0.1", 8090
B = "http://%s:%d/v1/chat/completions" % (HOST, PORT)
MODEL = "ornith-1.5-9b-k200"
GOLDEN = "你好！有什么我可以帮你的吗？😊"
FIFO = "/home/caden/ornc/.ornq_cb.fifo"
OUTD = "/home/caden/ornc/accept.post32"
LOG = os.path.join(OUTD, "kill.log")
LOGW = "/home/caden/ornc/serve2.log"
os.makedirs(OUTD, exist_ok=True)


def log(*a):
    m = " ".join(str(x) for x in a)
    print(m, flush=True)
    with open(LOG, "a") as f:
        f.write(m + "\n")


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.rstrip()


def ask(msgs, mt=24, timeout=600, raw_sock=None):
    body = json.dumps({"model": MODEL, "temperature": 0, "messages": msgs, "max_tokens": mt}).encode()
    if raw_sock is None:
        t0 = time.time()
        d = json.loads(urllib.request.urlopen(
            urllib.request.Request(B, data=body, headers={"Content-Type": "application/json"}),
            timeout=timeout).read().decode())
        return d["choices"][0]["message"]["content"], d.get("k200") or {}, time.time() - t0
    req = ("POST /v1/chat/completions HTTP/1.1\r\nHost: %s:%d\r\nContent-Type: application/json\r\n"
           "Content-Length: %d\r\nConnection: close\r\n\r\n" % (HOST, PORT, len(body))).encode() + body
    raw_sock.sendall(req)


def alive_engine():
    return sh("pgrep -af 'orn3 --n' | grep -v grep | wc -l")


def main():
    log("=== acc_kill32 开始 %s ===" % time.strftime("%F %T"))
    t0, _, w0 = ask([{"role": "user", "content": "你好"}], mt=24)
    log("基线 '你好': wall=%.2fs 逐字节==黄金题=%s" % (w0, t0 == GOLDEN))

    # ---------------- A) 客户端被 kill ----------------
    s = socket.create_connection((HOST, PORT), timeout=10)
    ask([{"role": "user", "content": "请写一段 400 字以上的中文说明，介绍昆仑芯 K200 的架构。"}],
        mt=400, raw_sock=s)
    time.sleep(3.0)
    s.close()                                  # ★ 模拟客户端被 kill (RST/关闭, 不等回答)
    log("A) 已发出长请求 (max_tokens=400) 并在 3.0s 后【直接关闭客户端 socket】")
    t1 = time.time()
    c1, k1, w1 = ask([{"role": "user", "content": "你好"}], mt=24)
    log("A) kill 后紧接着 '你好': wall=%.2fs (要求<=3s=%s) 逐字节==黄金题=%s 答=%r"
        % (w1, w1 <= 3.0, c1 == GOLDEN, c1))
    log("A) 网关取消日志: %s" % (sh("grep -a '客户端中断' %s | tail -3" % LOGW) or "<无>"))
    log("A) 引擎取消日志: %s" % (sh("grep -aE '__SDEL__|预填充被取消' %s | tail -3" % LOGW) or "<无>"))
    log("A) /health: %s" % sh("curl -s -m 6 http://127.0.0.1:8090/health | head -c 200"))
    log("A) 引擎进程数 = %s (必须 >=1, 无需重启)" % alive_engine())

    # ---------------- B) FIFO 半截帧 + 完整命令 ----------------
    n_before = sh("grep -ac '半截残留帧\\|丢弃不完整命令帧\\|丢弃非法命令帧' %s" % LOGW)
    with open(FIFO, "wb", buffering=0) as f:
        f.write(b"XXX-HALF-FRAME-NO-NEWLINE-32nd-round")
    time.sleep(0.5)
    with open(FIFO, "wb", buffering=0) as f:
        f.write(b"!SSTAT\n")
    time.sleep(1.0)
    n_after = sh("grep -ac '半截残留帧\\|丢弃不完整命令帧\\|丢弃非法命令帧' %s" % LOGW)
    log("B) 注入半截帧 + 完整命令 ⇒ 引擎告警/重同步行数 %s -> %s (必须增加)" % (n_before, n_after))
    log("B) 证据: %s" % (sh("grep -aE '半截残留帧|丢弃非法命令帧' %s | tail -2" % LOGW) or "<无>"))
    t2, k2, w2 = ask([{"role": "user", "content": "你好"}], mt=24)
    log("B) 之后 '你好': wall=%.2fs 逐字节==黄金题=%s │ 服务照常=%s" % (w2, t2 == GOLDEN, t2 == GOLDEN))

    # ---------------- C) len= 校验 (半截帧不许入槽) ----------------
    with open(FIFO, "wb", buffering=0) as f:
        f.write(b"!SADD 999998 32 len=100 b64:QUJD\n")     # 声明 100 字节, 实际 3 字节
    time.sleep(1.0)
    log("C) 证据(拒收半截帧): %s" % (sh("grep -aE '丢弃不完整命令帧|truncated-frame' %s | tail -2" % LOGW) or "<无>"))
    t3, k3, w3 = ask([{"role": "user", "content": "你好"}], mt=24)
    log("C) 之后 '你好': wall=%.2fs 逐字节==黄金题=%s" % (w3, t3 == GOLDEN))

    # ---------------- 台账 ----------------
    log("异常累计 = %s" % sh("grep -ac 'Exception in kernel execution' /var/log/kern.log"))
    log("reset_count = %s" % sh("cat /proc/xpu/dev0/reset_count /proc/xpu/dev1/reset_count | tr '\\n' ' '"))
    log("state = %s" % sh("cat /proc/xpu/dev0/state /proc/xpu/dev1/state | tr '\\n' ' '"))
    log("引擎进程数 = %s" % alive_engine())
    log("=== acc_kill32 结束 %s ===" % time.strftime("%F %T"))


main()
