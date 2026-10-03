#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cb_accept29.py — 第29轮文本验收 (可指定端口): 黄金题逐字节 + 6 条 vs accept.post23 + 长 prompt 5 次
#   用法: cb_accept29.py <port> <tag>
import json, os, subprocess, sys, time, urllib.request

P = int(sys.argv[1]) if len(sys.argv) > 1 else 8091
TAG = sys.argv[2] if len(sys.argv) > 2 else "p29"
B = "http://127.0.0.1:%d/v1/chat/completions" % P
O = "/home/caden/ornc/accept.post29"
GOLDEN = "你好！有什么可以帮你的吗？😊".replace("我可以帮", "我可以帮")
GOLDEN = "你好！有什么我可以帮你的吗？😊"


def ask(msgs, mt, timeout=900):
    body = json.dumps({"model": "ornith-1.5-9b-k200", "temperature": 0,
                       "messages": msgs, "max_tokens": mt}).encode()
    req = urllib.request.Request(B, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    return d["choices"][0]["message"]["content"], d.get("k200", {}) or {}, time.time() - t0


def log(fh, s):
    print(s, flush=True)
    with open(fh, "a") as f:
        f.write(s + "\n")


def main():
    os.makedirs(O, exist_ok=True)
    fh = os.path.join(O, "text_%s.log" % TAG)
    log(fh, "=== cb_accept29[port=%d tag=%s] 开始 %s ===" % (P, TAG, time.strftime("%F %T")))
    c, k, w = ask([{"role": "user", "content": "你好"}], 24)
    log(fh, "黄金题: ok=%s  %r  wall=%.2fs" % (c == GOLDEN, c, w))
    # 长 prompt 5 次 (DSH 形态 5742 字符/14 条, 问 3+4)
    filler = "你是编码助手。以下是项目规范：所有函数必须有类型注解；提交前跑测试；不要改公共接口。"
    sysmsg = {"role": "system", "content": filler * 40}
    hist = []
    for i in range(6):
        hist.append({"role": "user", "content": ("这是第 %d 轮的历史上下文，" % i) + "补充说明。" * 60})
        hist.append({"role": "assistant", "content": "明白了，我会按规范处理。" * 30})
    msgs = [sysmsg] + hist + [{"role": "user", "content": "只回答一个数字：3 + 4 等于几？"}]
    log(fh, "长 prompt: %d 条 / %d 字符" % (len(msgs), sum(len(m["content"]) for m in msgs)))
    ans5 = []
    for i in range(5):
        try:
            c, k, w = ask(msgs, 32)
            ans5.append(c)
            log(fh, "  第%d次: %r  wall=%.1fs  prefill=%s  reuse=%s  new=%s" %
                (i + 1, c[:40], w, k.get("prompt_tokens"), k.get("prefill_reused_tokens"), k.get("lcp_new_tokens")))
        except Exception as e:
            ans5.append("<ERR %r>" % (e,)); log(fh, "  第%d次 ERR %r" % (i + 1, e))
    all7 = all("7" in (a or "") for a in ans5)
    same = len(set(ans5)) == 1
    log(fh, "长 prompt 5 次: 全含'7'=%s  逐字节一致=%s" % (all7, same))
    log(fh, "--- accept6 (port=%d) ---" % P)
    r = subprocess.run(["python3", "/home/caden/ornc/accept6.py", str(P),
                        os.path.join(O, "accept6_%s.log" % TAG), "64"], capture_output=True, text=True)
    for ln in (r.stdout or "").splitlines()[-4:]:
        log(fh, "  " + ln[:220])
    base = "/home/caden/ornc/accept.post23/accept6.log"
    if os.path.exists(base):
        r2 = subprocess.run(["python3", "/home/caden/orn_engine/cmp_txt21.py",
                             os.path.join(O, "accept6_%s.log" % TAG), base],
                            capture_output=True, text=True)
        for ln in (r2.stdout or "").splitlines():
            log(fh, "  " + ln[:220])
    else:
        log(fh, "  (基线 %s 不存在, 跳过逐字节对拍)" % base)
    for nm, c in (("异常累计", "grep -ac 'Exception in kernel execution' /var/log/kern.log"),
                  ("reset_count", "cat /proc/xpu/dev0/reset_count /proc/xpu/dev1/reset_count | tr '\\n' ' '"),
                  ("state", "cat /proc/xpu/dev0/state /proc/xpu/dev1/state | tr '\\n' ' '"),
                  ("crontab行数", "crontab -l | wc -l")):
        log(fh, "%s = %s" % (nm, subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()))
    log(fh, "=== 结束 %s ===" % time.strftime("%F %T"))


main()
