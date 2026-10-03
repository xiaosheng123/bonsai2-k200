#!/usr/bin/env python3
# stab_test.py [TAG] [N] —— 长稳 + 冷启动 + 黄金 + 长 prompt 综合测试 (只经 8090)
import json, time, sys, urllib.request, urllib.error
PORT = 8090
TAG = sys.argv[1] if len(sys.argv) > 1 else "s"
N   = int(sys.argv[2]) if len(sys.argv) > 2 else 22
OUT = "/home/caden/ornc/stab_%s.log" % TAG
GOLDEN = "你好！有什么我可以帮你的吗？😊"

def log(s):
    print(s, flush=True)
    with open(OUT, "a") as f: f.write(s + "\n")

def health():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8090/health", timeout=6) as r:
            return json.load(r)
    except Exception as e:
        return {"status": "err", "e": repr(e)}

def ask(q, mx, timeout=2400):
    body = json.dumps({"model": "ornith-1.5-9b-k200", "temperature": 0,
                       "messages": [{"role": "user", "content": q}], "max_tokens": mx}).encode()
    req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"], d.get("usage", {}), time.time() - t0

def filler(nch, nonce):
    s = "请阅读下面这段材料，然后只用一句话总结它的核心观点。材料如下："
    u = "在数字化转型过程中，企业需要重新审视自身的组织结构、数据治理能力与业务流程，"
    while len(s) < nch: s += u
    return s + "（编号 %s）" % nonce

def main():
    log("=== stab_test TAG=%s %s N=%d ===" % (TAG, time.strftime("%F %T"), N))
    h = health(); log("[health0] %s" % json.dumps(h, ensure_ascii=False))
    ok = fail = 0
    # 0) 冷启动首请求计时 (等服务 ready 后立刻发)
    t0 = time.time()
    while True:
        h = health()
        if h.get("ready"): break
        if time.time() - t0 > 300: log("[cold] 300s 仍未 ready"); break
        time.sleep(2)
    tready = time.time() - t0
    log("[cold] /health ready=true 用时 %.1fs (从脚本开始) => 立刻发第一发「你好」" % tready)
    try:
        txt, u, dt = ask("你好", 16)
        log("[cold] 第一发「你好」墙钟 %.2fs  逐字节一致=%s 原文=%r usage=%s" % (dt, txt == GOLDEN, txt, u))
        ok += 1
    except Exception as e:
        log("[cold] 第一发失败: %r" % (e,)); fail += 1
    # 1) 混合长稳
    kinds = [
        ("短", lambda i: "第%d个问题：1+1等于几？请只回答数字。" % i, 24),
        ("中", lambda i: filler(200, "%s-m%d" % (TAG, i)), 24),
        ("长", lambda i: filler(700, "%s-L%d" % (TAG, i)), 24),
        ("短2", lambda i: "第%d个问题：把“今天天气不错”翻译成英文，只给译文。" % i, 24),
    ]
    for i in range(N):
        nm, fn, mx = kinds[i % len(kinds)]
        q = fn(i)
        try:
            txt, u, dt = ask(q, mx)
            good = bool(txt.strip())
            if good: ok += 1
            else: fail += 1
            log("[%02d/%d %s] wall=%.2fs prompt_tok=%s comp=%s ok=%s resp=%r" %
                (i + 1, N, nm, dt, u.get("prompt_tokens"), u.get("completion_tokens"), good, txt[:60]))
        except Exception as e:
            fail += 1
            log("[%02d/%d %s] ✗ 失败: %r" % (i + 1, N, nm, e))
    h = health()
    log("[health_end] %s" % json.dumps(h, ensure_ascii=False))
    log("=== 长稳结果: 成功 %d / 失败 %d (共 %d 次请求, 含冷启动首请求) | 服务 %s ===" %
        (ok, fail, N + 1, "存活" if h.get("engine") == "alive" else "DEAD"))
    log("=== done %s ===" % time.strftime("%F %T"))

main()
