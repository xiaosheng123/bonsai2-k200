#!/usr/bin/env python3
# agg.py — 聚合吞吐实测: 2 个请求【并发】打到 8090, 报各自 wall/耗时/tok/s 与聚合 tok/s
import json, time, threading, urllib.request

RES = {}
def ask(tag, q, mt):
    body = json.dumps({"messages": [{"role": "user", "content": q}], "max_tokens": mt}).encode()
    req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read().decode())
    RES[tag] = (time.time() - t0, d["usage"]["completion_tokens"],
                d["choices"][0]["message"]["content"], d.get("k200", {}))

def run(parallel=True, mt=8):
    RES.clear()
    jobs = [("A", "你好", mt), ("B", "1+1等于几", mt)]
    t0 = time.time()
    if parallel:
        ths = [threading.Thread(target=ask, args=j) for j in jobs]
        for t in ths: t.start()
        for t in ths: t.join()
    else:
        for j in jobs: ask(*j)
    wall = time.time() - t0
    tot = sum(v[1] for v in RES.values())
    print("--- %s ---" % ("2 请求并发" if parallel else "2 请求串行"))
    for k in sorted(RES):
        dt, n, txt, kk = RES[k]
        print("  %s: wall=%.1fs %d tok  %.3f tok/s  回复=%r" % (k, dt, n, n / dt, txt[:40]))
        print("      引擎: %s" % kk.get("engine_prefill", "")[:150])
    print("  两次请求总 wall=%.1fs 总 token=%d  聚合=%.3f tok/s" % (wall, tot, tot / wall))
    return tot / wall

if __name__ == "__main__":
    print("========= (1) 串行基线 =========")
    run(False)
    print("========= (2) 并发 =========")
    run(True)
