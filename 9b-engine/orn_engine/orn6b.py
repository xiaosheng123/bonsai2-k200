#!/usr/bin/env python3
# orn6b.py — 经 8090 跑 6 条任务型提问; 引擎崩了就 soft_reset + 重启服务重试
import os, sys, json, time, subprocess, urllib.request, urllib.error

OUT   = "/home/caden/orn_engine/orn6.log"
SRVL  = "/home/caden/orn_engine/srv.log"
ENG   = "/home/caden/orn_engine/orn"
MODEL = "/home/caden/orn/Ornith-1.5-9B-Q8_0.gguf"
PORT  = 8090
QS = ["你好",
      "1+1等于几",
      "用三句话解释什么是光合作用",
      "把“今天天气不错，我们出去走走吧”翻译成英文",
      "写一个Python函数输入整数列表返回最大值索引(只给代码)",
      "我明天要交一份季度税务报告，列5条检查清单"]

def log(*a):
    print(*a, flush=True)
    with open(OUT, "a") as f:
        f.write(" ".join(str(x) for x in a) + "\n")

def sh(cmd):
    subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True)

def start_engine():
    sh("pkill -9 -f 'serve[r].py'"); sh("pkill -9 -f 'orn_engin[e]/orn'")
    time.sleep(2)
    for d in ("0", "1"):
        sh("/usr/local/xpu-4.33.0/tools/soft_reset " + d)
    time.sleep(2)
    env = dict(os.environ)
    env.update({"K200_ENGINE": ENG, "K200_MODEL_PATH": MODEL, "K200_PORT": str(PORT),
                "K200_MODEL": "ornith-1.5-9b-k200", "K200_NGEN": "48", "K200_MAXTOK": "64",
                "LD_LIBRARY_PATH": "/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib"})
    f = open(SRVL, "a")
    subprocess.Popen(["python3", "/home/caden/k200llm/server.py"], env=env, stdout=f,
                     stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(80):
        time.sleep(3)
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=2)
            return True
        except Exception:
            pass
    return False

def ask(q, mx=48, timeout=1500):
    body = json.dumps({"model": "ornith-1.5-9b-k200", "temperature": 0,
                       "messages": [{"role": "user", "content": q}], "max_tokens": mx}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    dt = time.time() - t0
    txt = d["choices"][0]["message"]["content"]
    n = d.get("usage", {}).get("completion_tokens", -1)
    return txt, n, dt

def main():
    open(OUT, "w").close()
    log("=== orn 6 题验收 (Q8_0, 双芯, 8090) 开始 %s ===" % time.strftime("%H:%M:%S"))
    for i, q in enumerate(QS, 1):
        ok = False
        for att in range(1, 4):
            if att == 1 or not ok:
                if not start_engine():
                    log("Q%d 第%d次: 引擎起不来" % (i, att)); continue
            # 探针: 短请求确认引擎活着
            try:
                t, n, dt = ask("1+1", 4, timeout=900)
                log("[probe] content=%r tokens=%d %.1fs" % (t[:40], n, dt))
            except Exception as e:
                log("[probe] 失败: %s" % e); sh("pkill -9 -f 'serve[r].py'"); continue
            try:
                t, n, dt = ask(q, 48)
            except Exception as e:
                log("Q%d 第%d次异常: %s" % (i, att, e)); continue
            if t.strip() == "":
                log("Q%d 第%d次: 内容为空 (引擎可能半路崩)" % (i, att)); continue
            log("\n########## Q%d: %s" % (i, q))
            log("回复: %s" % t)
            log("[%d tok, %.1fs, %.3f tok/s]" % (n, dt, (n / dt) if dt else 0))
            ok = True
            break
        if not ok:
            log("Q%d *** 3 次都没拿到有效回复 ***" % i)
    log("ALL6_DONE %s" % time.strftime("%H:%M:%S"))

main()
