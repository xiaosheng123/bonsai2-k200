#!/usr/bin/env python3
# =============================================================================
# pfbench.py —— 引擎级预填充/文本回归台 (主机侧, 不碰卡; 卡要独占 => 由 safe_run 包)
#
#   ★ prompt 由【生产网关 serve2.py 的函数】生成 (_TRIM / _est_tok / build_prompt),
#     因此这里测的每一字节就是线上会发给引擎的字节。
#   ★ 一次引擎启动喂多条请求 (引擎串行处理), 每条请求记录:
#       __TOK__ 解出的原文 (逐字节可比) + 引擎自报的 "prompt=N tok 用时 ... ms/tok" 行
#       + K200_PROF 的 PPROF 分桶 (预填充耗时分解)
#   用法: pfbench.py <引擎路径> <输出json> [标签]
# =============================================================================
import os, sys, json, time, subprocess, threading, queue, base64, re, hashlib

LDP   = "/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib"
MODEL = "/home/caden/orn/Ornith-1.5-9B-Q8_0.gguf"
SAFE  = "/home/caden/orn_engine/safe_run.sh"
sys.path.insert(0, "/home/caden/ornc")
import serve2                                        # 只 import, 不会起服务 (启动在 __main__ 里)

ENG = sys.argv[1]
OUT = sys.argv[2]
TAG = sys.argv[3] if len(sys.argv) > 3 else "x"
FIFO = "/tmp/pfb_%d.fifo" % os.getpid()

def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

# ---------------- 请求集 ----------------
QS6 = ["你好", "1+1等于几", "用三句话解释什么是光合作用",
       "把“今天天气不错，我们出去走走吧”翻译成英文",
       "写一个Python函数输入整数列表返回最大值索引(只给代码)",
       "我明天要交一份季度税务报告，列5条检查清单"]

def long_dsh_14():                                    # 与 longtest.py 完全一致的长 prompt
    filler = "你是编码助手。以下是项目规范：所有函数必须有类型注解；提交前跑测试；不要改公共接口。"
    sysmsg = {"role": "system", "content": filler * 40}
    hist = []
    for i in range(6):
        hist.append({"role": "user", "content": ("这是第 %d 轮的历史上下文，" % i) + "补充说明。" * 60})
        hist.append({"role": "assistant", "content": "明白了，我会按规范处理。" * 30})
    last = {"role": "user", "content": "只回答一个数字：3 + 4 等于几？"}
    return [sysmsg] + hist + [last]

def long_plain(nq, per):                              # 纯长文本 (打满 MAXT 用)
    out = [{"role": "system", "content": "你是严谨的技术助手。" }]
    body = "背景资料：本项目的接口约定、字段含义、错误码与部署约束如下所述。" * per
    for i in range(nq):
        out.append({"role": "user", "content": "第%d段资料：%s" % (i, body)})
        out.append({"role": "assistant", "content": "已收到第%d段资料，我会记住。" % i})
    out.append({"role": "user", "content": "只回答一个数字：5 + 6 等于几？"})
    return out

def build_reqs():
    R = []
    gl = [{"role": "user", "content": "你好"}]
    R.append(("golden", serve2.build_prompt(gl), 24, "g0"))
    for i, q in enumerate(QS6, 1):
        R.append(("q%d" % i, serve2.build_prompt([{"role": "user", "content": q}]), 64, "q%d" % i))
    L14 = long_dsh_14()
    # (a) 线上真路径: 经过 _TRIM(budget=default) 的 14 条
    t900  = serve2._TRIM(list(L14), 900)
    t1700 = serve2._TRIM(list(L14), 1700)
    R.append(("long14_trim900",  serve2.build_prompt(t900),  32, "lt9"))
    R.append(("long14_trim1700", serve2.build_prompt(t1700), 32, "lt17"))
    R.append(("long14_raw",      serve2.build_prompt(L14),   32, "ltraw"))
    # (b) 更长: 目标 ~1400 tok / ~1900 tok (跨过旧的 1024 上限)
    P1 = long_plain(3, 12)
    P2 = long_plain(4, 13)
    R.append(("plain1400", serve2.build_prompt(P1), 24, "p1"))
    R.append(("plain1900", serve2.build_prompt(P2), 24, "p2"))
    return R, dict(n14=len(L14), chars14=sum(len(m["content"]) for m in L14),
                   kept900=len(t900), kept1700=len(t1700),
                   est900=int(sum(serve2._clen(m) for m in t900)),
                   est1700=int(sum(serve2._clen(m) for m in t1700)))

# ---------------- 引擎驱动 (照抄 serve2.Engine 的 FIFO 写法) ----------------
class Eng:
    def __init__(self, eng):
        if os.path.exists(FIFO):
            os.remove(FIFO)
        os.mkfifo(FIFO, 0o600)
        self.wfd = os.open(FIFO, os.O_RDWR)
        self.rfd = os.open(FIFO, os.O_RDWR)
        inner = ("exec env LD_LIBRARY_PATH=%s K200_PROF=1 %s --n 64 --model %s < %s"
                 % (LDP, eng, MODEL, FIFO))
        cmd = [SAFE, "-n", "pfbench", "-t", "1500", "--", "/bin/sh", "-c", inner]
        self.p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, bufsize=0)
        self.q = queue.Queue()
        self.ready = False
        self.t_ready = None
        self.t0 = time.time()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        t0 = time.time()
        try:
            while True:
                line = self.p.stdout.readline()
                if not line:
                    break
                s = line.decode("utf-8", "replace").rstrip("\r\n")
                if "权重常驻 HBM" in s and not self.ready:
                    self.ready = True; self.t_ready = time.time() - t0
                self.q.put(s)
        finally:
            self.q.put(None)

    def wait_ready(self, to=600):
        t0 = time.time()
        while time.time() - t0 < to:
            if self.ready:
                return True
            if self.p.poll() is not None:
                return False
            time.sleep(1)
        return False

    def req(self, prompt, mt, sid, to=1500):
        b64 = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
        os.write(self.wfd, ("@%d@sid=%s@b64:%s\n" % (mt, sid, b64)).encode("ascii"))
        t0 = time.time()
        out, raw, prof, pre, pprof, started, fin = [], "", [], "", [], False, "?"
        while True:
            try:
                s = self.q.get(timeout=to)
            except queue.Empty:
                return dict(err="timeout", text="")
            if s is None:
                return dict(err="engine exit rc=%s" % self.p.poll(), text="")
            if s.startswith("[orn][PPROF]"):
                pprof.append(s)
            elif s.startswith("[orn][PROF]"):
                prof.append(s)
            elif "结束原因=" in s:
                fin = s.split("结束原因=")[1].split()[0]
            elif "复用" in s and "prompt=" in s:
                pre = s
            elif s == "__BEGIN__":
                started = True; continue
            elif s == "__END__":
                break
            elif started and s.startswith("__TOK__ "):
                try:
                    raw += bytes.fromhex(s[8:].strip()).decode("utf-8", "replace")
                except Exception:
                    pass
        dt = time.time() - t0
        txt = raw
        for m in ("<|im_end|>", "<|im_start|>", "</think>", "<think>", "<s>"):
            i = txt.find(m)
            if i >= 0:
                txt = txt[:i]
        ptok = 0; reused = 0; msp = None
        m = re.search(r"prompt=(\d+) tok", pre or "")
        if m: ptok = int(m.group(1))
        m = re.search(r"复用(\d+) tok", pre or "")
        if m: reused = int(m.group(1))
        m = re.search(r"\(([\d.]+) ms/tok", pre or "")
        if m: msp = float(m.group(1))
        return dict(text=txt.strip(), raw=raw, wall=round(dt, 2), prompt_tok=ptok, reused=reused,
                    ms_per_tok=msp, finish=fin, prefill_line=pre, pprof=pprof, prof=prof)

def main():
    reqs, meta = build_reqs()
    E = Eng(ENG)
    ok = E.wait_ready(600)
    res = dict(tag=TAG, engine=ENG, engine_md5=md5f(ENG), meta=meta,
               ready=ok, startup_s=round(E.t_ready or -1, 1))
    print("[pfbench] %s 引擎=%s md5=%s ready=%s 启动=%.1fs" % (TAG, ENG, res["engine_md5"], ok, res["startup_s"]), flush=True)
    items = []
    for name, prompt, mt, sid in reqs:
        r = E.req(prompt, mt, sid)
        r["name"] = name; r["max_tokens"] = mt
        items.append(r)
        print("[pfbench] %-14s prompt_tok=%-5s reused=%-5s ms/tok=%-7s finish=%-7s wall=%-7s text=%r"
              % (name, r.get("prompt_tok"), r.get("reused"), r.get("ms_per_tok"), r.get("finish"), r.get("wall"),
                 (r.get("text") or "")[:60]), flush=True)
        for L in r.get("pprof", []):
            print("           " + L, flush=True)
        if r.get("err"):
            print("           ERR " + r["err"], flush=True)

    # ---- ★ 会话续算 (DSH 真用法: 多轮对话 => 只 prefill 新增 token) ----
    if os.environ.get("PFB_SESS", "1") == "1":
        L = long_dsh_14()
        ra = E.req(serve2.build_prompt(L), 24, "sess1")
        ra["name"] = "sess_a"; ra["max_tokens"] = 24
        items.append(ra)
        print("[pfbench] sess_a  prompt_tok=%s reused=%s ms/tok=%s text=%r"
              % (ra.get("prompt_tok"), ra.get("reused"), ra.get("ms_per_tok"), (ra.get("text") or "")[:60]), flush=True)
        if ra.get("text"):
            L2 = list(L) + [{"role": "assistant", "content": ra["text"]},
                            {"role": "user", "content": "那 5 + 6 呢？只回答数字。"}]
            rb = E.req(serve2.build_prompt(L2), 16, "sess1")
            rb["name"] = "sess_b"; rb["max_tokens"] = 16
            items.append(rb)
            print("[pfbench] sess_b  prompt_tok=%s reused=%s ms/tok=%s text=%r"
                  % (rb.get("prompt_tok"), rb.get("reused"), rb.get("ms_per_tok"), (rb.get("text") or "")[:60]), flush=True)

    res["items"] = items
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    # ★ 收尾: 显式按名字杀引擎 (绝不用 killpg —— safe_run 的目标在独立进程组里,
    #   杀错组会把带病引擎留在卡上)
    os.system("pkill -f '%s --n 64' >/dev/null 2>&1" % ENG)
    time.sleep(3)
    os.system("pkill -9 -f '%s --n 64' >/dev/null 2>&1" % ENG)
    print("[pfbench] 写入 %s ; 引擎已按名清理" % OUT, flush=True)

main()
