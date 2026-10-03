#!/usr/bin/env python3
# ornserve.py —— 8090 单端口 OpenAI 兼容服务, 后端 = Ornith-1.5-9B-Q8_0 在昆仑 K200 (双芯)
#
# 三条硬规矩:
#   1) 引擎进程整个跑在 safe_run.sh 安全网下: 卡上一旦出异常, safe_run 立即 SIGKILL 引擎,
#      服务随后判定 FAILED 且【不自动重启】(有异常就判失败且不重试)。
#   2) 强制关思考: 提示里 assistant 回合直接写死空 <think></think> 块 (与 CPU 基准
#      llama-cli --reasoning off 等价), 否则模型会先吐一长段思考链。
#   3) 不做任何 soft_reset / 重试 / pkill 别人的进程。
import os, sys, json, base64, time, threading, subprocess, http.server, socketserver

ENG      = os.environ.get("K200_ENGINE", "/home/caden/orn_engine/orn")
MODEL    = os.environ.get("K200_MODEL_PATH", "/home/caden/orn/Ornith-1.5-9B-Q8_0.gguf")
PORT     = int(os.environ.get("K200_PORT", "8090"))
NGEN     = os.environ.get("K200_NGEN", "64")
MODEL_ID = os.environ.get("K200_MODEL", "ornith-1.5-9b-k200")
SAFEBIN  = os.environ.get("K200_SAFEBIN", "/home/caden/orn_engine/safe_run.sh")
SAFEMAX  = os.environ.get("K200_SAFEMAXSEC", "14400")
LDP      = "/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib"
# 强制关思考模板 (可用 K200_TMPL_IN 覆盖, {q} 为提问)
TMPL = os.environ.get("K200_TMPL_IN",
    "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

def log(*a):
    print("[ornserve %s]" % time.strftime("%H:%M:%S"), *a, flush=True)

def build_prompt(messages):
    """ChatML + 强制空 think 块。多轮时逐回合拼接, 最后一轮走 TMPL。"""
    parts = []
    for m in messages[:-1]:
        role = m.get("role", "user")
        c = m.get("content", "")
        if isinstance(c, list):
            c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
        parts.append("<|im_start|>%s\n%s<|im_end|>\n" % (role, c))
    last = messages[-1].get("content", "")
    if isinstance(last, list):
        last = " ".join(x.get("text", "") for x in last if isinstance(x, dict))
    parts.append(TMPL.replace("{q}", str(last)))
    return "".join(parts)

class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.dead = False
        self.reason = ""
        self.ntok_total = 0
        self.gen_secs = 0.0
        self.q = __import__("queue").Queue()
        self.ready = False
        cmd = ["stdbuf", "-oL", SAFEBIN, "-n", "orn_serve", "-t", str(SAFEMAX), "--",
               "env", "LD_LIBRARY_PATH=" + LDP, ENG, "--n", str(NGEN), "--model", MODEL]
        log("启动引擎(在安全网下):", " ".join(cmd))
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, bufsize=0)
        log("引擎 pid=%d, 等权重常驻 HBM(约 12-40s)..." % self.p.pid)
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        try:
            while True:
                line = self.p.stdout.readline()
                if not line:
                    break
                s = line.decode("utf-8", "replace").rstrip("\r\n")
                if "[orn] 就绪" in s or "就绪." in s:
                    self.ready = True
                    log("引擎就绪")
                elif s.startswith("[orn]") or s.startswith("XPURT") or s.startswith("[WARN") \
                        or s.startswith("[safe_run]"):
                    log("引擎|", s[:160])
                self.q.put(s)
        finally:
            self.q.put(None)   # EOF 哨兵

    def alive(self):
        return (not self.dead) and self.p.poll() is None

    def generate(self, prompt, max_tokens, on_token=None):
        with self.lock:
            if not self.alive():
                raise RuntimeError("引擎已退出或已被安全网判定失败: %s" % self.reason)
            req = "@%d@b64:%s\n" % (max_tokens,
                                    base64.b64encode(prompt.encode("utf-8", "ignore")).decode("ascii"))
            try:
                self.p.stdin.write(req.encode("ascii"))
                self.p.stdin.flush()
            except Exception as e:
                self.dead = True; self.reason = "写引擎失败: %r" % (e,)
                raise RuntimeError(self.reason)
            t0 = time.time()
            out, ntok, started, pbuf = [], 0, False, b""
            try:
                while True:
                    try:
                        line = self.q.get(timeout=1800)
                    except Exception:
                        raise RuntimeError("引擎 1800s 无输出, 判为卡死")
                    if line is None:
                        self.dead = True
                        self.reason = ("引擎进程死掉了(rc=%s) —— 若 safe_run 判定为卡异常, 按规矩不再重试"
                                       % self.p.poll())
                        raise RuntimeError(self.reason)
                    if line == "__BEGIN__":
                        started = True; continue
                    if line == "__END__":
                        break
                    if not started or not line.startswith("__TOK__ "):
                        continue
                    try:
                        tb = bytes.fromhex(line.split(" ", 2)[2])
                    except Exception:
                        tb = b""
                    ntok += 1
                    pbuf += tb
                    try:
                        s = pbuf.decode("utf-8"); pbuf = b""
                    except UnicodeDecodeError as e:
                        if e.start > 0:
                            s = pbuf[:e.start].decode("utf-8", "replace"); pbuf = pbuf[e.start:]
                        else:
                            s = ""
                    if s:
                        out.append(s)
                        if on_token: on_token(s)
                if pbuf:
                    out.append(pbuf.decode("utf-8", "replace"))
            except RuntimeError:
                raise
            dt = time.time() - t0
            self.ntok_total += ntok; self.gen_secs += dt
            txt = "".join(out)
            for m in ("<|im_end|>", "<|im_start|>", "</think>", "<think>", "<s>"):
                i = txt.find(m)
                if i >= 0: txt = txt[:i]
            return txt.strip(), ntok, dt

ENGINE = None

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        try: self.wfile.write(b)
        except Exception: pass
    def do_GET(self):
        if ENGINE.alive():
            e = {"status": "ok", "engine": "alive", "model": MODEL_ID,
                 "tok_total": ENGINE.ntok_total, "gen_secs": round(ENGINE.gen_secs, 1),
                 "tok_per_s": round(ENGINE.ntok_total / ENGINE.gen_secs, 3) if ENGINE.gen_secs else None}
            self._send(200, e)
        else:
            self._send(503, {"status": "failed", "engine": "dead", "reason": ENGINE.reason,
                             "note": "卡异常/引擎死亡 => 按安全规矩不重试"})
    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._send(404, {"error": {"message": "use POST /v1/chat/completions"}}); return
        n = int(self.headers.get("Content-Length", "0"))
        try:
            req = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            self._send(400, {"error": {"message": "bad json: %r" % (e,)}}); return
        msgs = req.get("messages") or [{"role": "user", "content": req.get("prompt", "")}]
        mt = int(req.get("max_tokens") or NGEN)
        if mt <= 0 or mt > 512: mt = int(NGEN)
        prompt = build_prompt(msgs)
        if not ENGINE.alive():
            self._send(503, {"error": {"message": "engine dead: %s" % ENGINE.reason}}); return
        try:
            txt, ntok, dt = ENGINE.generate(prompt, mt)
        except Exception as e:
            self._send(500, {"error": {"message": str(e)}}); return
        self._send(200, {"id": "chatcmpl-k200-%d" % int(time.time() * 1000),
                         "object": "chat.completion", "created": int(time.time()), "model": MODEL_ID,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": txt},
                                      "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 0, "completion_tokens": ntok, "total_tokens": ntok},
                         "k200": {"gen_secs": round(dt, 2), "tok_per_s": round(ntok / dt, 3) if dt else None}})

class TS(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

if __name__ == "__main__":
    ENGINE = Engine()
    srv = TS(("0.0.0.0", PORT), H)
    log("监听 0.0.0.0:%d (引擎在后台加载权重)" % PORT)
    srv.serve_forever()
