#!/usr/bin/env python3
# =============================================================================
# serve3.py — 8090(+8091) 单/双端口 OpenAI 兼容服务, 后端 = orn3.cb
#             (第29轮: 多槽位连续批处理 continuous batching)
#
# 与 serve2.py 的关系: 从 serve2.py 派生, HTTP 层 (OpenAI /v1/chat/completions、
#   SSE 流式、/health、/v1/models、_TRIM 预算、图像预处理) 【一行语义都没改】;
#   只把"一次一个请求"的引擎调用换成"提交进批、按 cid 路由 token"的并发调用。
#
# ★ 端口与并发档 (K200_PORT / K200_PORT2):
#     主口 (K200_PORT, 生产 8090): solo=K200_SOLO (默认 1)
#          solo=1 ⇒ 该请求只在【没有任何别的活跃槽】时进批 ⇒ 永远 M=1
#          ⇒ 与老单会话路径逐位一致, 生产流量在验收期间零风险。
#     副口 (K200_PORT2, 隔离 8091):  solo=K200_SOLO2 (默认 0) ⇒ 真并发批处理。
#   K200_PORT2=0 表示不额外开第二个端口。
#
# ★ 引擎的 stdin 走 FIFO (与 serve2 同因: 经 safe_run 后子进程 fd0 会立刻 EOF)。
# ★ 引擎整个跑在 safe_run.sh 安全网下; 卡上一出异常引擎被杀 ⇒ 判失败,
#   绝不自动重启/重试, 绝不 soft_reset。
# =============================================================================
import os, sys, json, base64, time, threading, subprocess, http.server, socketserver, queue, errno, hashlib, re

HERE     = os.path.dirname(os.path.abspath(__file__))
ENG      = os.environ.get("K200_ENGINE", "/home/caden/orn_engine/orn3.cb")
MODEL    = os.environ.get("K200_MODEL_PATH", "/home/caden/orn/Ornith-1.5-9B-Q8_0.gguf")
PORT     = int(os.environ.get("K200_PORT", "8090"))
PORT2    = int(os.environ.get("K200_PORT2", "0"))          # 0 = 只开一个端口
NGEN     = os.environ.get("K200_NGEN", "512")
MODEL_ID = os.environ.get("K200_MODEL", "ornith-1.5-9b-k200")
SAFEBIN  = os.environ.get("K200_SAFEBIN", "/home/caden/orn_engine/safe_run.sh")
SAFEMAX  = os.environ.get("K200_SAFEMAXSEC", "28800")
LDP      = "/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib"
FIFO     = os.environ.get("K200_FIFO", "/home/caden/ornc/.ornq_cb.fifo")
TMPL     = os.environ.get("K200_TMPL_IN",
    "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
SEP      = os.environ.get("K200_TMPL_SEP", "<|im_end|>\n<|im_start|>")
RAWTOK   = bool(int(os.environ.get("K200_RAWTOK", "0")))
RAWLOG   = os.environ.get("K200_RAWLOG", "/home/caden/ornc/raw_tok.log")
VIS_CH   = os.environ.get("K200_VIS_CH", "96")
VIS_CHA  = os.environ.get("K200_VIS_CHA", "0")
VIS_FOLD = os.environ.get("K200_VIS_FOLD", "1")
PROF     = os.environ.get("K200_PROF", "0")
SLOTS    = os.environ.get("K200_SLOTS", "4")
CKPT     = os.environ.get("K200_CKPT", "8")
SOLO     = int(os.environ.get("K200_SOLO", "1"))           # 主口
SOLO2    = int(os.environ.get("K200_SOLO2", "0"))          # 副口

IMGDIR   = os.environ.get("K200_IMGDIR", "/tmp/ornimg")
MM_MAXPX = 4194304
MM_MINPX = 8192


def log(*a):
    print("[serve %s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def mm_calc_size(W, H):
    w = max(32, int(round(W / 32.0) * 32)); h = max(32, int(round(H / 32.0) * 32))
    if h * w > MM_MAXPX:
        beta = (H * W / float(MM_MAXPX)) ** 0.5
        h = max(32, int(H / beta) // 32 * 32); w = max(32, int(W / beta) // 32 * 32)
    elif h * w < MM_MINPX:
        beta = (float(MM_MINPX) / (H * W)) ** 0.5
        w = max(32, int((W * beta + 31) // 32 * 32)); h = max(32, int((H * beta + 31) // 32 * 32))
    return w, h


def prep_image(url):
    import io as _io
    import numpy as np
    from PIL import Image
    if url.startswith("data:"):
        b = base64.b64decode(url.split(",", 1)[1])
    elif url.startswith("http://") or url.startswith("https://"):
        import urllib.request
        with urllib.request.urlopen(url, timeout=60) as r:
            b = r.read(64 * 1024 * 1024)
    else:
        with open(url, "rb") as f:
            b = f.read()
    im = Image.open(_io.BytesIO(b)).convert("RGB")
    W0, H0 = im.size
    tw, th = mm_calc_size(W0, H0)
    if (tw, th) == (W0, H0):
        v = (np.asarray(im, np.uint8).astype(np.float32) / 255.0 - 0.5) / 0.5
    else:
        scale = min(tw / float(W0), th / float(H0))
        nw, nh = min(tw, int(np.ceil(W0 * scale))), min(th, int(np.ceil(H0 * scale)))
        r = np.asarray(im.resize((nw, nh), Image.BICUBIC), np.uint8).astype(np.float32)
        v = np.zeros((th, tw, 3), np.float32)
        ox, oy = (tw - nw) // 2, (th - nh) // 2
        v[oy:oy + nh, ox:ox + nw, :] = (r / 255.0 - 0.5) / 0.5
    os.makedirs(IMGDIR, exist_ok=True)
    hp = os.path.join(IMGDIR, "img_%s_%dx%d.f32" % (hashlib.sha1(b).hexdigest()[:16], tw, th))
    np.ascontiguousarray(v.transpose(2, 0, 1)).astype("<f4").tofile(hp)
    return hp, tw, th


def split_content(c):
    if isinstance(c, list):
        txt, urls = [], []
        for x in c:
            if not isinstance(x, dict):
                txt.append(str(x)); continue
            if x.get("type") == "image_url":
                iu = x.get("image_url")
                urls.append(iu.get("url", "") if isinstance(iu, dict) else str(iu or ""))
            elif x.get("type") == "text" or "text" in x:
                txt.append(x.get("text", ""))
        return " ".join(txt), urls
    return ("" if c is None else str(c)), []


def _est_tok(t):
    n = 0
    for ch in t:
        o = ord(ch)
        if 0x2E80 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7FF or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF:
            n += 1
        else:
            n += 0.25
    return n


def _clen(m):
    c = m.get("content")
    if isinstance(c, list):
        c = "".join((x.get("text") or "") for x in c if isinstance(x, dict) and x.get("type") == "text")
    return _est_tok(c or "")


def _TRIM(msgs, budget=None):
    if budget is None:
        budget = int(os.environ.get("K200_PROMPT_BUDGET", "600"))
    if not msgs:
        return msgs
    total = sum(_clen(m) for m in msgs)
    if total <= budget:
        return msgs
    out = [msgs[0]] if msgs[0].get("role") == "system" else []
    used = sum(_clen(m) for m in out)
    if used > budget * 0.5:
        sc = (msgs[0].get("content") or "")
        lim = int(budget * 0.35)
        acc, cut = 0, len(sc)
        for idx, ch in enumerate(sc):
            acc += _est_tok(ch)
            if acc > lim:
                cut = idx
                break
        head = sc[:cut]
        for sep in ("。", "！", "？", "\n", "；", ". ", "! ", "? "):
            k = head.rfind(sep)
            if k > len(head) * 0.5:
                head = head[:k + len(sep)]
                break
        if head.rstrip() and not head.rstrip().endswith(("。", "！", "？", ".", "!", "?")):
            head = head.rstrip() + "。"
        out = [dict(msgs[0], content=head)]
        used = _clen(out[0])
    tail = [msgs[-1]]
    used += _clen(msgs[-1])
    i = len(msgs) - 2
    while i >= len(out) and used + _clen(msgs[i]) <= budget:
        tail.insert(0, msgs[i]); used += _clen(msgs[i]); i -= 1
    out = out + tail
    if len(out) < len(msgs):
        log("prompt 过长: %d 条/估 %d tok ⇒ 裁到 %d 条/估 %d tok (预算 %d)"
            % (len(msgs), total, len(out), used, budget))
    return out


def build_prompt(messages, vision=None):
    out = []
    for m in messages[:-1]:
        c, _u = split_content(m.get("content", ""))
        role = m.get("role", "user")
        if role == "assistant":
            out.append("<|im_start|>assistant\n<think>\n\n</think>\n\n%s<|im_end|>\n" % c)
        else:
            out.append("<|im_start|>%s\n%s<|im_end|>\n" % (role, c))
    last, _u = split_content(messages[-1].get("content", ""))
    if vision:
        pre = "".join("<|vision_start|>" + "<|image_pad|>" * int(n) + "<|vision_end|>" for n in vision)
        last = pre + str(last)
    out.append(TMPL.replace("{q}", str(last)))
    return "".join(out)


class Route(object):
    __slots__ = ("q",)

    def __init__(self):
        self.q = queue.Queue()


class Engine(object):
    def __init__(self):
        self.lock = threading.Lock()      # 老串行语义 (legacy 模式用)
        self.wlock = threading.Lock()     # FIFO 写原子性
        self.rlock = threading.Lock()
        self.vlock = threading.Lock()     # 视觉塔串行
        self.dead = False
        self.reason = ""
        self.ready = False
        self.ntok_total = 0
        self.gen_secs = 0.0
        self.nreq = 0
        self.q = queue.Queue(maxsize=4000)      # 老路径/日志旁路 (有界, 满了就丢)
        self.ctl = queue.Queue()
        self.routes = {}
        self.cid = 0
        self.last_prompt_tok = 0
        self.last_reused = 0
        self.last_vis_info = ""
        self.vis_fp = 0
        self.vis_n = 0
        self.slot_stat = ""
        self.active_now = 0
        if os.path.exists(FIFO):
            os.remove(FIFO)
        os.mkfifo(FIFO, 0o600)
        self.wfd = os.open(FIFO, os.O_RDWR)
        self.rfd = os.open(FIFO, os.O_RDWR)
        inner = ("exec env LD_LIBRARY_PATH=%s VIS_CH=%s VIS_CHA=%s VIS_FOLD=%s K200_PROF=%s "
                 "K200_SLOTS=%s K200_CKPT=%s %s --n %s --model %s < %s"
                 % (LDP, VIS_CH, VIS_CHA, VIS_FOLD, PROF, SLOTS, CKPT, ENG, NGEN, MODEL, FIFO))
        cmd = [SAFEBIN, "-n", "orn_serve", "-t", str(SAFEMAX), "--", "/bin/sh", "-c", inner]
        log("启动引擎(安全网下):", " ".join(cmd))
        self.p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, bufsize=0)
        log("safe_run pid=%d, FIFO=%s, 等权重常驻 HBM(~20s)..." % (self.p.pid, FIFO))
        threading.Thread(target=self._reader, daemon=True).start()

    # ---------------- 读线程: 一行一行解析并【按 cid 路由】 ----------------
    def _reader(self):
        try:
            while True:
                line = self.p.stdout.readline()
                if not line:
                    break
                s = line.decode("utf-8", "replace").rstrip("\r\n")
                if "权重常驻 HBM" in s:
                    self.ready = True; log("引擎就绪(权重已常驻 HBM)")
                elif "就绪" in s:
                    log("引擎加载中|", s[:120])
                elif s.startswith(("[orn]", "XPURT", "[WARN", "[safe_run]", "[vis]")):
                    log("引擎|", s[:180])
                self._route(s)
        finally:
            self.dead = True
            try:
                self.q.put(None)
            except queue.Full:
                pass
            with self.rlock:
                for k, r in list(self.routes.items()):
                    r.q.put(("eof",))
            self.ctl.put("__ENGINE_DEAD__")

    def _route(self, s):
        if s.startswith("__STOK__ "):
            p = s.split(None, 2)
            try:
                cid = int(p[1])
            except Exception:
                return
            r = self.routes.get(cid)
            if r:
                r.q.put(("tok", p[2] if len(p) > 2 else ""))
            return
        if s.startswith("__SDONE__ "):
            p = s.split()
            try:
                cid = int(p[1])
            except Exception:
                return
            r = self.routes.get(cid)
            self.active_now = max(0, self.active_now - 1)
            if r:
                r.q.put(("done", p[2] if len(p) > 2 else "stop", int(p[3]) if len(p) > 3 else 0))
            return
        if s.startswith("__SFP__ "):
            p = s.split(); cid = int(p[1]); r = self.routes.get(cid)
            if r:
                r.q.put(("fp", s))
            return
        if s.startswith("__SPREF__ "):
            p = s.split(); cid = int(p[1]); r = self.routes.get(cid)
            if r:
                r.q.put(("pref", s))
            return
        if s.startswith("__SACTIVE__ "):
            p = s.split(); cid = int(p[1]); r = self.routes.get(cid)
            self.active_now += 1
            if r:
                r.q.put(("active", s))
            return
        if s.startswith("__SADD__ "):
            p = s.split(); cid = int(p[1]); r = self.routes.get(cid)
            if r:
                r.q.put(("add", s))
            return
        if s.startswith("__SREJ__ "):
            p = s.split(); cid = int(p[1]); r = self.routes.get(cid)
            if r:
                r.q.put(("rej", s))
            return
        if s.startswith(("__VIS__", "__VISR__", "__VISIMG__", "__VISQ__", "__SLOTS__",
                         "__SSTAT__", "__SPAUSE__", "__SRESUME__", "__SDEL__")):
            self.ctl.put(s); return
        try:
            self.q.put(s)
        except queue.Full:
            pass

    # ---------------- 底层 ----------------
    def alive(self):
        return (not self.dead) and self.p.poll() is None

    def _write_all(self, data):
        while data:
            try:
                n = os.write(self.wfd, data)
                data = data[n:]
            except OSError as e:
                if e.errno == errno.EINTR:
                    continue
                raise

    def send(self, line):
        with self.wlock:
            self._write_all((line + "\n").encode("ascii"))

    def _ctl_wait(self, prefix, timeout=30.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                s = self.ctl.get(timeout=1.0)
            except queue.Empty:
                if not self.alive():
                    raise RuntimeError("引擎退出: %s" % self.reason)
                continue
            if s.startswith(prefix):
                return s
        return None

    # ---------------- 视觉塔 (串行; 返回 token 数 + 图像指纹) ----------------
    def vis_send(self, path, W, H, timeout=900):
        """★ 调用方必须持有 vlock (串行化 !VISR/!VIS/!VISIMG 三连)"""
        self.send("!VIS %s %d %d" % (path, W, H))
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                s = self.ctl.get(timeout=1.0)
            except queue.Empty:
                if not self.alive():
                    raise RuntimeError("引擎在视觉编码中退出")
                continue
            if s.startswith("[vis] 完成"):
                self.last_vis_info = s
            if s.startswith("__VIS__"):
                p = s.split()
                if len(p) >= 2 and p[1].isdigit():
                    n = int(p[1])
                    self.send("!VISIMG")
                    s2 = self._ctl_wait("__VISIMG__", 30)
                    fp = int(s2.split()[2], 16) if s2 and len(s2.split()) >= 3 else 0
                    self.vis_n, self.vis_fp = n, fp
                    return n, fp
                raise RuntimeError("视觉塔失败: %s" % s)
        raise RuntimeError("视觉编码超时")

    def vis_reset(self):
        try:
            self.send("!VISR")
            self._ctl_wait("__VISR__", 10)
        except Exception:
            pass
        self.vis_fp = 0; self.vis_n = 0

    def sstat(self, timeout=5):
        try:
            self.send("!SSTAT")
            s = self._ctl_wait("__SSTAT__", timeout)
            if s:
                self.slot_stat = s.split(None, 1)[1] if " " in s else ""
        except Exception:
            pass
        return self.slot_stat

    # ---------------- 生成 (进批; 按 cid 路由) ----------------
    def generate(self, prompt, max_tokens, on_token=None, sid="", solo=1, imgfp=0, timeout=2400):
        if not self.alive():
            raise RuntimeError("引擎不可用: %s" % (self.reason or "进程已退出"))
        with self.wlock:
            self.cid += 1
            cid = self.cid
            r = Route()
            with self.rlock:
                self.routes[cid] = r
            img = ("img=%016x " % imgfp) if imgfp else ""
            b64 = base64.b64encode(prompt.encode("utf-8", "ignore")).decode("ascii")
            line = "!SADD %d %d sid=%s %ssolo=%d b64:%s" % (cid, int(max_tokens), (sid or "-"), img, int(solo), b64)
            try:
                self._write_all((line + "\n").encode("ascii"))
            except OSError as e:
                with self.rlock:
                    self.routes.pop(cid, None)
                self.dead = True; self.reason = "写 FIFO 失败: %r" % (e,)
                raise RuntimeError(self.reason)
        t0 = time.time()
        out, ntok, pbuf = [], 0, b""
        meta, fps, reason = {}, None, "stop"
        self.last_prompt_tok, self.last_reused = 0, 0
        try:
            while True:
                try:
                    it = r.q.get(timeout=timeout)
                except queue.Empty:
                    raise RuntimeError("引擎 %ds 无输出, 判卡死" % timeout)
                if it[0] == "eof" or it is None:
                    self.dead = True
                    self.reason = ("引擎进程结束(rc=%s) —— 若是 safe_run 判定卡异常, 按规矩不再重试"
                                   % self.p.poll())
                    raise RuntimeError(self.reason)
                k = it[0]
                if k == "tok":
                    hx = it[1].strip()
                    tb = bytes.fromhex(hx) if hx else b""
                    if RAWTOK:
                        try:
                            with open(RAWLOG, "a") as rf:
                                rf.write("%s\n" % hx)
                        except Exception:
                            pass
                    ntok += 1; pbuf += tb
                    try:
                        s = pbuf.decode("utf-8"); pbuf = b""
                    except UnicodeDecodeError as e:
                        if e.start > 0:
                            s = pbuf[:e.start].decode("utf-8", "replace"); pbuf = pbuf[e.start:]
                        else:
                            s = ""
                    if s:
                        out.append(s)
                        if on_token:
                            on_token(s)
                elif k == "done":
                    reason = it[1]; break
                elif k == "fp":
                    fps = self._mkfp(it[1])
                elif k in ("pref", "active", "add", "rej"):
                    meta[k] = it[1]
                    if k == "pref":
                        try:
                            d = dict(x.split("=", 1) for x in it[1].split()[2:])
                            self.last_reused = int(d.get("reused", 0))
                            self.last_prompt_tok = int(d.get("ptok", 0))
                            meta["reused"] = int(d.get("reused", 0))
                            meta["new"] = int(d.get("new", 0))
                            meta["prefill_s"] = float(d.get("prefill_s", 0))
                            meta["lcp"] = int(d.get("lcp", 0))
                        except Exception:
                            pass
                if it[0] == "rej":
                    raise RuntimeError("引擎拒绝: %s" % it[1])
        finally:
            with self.rlock:
                self.routes.pop(cid, None)
        dt = time.time() - t0
        self.ntok_total += ntok; self.gen_secs += dt; self.nreq += 1
        if pbuf:
            out.append(pbuf.decode("utf-8", "replace"))
        txt = "".join(out)
        for m in ("<|im_end|>", "<|im_start|>", "</think>", "<think>", "<s>"):
            i = txt.find(m)
            if i >= 0: txt = txt[:i]
        return txt.strip(), ntok, dt, meta, fps, reason

    @staticmethod
    def _mkfp(line):
        m = re.search(r"__SFP__ (\d+) (\d+) ids=([0-9a-f]+) bytes=([0-9a-f]+) prompt=([0-9a-f]+)\((\d+) tok\)", line)
        if not m:
            return None
        return "★ GEN %s tok ids=%s bytes=%s prompt=%s(%s tok)" % (m.group(2), m.group(3), m.group(4), m.group(5), m.group(6))


ENGINE = None


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _solo(self):
        p = self.server.server_port
        if PORT2 and p == PORT2:
            return SOLO2
        return SOLO

    def _wait_ready(self, maxt=None):
        if maxt is None:
            maxt = float(os.environ.get("K200_WAIT_MAX", "1800"))
        t0 = time.time()
        while not ENGINE.ready:
            if not ENGINE.alive():
                return False
            if (time.time() - t0) >= maxt:
                return False
            time.sleep(1.0)
        return True

    def _send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        try:
            self.wfile.write(b)
        except Exception:
            pass

    def do_GET(self):
        p = self.path.rstrip("/")
        if p in ("/health", "/healthz"):
            if ENGINE.alive():
                st = ENGINE.sstat()
                self._send(200, {"status": "ok" if ENGINE.ready else "loading",
                                 "engine": "alive", "ready": ENGINE.ready, "note": "",
                                 "model": MODEL_ID, "requests": ENGINE.nreq,
                                 "tok_total": ENGINE.ntok_total,
                                 "gen_secs": round(ENGINE.gen_secs, 1),
                                 "tok_per_s": round(ENGINE.ntok_total / ENGINE.gen_secs, 3) if ENGINE.gen_secs else None,
                                 "origin": "serve3-cb", "slots_env": SLOTS, "ckpt": CKPT,
                                 "port": self.server.server_port, "solo": self._solo(),
                                 "slots": st})
            else:
                self._send(503, {"status": "failed", "engine": "dead", "reason": ENGINE.reason,
                                 "note": "卡异常/引擎死亡 => 按安全规矩不重试, 不 soft_reset"})
            return
        if p in ("/v1/models", "/models"):
            self._send(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model",
                                                         "owned_by": "k200-ornith"}]}); return
        self._send(404, {"error": {"message": "GET /health | /v1/models"}})

    def _stream(self, req, prompt, mt, sid):
        cid = "chatcmpl-k200-%d" % int(time.time() * 1000)
        created = int(time.time())
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def ev(obj):
                self.wfile.write(("data: %s\n\n" % json.dumps(obj, ensure_ascii=False)).encode("utf-8"))
                self.wfile.flush()

            ev({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})

            def cb(tb):
                c = tb if isinstance(tb, str) else tb.decode("utf-8", "ignore")
                if not c:
                    return
                ev({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {"content": c}, "finish_reason": None}]})

            try:
                txt, ntok, dt, meta, fps, reason = ENGINE.generate(
                    prompt, mt, on_token=cb, sid=sid, solo=self._solo(), imgfp=ENGINE.vis_fp)
            except Exception as e:
                ev({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                    "k200": {"error": str(e)}})
                self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush(); return
            ev({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "k200": {"gen_secs": round(dt, 2), "tok_per_s": round(ntok / dt, 4) if dt else None,
                         "prompt_tokens": ENGINE.last_prompt_tok,
                         "prefill_reused_tokens": ENGINE.last_reused, "sid": sid,
                         "gen_fingerprint": fps, "cb": meta, "solo": self._solo()}})
            if (req.get("stream_options") or {}).get("include_usage"):
                ev({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                    "choices": [], "usage": {"prompt_tokens": ENGINE.last_prompt_tok or 13,
                                             "completion_tokens": ntok,
                                             "total_tokens": (ENGINE.last_prompt_tok or 13) + ntok}})
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                self._send(500, {"error": {"message": str(e)}})
            except Exception:
                pass

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._send(404, {"error": {"message": "use POST /v1/chat/completions"}}); return
        n = int(self.headers.get("Content-Length", "0"))
        try:
            req = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            self._send(400, {"error": {"message": "bad json: %r" % (e,)}}); return
        msgs = _TRIM(req.get("messages")) or [{"role": "user", "content": req.get("prompt", "")}]
        mt = int(req.get("max_tokens") or req.get("max_completion_tokens") or NGEN)
        if mt <= 0: mt = int(NGEN)
        _cap = int(os.environ.get("K200_MAXTOK", "8192"))
        if mt > _cap: mt = _cap
        if not ENGINE.alive():
            self._send(503, {"error": {"message": "engine dead: %s" % ENGINE.reason}}); return
        if not ENGINE.ready:
            log("引擎未就绪, 请求排队等待 (上限 %ss)" % os.environ.get("K200_WAIT_MAX", "1800"))
            if not self._wait_ready():
                self._send(503, {"error": {"message": "engine loading: 权重尚未常驻 HBM (约需 75s), 请稍后重试"}}); return
        solo = self._solo()
        vision = []
        imgfp = 0
        try:
            _t, urls = split_content(msgs[-1].get("content", ""))
            if urls:
                with ENGINE.vlock:                       # ★ 视觉塔 + 取指纹: 全程串行, 防串图
                    ENGINE.vis_reset()
                    tvis = time.time()
                    for u in urls:
                        if not u:
                            continue
                        hp, iw, ih = prep_image(u)
                        nn, fp = ENGINE.vis_send(hp, iw, ih)
                        vision.append(nn)
                        imgfp = fp
                        log("图像 %s %dx%d -> %d token fp=%016x" % (os.path.basename(hp), iw, ih, nn, fp))
                    log("图像编码合计 %.2fs (%d 张) solo=%d" % (time.time() - tvis, len(vision), solo))
                    prompt = build_prompt(msgs, vision)
                    if req.get("stream"):
                        return self._stream(req, prompt, mt, self._sid(msgs))
                    t0 = time.time()
                    txt, ntok, dt, meta, fps, reason = ENGINE.generate(
                        prompt, mt, sid=self._sid(msgs), solo=solo, imgfp=imgfp)
        except Exception as e:
            log("图像处理失败:", repr(e))
            self._send(400, {"error": {"message": "image: %r" % (e,)}}); return
        if not vision:
            prompt = build_prompt(msgs, vision)
            if req.get("stream"):
                return self._stream(req, prompt, mt, self._sid(msgs))
            t0 = time.time()
            try:
                txt, ntok, dt, meta, fps, reason = ENGINE.generate(
                    prompt, mt, sid=self._sid(msgs), solo=solo, imgfp=0)
            except Exception as e:
                self._send(500, {"error": {"message": str(e)}}); return
        self._send(200, {"id": "chatcmpl-k200-%d" % int(time.time() * 1000),
                         "object": "chat.completion", "created": int(time.time()), "model": MODEL_ID,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": txt},
                                      "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": ENGINE.last_prompt_tok or 13,
                                   "completion_tokens": ntok,
                                   "total_tokens": (ENGINE.last_prompt_tok or 13) + ntok},
                         "k200": {"gen_secs": round(dt, 2), "wall_secs": round(time.time() - t0, 2),
                                  "vision_tokens": vision, "image_encode": ENGINE.last_vis_info[:160],
                                  "tok_per_s": round(ntok / dt, 4) if dt else None,
                                  "prompt_chars": len(prompt), "nothink": True, "sid": self._sid(msgs),
                                  "prompt_tokens": ENGINE.last_prompt_tok,
                                  "prefill_reused_tokens": ENGINE.last_reused,
                                  "lcp_reused_tokens": meta.get("reused", ENGINE.last_reused),
                                  "lcp_new_tokens": meta.get("new", 0),
                                  "prefill_secs": round(meta.get("prefill_s", 0), 3),
                                  "finish_reason_raw": reason,
                                  "gen_fingerprint": fps, "cb": meta, "solo": solo,
                                  "engine_prefill": (meta.get("active") or meta.get("add") or "")[:200]}})

    @staticmethod
    def _sid(msgs):
        try:
            first = ""
            for m in msgs:
                if m.get("role") == "user":
                    c = m.get("content", "")
                    first = " ".join(x.get("text", "") for x in c if isinstance(x, dict)) if isinstance(c, list) else str(c)
                    break
            return hashlib.sha1(first.encode("utf-8")).hexdigest()[:16] if first else "-"
        except Exception:
            return "-"


class TS(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    ENGINE = Engine()
    if PORT2 and PORT2 != PORT:
        srv2 = TS(("0.0.0.0", PORT2), H)
        threading.Thread(target=srv2.serve_forever, daemon=True).start()
        log("监听 0.0.0.0:%d (solo=%d 真并发档)" % (PORT2, SOLO2))
    srv = TS(("0.0.0.0", PORT), H)
    log("监听 0.0.0.0:%d (solo=%d) | 引擎在后台加载" % (PORT, SOLO))
    srv.serve_forever()
