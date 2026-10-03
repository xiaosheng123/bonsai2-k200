#!/usr/bin/env python3
# =============================================================================
# serve.py —— 8090 单端口 OpenAI 兼容服务, 后端 = 已验证正确的 orn 引擎 (Q8_0 双芯)
#
# 设计要点 (为什么这么写):
#   1) 后端用【已经离线验证过输出正确】的二进制, 内核 = kq8 + HW_CORE=16 + co<=16。
#      不做任何性能花样 —— 现在唯一目标是"能出正确的话"。
#   2) 引擎的 stdin 走【FIFO】而不是 Popen(stdin=PIPE): 实测经 safe_run 后子进程的
#      fd0 会立刻 EOF(探明前先用最稳的写法绕开), FIFO + O_RDWR 常开则永不 EOF。
#   3) 引擎整个跑在 safe_run.sh 安全网下: 卡上一出异常就被立即 SIGKILL, 服务判 FAILED,
#      【绝不自动重启/重试】, 也绝不 soft_reset。
#   4) 强制关思考: assistant 回合里直接写死空 <think></think> 块(与 CPU 基准 --reasoning off 等价)。
#
# ★ 命名约定 (与 ornc/start.sh / stop.sh / restart.sh / status.sh 对齐, 不许再各叫各的):
#     本文件 = 服务进程名 (serve2.py); 引擎二进制 = /home/caden/orn_engine/orn3。
#     改任一名字都必须同步改 stop.sh 的 PAT_* —— 否则会「停不掉」, 把带病引擎留在卡上。
# =============================================================================
import os, sys, json, base64, time, threading, subprocess, http.server, socketserver, queue, errno, hashlib, re

HERE     = os.path.dirname(os.path.abspath(__file__))
ENG      = os.environ.get("K200_ENGINE", "/home/caden/orn_engine/orn")
MODEL    = os.environ.get("K200_MODEL_PATH", "/home/caden/orn/Ornith-1.5-9B-Q8_0.gguf")
PORT     = int(os.environ.get("K200_PORT", "8090"))
NGEN     = os.environ.get("K200_NGEN", "64")
MODEL_ID = os.environ.get("K200_MODEL", "ornith-1.5-9b-k200")
SAFEBIN  = os.environ.get("K200_SAFEBIN", "/home/caden/orn_engine/safe_run.sh")
SAFEMAX  = os.environ.get("K200_SAFEMAXSEC", "14400")
LDP      = "/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib"
FIFO     = os.environ.get("K200_FIFO", "/tmp/ornq_in.fifo")
TMPL     = os.environ.get("K200_TMPL_IN",
    "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
SEP      = os.environ.get("K200_TMPL_SEP", "<|im_end|>\n<|im_start|>")
RAWTOK   = bool(int(os.environ.get("K200_RAWTOK", "0")))
RAWLOG   = os.environ.get("K200_RAWLOG", "/home/caden/ornc/raw_tok.log")
# ★ 视觉塔调档变量: 由 serve2 显式塞进引擎子进程 env (K200_VIS_* 可覆盖; 默认值即交付档)
VIS_CH   = os.environ.get("K200_VIS_CH", "96")
VIS_CHA  = os.environ.get("K200_VIS_CHA", "0")
VIS_FOLD = os.environ.get("K200_VIS_FOLD", "1")
# ★ 第25轮: 网关裁剪与【客户端超时】挂钩 (判据不是"历史留得越多越好", 而是"首字 ≤25s")
CTX_LIMIT     = int(os.environ.get("K200_CTX_LIMIT", "2048"))            # 引擎位置上限 MAXT
CLIENT_TIMEOUT_MS = int(os.environ.get("K200_CLIENT_TIMEOUT_MS", "30000"))   # 中转站/客户端超时
PREFILL_TARGET_MS = int(os.environ.get("K200_PREFILL_TARGET_MS", "25000"))   # 首字目标 (留超时余量)
PREFILL_MS_FALLBACK = float(os.environ.get("K200_PREFILL_MS_FALLBACK", "26.0"))  # 实测前兜底 ms/tok
IMG_TOK_RESERVE   = int(os.environ.get("K200_IMG_TOK_RESERVE", "700"))   # 带图请求给图像 token 留位

# ============================================================================
# ★ VIS_IMAGE_PATCHED: 支持 content 数组里的 {"type":"image_url",...}
#   流程: 取图 -> 照 llama.cpp mtmd 预处理成平面 f32 -> 发 !VIS 命令给引擎
#         (引擎在卡上跑视觉塔) -> 拼 <|vision_start|><|image_pad|>*n<|vision_end|> + 文本
#   ★ 纯文本请求: 下面所有分支都不触发, 与原路径逐字节相同。
# ============================================================================
IMGDIR   = os.environ.get("K200_IMGDIR", "/tmp/ornimg")
MM_MAXPX = 4194304
MM_MINPX = 8192

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
    """OpenAI image_url -> (f32路径, W, H); 平面布局 x + W*y + W*H*c, 值=(px/255-0.5)/0.5"""
    import io as _io, hashlib
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
    """content -> (文本, [图像 url])"""
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

def log(*a):
    print("[serve %s]" % time.strftime("%H:%M:%S"), *a, flush=True)

def _est_tok(t):
    """★ EST_TOK: 中文感知的 token 估算 (CJK 1 字 ≈ 1 token, 非 CJK 4 字符 ≈ 1 token)。
       之前用 2.6 字符/token 的英文口径估中文, 导致"裁完仍超标" ⇒ 引擎仍丢尾 ⇒ 仍然空回复。"""
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
    """★ 引擎 MAXT=2048 (第25轮由 1024 抬起; 抬前 MAXT=1024 且截断时【丢尾】(orn3.cpp:1688
       `ids.resize(i)`): 长 prompt 会把用户最后的问题丢掉, 且截断点会把 ChatML 最后一条消息切成
       半截(无 <|im_end|>) ⇒ 模型当场吐 EOS ⇒ 生成 0 token ⇒ 前端显示"空响应"。
       这里在网关侧按 token 估算裁掉中间回合, 保证【system 头 + 最近若干条(含最后一条问题)】留下,
       且总估算 ≤ budget(默认 1700 = 引擎 MAXT 2048 留 20% 余量给生成/包装)。
       逻辑与 MAXT=1024 时代逐行一致 —— 只调大了预算数字。"""
    if not msgs:
        return msgs
    total = sum(_clen(m) for m in msgs)
    if total <= budget:
        return msgs
    out = [msgs[0]] if msgs[0].get("role") == "system" else []
    used = sum(_clen(m) for m in out)
    if used > budget * 0.5:                # 光 system 就占了半预算 ⇒ system 也砍
        # ★ SYS_CUT: 砍到句子边界为止 (半句话结尾会把模型带偏 —— 实测问"3+4"答"3")
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
    tail = [msgs[-1]]                      # ★ 最后一条(用户当前问题)无条件保留
    used += _clen(msgs[-1])
    i = len(msgs) - 2
    while i >= len(out) and used + _clen(msgs[i]) <= budget:
        tail.insert(0, msgs[i]); used += _clen(msgs[i]); i -= 1
    out = out + tail
    if len(out) < len(msgs):
        log("prompt 过长(字符估算): %d 条/估 %d tok ⇒ 裁到 %d 条/估 %d tok (保 system 头 + 最近消息尾; 引擎上限 MAXT=%d, 预算 %d 估算 tok / 预计预填充 %.1fs / 客户端超时 %.0fs)"
            % (len(msgs), total, len(out), used, CTX_LIMIT, budget,
               PREFILL_MS_FALLBACK * budget / 1000.0, CLIENT_TIMEOUT_MS / 1000.0))
    return out

def _budget_real_tok(ms_per_tok=None, reserve=0):
    """★ 按【真实 token】的预算 = 首字目标 25s / 实测 ms/tok, 再减带图保留位, 夹在 [64, CTX_LIMIT-64]
       ⇒ 预填充越快, 预算自动越大 (先提速再放开, 不用改配置)。"""
    m = ms_per_tok if (ms_per_tok and ms_per_tok > 0) else PREFILL_MS_FALLBACK
    b = int(PREFILL_TARGET_MS / m) - int(reserve)
    return max(64, min(b, CTX_LIMIT - 64))

def _TRIM_REAL(msgs, engine, reserve=0):
    """★ 主裁剪闸: 用引擎 !TOK 拿【真实 token 数】, 超预算就【整条丢头保尾】。
       - 绝不丢最后一条消息 (用户当前问题) ✓   - 绝不切碎消息 ✓
       - ≤预算的请求一字不改 ✓ (短 prompt/黄金题逐字节不变)
       - !TOK 不可用 ⇒ 返回 None, 调用方退回 _TRIM 字符估算"""
    if not msgs:
        return msgs
    budget = _budget_real_tok(getattr(engine, "last_ms_tok", None), reserve)
    def toks(ms):
        return engine.count_tokens(build_prompt(ms))
    try:
        n = toks(msgs)
    except Exception as e:
        log("!TOK 真实分词不可用, 退回字符估算裁剪:", repr(e))
        return None
    if n <= budget:
        return msgs
    log("prompt 真实 %d tok > 实时预算 %d tok (实测 %s ms/tok, 首字目标 %.1fs, 客户端超时 %.0fs) ⇒ 整条丢头保尾"
        % (n, budget, getattr(engine, "last_ms_tok", None) or ("%.1f(兜底)" % PREFILL_MS_FALLBACK),
           PREFILL_TARGET_MS / 1000.0, CLIENT_TIMEOUT_MS / 1000.0))
    out = list(msgs)
    tgt = max(1, int(len(out) * budget / float(n)))
    while len(out) > tgt and len(out) > 1:
        out.pop(0)
    while len(out) > 1:
        try:
            n2 = toks(out)
        except Exception:
            break
        if n2 <= budget:
            log("  丢头后: 保留 %d 条 / 真实 %d tok (预算 %d)" % (len(out), n2, budget))
            return out
        out.pop(0)
    log("  已丢到只剩最后一条: 真实 %d tok 仍超预算 %d ⇒ 交给引擎侧丢头保尾 (orn3 TAILKEEP)" % (n, budget))
    return out

def build_prompt(messages, vision=None):
    """ChatML + 强制空 think 块, 特殊 token 全部来自 GGUF 推导 (与引擎打印的 id 一致)。
       vision: 若非空 = 最后一条消息的图像 token 数列表 (图像 embedding 已先进引擎)"""
    out = []
    for m in messages[:-1]:
        c, _u = split_content(m.get("content", ""))
        role = m.get("role", "user")
        if role == "assistant":
            # ★ 历史里的 assistant 回合也要带上强制空 think 块 —— 否则历史与上一轮
            #   "已吃进引擎状态" 的 token 流在第 10 个 token 处就对不齐, 会话续算永不命中
            out.append("<|im_start|>assistant\n<think>\n\n</think>\n\n%s<|im_end|>\n" % c)
        else:
            out.append("<|im_start|>%s\n%s<|im_end|>\n" % (role, c))
    last, _u = split_content(messages[-1].get("content", ""))
    if vision:
        pre = "".join("<|vision_start|>" + "<|image_pad|>" * int(n) + "<|vision_end|>" for n in vision)
        last = pre + str(last)
    out.append(TMPL.replace("{q}", str(last)))
    return "".join(out)

class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.dead = False
        self.reason = ""
        self.ntok_total = 0
        self.gen_secs = 0.0
        self.nreq = 0
        self.q = queue.Queue()
        self.ready = False
        self.prefill_info = ""
        # 真实 prompt token 数 与 本次复用的 token 数 (从引擎那行 "prompt=N tok 用时 ..s (快照/会话/全量 复用M tok)" 解析)
        # 以前 usage.prompt_tokens 硬写 13, 多轮时就报错了 —— 现在报引擎实测值。
        self.last_prompt_tok = 0
        self.last_reused = 0
        self.last_ms_tok = None       # ★ 引擎自报的预填充 ms/tok (暴露到 /health)
        self.last_finish = "stop"     # ★ 引擎自报的结束原因 (stop/length) —— finish_reason 诚实来源
        self.last_vis_info = ""
        if os.path.exists(FIFO):
            os.remove(FIFO)
        os.mkfifo(FIFO, 0o600)
        # 常开 O_RDWR: FIFO 的读端永不 EOF, 写端永不收到 EPIPE
        self.wfd = os.open(FIFO, os.O_RDWR)
        self.rfd = os.open(FIFO, os.O_RDWR)
        # ★ VIS_*_PASSTHRU: 引擎的调档变量必须【显式】写进子进程 env ——
        #   实测: 只在 start.sh 里 export 不起作用 (svc_guard/cron 拉起的那条链没有这些变量)
        inner = ("exec env LD_LIBRARY_PATH=%s VIS_CH=%s VIS_CHA=%s VIS_FOLD=%s %s --n %s --model %s < %s"
                 % (LDP, VIS_CH, VIS_CHA, VIS_FOLD, ENG, NGEN, MODEL, FIFO))
        cmd = [SAFEBIN, "-n", "orn_serve", "-t", str(SAFEMAX), "--", "/bin/sh", "-c", inner]
        log("启动引擎(安全网下):", " ".join(cmd))
        self.p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, bufsize=0)
        log("safe_run pid=%d, FIFO=%s, 等权重常驻 HBM(~20s)..." % (self.p.pid, FIFO))
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        try:
            while True:
                line = self.p.stdout.readline()
                if not line:
                    break
                s = line.decode("utf-8", "replace").rstrip("\r\n")
                if "权重常驻 HBM" in s:
                    # ★ 关键修正: 以前是 if "就绪" in s —— 引擎在【灌权重一开始】就打
                    #   "chip0 api::Context 就绪", 于是 /health 早在权重就位前 ~75s 就报
                    #   ready=true; 客户端此时发出的请求只能在 FIFO 里干等,
                    #   这正是"重启后第一发「你好」墙钟 56.9s"的根因。
                    #   改为只认"权重常驻 HBM"那一行 —— 权重真的在 HBM 上了才算 ready。
                    self.ready = True; log("引擎就绪(权重已常驻 HBM)")
                elif "就绪" in s:
                    log("引擎加载中|", s[:120])
                elif s.startswith(("[orn]", "XPURT", "[WARN", "[safe_run]")):
                    log("引擎|", s[:400])      # ★ 原来 170 会截掉 PPROF 预填充分桶行
                self.q.put(s)
        finally:
            self.q.put(None)

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

    def vis_send(self, path, W, H, timeout=900):
        """发 !VIS <path> <W> <H> -> 等 __VIS__ <n_tok>; 返回图像 token 数"""
        with self.lock:
            if not self.alive():
                raise RuntimeError("引擎不可用: %s" % (self.reason or "进程已退出"))
            self._write_all(("!VIS %s %d %d\n" % (path, W, H)).encode("ascii"))
            t0 = time.time()
            while time.time() - t0 < timeout:
                try:
                    line = self.q.get(timeout=1.0)
                except queue.Empty:
                    if not self.alive():
                        raise RuntimeError("引擎在视觉编码中退出")
                    continue
                if line is None:
                    raise RuntimeError("引擎退出")
                if line.startswith("[vis] 完成"):        # ★ VISINFO_FIX: 视觉塔耗时在此被消费
                    self.last_vis_info = line
                if line.startswith("__VIS__"):
                    p = line.split()
                    if len(p) >= 2 and p[1].isdigit():
                        return int(p[1])
                    raise RuntimeError("视觉塔失败: %s" % line)
            raise RuntimeError("视觉编码超时")

    def count_tokens(self, text, timeout=120):
        """★ !TOK: 只分词不生成 -> 真实 token 数 (网关按真实数裁, 不再靠字符估算)"""
        with self.lock:
            if not self.alive():
                raise RuntimeError("引擎不可用: %s" % (self.reason or "进程已退出"))
            b = base64.b64encode(text.encode("utf-8", "ignore")).decode("ascii")
            self._write_all(("!TOK@b64:%s\n" % b).encode("ascii"))
            t0 = time.time()
            while time.time() - t0 < timeout:
                try:
                    line = self.q.get(timeout=1.0)
                except queue.Empty:
                    if not self.alive():
                        raise RuntimeError("引擎在分词中退出")
                    continue
                if line is None:
                    raise RuntimeError("引擎退出")
                if line.startswith("TOK="):
                    return int(line[4:].split()[0])
            raise RuntimeError("引擎分词超时")

    def vis_reset(self):
        with self.lock:
            if self.alive():
                self._write_all(b"!VISR\n")

    def generate(self, prompt, max_tokens, on_token=None, sid=""):
        with self.lock:
            if not self.alive():
                raise RuntimeError("引擎不可用: %s" % (self.reason or "进程已退出"))
            sids = ("sid=%s@" % sid) if sid else ""
            req = "@%d@%sb64:%s\n" % (max_tokens, sids,
                                    base64.b64encode(prompt.encode("utf-8", "ignore")).decode("ascii"))
            try:
                self._write_all(req.encode("ascii"))
            except OSError as e:
                self.dead = True; self.reason = "写 FIFO 失败: %r" % (e,)
                raise RuntimeError(self.reason)
            t0 = time.time()
            out, ntok, started, pbuf = [], 0, False, b""
            self.last_prompt_tok, self.last_reused, self.last_ms_tok, self.last_finish = 0, 0, None, "stop"
            while True:
                try:
                    line = self.q.get(timeout=2400)
                except queue.Empty:
                    raise RuntimeError("引擎 2400s 无输出, 判卡死")
                if line is None:
                    self.dead = True
                    self.reason = ("引擎进程结束(rc=%s) —— 若是 safe_run 判定卡异常, 按规矩不再重试"
                                   % self.p.poll())
                    raise RuntimeError(self.reason)
                if "复用" in line:
                    self.prefill_info = line
                    # 引擎原始行: "[orn] prompt=13 tok 用时 23.09s (全量 复用0 tok) | 生成 ..."
                    try:
                        m = re.search(r"prompt=(\d+) tok", line)
                        if m: self.last_prompt_tok = int(m.group(1))
                        m = re.search(r"复用(\d+) tok", line)
                        if m: self.last_reused = int(m.group(1))
                        m = re.search(r"\(([\d.]+) ms/tok", line)
                        if m: self.last_ms_tok = float(m.group(1))
                    except Exception:
                        pass
                if "结束原因=" in line:
                    # ★ FINISH_FIX: 引擎在 pos>=MAXT 或 max_tokens 处停下时自报 length,
                    #   自然(停用词/EOS)结束才报 stop —— 网关据此回诚实的 finish_reason。
                    try:
                        self.last_finish = line.split("结束原因=")[1].split()[0]
                    except Exception:
                        self.last_finish = "stop"
                if line.startswith("[vis] 完成"):
                    self.last_vis_info = line
                if line == "__BEGIN__":
                    started = True; continue
                if line == "__END__":
                    break
                if not started or not line.startswith("__TOK__ "):
                    continue
                try:
                    tb = bytes.fromhex(line[8:].strip())
                except Exception:
                    tb = b""
                if RAWTOK:
                    try:
                        with open(RAWLOG, "a") as rf:
                            rf.write("%s\n" % line[8:].strip())
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
                    if on_token: on_token(s)
            dt = time.time() - t0
            self.ntok_total += ntok; self.gen_secs += dt; self.nreq += 1
            if pbuf:
                out.append(pbuf.decode("utf-8", "replace"))
            txt = "".join(out)
            for m in ("<|im_end|>", "<|im_start|>", "</think>", "<think>", "<s>"):
                i = txt.find(m)
                if i >= 0: txt = txt[:i]
            return txt.strip(), ntok, dt

ENGINE = None

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _wait_ready(self, maxt=None):
        # ★ WAITREADY: 未就绪时【排队等待】而不是回 503。
        import os as _os, time as _t
        if maxt is None:
            maxt = float(_os.environ.get("K200_WAIT_MAX", "1800"))
        t0 = _t.time()
        while not ENGINE.ready:
            if not ENGINE.alive():
                return False
            if (_t.time() - t0) >= maxt:
                return False
            _t.sleep(1.0)
        return True
    def _send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        try: self.wfile.write(b)
        except Exception: pass
    def do_GET(self):
        p = self.path.rstrip("/")
        if p in ("/health", "/healthz"):
            if ENGINE.alive():
                self._send(200, {"status": "ok" if ENGINE.ready else "loading",
                                 "engine": "alive", "ready": ENGINE.ready,
                                 "note": "" if ENGINE.ready else "引擎在灌权重到两芯 HBM (约 75s), 此刻发请求会被 503 拒绝",
                                 "model": MODEL_ID, "requests": ENGINE.nreq,
                                 "tok_total": ENGINE.ntok_total,
                                 "gen_secs": round(ENGINE.gen_secs, 1),
                                 "tok_per_s": round(ENGINE.ntok_total / ENGINE.gen_secs, 3) if ENGINE.gen_secs else None,
                                 # ★ 第25轮: MAXT 抬到 2048 后的上下文/预填充可见性
                                 "context_limit": 2048,
                                 "prompt_budget_tok": int(os.environ.get("K200_PROMPT_BUDGET", "600")),
                                 "prefill_target_ms": PREFILL_TARGET_MS,
                                 "client_timeout_ms": CLIENT_TIMEOUT_MS,
                                 "last_real_budget_tok": _budget_real_tok(ENGINE.last_ms_tok, 0),
                                 "last_prompt_tok": ENGINE.last_prompt_tok,
                                 "last_prefill_reused_tok": ENGINE.last_reused,
                                 "last_prefill_ms_per_tok": ENGINE.last_ms_tok,
                                 "last_finish_reason": ENGINE.last_finish,
                                 "last_prefill_est_secs": (round(ENGINE.last_ms_tok * (ENGINE.last_prompt_tok - ENGINE.last_reused) / 1000.0, 1)
                                                           if ENGINE.last_ms_tok else None)})
            else:
                self._send(503, {"status": "failed", "engine": "dead", "reason": ENGINE.reason,
                                 "note": "卡异常/引擎死亡 => 按安全规矩不重试, 不 soft_reset"})
            return
        if p in ("/v1/models", "/models"):
            self._send(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model",
                                                         "owned_by": "k200-ornith"}]}); return
        self._send(404, {"error": {"message": "GET /health | /v1/models"}})
    # ============ OpenAI 兼容: stream=true 的 SSE 流式 ============
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
                txt, ntok, dt = ENGINE.generate(prompt, mt, on_token=cb, sid=sid)
            except Exception as e:
                ev({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                    "k200": {"error": str(e)}})
                self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush(); return

            # ★ 收尾: 必须发带 finish_reason 的最终 chunk, 否则客户端报 "Stream ended without finish_reason"
            ev({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": ENGINE.last_finish}],
                "k200": {"gen_secs": round(dt, 2), "tok_per_s": round(ntok / dt, 4) if dt else None,
                         "prompt_tokens": ENGINE.last_prompt_tok,
                         "generated_tokens": ntok, "context_limit": 2048,
                         "prefill_ms_per_tok": ENGINE.last_ms_tok,
                         "finish_reason": ENGINE.last_finish,
                         "prefill_reused_tokens": ENGINE.last_reused, "sid": sid}})
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
        msgs_raw = req.get("messages")
        mt = int(req.get("max_tokens") or req.get("max_completion_tokens") or NGEN)
        if mt <= 0: mt = int(NGEN)
        # ★ MAXTOK_FIX: 原来 `mt > 512 ⇒ 夹到 NGEN(64)` 会把客户端请求(如 4096)砍成 64 token
        #   ⇒ 症状"回复一点儿就断了"。这里只设一个很宽的上限, 真正的边界交给引擎
        #   (引擎在 pos < MAXT 处自行停止)。可用 K200_MAXTOK 调整。
        _cap = int(os.environ.get("K200_MAXTOK", "8192"))
        if mt > _cap: mt = _cap
        if not ENGINE.alive():
            self._send(503, {"error": {"message": "engine dead: %s" % ENGINE.reason}}); return
        if not ENGINE.ready:
            # ★ WAITREADY: 等权重常驻 HBM 再答, 不向上游抛失败(否则中转站会禁用渠道)
            log("引擎未就绪, 请求排队等待 (上限 %ss)" % os.environ.get("K200_WAIT_MAX", "1800"))
            if not self._wait_ready():
                self._send(503, {"error": {"message": "engine loading: 权重尚未常驻 HBM (约需 75s), 请稍后重试"}}); return
        # ★★ REAL_TRIM: 引擎已就绪 ⇒ 先用 !TOK 拿【真实 token 数】再决定裁多少 (整条丢头保尾)
        try:
            _tt, _urls0 = split_content((msgs_raw or [{}])[-1].get("content", "")) if msgs_raw else ("", [])
        except Exception:
            _urls0 = []
        if msgs_raw:
            try:
                msgs = _TRIM_REAL(msgs_raw, ENGINE, reserve=(IMG_TOK_RESERVE if _urls0 else 0))
            except Exception as e:
                log("真实 token 裁剪异常, 退回估算裁剪:", repr(e))
                msgs = None
            if not msgs:
                msgs = _TRIM(msgs_raw) or msgs_raw
        else:
            msgs = [{"role": "user", "content": req.get("prompt", "")}]
        # ★ VIS_ORDER_FIX: 引擎已就绪, 再送图
        vision = []
        try:
            _t, urls = split_content(msgs[-1].get("content", ""))
            if urls:
                if not ENGINE.ready and not self._wait_ready():
                    self._send(503, {"error": {"message": "engine loading"}}); return
                ENGINE.vis_reset()
                tvis = time.time()
                for u in urls:
                    if not u:
                        continue
                    hp, iw, ih = prep_image(u)
                    n = ENGINE.vis_send(hp, iw, ih)
                    vision.append(n)
                    log("图像 %s %dx%d -> %d token" % (os.path.basename(hp), iw, ih, n))
                log("图像编码合计 %.2fs (%d 张)" % (time.time() - tvis, len(vision)))
        except Exception as e:
            log("图像处理失败:", repr(e))
            self._send(400, {"error": {"message": "image: %r" % (e,)}}); return
        prompt = build_prompt(msgs, vision)
        # 会话 id: 用【首条 user 消息】做键 => 同一会话的多轮请求共享 sid, 引擎才能续算 KV/状态
        try:
            first = ""
            for m in msgs:
                if m.get("role") == "user":
                    c = m.get("content", "")
                    first = " ".join(x.get("text", "") for x in c if isinstance(x, dict)) if isinstance(c, list) else str(c)
                    break
            sid = hashlib.sha1(first.encode("utf-8")).hexdigest()[:16]
        except Exception:
            sid = ""
        if req.get("stream"):
            return self._stream(req, prompt, mt, sid)
        t0 = time.time()
        try:
            ENGINE.prefill_info = ""
            txt, ntok, dt = ENGINE.generate(prompt, mt, sid=sid)
        except Exception as e:
            self._send(500, {"error": {"message": str(e)}}); return
        self._send(200, {"id": "chatcmpl-k200-%d" % int(time.time() * 1000),
                         "object": "chat.completion", "created": int(time.time()), "model": MODEL_ID,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": txt},
                                      "finish_reason": ENGINE.last_finish}],
                         "usage": {"prompt_tokens": ENGINE.last_prompt_tok or 13,
                                   "completion_tokens": ntok,
                                   "total_tokens": (ENGINE.last_prompt_tok or 13) + ntok},
                         "k200": {"gen_secs": round(dt, 2), "wall_secs": round(time.time() - t0, 2),
                                  "finish_reason": ENGINE.last_finish,
                                  "generated_tokens": ntok, "context_limit": 2048,
                                  "prefill_ms_per_tok": ENGINE.last_ms_tok,
                                  "vision_tokens": vision, "image_encode": ENGINE.last_vis_info[:160],
                                  "tok_per_s": round(ntok / dt, 4) if dt else None,
                                  "prompt_chars": len(prompt), "nothink": True, "sid": sid,
                                  "prompt_tokens": ENGINE.last_prompt_tok,
                                  "prefill_reused_tokens": ENGINE.last_reused,
                                  "engine_prefill": ENGINE.prefill_info.strip()[:200]}})

class TS(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

if __name__ == "__main__":
    ENGINE = Engine()
    srv = TS(("0.0.0.0", PORT), H)
    log("监听 0.0.0.0:%d (引擎在后台加载)" % PORT)
    srv.serve_forever()
