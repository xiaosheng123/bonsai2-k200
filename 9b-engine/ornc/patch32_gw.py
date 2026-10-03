#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# patch32_gw.py — 第32轮网关改动 (serve2.py -> serve2.new32.py)
#   ① 入队前的【冷预填充代价闸门】: 用引擎真实 tokenizer 问 !CPREF, 超限立刻明确报错
#      (HTTP 200 + content 说明 —— 不用 4xx/5xx: 上游 New API 中转站会把报错渠道自动禁用 ✗)
#   ② 客户端中途断开(BrokenPipe) ⇒ 立刻 !SDEL 取消引擎侧槽, 不让槽空跑到 MAXT
#   ③ /health 暴露 max_cold_prefill_tok 等 (与 engine_maxt/prompt_budget_tok 同风格)
#   ④ FIFO 不再每次重启 remove+mkfifo (引擎才能活过网关重启) + 把 K200_FIFO/上限传给引擎
#   每个替换断言命中次数, 不对就整体失败。
# =============================================================================
import sys

SRC = "/home/caden/ornc/serve2.py"
DST = "/home/caden/ornc/serve2.new32.py"

orig = open(SRC, encoding="utf-8").read()
s = orig
applied = []


def rep(name, old, new, expect=1):
    global s
    n = s.count(old)
    if n != expect:
        print("FATAL[%s]: 命中 %d 次 (期望 %d) ⇒ 中止, 不改文件" % (name, n, expect))
        sys.exit(2)
    s = s.replace(old, new)
    applied.append(name)
    print("  ok  %s" % name)


# ------------------------------------------------------------- 1) import stat
rep("import-stat",
    "import os, sys, json, base64, time, threading, subprocess, http.server, socketserver, queue, errno, hashlib, re\n",
    "import os, sys, json, base64, time, threading, subprocess, http.server, socketserver, queue, errno, hashlib, re, stat\n")

# ------------------------------------------------------------- 2) 冷预填充上限开关
rep("env-maxcold",
    'ENG_MAXT      = int(os.environ.get("K200_ENGINE_MAXT", "8192"))\n',
    'ENG_MAXT      = int(os.environ.get("K200_ENGINE_MAXT", "8192"))\n'
    '# ★★ 第32轮: 【失败要快】冷预填充上限 (token) ★★\n'
    '#   真实事故: 用户 14685 tok 的 agent 载荷 ⇒ 引擎头部砍到 8192 ⇒ 冷预填充 ~13 分钟 ⇒\n'
    '#   唯一引擎通道被独占 ⇒ 后续请求全排队 ⇒ 用户观感="服务死了"。现在【入队之前】就用引擎自己的\n'
    '#   tokenizer 问清"本次要冷预填充多少 token", 超限且无前缀可复用 ⇒ 立刻明确报错, 绝不塞进引擎。\n'
    '#   0 = 关闭该保护。\n'
    'MAXCOLDPREFILL = int(os.environ.get("K200_MAX_COLD_PREFILL_TOK", "3000"))\n')

# ------------------------------------------------------------- 3) 拒绝话术 / 时间估计
rep("cold-rej-note",
    'EMPTY_NOTE = ("（K200 引擎本次没有生成任何 token：原因=%s。通常是输入被上下文上限截断，"\n'
    '              "请缩短输入、或把问题单独放在最后一句后重试。）")\n',
    'EMPTY_NOTE = ("（K200 引擎本次没有生成任何 token：原因=%s。通常是输入被上下文上限截断，"\n'
    '              "请缩短输入、或把问题单独放在最后一句后重试。）")\n'
    '\n'
    '# ★★ 第32轮: 冷预填充超限的明确错误 ★★\n'
    '#   为什么用 HTTP 200 + content 而不是 4xx/5xx: 客户端多经 NAS 上的 New API 中转站, 它会把\n'
    '#   【报错的渠道自动禁用】⇒ 之后即使服务正常, 客户端仍一直收到 "No available channel" ✗。\n'
    '#   所以把"拒绝原因"当正文回给用户 (人能看到、可自救), 而不是回错误码。\n'
    'COLD_REJ_NOTE = ("（K200 服务拒绝本次请求：输入过大，且没有可命中的前缀缓存可利用。\\n"\n'
    '                 "  本请求需要【冷预填充 %d tok】> 上限 %d tok（可复用 %d tok 已被扣掉）。\\n"\n'
    '                 "  按实测 %s 估算，冷预填充约需 %s —— 会把唯一的引擎通道占满、让之后所有请求"\n'
    '                 "排队等待（这正是用户看到的「发你好两分钟没回复」的原因），所以在入队前就拒绝。\\n"\n'
    '                 "  请缩短输入（或先发一个短一点的请求把共享前缀跑热）后重试。）")\n'
    '\n'
    '\n'
    'def _cold_rej_msg(tok, reused, maxcold):\n'
    '    msp = 0.0\n'
    '    try:\n'
    '        msp = ENGINE.last_prefill_ms_per_tok\n'
    '    except Exception:\n'
    '        msp = 0.0\n'
    '    if not msp or msp <= 0:\n'
    '        msp = 50.0\n'
    '    cold = max(0, int(tok) - int(reused))\n'
    '    secs = cold * msp / 1000.0\n'
    '    t = ("%.0f 秒" % secs) if secs < 120 else ("%.1f 分钟" % (secs / 60.0))\n'
    '    return COLD_REJ_NOTE % (cold, maxcold, int(reused), "%.0f ms/tok" % msp, t)\n')

# ------------------------------------------------------------- 4) Engine: 计数器 + 探针
rep("engine-counters",
    "        self.last_prompt_tok = 0\n        self.last_reused = 0\n",
    "        self.last_prompt_tok = 0\n        self.last_reused = 0\n"
    "        self.last_prefill_ms_per_tok = 0.0      # ★ 第32轮: 实测冷预填充速度(给拒绝话术估时间)\n"
    "        self.last_cpref = {}                    # ★ 第32轮: 最近一次 !CPREF 探针结果\n"
    "        self.cold_reject_total = 0               # ★ 第32轮: 因冷预填充超限被拒的请求数\n"
    "        self.last_cold_reject = {}\n")

rep("cpref-method",
    "    # ---------------- 生成 (进批; 按 cid 路由) ----------------\n",
    '''    def cpref(self, prompt, imgfp=0, timeout=120):
        """★★ 第32轮: 入队前向引擎咨询【真实 token 数 + 可复用前缀长度】(只读, 不占槽) ★★
        判定必须用真实 token 数: 字符估算对中文/数字误差可达 3~4 倍 ✗ (引擎 tokenizer 才是真值)。
        返回 dict(tok, raw, reused, cold, coldmax, lcp, haspad, tk_ms); 探针失败 ⇒ None (调用方按保守口径)。"""
        try:
            if not self.alive():
                return None
            raw = prompt.encode("utf-8", "ignore")
            b64 = base64.b64encode(raw).decode("ascii")
            img = ("img=%016x " % imgfp) if imgfp else ""
            line = "!CPREF %slen=%d b64:%s" % (img, len(raw), b64)
            with self.clk:
                self.send(line)
                s = self._ctl_wait("__CPREF__", timeout)
            if not s:
                return None
            d = {}
            for x in s.split()[1:]:
                if "=" in x:
                    k, v = x.split("=", 1)
                    try:
                        d[k] = int(v)
                    except ValueError:
                        try:
                            d[k] = float(v)
                        except ValueError:
                            d[k] = v
            if int(d.get("tok", -1)) < 0:
                self.last_cpref = dict(d, ok=False)
                return None
            d["cold"] = int(d.get("tok", 0)) - int(d.get("reused", 0))
            self.last_cpref = dict(d, ok=True)
            log("CPREF 探针(引擎 tokenizer): tok=%s raw=%s 可复用=%s 冷预填充=%s (上限 %s) tk_ms=%s"
                % (d.get("tok"), d.get("raw"), d.get("reused"), d["cold"], d.get("coldmax"), d.get("tk_ms")))
            return d
        except Exception as e:
            log("CPREF 探针失败:", repr(e))
            return None

    # ---------------- 生成 (进批; 按 cid 路由) ----------------
''')

# ------------------------------------------------------------- 5) generate: 取消 + 提速台账
rep("gen-cancel",
    "        finally:\n            with self.rlock:\n                self.routes.pop(cid, None)\n        dt = time.time() - t0\n",
    '''        except BaseException as _e:
            # ★★ 第32轮: 客户端中途断开(BrokenPipe/ConnectionReset 或任何异常) ⇒ 立刻 !SDEL 取消
            #   引擎侧的槽。旧行为: 路由被摘掉但引擎继续算到 maxtok/MAXT (客户端都死了槽还占几分钟,
            #   而 solo 语义下这会让后续请求全排队 ⇒ 又一处"服务像死了")。!SDEL 幂等。
            try:
                self.send("!SDEL %d" % cid)
                log("★ 客户端中断/异常 ⇒ 已发 !SDEL %d 取消引擎侧槽 (%s)" % (cid, type(_e).__name__))
            except Exception:
                pass
            raise
        finally:
            with self.rlock:
                self.routes.pop(cid, None)
        dt = time.time() - t0
''')

rep("gen-mspt",
    '                            meta["prefill_s"] = float(d.get("prefill_s", 0))\n',
    '                            meta["prefill_s"] = float(d.get("prefill_s", 0))\n'
    '                            _new = int(d.get("new", 0))\n'
    '                            if _new > 0:      # ★ 第32轮: 记实测冷预填充 ms/tok (给拒绝话术估时间)\n'
    '                                self.last_prefill_ms_per_tok = 1000.0 * meta["prefill_s"] / _new\n')

# ------------------------------------------------------------- 6) ctl 路由: __CPREF__
rep("route-cpref",
    '        if s.startswith(("__VIS__", "__VISR__", "__VISIMG__", "__VISQ__", "__SLOTS__",\n'
    '                         "__SSTAT__", "__SPAUSE__", "__SRESUME__", "__SDEL__")):\n',
    '        if s.startswith(("__VIS__", "__VISR__", "__VISIMG__", "__VISQ__", "__SLOTS__",\n'
    '                         "__SSTAT__", "__SPAUSE__", "__SRESUME__", "__SDEL__", "__CPREF__")):\n')

# ------------------------------------------------------------- 7) FIFO 不再重建 (引擎才能活过网关重启)
rep("fifo-keep",
    "        if os.path.exists(FIFO):\n            os.remove(FIFO)\n        os.mkfifo(FIFO, 0o600)\n",
    "        # ★★ 第32轮: FIFO 文件【不再每次重启就 remove+mkfifo】★★\n"
    "        #   路径被重建时, 引擎手里那个已打开的 fd 指向的是【被 unlink 的旧 inode】⇒ 网关重启后\n"
    "        #   引擎再也收不到命令 (事故现场: kill 网关后 /health 与所有请求都不响应, 只能 stop/start 80s ✗)。\n"
    "        #   现在同一个 inode 跨网关重启存活 ⇒ 引擎与它的检查点表(热缓存)一起活下来 ✓\n"
    "        if not os.path.exists(FIFO):\n"
    "            os.mkfifo(FIFO, 0o600)\n"
    "        elif not stat.S_ISFIFO(os.stat(FIFO).st_mode):\n"
    "            os.remove(FIFO); os.mkfifo(FIFO, 0o600)\n")

# ------------------------------------------------------------- 8) 给引擎传 K200_FIFO / 上限
rep("inner-env",
    '        inner = ("exec env LD_LIBRARY_PATH=%s VIS_CH=%s VIS_CHA=%s VIS_FOLD=%s K200_PROF=%s "\n'
    '                 "K200_SLOTS=%s K200_CKPT=%s K200_MAXT=%s %s --n %s --model %s < %s"\n'
    '                 % (LDP, VIS_CH, VIS_CHA, VIS_FOLD, PROF, SLOTS, CKPT, ENG_MAXT, ENG, NGEN, MODEL, FIFO))\n',
    '        inner = ("exec env LD_LIBRARY_PATH=%s VIS_CH=%s VIS_CHA=%s VIS_FOLD=%s K200_PROF=%s "\n'
    '                 "K200_SLOTS=%s K200_CKPT=%s K200_MAXT=%s K200_FIFO=%s K200_MAX_COLD_PREFILL_TOK=%s "\n'
    '                 "%s --n %s --model %s < %s"\n'
    '                 % (LDP, VIS_CH, VIS_CHA, VIS_FOLD, PROF, SLOTS, CKPT, ENG_MAXT, FIFO, MAXCOLDPREFILL,\n'
    '                    ENG, NGEN, MODEL, FIFO))\n')

# ------------------------------------------------------------- 9) /health 暴露
rep("health-fields",
    '                                 "prompt_budget_tok": PROMPT_BUDGET, "engine_maxt": ENG_MAXT,\n',
    '                                 "prompt_budget_tok": PROMPT_BUDGET, "engine_maxt": ENG_MAXT,\n'
    '                                 # ★ 第32轮: 冷预填充闸门当前生效值 + 最近一次判据 (照 engine_maxt/prompt_budget_tok 风格)\n'
    '                                 "max_cold_prefill_tok": MAXCOLDPREFILL,\n'
    '                                 "cold_reject_total": ENGINE.cold_reject_total,\n'
    '                                 "last_cold_reject": ENGINE.last_cold_reject,\n'
    '                                 "last_cpref": ENGINE.last_cpref,\n'
    '                                 "prefill_ms_per_tok": (round(ENGINE.last_prefill_ms_per_tok, 2)\n'
    '                                                        if ENGINE.last_prefill_ms_per_tok else None),\n')

# ------------------------------------------------------------- 10) 闸门 + 拒绝响应
rep("guard-methods",
    "    def _send(self, code, obj):\n",
    '''    # ================= ★★ 第32轮: 入队前的冷预填充代价闸门 ★★ =================
    def _cold_guard(self, prompt, imgfp=0, msgs=None):
        """入队【之前】判定: 本次请求要【冷预填充】多少 token (真实 token 数, 来自引擎的 !CPREF)。
        判据: 冷预填充 tok (ids - 可复用前缀) > K200_MAX_COLD_PREFILL_TOK ⇒ 立刻拒绝。
        为什么在网关这一层: 引擎是单线程的, 一旦开算冷预填充, 命令循环被整段占住 ⇒ /health 与
        后续所有请求一起排队。入队前拒绝 = 唯一的引擎通道永远只接"付得起的活"。"""
        if MAXCOLDPREFILL <= 0:
            return None
        est = _est_tok(prompt)
        if est * 3.0 <= MAXCOLDPREFILL:      # 连估算的 3 倍都够不到门槛 ⇒ 不必打扰引擎
            return None
        cp = ENGINE.cpref(prompt, imgfp)
        if cp is None:
            # 探针失败(引擎忙/死/帧不完整) ⇒ 保守: 用字符估算, 宁可误拒也不把长活塞进引擎
            if est > MAXCOLDPREFILL:
                rej = {"tok": int(est), "raw": int(est), "reused": 0, "cold": int(est),
                       "coldmax": MAXCOLDPREFILL, "how": "est(探针失败, 保守口径)"}
                ENGINE.cold_reject_total += 1
                ENGINE.last_cold_reject = dict(rej)
                log("★ 拒绝(冷预填充超限, 探针失败按估算): est=%.0f tok > 上限 %d" % (est, MAXCOLDPREFILL))
                return rej
            return None
        cold = int(cp.get("tok", 0)) - int(cp.get("reused", 0))
        if cold > MAXCOLDPREFILL:
            rej = dict(cp, cold=cold, coldmax=MAXCOLDPREFILL, how="engine-tokenizer(!CPREF)")
            ENGINE.cold_reject_total += 1
            ENGINE.last_cold_reject = dict(rej)
            log("★ 拒绝(冷预填充超限): 真实 prompt=%s tok 可复用=%s ⇒ 冷预填充=%d > 上限 %d ⇒ "
                "入队前直接报错, 不占用引擎通道 (结算: 这一步本来要占 ~%.0fs) ✗"
                % (cp.get("tok"), cp.get("reused"), cold, MAXCOLDPREFILL,
                   cold * (ENGINE.last_prefill_ms_per_tok or 50.0) / 1000.0))
            return rej
        return None

    def _reject(self, req, sid, rej):
        """把"冷预填充超限"当【正文】回给用户 (HTTP 200) —— 见 COLD_REJ_NOTE 里为什么不用 4xx/5xx。"""
        msg = _cold_rej_msg(int(rej.get("tok", 0)), int(rej.get("reused", 0)), MAXCOLDPREFILL)
        meta = {"rejected": "cold-prefill-too-large",
                "prompt_tokens": int(rej.get("tok", 0)), "prompt_tokens_raw": int(rej.get("raw", 0)),
                "prefill_reused_tokens": int(rej.get("reused", 0)),
                "cold_prefill_tokens": int(rej.get("cold", 0)),
                "max_cold_prefill_tok": MAXCOLDPREFILL, "sid": sid,
                "measured_by": rej.get("how", "?"), "engine_cpref": ENGINE.last_cpref}
        if not req.get("stream"):
            self._send(200, {"id": "chatcmpl-k200-%d" % int(time.time() * 1000),
                             "object": "chat.completion", "created": int(time.time()), "model": MODEL_ID,
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": msg},
                                          "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": meta["prompt_tokens"], "completion_tokens": 0,
                                       "total_tokens": meta["prompt_tokens"]},
                             "k200": meta})
            return
        cid = "chatcmpl-k200-%d" % int(time.time() * 1000)
        try:                                  # 流式客户端也要拿到一帧合法 SSE + [DONE]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            for obj in ({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": MODEL_ID, "choices": [{"index": 0, "delta": {"role": "assistant", "content": msg},
                                                         "finish_reason": None}]},
                        {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": MODEL_ID, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                         "k200": meta}):
                self.wfile.write(("data: %s\\n\\n" % json.dumps(obj, ensure_ascii=False)).encode("utf-8"))
            self.wfile.write(b"data: [DONE]\\n\\n")
            self.wfile.flush()
        except Exception:
            pass

    def _send(self, code, obj):
''')

# ------------------------------------------------------------- 11) do_POST 两个分支接闸门
rep("guard-call-vision",
    '                    prompt = build_prompt(msgs, vision, tools)\n'
    '                    if req.get("stream"):\n'
    '                        return self._stream(req, prompt, mt, self._sid(msgs), tools)\n',
    '                    prompt = build_prompt(msgs, vision, tools)\n'
    '                    _rej = self._cold_guard(prompt, imgfp, msgs)      # ★ 第32轮: 入队前闸门\n'
    '                    if _rej:\n'
    '                        return self._reject(req, self._sid(msgs), _rej)\n'
    '                    if req.get("stream"):\n'
    '                        return self._stream(req, prompt, mt, self._sid(msgs), tools)\n')

rep("guard-call-text",
    '        if not vision:\n'
    '            prompt = build_prompt(msgs, vision, tools)\n'
    '            if req.get("stream"):\n'
    '                return self._stream(req, prompt, mt, self._sid(msgs), tools)\n',
    '        if not vision:\n'
    '            prompt = build_prompt(msgs, vision, tools)\n'
    '            _rej = self._cold_guard(prompt, 0, msgs)             # ★ 第32轮: 入队前闸门\n'
    '            if _rej:\n'
    '                return self._reject(req, self._sid(msgs), _rej)\n'
    '            if req.get("stream"):\n'
    '                return self._stream(req, prompt, mt, self._sid(msgs), tools)\n')

open(DST, "w", encoding="utf-8").write(s)
print("== 已写出 %s (%d 字节, 原 %d) ==" % (DST, len(s), len(orig)))
print("== 应用的改动 %d 项: %s ==" % (len(applied), ", ".join(applied)))
