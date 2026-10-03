#!/usr/bin/env python3
"""给 serve2.py 补 OpenAI 兼容的 SSE 流式 (stream=true)。幂等: 已打过就跳过。"""
import sys, shutil, py_compile

P = "/home/caden/ornc/serve2.py"
s = open(P, encoding="utf-8").read()
if "text/event-stream" in s:
    print("已支持流式, 跳过"); sys.exit(0)
shutil.copy(P, P + ".bak.sse")

SSE = "\n".join([
    '    # ============ OpenAI 兼容: stream=true 的 SSE 流式 ============',
    '    def _stream(self, req, prompt, mt, sid):',
    '        cid = "chatcmpl-k200-%d" % int(time.time() * 1000)',
    '        created = int(time.time())',
    '        try:',
    '            self.send_response(200)',
    '            self.send_header("Content-Type", "text/event-stream; charset=utf-8")',
    '            self.send_header("Cache-Control", "no-cache")',
    '            self.send_header("Connection", "close")',
    '            self.end_headers()',
    '            self.close_connection = True',
    '',
    '            def ev(obj):',
    '                self.wfile.write(("data: %s\\n\\n" % json.dumps(obj, ensure_ascii=False)).encode("utf-8"))',
    '                self.wfile.flush()',
    '',
    '            ev({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,',
    '                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})',
    '',
    '            def cb(tb):',
    '                c = tb.decode("utf-8", "ignore")',
    '                if not c:',
    '                    return',
    '                ev({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,',
    '                    "choices": [{"index": 0, "delta": {"content": c}, "finish_reason": None}]})',
    '',
    '            try:',
    '                txt, ntok, dt = ENGINE.generate(prompt, mt, on_token=cb, sid=sid)',
    '            except Exception as e:',
    '                ev({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,',
    '                    "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],',
    '                    "k200": {"error": str(e)}})',
    '                self.wfile.write(b"data: [DONE]\\n\\n"); self.wfile.flush(); return',
    '',
    '            # ★ 收尾: 必须发带 finish_reason 的最终 chunk, 否则客户端报 "Stream ended without finish_reason"',
    '            ev({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,',
    '                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],',
    '                "k200": {"gen_secs": round(dt, 2), "tok_per_s": round(ntok / dt, 4) if dt else None,',
    '                         "prompt_tokens": ENGINE.last_prompt_tok,',
    '                         "prefill_reused_tokens": ENGINE.last_reused, "sid": sid}})',
    '            if (req.get("stream_options") or {}).get("include_usage"):',
    '                ev({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,',
    '                    "choices": [], "usage": {"prompt_tokens": ENGINE.last_prompt_tok or 13,',
    '                                             "completion_tokens": ntok,',
    '                                             "total_tokens": (ENGINE.last_prompt_tok or 13) + ntok}})',
    '            self.wfile.write(b"data: [DONE]\\n\\n"); self.wfile.flush()',
    '        except (BrokenPipeError, ConnectionResetError):',
    '            pass',
    '        except Exception as e:',
    '            try:',
    '                self._send(500, {"error": {"message": str(e)}})',
    '            except Exception:',
    '                pass',
    '',
])

anchor = "    def do_POST(self):"
assert anchor in s, "找不到 do_POST"
s = s.replace(anchor, SSE + anchor, 1)

# generate() 若未逐 token 回调, 补上
if "on_token(" not in s:
    old = "                ntok += 1; pbuf += tb"
    assert old in s, "找不到 token 计数行"
    s = s.replace(old, old + "\n                if on_token:\n                    try: on_token(tb)\n                    except Exception: on_token = None", 1)
    print("[patch] 已在 generate() 补 on_token 回调")

# do_POST 里分流到流式
old2 = "        t0 = time.time()\n        try:\n            ENGINE.prefill_info = \"\""
assert old2 in s, "找不到 do_POST 生成起点"
s = s.replace(old2, "        if req.get(\"stream\"):\n            return self._stream(req, prompt, mt, sid)\n" + old2, 1)

# 一些客户端用 max_completion_tokens
s = s.replace('        mt = int(req.get("max_tokens") or NGEN)',
              '        mt = int(req.get("max_tokens") or req.get("max_completion_tokens") or NGEN)', 1)

open(P, "w", encoding="utf-8").write(s)
py_compile.compile(P, doraise=True)
print("[patch] SSE 流式已加入, 语法 OK; 备份:", P + ".bak.sse")
