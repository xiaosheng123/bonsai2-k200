#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# acc_w2.py — 第32轮 ①③验收 + ②的生产配置(闸门开)验收, 需在 K200_MAX_COLD_PREFILL_TOK=3000 窗口跑
#   ① 大冷请求 (DSH 形态: system+tools ≈ 11000+ tok) ⇒ 必须【秒回明确错误】;
#      紧接着发"你好" ⇒ 必须 <=3s 正常回答 (逐字节黄金题)。
#   ② 闸门开着也能用: 多消息 <=3000 tok 冷步进把 >=5000 tok 的共享前缀跑热 ⇒
#      第三个 session(不同 sid)必须 reused >=5000、预填充 <=3s, 且与全命中重发位级一致。
# =============================================================================
import hashlib, json, os, subprocess, time, urllib.request

B = "http://127.0.0.1:8090/v1/chat/completions"
MODEL = "ornith-1.5-9b-k200"
GOLDEN = "你好！有什么我可以帮你的吗？😊"
OUTD = "/home/caden/ornc/accept.post32"
LOG = os.path.join(OUTD, "w2.log")
LOGW = "/home/caden/ornc/serve2.log"
os.makedirs(OUTD, exist_ok=True)


def log(*a):
    m = " ".join(str(x) for x in a)
    print(m, flush=True)
    with open(LOG, "a") as f:
        f.write(m + "\n")


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.rstrip()


def eng_lines(n=6, pat="入槽|跨会话|LCP|拒绝|CPREF|预填充上限"):
    return sh("grep -aE '%s' %s | tail -%d" % (pat, LOGW, n))


def ask(msgs, mt=24, timeout=2400, tools=None, stream=False):
    req = {"model": MODEL, "temperature": 0, "messages": msgs, "max_tokens": mt}
    if tools:
        req["tools"] = tools
    body = json.dumps(req).encode()
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(
        urllib.request.Request(B, data=body, headers={"Content-Type": "application/json"}),
        timeout=timeout).read().decode())
    k = d.get("k200") or {}
    txt = d["choices"][0]["message"]["content"]
    return txt, k, time.time() - t0


LINE = ("[%s] tool_definition name=get_weather_forecast arguments={city:string, days:int} "
        "returns={daily:[{date:string,temp_c:float,rain_mm:float}]} description: Fetch the deterministic "
        "weather forecast for one city over a bounded number of days; call this whenever the user asks "
        "about weather, rain, temperature, travel planning or outdoor scheduling.\n")


def filler(n, tag):
    return LINE.replace("%s", tag) * n


def toolset(n=20):
    out = []
    for i in range(n):
        out.append({"type": "function", "function": {
            "name": "tool_%02d_long_name" % i,
            "description": "Deterministic helper #%d: reads project files, applies a bounded transformation and "
                           "returns a structured diff summary together with the list of touched symbols." % i,
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "absolute path inside the project"},
                "mode": {"type": "string", "enum": ["read", "diff", "apply"]},
                "limit": {"type": "integer", "description": "maximum number of records to return"}},
                "required": ["path"]}}})
    return out


def main():
    log("=== acc_w2 开始 %s ===" % time.strftime("%F %T"))
    h = json.loads(urllib.request.urlopen("http://127.0.0.1:8090/health", timeout=10).read().decode())
    log("health: max_cold_prefill_tok=%s (要求 3000) engine_maxt=%s ready=%s cold_reject_total=%s" %
        (h.get("max_cold_prefill_tok"), h.get("engine_maxt"), h.get("ready"), h.get("cold_reject_total")))
    log("md5: orn3=%s serve2.py=%s" % (sh("md5sum /home/caden/orn_engine/orn3 | cut -c1-32"),
                                       sh("md5sum /home/caden/ornc/serve2.py | cut -c1-32")))

    # ================= ① 大冷请求必须秒回明确错误 =================
    big_sys = ("You are a coding agent working in the repository. Follow the project conventions: "
               "type annotations everywhere, tests before commit, never change public interfaces.\n"
               + filler(230, "BIG32"))
    dsh = [{"role": "system", "content": big_sys},
           {"role": "user", "content": "只回答一个数字：3 加 4 等于几？"}]
    txt, k, w = ask(dsh, mt=32, tools=toolset(20))
    log("① 大冷请求(DSH 形态): wall=%.2fs 拒绝标记=%s" % (w, k.get("rejected")))
    log("   正文: %s" % (txt or "").replace("\n", " ")[:300])
    log("   k200: prompt_tokens=%s reused=%s cold=%s max=%s measured_by=%s" %
        (k.get("prompt_tokens"), k.get("prefill_reused_tokens"), k.get("cold_prefill_tokens"),
         k.get("max_cold_prefill_tok"), k.get("measured_by")))
    log("   ★ 秒回(<20s)=%s; 明确错误(正文含'输入过大'或'冷预填充')=%s" %
        (w < 20, ("输入过大" in txt) or ("冷预填充" in txt)))
    log("   引擎台账(不应有本次的预填充):\n" + eng_lines(6, "CPREF|拒绝|入槽"))

    # ---- 紧接着发"你好" ⇒ <=3s 正常回答 ----
    t_g, kg, wg = ask([{"role": "user", "content": "你好"}], mt=24)
    log("① 紧接着 '你好': wall=%.2fs 逐字节==黄金题=%s 答=%r │ prefill=%ss reused=%s" %
        (wg, t_g == GOLDEN, t_g, kg.get("prefill_secs"), kg.get("prefill_reused_tokens")))
    log("   ★ 你好 <=3s = %s" % (wg <= 3.0))

    # ================= ② 闸门开着: 冷步进跑热 >=5000 tok 共享前缀 =================
    _, kc, _ = ask([{"role": "system", "content": filler(20, "CAL32")},
                    {"role": "user", "content": "只回答一个数字：1 加 1 等于几？"}], mt=8)
    per = max(1.0, float(kc.get("prompt_tokens", 0)) / 20.0)
    log("标定: %.2f tok/行" % per)
    n1, n2 = int(2800.0 / per), int(2900.0 / per)
    m1, m2 = filler(n1, "M1A"), filler(n2, "M2B")
    S = m1 + m2
    log("共享前缀: m1=%d 行 m2=%d 行 (估 %d tok, 要求 >=5000)" % (n1, n2, int((n1 + n2) * per)))

    t_w1, k_w1, w_w1 = ask([{"role": "system", "content": m1},
                            {"role": "user", "content": "只回答一个数字：2 加 2 等于几？"}], mt=16)
    log("②-1 冷步进1 [m1]: prompt=%s reused=%s prefill=%ss wall=%.1fs (要求冷预填充<=3000 才不会被闸门拒)" %
        (k_w1.get("prompt_tokens"), k_w1.get("prefill_reused_tokens"), k_w1.get("prefill_secs"), w_w1))
    t_w2, k_w2, w_w2 = ask([{"role": "system", "content": S},
                            {"role": "user", "content": "只回答一个数字：8 加 1 等于几？"}], mt=16)
    log("②-2 冷步进2 [m1+m2]: prompt=%s reused=%s prefill=%ss wall=%.1fs" %
        (k_w2.get("prompt_tokens"), k_w2.get("prefill_reused_tokens"), k_w2.get("prefill_secs"), w_w2))
    log("   引擎台账:\n" + eng_lines(6))

    qz = "只回答一个数字：9 加 9 等于几？"
    t_h1, k_h1, w_h1 = ask([{"role": "system", "content": S}, {"role": "user", "content": qz}], mt=16)
    log("②-3 第三个 session 发同一大前缀+新问题: prompt=%s reused=%s prefill=%ss wall=%.2fs 答=%r" %
        (k_h1.get("prompt_tokens"), k_h1.get("prefill_reused_tokens"), k_h1.get("prefill_secs"), w_h1, t_h1))
    log("   ★ 跨会话 reused>=5000 = %s; 预填充<=3s = %s" %
        (int(k_h1.get("prefill_reused_tokens") or 0) >= 5000, float(k_h1.get("prefill_secs") or 9) <= 3.0))
    t_h2, k_h2, w_h2 = ask([{"role": "system", "content": S}, {"role": "user", "content": qz}], mt=16)
    log("②-4 第四个 client 重发同一 prompt: reused=%s prefill=%ss wall=%.2fs ★指纹==②-3 = %s 回答逐字节相同=%s" %
        (k_h2.get("prefill_reused_tokens"), k_h2.get("prefill_secs"), w_h2,
         k_h2.get("gen_fingerprint") == k_h1.get("gen_fingerprint"), t_h2 == t_h1))
    log("   指纹: ②-3=%s / ②-4=%s" % (k_h1.get("gen_fingerprint"), k_h2.get("gen_fingerprint")))
    log("   引擎台账(跨会话命中证据):\n" + eng_lines(8, "跨会话|入槽|LCP|检查点表"))

    log("--- 卡台账 ---")
    log("异常累计 = %s" % sh("grep -ac 'Exception in kernel execution' /var/log/kern.log"))
    log("reset_count = %s" % sh("cat /proc/xpu/dev0/reset_count /proc/xpu/dev1/reset_count | tr '\\n' ' '"))
    log("state = %s" % sh("cat /proc/xpu/dev0/state /proc/xpu/dev1/state | tr '\\n' ' '"))
    log("=== acc_w2 结束 %s ===" % time.strftime("%F %T"))


main()
