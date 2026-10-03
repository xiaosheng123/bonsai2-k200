#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# acc_w1.py — 第32轮 ② 验收 Window-1 (需在 K200_MAX_COLD_PREFILL_TOK=0 的窗口内跑)
#   ① 冷跑: [大 system(≈6000 tok) + 问题 q1]  —— 这一次是真冷 (reused=0), 记 ★GEN 指纹
#   ② 第二个客户端发【同一 prompt】   —— 必须 复用≈全 prompt、预填充≈0、★GEN 指纹与冷跑【逐字节相同】
#   ③ 第三个客户端发【同一大前缀 + 不同问题】(sid 不同) —— 必须 reused≈前缀长度、预填充<=3s
#   ④ ④ 再发一次 ③ 的 prompt (第4个客户端) —— 全命中, 指纹与 ③ 相同 (位级一致)
# =============================================================================
import hashlib, json, os, subprocess, sys, time, urllib.request

B = "http://127.0.0.1:8090/v1/chat/completions"
MODEL = "ornith-1.5-9b-k200"
OUTD = "/home/caden/ornc/accept.post32"
LOG = os.path.join(OUTD, "w1.log")
LOGW = "/home/caden/ornc/serve2.log"
os.makedirs(OUTD, exist_ok=True)


def log(*a):
    m = " ".join(str(x) for x in a)
    print(m, flush=True)
    with open(LOG, "a") as f:
        f.write(m + "\n")


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.rstrip()


def eng_lines(n=6, pat="入槽|跨会话|LCP|预填充"):
    return sh("grep -aE '%s' %s | tail -%d" % (pat, LOGW, n))


def ask(msgs, mt=24, timeout=2400):
    body = json.dumps({"model": MODEL, "temperature": 0, "messages": msgs, "max_tokens": mt}).encode()
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


def main():
    log("=== acc_w1 开始 %s ===" % time.strftime("%F %T"))
    h = json.loads(urllib.request.urlopen("http://127.0.0.1:8090/health", timeout=10).read().decode())
    log("health: max_cold_prefill_tok=%s engine_maxt=%s ready=%s" %
        (h.get("max_cold_prefill_tok"), h.get("engine_maxt"), h.get("ready")))
    log("md5: orn3=%s serve2.py=%s" % (sh("md5sum /home/caden/orn_engine/orn3 | cut -c1-32"),
                                       sh("md5sum /home/caden/ornc/serve2.py | cut -c1-32")))

    # ---- 0) 标定: 用【不同前缀】的小载荷量出 token/行 (小请求, 十几秒) ----
    _, kc, _ = ask([{"role": "system", "content": filler(20, "CALIB7Z")},
                    {"role": "user", "content": "只回答一个数字：1 加 1 等于几？"}], mt=8)
    per = max(1.0, float(kc.get("prompt_tokens", 0)) / 20.0)
    log("标定: 20 行 ⇒ prompt_tokens=%s ⇒ %.2f tok/行" % (kc.get("prompt_tokens"), per))
    N = int(5300.0 / per)   # ★ 冷跑必须 >=5000 tok (验收要求), 同时 <= 本窗口的闸门值(6000) 才跑得起来
    S = filler(N, "S32")
    q1 = "只回答一个数字：3 加 4 等于几？"
    q2 = "只回答一个数字：5 加 6 等于几？"
    log("大 system: %d 行 (估 %d tok)" % (N, int(N * per)))

    # ---- ① 冷跑 ----
    txt1, k1, w1 = ask([{"role": "system", "content": S}, {"role": "user", "content": q1}], mt=24)
    log("① 冷跑(客户端1/q1): prompt_tokens=%s reused=%s prefill=%ss wall=%.1fs 答=%r" %
        (k1.get("prompt_tokens"), k1.get("prefill_reused_tokens"), k1.get("prefill_secs"), w1, txt1))
    log("   ① ★GEN 指纹 = %s" % k1.get("gen_fingerprint"))
    log("   ① 引擎台账:\n" + eng_lines(4))
    fp1, a1 = k1.get("gen_fingerprint"), hashlib.sha1(txt1.encode()).hexdigest()[:12]
    ptok = int(k1.get("prompt_tokens") or 0)

    # ---- ② 同一 prompt, 另一个客户端 ----
    txt2, k2, w2 = ask([{"role": "system", "content": S}, {"role": "user", "content": q1}], mt=24)
    fp2 = k2.get("gen_fingerprint")
    log("② 同一 prompt(客户端2): reused=%s prefill=%ss wall=%.2fs 答=%r" %
        (k2.get("prefill_reused_tokens"), k2.get("prefill_secs"), w2, txt2))
    log("   ② ★GEN 指纹 = %s" % fp2)
    log("   ★ 复用 tok = %s (要求 ≈ prompt=%s); 预填充 %.2fs (要求 <=3s); 指纹逐字节相同=%s; 回答逐字节相同=%s" %
        (k2.get("prefill_reused_tokens"), ptok, float(k2.get("prefill_secs") or 9),
         fp2 == fp1, txt2 == txt1))

    # ---- ③ 同一大前缀 + 不同问题 (sid 不同) ----
    txt3, k3, w3 = ask([{"role": "system", "content": S}, {"role": "user", "content": q2}], mt=24)
    fp3 = k3.get("gen_fingerprint")
    log("③ 同前缀+不同问题(客户端3/q2): prompt=%s reused=%s prefill=%ss wall=%.2fs 答=%r" %
        (k3.get("prompt_tokens"), k3.get("prefill_reused_tokens"), k3.get("prefill_secs"), w3, txt3))
    log("   ★ 跨会话复用 tok = %s (要求 ≈ 前缀长度); 预填充 %.2fs (要求 <=3s)" %
        (k3.get("prefill_reused_tokens"), float(k3.get("prefill_secs") or 9)))
    log("   ③ 引擎台账:\n" + eng_lines(6))

    # ---- ④ 再发 ③ 的 prompt: 全命中, 与 ③ 位级一致 ----
    txt4, k4, w4 = ask([{"role": "system", "content": S}, {"role": "user", "content": q2}], mt=24)
    fp4 = k4.get("gen_fingerprint")
    log("④ 重发 ③(客户端4): reused=%s prefill=%ss wall=%.2fs" %
        (k4.get("prefill_reused_tokens"), k4.get("prefill_secs"), w4))
    log("   ★ ④ 指纹 == ③ 指纹 = %s; 回答逐字节相同=%s" % (fp4 == fp3, txt4 == txt3))

    log("--- 卡台账 ---")
    log("异常累计 = %s" % sh("grep -ac 'Exception in kernel execution' /var/log/kern.log"))
    log("reset_count = %s" % sh("cat /proc/xpu/dev0/reset_count /proc/xpu/dev1/reset_count | tr '\\n' ' '"))
    log("state = %s" % sh("cat /proc/xpu/dev0/state /proc/xpu/dev1/state | tr '\\n' ' '"))
    log("=== acc_w1 结束 %s ===" % time.strftime("%F %T"))


main()
