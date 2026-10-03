#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib
cd /home/caden/k200llm
cat > /home/caden/k200llm/tmpl_new.txt <<'TEOF'
<s><|im_start|>user
{q}<|im_end|>
<|im_start|>assistant
<think>

</think>

TEOF
export K200_TMPL="$(cat /home/caden/k200llm/tmpl_new.txt)"
pkill -9 -f server.py; pkill -9 -f lb.py; sleep 2
/usr/local/xpu-4.33.0/tools/soft_reset 0 >/dev/null 2>&1
/usr/local/xpu-4.33.0/tools/soft_reset 1 >/dev/null 2>&1
XPU_VISIBLE_DEVICES=0 env -u K200_DUAL -u K200_I8 -u K200_I8G K200_ENGINE=/home/caden/k200llm/mini.th K200_PORT=8091 nohup python3 server.py > y0.log 2>&1 &
sleep 42
XPU_VISIBLE_DEVICES=1 env -u K200_DUAL -u K200_I8 -u K200_I8G K200_ENGINE=/home/caden/k200llm/mini.th K200_PORT=8092 nohup python3 server.py > y1.log 2>&1 &
sleep 42
nohup python3 lb.py > lb5.log 2>&1 &
sleep 4
echo "== 健康 =="
for p in 8091 8092 8090; do echo -n " $p: "; timeout 6 curl -s http://127.0.0.1:$p/health || echo -n DOWN; echo; done
echo "== 5 条实测 =="
python3 - <<'PEOF'
import json, urllib.request, time
qs = ["你好", "1+1等于几", "用一句话介绍北京", "把“今天天气不错，我们出去走走吧”翻译成英文", "我明天要交一份季度税务报告，列5条检查清单"]
for q in qs:
    b = json.dumps({"model": "minicpm5-2b-k200", "messages": [{"role": "user", "content": q}], "max_tokens": 96}).encode()
    r = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions", data=b, headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        d = json.loads(urllib.request.urlopen(r, timeout=120).read().decode())
        c = d["choices"][0]["message"]["content"]
        print("Q: %s\n   → %r  (%.1fs)" % (q, c[:160], time.time()-t))
    except Exception as e:
        print("Q: %s\n   ✗ %s" % (q, repr(e)[:70]))
PEOF
