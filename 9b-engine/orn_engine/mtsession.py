#!/usr/bin/env python3
# mtsession.py —— 模拟 DSH 多轮会话: 第1轮全量, 后续轮只在尾部加一小段
#   关注: 每轮的 (真实 prompt_tok, 复用 tok, ms/tok, 墙钟) 与回答正确性/finish_reason
import json, time, urllib.request, sys

B = "http://127.0.0.1:8090/v1/chat/completions"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/caden/ornc/accept.post25b/mtsession.json"

def post(msgs, mt=32, timeout=600):
    body = {"model": "ornith-1.5-9b-k200", "messages": msgs, "max_tokens": mt}
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(urllib.request.Request(
        B, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}),
        timeout=timeout).read().decode())
    ch = d["choices"][0]
    return dict(text=ch["message"]["content"], finish=ch.get("finish_reason"),
                k=d.get("k200") or {}, wall=round(time.time() - t0, 2))

SYS = "你是编码助手。项目规范：所有函数必须有类型注解；提交前跑测试；不要改公共接口。"
# 第 1 轮: 系统 + 一段历史 + 问题 (总长约 600 估算 tok)
msgs = [{"role": "system", "content": SYS * 6},
        {"role": "user", "content": "背景资料：" + "本项目的接口约定与部署约束如下。" * 20},
        {"role": "assistant", "content": "已了解项目规范。"},
        {"role": "user", "content": "只回答一个数字：2 + 3 等于几？"}]

res = []
for turn in range(1, 7):
    r = post(msgs, 24 if turn == 1 else 16)
    k = r["k"]
    line = ("第%d轮: prompt_tok=%-5s 复用=%-5s prefill_ms/tok=%-7s 墙钟=%-6s finish=%-7s 答=%r"
            % (turn, k.get("prompt_tokens"), k.get("prefill_reused_tokens"),
               k.get("prefill_ms_per_tok"), r["wall"], r["finish"], r["text"][:40]))
    print(line, flush=True)
    res.append(dict(turn=turn, text=r["text"], finish=r["finish"], wall=r["wall"], k=k))
    if not r["text"]:
        print("  ★ 空回复, 中止", flush=True); break
    # 下一轮 = 历史 + 上一轮回答 + 新问题 (DSH 行为: 整段重发)
    msgs = list(msgs) + [{"role": "assistant", "content": r["text"]},
                         {"role": "user", "content": "再算一个：第%d轮，4 + %d 等于几？只回答数字。" % (turn + 1, turn)}]

with open(OUT, "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("写入 %s" % OUT, flush=True)
