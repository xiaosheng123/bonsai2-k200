#!/usr/bin/env python3
# srvprobe.py —— 8090 服务态 A/B 探针 (不需要卡窗口: 就是正常调线上服务)
#   目的: 同一批请求在【MAXT=1024 旧引擎】与【MAXT=2048 新引擎】上的
#         (a) 逐字节回答  (b) 真实 prompt_tok 与 ms/tok  (c) finish_reason 诚实性
#         (d) 会话续算复用 token 数  (e) 超长 prompt 是否空回复
#   用法: srvprobe.py <输出json> <标签>
import json, sys, time, urllib.request, urllib.error

B = "http://127.0.0.1:8090/v1/chat/completions"
OUT = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else "x"

def post(body, timeout=900):
    req = urllib.request.Request(B, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        return dict(err="HTTP %s %s" % (e.code, e.read().decode("utf-8", "replace")[:200]), wall=time.time() - t0)
    except Exception as e:
        return dict(err=repr(e), wall=time.time() - t0)
    ch = (d.get("choices") or [{}])[0]
    return dict(text=(ch.get("message") or {}).get("content", ""),
                finish=ch.get("finish_reason"), k200=d.get("k200") or {},
                usage=d.get("usage") or {}, wall=round(time.time() - t0, 2))

def msg(path, role, content):
    return {"role": role, "content": content}

FILL = "背景资料：本项目的接口约定、字段含义、错误码与部署约束如下所述。"
def big(k, tail="只回答一个数字：7 + 8 等于几？"):
    return (FILL * k) + "\n\n" + tail

def long14():
    filler = "你是编码助手。以下是项目规范：所有函数必须有类型注解；提交前跑测试；不要改公共接口。"
    h = [msg(None, "system", filler * 40)]
    for i in range(6):
        h.append(msg(None, "user", ("这是第 %d 轮的历史上下文，" % i) + "补充说明。" * 60))
        h.append(msg(None, "assistant", "明白了，我会按规范处理。" * 30))
    h.append(msg(None, "user", "只回答一个数字：3 + 4 等于几？"))
    return h

R = []

def rec(name, req):
    r = post(req)
    r["name"] = name
    R.append(r)
    k = r.get("k200") or {}
    print("[%s] %-18s prompt_tok=%-5s ms/tok=%-7s finish=%-8s wall=%-6s text=%r%s"
          % (TAG, name, k.get("prompt_tokens"), k.get("prefill_ms_per_tok") or (k.get("engine_prefill") or "").split("(")[-1][:5],
             r.get("finish"), r.get("wall"), (r.get("text") or "")[:50],
             "" if not r.get("err") else " ERR=" + r["err"][:120]), flush=True)
    return r

def main():
    # 1) 黄金题 (逐字节判据)
    rec("golden", {"model": "ornith-1.5-9b-k200", "messages": [msg(None, "user", "你好")], "max_tokens": 24})
    # 2) 14 条长会话 (走 _TRIM 真路径)
    h = long14()
    rec("long14_messages", {"model": "ornith-1.5-9b-k200", "messages": h, "max_tokens": 32})
    # 3) 递增尺寸的超长 prompt (走 prompt 字段 => 不过 _TRIM, 直接压引擎位置上限)
    for k in (20, 50, 80, 95):
        rec("plain_k%02d" % k, {"model": "ornith-1.5-9b-k200", "prompt": big(k), "max_tokens": 16})
    # 4) 会话续算: 同一 sid (首条 user 决定) 两轮
    h1 = long14()
    r1 = rec("sess_t1", {"model": "ornith-1.5-9b-k200", "messages": h1, "max_tokens": 24})
    if r1.get("text"):
        h2 = list(h1) + [msg(None, "assistant", r1["text"]), msg(None, "user", "那 5 + 6 呢？只回答数字。")]
        rec("sess_t2", {"model": "ornith-1.5-9b-k200", "messages": h2, "max_tokens": 16})
    with open(OUT, "w") as f:
        json.dump({"tag": TAG, "items": R}, f, ensure_ascii=False, indent=1)
    print("[%s] 写入 %s" % (TAG, OUT), flush=True)

main()
