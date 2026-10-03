#!/usr/bin/env python3
# WAITREADY 补丁: 未就绪时排队等待, 而不是回 503
# 原因: 上游 New API 中转站见到失败会把渠道【自动禁用】, 之后即使我们恢复,
#       客户端(DSH)也会一直收到 model_not_found("No available channel ...")。
P = "/home/caden/ornc/serve2.py"
s = open(P, encoding="utf-8").read()
if "WAITREADY" in s:
    print("already patched"); raise SystemExit(0)

anchor = "    def log_message(self, *a): pass\n"
assert s.count(anchor) == 1, "anchor count=%d" % s.count(anchor)
helper = anchor + '''    def _wait_ready(self, maxt=None):
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
'''
s = s.replace(anchor, helper)

old1 = '''        if not ENGINE.ready:
            # ★ 关键修正: 未就绪时直接 503 fail-fast (与文本路径一致)
            self._send(503, {"error": {"message": "engine loading: 权重尚未常驻 HBM (约需 75s), 请稍后重试"}}); return'''
new1 = '''        if not ENGINE.ready:
            # ★ WAITREADY: 等权重常驻 HBM 再答, 不向上游抛失败(否则中转站会禁用渠道)
            log("引擎未就绪, 请求排队等待 (上限 %ss)" % os.environ.get("K200_WAIT_MAX", "1800"))
            if not self._wait_ready():
                self._send(503, {"error": {"message": "engine loading: 权重尚未常驻 HBM (约需 75s), 请稍后重试"}}); return'''
assert s.count(old1) == 1, "old1 count=%d" % s.count(old1)
s = s.replace(old1, new1)

old2 = '''                if not ENGINE.ready:
                    self._send(503, {"error": {"message": "engine loading"}}); return'''
new2 = '''                if not ENGINE.ready and not self._wait_ready():
                    self._send(503, {"error": {"message": "engine loading"}}); return'''
assert s.count(old2) == 1, "old2 count=%d" % s.count(old2)
s = s.replace(old2, new2)

open(P, "w", encoding="utf-8").write(s)
print("patched ok, bytes=%d" % len(s))
