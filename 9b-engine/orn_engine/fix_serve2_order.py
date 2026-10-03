import sys
P='/home/caden/ornc/serve2.py'
s=open(P,encoding='utf-8').read()
if 'VIS_ORDER_FIX' in s: print('已修'); sys.exit(0)
old='''        # ★ VIS: 最后一条消息里的图像先送进引擎编码
        vision = []'''
new='''        if not ENGINE.alive():
            self._send(503, {"error": {"message": "engine dead: %s" % ENGINE.reason}}); return
        if not ENGINE.ready:
            # ★ 关键修正: 未就绪时直接 503 fail-fast (与文本路径一致)
            self._send(503, {"error": {"message": "engine loading: 权重尚未常驻 HBM (约需 75s), 请稍后重试"}}); return
        # ★ VIS_ORDER_FIX: 引擎已就绪, 再送图
        vision = []'''
assert s.count(old)==1
s=s.replace(old,new)
old2='''        if not ENGINE.alive():
            self._send(503, {"error": {"message": "engine dead: %s" % ENGINE.reason}}); return
        if not ENGINE.ready:
            # ★ 关键修正: 未就绪时直接 503 fail-fast, 不要让请求在 FIFO 里干等整个灌权重过程
            #   (以前 /health 报 ready=true 但引擎还在灌权重 => 冷启动首请求墙钟 56.9s)
            self._send(503, {"error": {"message": "engine loading: 权重尚未常驻 HBM (约需 75s), 请稍后重试"}}); return
'''
# 删除原先靠后的那对检查 (第二次出现)
i = s.find(old2, s.find(new))
assert i > 0
s = s[:i] + s[i+len(old2):]
open(P,'w',encoding='utf-8').write(s)
import ast; ast.parse(s); print('serve2.py 顺序修正 OK')
