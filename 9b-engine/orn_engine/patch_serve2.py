#!/usr/bin/env python3
# serve2.py 增加 image_url 支持 (纯增量; 纯文本请求路径一字未动)
import sys
P = '/home/caden/ornc/serve2.py'
s = open(P, encoding='utf-8').read()
if 'VIS_IMAGE_PATCHED' in s:
    print('已打过补丁'); sys.exit(0)

# 1) 常量 + 图像预处理 + content 拆分
anchor = 'RAWLOG   = os.environ.get("K200_RAWLOG", "/home/caden/ornc/raw_tok.log")\n'
assert anchor in s
s = s.replace(anchor, anchor + '''
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
''')

# 2) build_prompt 支持 vision 块
old = '''def build_prompt(messages):
    """ChatML + 强制空 think 块, 特殊 token 全部来自 GGUF 推导 (与引擎打印的 id 一致)。"""
    out = []
    for m in messages[:-1]:
        c = m.get("content", "")
        if isinstance(c, list):
            c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
        role = m.get("role", "user")'''
new = '''def build_prompt(messages, vision=None):
    """ChatML + 强制空 think 块, 特殊 token 全部来自 GGUF 推导 (与引擎打印的 id 一致)。
       vision: 若非空 = 最后一条消息的图像 token 数列表 (图像 embedding 已先进引擎)"""
    out = []
    for m in messages[:-1]:
        c, _u = split_content(m.get("content", ""))
        role = m.get("role", "user")'''
assert old in s
s = s.replace(old, new)

old = '''    last = messages[-1].get("content", "")
    if isinstance(last, list):
        last = " ".join(x.get("text", "") for x in last if isinstance(x, dict))
    out.append(TMPL.replace("{q}", str(last)))
    return "".join(out)'''
new = '''    last, _u = split_content(messages[-1].get("content", ""))
    if vision:
        pre = "".join("<|vision_start|>" + "<|image_pad|>" * int(n) + "<|vision_end|>" for n in vision)
        last = pre + str(last)
    out.append(TMPL.replace("{q}", str(last)))
    return "".join(out)'''
assert old in s
s = s.replace(old, new)

# 3) Engine: 视觉命令
old = '''    def generate(self, prompt, max_tokens, on_token=None, sid=""):'''
new = '''    def vis_send(self, path, W, H, timeout=900):
        """发 !VIS <path> <W> <H> -> 等 __VIS__ <n_tok>; 返回图像 token 数"""
        with self.lock:
            if not self.alive():
                raise RuntimeError("引擎不可用: %s" % (self.reason or "进程已退出"))
            self._write_all(("!VIS %s %d %d\\n" % (path, W, H)).encode("ascii"))
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
                if line.startswith("__VIS__"):
                    p = line.split()
                    if len(p) >= 2 and p[1].isdigit():
                        return int(p[1])
                    raise RuntimeError("视觉塔失败: %s" % line)
            raise RuntimeError("视觉编码超时")

    def vis_reset(self):
        with self.lock:
            if self.alive():
                self._write_all(b"!VISR\\n")

    def generate(self, prompt, max_tokens, on_token=None, sid=""):'''
assert old in s
s = s.replace(old, new, 1)

# 4) reader 里抓视觉塔耗时行
old = '''                if line == "__BEGIN__":'''
new = '''                if line.startswith("[vis] 完成"):
                    self.last_vis_info = line
                if line == "__BEGIN__":'''
assert old in s
s = s.replace(old, new, 1)
old = '''        self.last_prompt_tok = 0
        self.last_reused = 0'''
new = '''        self.last_prompt_tok = 0
        self.last_reused = 0
        self.last_vis_info = ""'''
assert old in s
s = s.replace(old, new, 1)

# 5) do_POST: 取图 -> !VIS -> 拼 prompt
old = '''        prompt = build_prompt(msgs)
        if not ENGINE.alive():'''
new = '''        # ★ VIS: 最后一条消息里的图像先送进引擎编码
        vision = []
        try:
            _t, urls = split_content(msgs[-1].get("content", ""))
            if urls:
                if not ENGINE.ready:
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
        if not ENGINE.alive():'''
assert old in s
s = s.replace(old, new, 1)

# 6) 响应里加视觉信息
old = '''                         "k200": {"gen_secs": round(dt, 2), "wall_secs": round(time.time() - t0, 2),'''
new = '''                         "k200": {"gen_secs": round(dt, 2), "wall_secs": round(time.time() - t0, 2),
                                  "vision_tokens": vision, "image_encode": ENGINE.last_vis_info[:160],'''
assert old in s
s = s.replace(old, new, 1)

open(P, 'w', encoding='utf-8').write(s)
print('serve2.py 补丁完成')
