"""ggml 张量语义的 numpy 模拟 (修正版: 严格 ne0 最快 / 一维缓冲)
   张量 = dict(d=一维缓冲(np.float32, ggml 序), ne=(4,), nb=(4,字节), off=元素偏移)
   约定: 逻辑元素 (i0,i1,i2,i3) 在缓冲中的偏移 = i0*nb[0]/4 + i1*nb[1]/4 + ..."""
import numpy as np

def _rev(n):
    return tuple(range(n-1, -1, -1))

def flat_of(arr):
    """(i0,i1,..) 索引的多维数组 -> ggml 一维序 (i0 最快)"""
    arr = np.asarray(arr)
    if arr.ndim <= 1:
        return np.ascontiguousarray(arr).ravel()
    return np.ascontiguousarray(arr.transpose(_rev(arr.ndim))).ravel()

def from_flat(lin, ne):
    """ggml 一维序 -> (i0,i1,..) 索引数组"""
    ne = tuple(ne)
    ne = ne + (1,)*(4-len(ne))
    a = np.asarray(lin).reshape(tuple(ne[::-1]))
    return a.transpose(_rev(4))

def mk(d, ne, nb=None, off=0):
    ne = tuple(ne) + (1,)*(4-len(ne))
    if nb is None:
        nb = []
        s = 4
        for i in range(4):
            nb.append(s); s *= ne[i]
        nb = tuple(nb)
    return dict(d=np.asarray(d), ne=ne, nb=tuple(nb), off=off)

def mkarr(arr, ne=None):
    """由 (i0,i1,..) 索引数组建张量"""
    arr = np.asarray(arr, np.float32)
    if ne is None:
        ne = arr.shape
    return mk(flat_of(arr), ne)

def mat(t):
    d, ne, nb = t['d'], t['ne'], t['nb']
    idx = np.zeros((1, 1, 1, 1), np.int64)
    for i in range(4):
        if ne[i] > 1:
            sh = [1, 1, 1, 1]; sh[i] = ne[i]
            idx = idx + (np.arange(ne[i], dtype=np.int64)*(nb[i]//4)).reshape(sh)
    return d[t['off'] + idx]

def permute(t, a0, a1, a2, a3):
    ne = [0]*4; nb = [0]*4
    for i, ax in enumerate((a0, a1, a2, a3)):
        ne[ax] = t['ne'][i]; nb[ax] = t['nb'][i]
    return dict(d=t['d'], ne=tuple(ne), nb=tuple(nb), off=t['off'])

def cont(t, ne_new):
    """ggml_cont_*: dst 按 src 的逻辑线性序顺序填 -> 新张量连续"""
    ne_new = tuple(ne_new) + (1,)*(4-len(ne_new))
    assert int(np.prod(t['ne'])) == int(np.prod(ne_new)), (t['ne'], ne_new)
    lin = flat_of(mat(t))
    return mk(lin, ne_new)

def add(a, b):
    return mkarr(mat(a) + mat(b), a['ne'])

def relrms(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    return 100.0*np.sqrt(((A-B)**2).sum()/max((B*B).sum(), 1e-30))


# ---------------- 视觉塔常量/工具 ----------------
class VT:
    def __init__(self, g):
        self.g = g
        self.n_embd = 1152; self.n_head = 16; self.dh = 72
        self.n_layer = 27; self.eps = 1e-6; self.ffn = 4304

    def interp_align_corners(self, tab, W, H):
        """tab:(nside,nside,C) -> (W,H,C)  ggml bilinear align_corners"""
        nside = tab.shape[0]
        sf0 = (W-1)/(nside-1); sf1 = (H-1)/(nside-1)
        x = np.arange(W)/sf0; y = np.arange(H)/sf1
        x0 = np.clip(np.floor(x).astype(np.int64), 0, nside-1)
        x1 = np.clip(x0+1, 0, nside-1)
        y0 = np.clip(np.floor(y).astype(np.int64), 0, nside-1)
        y1 = np.clip(y0+1, 0, nside-1)
        dx = np.clip(x-x0, 0, 1)[:, None, None]; dy = np.clip(y-y0, 0, 1)[None, :, None]
        a = tab[x0][:, y0]; b = tab[x1][:, y0]; c = tab[x0][:, y1]; d = tab[x1][:, y1]
        return (a*(1-dx)*(1-dy) + b*dx*(1-dy) + c*(1-dx)*dy + d*dx*dy).astype(np.float32)

    def load_img(self, path):
        """-> pix 一维缓冲 (ggml 序, ne=[W,H,3]), W, H"""
        from PIL import Image
        im = Image.open(path).convert('RGB')
        a = np.asarray(im, np.uint8).astype(np.float32)
        H, W = a.shape[0], a.shape[1]
        v = (a/255.0 - 0.5)/0.5                      # (H,W,3)
        vol = np.ascontiguousarray(np.transpose(v, (1, 0, 2)))   # (W,H,3)
        return flat_of(vol), W, H

    def conv_patch(self, inp_raw, W0, W1, bias, ow, oh):
        """conv2d(两半相加) -> 张量 ne=[ow,oh,1152,1]"""
        x = mat(inp_raw)                       # (W,H,3,1)
        K = mat(W0) + mat(W1)                  # (16,16,3,1152)
        out = np.zeros((ow, oh, 1152), np.float32)
        for i in range(16):
            for j in range(16):
                out += x[i::16, j::16, :, 0] @ K[i, j]
        return mkarr(out + bias[None, None, :], (ow, oh, 1152, 1))

    def merge_chain(self, tens):
        """照抄 qwen3vl.cpp: [w,h,c,b] -> [c,w,h,b] -> cont -> reshape -> permute -> cont"""
        ow, oh = tens['ne'][0], tens['ne'][1]
        t = permute(tens, 1, 2, 0, 3)
        t = cont(t, (1152*2, ow//2, oh, 1))
        t = cont(t, (1152*2, ow//2, 2, oh//2))
        t = permute(t, 0, 2, 1, 3)
        t = cont(t, (1152, ow*oh, 1))
        return mat(t)[:, :, 0, 0]              # (1152, n_pos)

    def pos_grid(self, ow, oh):
        """ggml 序列的位置编码: 先取 (c,p) -> reshape (48,48) -> [p%48, p//48, c] -> 插值"""
        a = self.g.np('v.position_embd.weight')          # (1152, 2304) 索引 [c,p]
        tab = a.reshape(1152, 48, 48).transpose(2, 1, 0)  # [p%48, p//48, c]
        if ow == 48 and oh == 48:
            return np.ascontiguousarray(tab)
        return self.interp_align_corners(np.ascontiguousarray(tab), ow, oh)

    def head(self, imgpath):
        W0 = mkarr(self.g.np('v.patch_embd.weight'))
        W1 = mkarr(self.g.np('v.patch_embd.weight.1'))
        bias = self.g.np('v.patch_embd.bias').astype(np.float32)
        pix, W, H = self.load_img(imgpath)
        inp_raw = mk(pix, (W, H, 3, 1))
        ow, oh = W//16, H//16
        conv = self.conv_patch(inp_raw, W0, W1, bias, ow, oh)
        a = self.merge_chain(conv)
        pg = mkarr(self.pos_grid(ow, oh), (ow, oh, 1152, 1))
        b = self.merge_chain(pg)
        return a, b                                # (1152,n_pos) 两次
