#!/usr/bin/env python3
"""mkimg.py <img> <out.f32> [maxside]
   照 llama.cpp mtmd 的预处理: 目标尺寸 = 32 对齐 + 保持比例 + PAD_CEIL 黑边,
   像素值 (px/255-0.5)/0.5, 平面布局 x + W*y + W*H*c。
   若目标尺寸 == 原尺寸 => 直接拷贝不做缩放 (llama.cpp 同路径)。"""
import sys, os
import numpy as np
from PIL import Image

MAXPX = 4194304
MINPX = 8192


def calc_size(W, H):
    w = max(32, round(W / 32) * 32)
    h = max(32, round(H / 32) * 32)
    if h * w > MAXPX:
        beta = (H * W / MAXPX) ** 0.5
        h = max(32, int(H / beta) // 32 * 32)
        w = max(32, int(W / beta) // 32 * 32)
    elif h * w < MINPX:
        beta = (MINPX / (H * W)) ** 0.5
        w = max(32, int(np.ceil(W * beta / 32) * 32))
        h = max(32, int(np.ceil(H * beta / 32) * 32))
    return w, h


def main():
    img, outp = sys.argv[1], sys.argv[2]
    im = Image.open(img).convert('RGB')
    W, H = im.size
    tw, th = calc_size(W, H)
    if (tw, th) == (W, H):
        a = np.asarray(im, np.uint8).astype(np.float32)
        v = (a / 255.0 - 0.5) / 0.5
    else:
        scale = min(tw / W, th / H)
        nw, nh = min(tw, int(np.ceil(W * scale))), min(th, int(np.ceil(H * scale)))
        r = np.asarray(im.resize((nw, nh), Image.BICUBIC), np.uint8).astype(np.float32)
        canvas = np.zeros((th, tw, 3), np.float32)
        ox, oy = (tw - nw) // 2, (th - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw, :] = (r / 255.0 - 0.5) / 0.5
        v = canvas
    x = np.ascontiguousarray(np.transpose(v, (2, 0, 1)))   # [c][y][x] C 序 -> x + W*y + W*H*c
    x.astype('<f4').tofile(outp)
    print('MKIMG %s %dx%d -> %dx%d (%d floats=%d bytes)' % (os.path.basename(img), W, H, tw, th, x.size, x.size * 4))
    print('SIZE %d %d' % (tw, th))


main()
