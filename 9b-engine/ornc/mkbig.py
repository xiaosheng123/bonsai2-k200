#!/usr/bin/env python3
# 768x768 测试图 (模型原生分辨率: 768/16=48 patch, 不做 resize 也不做 pos-emb 插值)
from PIL import Image, ImageDraw, ImageFont
import os
W = H = 768
out = "/home/caden/ornc/mmtests"
os.makedirs(out, exist_ok=True)

# 1) 文字: 大写 K200
im = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(im)
f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 300)
d.text((60, 200), "K200", font=f, fill=(0, 0, 170))
d.rectangle([20, 20, W-20, H-20], outline=(0, 0, 0), width=10)
im.save(out + "/big_text.png")

# 2) 几何: 红圆 + 绿方 + 蓝三角
im = Image.new("RGB", (W, H), (245, 245, 245)); d = ImageDraw.Draw(im)
d.ellipse([60, 60, 320, 320], fill=(210, 30, 30))          # 红圆 (左上)
d.rectangle([440, 60, 700, 320], fill=(30, 170, 30))        # 绿方 (右上)
d.polygon([(384, 430), (620, 700), (148, 700)], fill=(30, 60, 210))  # 蓝三角 (下中)
im.save(out + "/big_shapes.png")

# 3) 表格 3x3
im = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(im)
f2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 90)
rows = [["A", "1", "7"], ["B", "2", "8"], ["C", "3", "9"]]
cw = rh = 200; x0 = y0 = 84
for i, r in enumerate(rows):
    for j, c in enumerate(r):
        x = x0 + j*cw; y = y0 + i*rh
        d.rectangle([x, y, x+cw, y+rh], outline=(0, 0, 0), width=6)
        d.text((x+70, y+45), c, font=f2, fill=(0, 0, 0))
im.save(out + "/big_table.png")
print("ok", sorted(os.listdir(out)))
