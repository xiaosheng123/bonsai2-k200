// vis.h — Ornith 视觉塔 (SigLIP-so400m-16-768 + qwen3vl_merger) 在 K200 上的实现
#pragma once
#include <vector>

// 惰性加载 mmproj 权重 (幂等); 返回 0 = OK
int vis_init(const char* mmproj_path);
int vis_loaded();

// in: 平面 f32, 长度 W*H*3, 索引 x + W*y + W*H*c, 值域已归一化 (px/255-0.5)/0.5
// out: 图像 embedding, 长度 n_tok*4096, 索引 o + 4096*t
// dumpdir: 非空则落盘逐阶段结果
int vis_forward(const float* in, int W, int H, std::vector<float>& out, int& n_tok, const char* dumpdir);

// 极小尺寸 gemm_int8 语义自检 (上卡前先跑)
int vis_gemm_selftest();

// ★ 第 19 轮: A 常驻卡上的分块 gemm 自检 (极小尺寸, 与老路径对拍)
int vis_cardA_selftest();
