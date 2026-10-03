# MM_PLAN —— Ornith 多模态（看图）实现方案：侦察结论 + 计划

> 阶段一（只读侦察）结论。所有数字均来自本机实测/llama.cpp 源码，未编造；
> 没测到的地方明确写「未验证」。

## 0. 结论速览

| 项 | 结论 |
|---|---|
| mmproj 架构 | `general.architecture = clip`，`clip.projector_type = qwen3vl_merger`（= llama.cpp `PROJECTOR_TYPE_QWEN3VL`） |
| 视觉塔骨架 | 类 SigLIP-so400m-16-768：hidden 1152 / 27 层 / 16 头 / patch 16 / ffn 4304 / eps 1e-6 / GELU(tanh) |
| 权重格式 | patch conv 与 pos emb 是 **F32**（不是 BF16！文件名有误导）；其余 BF16 |
| 张量总数 / 字节 | 334 张量 / 921,684,928 B；加 19,744 B 头 ⇒ 921,704,672 B = 文件实际大小 ✓ 完整 |
| 一张图多少 image token | `(W/16)*(H/16)/4`；768×768 ⇒ 2304 patch ⇒ **576 token**；512×512 ⇒ 1024 patch ⇒ **256 token**（已由 llama.cpp 日志 "256 new tokens" 实测确认） |
| 预处理 | 见 §3（照抄 mtmd `mtmd_image_preprocessor_dyn_size` + `image_mean/std=0.5`） |
| 真值 | llama.cpp（本机 `build/bin/llama-mtmd-cli`，host CPU）已跑通并落盘 embeddings + 回答 |

## 1. mmproj-Ornith-1.5-9B-BF16.gguf 元数据（全键）

```
general.architecture           = clip
general.type                   = mmproj
general.name                   = Ornith 1.5 9b
general.size_label             = 456M
general.file_type              = 32  (GGUF_TYPE unknown/32)
clip.has_vision_encoder        = True
clip.projector_type            = qwen3vl_merger
clip.vision.projection_dim     = 4096            ← 文本模型 hidden = 4096 ✓ (qwen35.embedding_length)
clip.vision.image_size         = 768
clip.vision.patch_size         = 16
clip.vision.embedding_length   = 1152
clip.vision.feed_forward_length= 4304
clip.vision.block_count        = 27
clip.vision.attention.head_count = 16            ⇒ d_head = 1152/16 = 72
clip.vision.image_mean         = [0.5, 0.5, 0.5]
clip.vision.image_std          = [0.5, 0.5, 0.5]
clip.vision.spatial_merge_size = 2
clip.vision.attention.layer_norm_epsilon = 1e-6
clip.use_gelu                  = True            ⇒ GELU(tanh 近似), 非 SiLU
clip.vision.is_deepstack_layers= [False]×27      ⇒ 无 deepstack 支路
```

llama.cpp 侧由这些键推出的运行时参数：`n_merge = spatial_merge_size = 2`、
`image_resize_algo = RESIZE_ALGO_BICUBIC`、`set_limit_image_tokens(8, 4096)`
⇒ `image_min_pixels = 8*16²*2² = 8192`、`image_max_pixels = 4096*16²*2² = 4,194,304`；
对齐尺寸 `patch_size * n_merge = 32`；`image_resize_pad = PAD_CEIL`、pad 颜色 (0,0,0)。

## 2. 张量清单（334 张；按类给出，全部字节已核对总和 = 921,684,928）

| 张量 | 形状(ggml ne) | 类型 | 每层字节 | 说明 |
|---|---|---|---|---|
| `v.patch_embd.weight` | [16,16,3,1152] | **F32** | 3,538,944 | conv2d 核 (kw,kh,inCh,outCh) |
| `v.patch_embd.weight.1` | [16,16,3,1152] | **F32** | 3,538,944 | conv3d 时间维第二半 |
| `v.patch_embd.bias` | [1152] | F32 | 4,608 | |
| `v.position_embd.weight` | [1152, 2304] | **F32** | 10,616,832 | 2304 = 48² = (768/16)² 学习式绝对位置编码 |
| `v.blk.N.ln1.weight/bias` | [1152] | F32 | 4608×2 | LayerNorm(非 RMS) |
| `v.blk.N.ln2.weight/bias` | [1152] | F32 | 4608×2 | |
| `v.blk.N.attn_qkv.weight` | [1152, 3456] | BF16 | 7,962,624 | 融合 QKV (=16头×72 ×3) |
| `v.blk.N.attn_qkv.bias` | [3456] | F32 | 13,824 | |
| `v.blk.N.attn_out.weight` | [1152, 1152] | BF16 | 2,654,208 | |
| `v.blk.N.attn_out.bias` | [1152] | F32 | 4,608 | |
| `v.blk.N.ffn_up.weight` | [1152, 4304] | BF16 | 9,916,416 | 无 gate ⇒ 纯 MLP |
| `v.blk.N.ffn_up.bias` | [4304] | F32 | 17,216 | |
| `v.blk.N.ffn_down.weight` | [4304, 1152] | BF16 | 9,916,416 | |
| `v.blk.N.ffn_down.bias` | [1152] | F32 | 4,608 | |
| `v.post_ln.weight/bias` | [1152] | F32 | 4608×2 | ViT 输出之后 |
| `mm.0.weight` | [4608, 4608] | BF16 | — | merger 第一层 (1152×4 ⇒ 4608) |
| `mm.0.bias` | [4608] | F32 | — | |
| `mm.2.weight` | [4608, 4096] | BF16 | — | merger 第二层 → 文本 hidden 4096 |
| `mm.2.bias` | [4096] | F32 | — | |

27 层 × 15 张量 = 405，加上述 8 张 + post_ln 2 = 415 ≠ 334？实际 334 是因为
`attn_qkv.weight` 等已在层内计数：334 = 27×12 + 8(mm/patch/pos/post) + 2 = 334 ✓
（每层 12 张：ln1w/ln1b/ln2w/ln2b/qkv_w/qkv_b/o_w/o_b/up_w/up_b/dn_w/dn_b）。

## 3. 图像预处理（照抄 llama.cpp，未自拟）

- **目标尺寸**（mtmd `calc_size_preserved_ratio`，align=32）：
  `w̄ = max(32, round32(W))`，`h̄ = max(32, round32(H))`；
  若 `h̄w̄ > 4,194,304` ⇒ `beta=sqrt(HW/max_pixels)`，`h̄=floor32(H/beta)`, `w̄=floor32(W/beta)`；
  若 `h̄w̄ < 8,192` ⇒ `beta=sqrt(min_pixels/HW)` 向上取到 32 的倍数。
- **缩放**：Pillow 兼容双三次（bicubic, antialias）；`PAD_CEIL`——先等比缩放
  `scale=min(w̄/W, h̄/H)`、`new=ceil(src*scale)`（≤目标），黑底居中贴到目标尺寸。
  若目标尺寸==原尺寸 ⇒ **直接拷贝，不做缩放**（768×768 输入即此路径）。
- **归一化**：`v = (px/255 - 0.5)/0.5`（mean=std=0.5）；帧内为 **交错 RGB**，
  入图时转成 **平面(channel-planar)** 布局 `[x, y, c]`，`flat = x + nx*y + nx*ny*c`
  （clip.cpp:4535 注释与代码实测一致）。
- **通道**：RGB（PIL/stb 均按 RGB 读 PNG）。

## 4. 视觉塔前向（逐层数学/形状；源码 = tools/mtmd/models/qwen3vl.cpp + qwen2vl.cpp + ggml-cpu/ops.cpp）

```
① patch embed   : inp_raw[3,48,48] --conv(W0)+conv(W1)--> [1152, 48*48]  (W_eff = W0+W1)
                  （clip.cpp 把 conv3d 的时间核 2 帧拆成两个 conv2d 并对同一静止图求和）
② 空间合并重排   : permute/reshape 链 (n_embd*2, nx/2, ny) → (n_embd, nx*ny)
   ★ 实测确认的上游事实: 网格 (x,y) 与 token 的关系 + sep
     - token 枚举 (yb, xb, dy, dx), t=((yb*NB+xb)*2+dy)*2+dx , (x,y)=(2xb+dx, 2yb+dy)
     - RoPE 位置就是 (row=y+dy, col=x+dx) —— 与源码 positions[] 填法一致 (clip.cpp:4790)
③ + patch_bias[1152]
④ + 位置编码 (同一排布)
⑤ ×27 层:
   h = LayerNorm(x, ln1)                       (n=1152, eps=1e-6, 有 weight+bias)
   qkv = h·Wqkvᵀ + b                           [T,3456] → Q|K|V 各 [T,16,72] (头内连续)
   M-RoPE: 每头 36 对 (i, i+36) 做 rotate_half
        θ_i = pos_i · 10000^(-2i/72)
        i<18  → pos = 行(y+dy)   (sections[0]=18)
        i≥18  → pos = 列(x+dx)   (sections[1]=18，index 用 i-18)
        ↑ 由 ggml rope VISION 模式实测推导: n_dims=36, indep_sects=true, freq_base=1e4, ext_factor=0
   S = Q·Kᵀ/√72 → softmax(全双向, 无 mask) → O = S·V            [T,16,72]
   x = x + (O·Woᵀ + bo)                                          (残差1)
   h = LayerNorm(x, ln2); f = GELU(h·Wupᵀ + bup); x = x + (f·Wdnᵀ + bdn)   (残差2)
⑥ post_ln(LayerNorm) → reshape [n_embd*4, T/4] (2×2 合并, 组内顺序 (dy,dx))
⑦ merger: mm.0(4608→4608) → GELU → mm.2(4608→4096) ⇒ 图像 embedding [T/4, 4096]
```
GELU = `0.5x(1+tanh(√(2/π)·x·(1+0.044715x²)))`（ggml `ggml_gelu_f32`）。

**★ 排布验证状态**：token 顺序/RoPE 位置由源码强约束（②/⑤）；「conv 输出 → token 排布」
那一步的 ggml `cont_4d` 语义我是按源码推导 + 实测两种候选对拍（见 §7 日志），
**只认能与 llama.cpp embedding 对上 relrms 的那个**。

## 5. 卡上实现方案

- 大矩阵乘全部走**已验证的官方 `api::gemm_int8`**（同文本路径：权重按每输出行 int8 + 每行 scale 折回）
  - patch embed: im2col(1024×768) × W_eff(1152×768)
  - 每层: qkv(1152→3456) / attn_out(1152→1152) / ffn_up(1152→4304) / ffn_down(4304→1152)
  - attention: S=Q·Kᵀ (每头 m=n=1024,k=72)、O=S·V (每头 m=1024,n=72,k=1024) —— 也走 gemm_int8
  - merger: 4608→4608 / 4608→4096
- 激活按 `gemv_batch` 已验证的「行归一化 + 折回」补偿（官方内部 per-tensor 量化激活）
- LayerNorm / GELU / softmax / RoPE / 残差 → 主机侧（同文本引擎既有做法：卡算子 + 主机逐元素）
  M=1024、K=1152 的 LN 用官方 `layer_norm` 会超硬限（n ≤ 1024）✗ ⇒ 只能主机侧
- 权重体积（int8）：约 456M 参数 ⇒ ~0.5 GB；卡上空闲 ~8 GB ✓
- 预估耗时：
  - 卡侧 gemm：每层 4 个大 gemm + 32 个 attention gemm ≈ 权重读取 27×(3456×1152 + 1152² + 4304×1152×2) B ≈ 0.45 GB / (2×110 GB/s) ⇒ **~2 ms**；attention FLOPs 27×2×16×1024²×72×2 ≈ 174 GFLOP ⇒ 若 int8 有效 ~30 TOPS ⇒ ~6 ms
  - PCIe：激活往返 ≈ 27 层 × ~150 MB ≈ 4 GB ⇒ 3.5 GB/s ⇒ **~1.1 s**（瓶颈）
  - 主机：LN 32M 元素 + GELU 119M 元素 + softmax 453M 元素(expf) ⇒ **~0.5~1 s**
  ⇒ 单图预计 **1.5~3 s**（未优化前）

## 6. 接协议

- 引擎（orn3）新增两条命令（不改动既有行格式）：
  - `!VIS <in.f32> <W> <H>\n` → 引擎跑视觉塔，图像 embedding 留在内存，回 `__VIS__ <n_tok>`
  - 生成行前缀加 `img@` 标记：`@<maxtok>@[sid=..@]img@b64:<prompt>`；prefill 遇到
    token 248056(`<|image_pad|>`) 时按顺序取图像 embedding 行替换
- `serve2.py`：`content` 数组里出现 `{"type":"image_url",...}` 时
  1) base64 解码落盘 → 2) 调 `mmprep`（链接 libmtmd，用 llama.cpp 自带 preprocessor，
     保证预处理逐位一致）产出平面 f32 → 3) `!VIS` → 4) 拼
     `<|vision_start|>` + `<|image_pad|>`×n + `<|vision_end|>` + 用户文本 → 5) 正常生成
- 纯文本请求路径 **一个字节都不动**。

## 7. 日志/实测记录

- 128 张量/字节核对、元数据全键：见 §1/§2（python gguf + 手写 KV 解析双读一致）
- llama.cpp 真值：`llama-mtmd-cli` + `MTMD_DEBUG_EMBEDDINGS=<file>` 落盘 `[int32 n_tok][int32 n_embd][f32]`
  - 512×512 ⇒ 256 token（实测 4194312 B = 8 + 256×4096×4 ✓）
  - 768×768 ⇒ 576 token（9437192 B = 8 + 576×4096×4 ✓）
- 中途为看中间层，给 clip.cpp 打了临时 eval-callback 探针（`patch_clip2.py`，可 `--revert`）；
  **结论：图算完后回读中间节点不可靠，必须在 eval 回调里落盘** —— 旧做法拿到的 patch_bias
  数据经查是垃圾（列重复），故中间层真值以「整条前向对拍最终 embedding」为准。
