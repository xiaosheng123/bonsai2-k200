# Bonsai 2 27B (TQ1_0 三元) 移植到昆仑 K200 — 工作存档

> **状态：硬件已出售，项目暂停归档。** 本仓库是完整的工作快照：
> 全部源码、解析脚本、验证结论、9B 引擎代码与适配路线。

## 目录

- [项目背景](#项目背景)
- [模型规格（已实测确认）](#模型规格已实测确认)
- [已完成的移植工作](#已完成的移植工作)
- [9B 引擎代码（Ornith-1.5-9B, qwen35 架构）](#9b-引擎代码)
- [Bonsai 2 适配 9B 引擎的方法（未实现，路线图）](#bonsai-2-适配-9b-引擎的方法)
- [目录结构](#目录结构)
- [关键结论速查](#关键结论速查)

---

## 项目背景

目标：把 **PrismML 的 Bonsai 2 27B**（Qwen3.8-27B 的**三元压缩**版本，1.75 bit/weight，5.95GB）
跑到昆仑 K200 双芯（2×8GB HBM）上。

Bonsai 2 与已有 9B 引擎（orn3.cb，跑 Ornith-1.5-9B）**架构完全相同**（都是 qwen35：
混合 SSM + 稀疏 attention），所以移植 = 新增三元 GEMM + FWHT（Hadamard）激活变换，
而不是新写整个推理引擎。

**为什么可行**：27B 模型三元压缩后只有 5.95GB，恰好能装进 K200 双芯 16GB HBM，
而 int8 解包会膨胀到 27GB（放不下）。三元 GEMM 只需加减法（权重 -1/0/+1），
理论带宽与 9B Q8 相同量级（约 20+ tok/s 级别）。

**为什么被放弃**：K200 已售出。之前实测 K200 自写 kernel 带宽只有官方 gemm_int8 的
1/50（2.58 vs 129.86 GB/s），而三元权重必须保持压缩态（无法解包成 int8 喂官方算子），
需要自写卡上解包 GEMM——性能风险大，加上硬件已出售，项目停止。

---

## 模型规格（已实测确认）

用 `scripts/parse_gguf3.py` 直接解析 `Ternary-Bonsai-2-27B-PTQ1_0.gguf`（5,946,648,928 字节）：

### 架构（qwen35，与 Ornith-1.5-9B 完全一致）

| 参数 | 值 |
|---|---|
| 层数 | 64 |
| embedding | 5120 |
| attention heads | 24 |
| KV heads | 4 |
| attention key_length | 256 |
| FFN 宽度 | 17408 |
| SSM state_size | 128 |
| SSM group_count | 16 |
| SSM inner_size | 6144 |
| SSM conv_kernel | 4 |
| SSM time_step_rank | 48 |
| full_attention_interval | 4（每 4 层一次真 attention，其余 SSM）|
| context | 262144 |

### 张量类型分布（851 个张量）

| GGUF type | 数量 | 含义 |
|---|---|---|
| 0 (F32) | 353 | 小权重/norm/bias |
| 30 (BF16) | 96 | recurrent 状态路径 |
| 143 (TQ1_0) | 402 | **全部大权重，三元格式** |

### prism.hadamard 元数据（旋转 + 符号翻转）

```
prism.hadamard.version    = 1
prism.hadamard.block_size = 1024
prism.hadamard.transform  = normalized-sylvester-walsh-hadamard
prism.hadamard.axis       = input-last-dimension
prism.hadamard.sign_mode  = explicit
prism.hadamard.sign_widths= [5120, 6144, 17408]   # embedding / ssm / ffn 输入维
prism.hadamard.sign_values= 28672 个 ±1           # = 5120+6144+17408
prism.hadamard.gdn_v_grouped = 1
prism.hadamard.weight_names  = 401 个              # 所有大权重（attn_qkv/gate/ssm_out/ffn_*/output）
prism.hadamard.inverse_weight_names = [token_embd.weight]
```

**含义**：
- 所有权重（401 个）**以旋转基存储**（量化前做 block=1024 的 Sylvester-Hadamard 旋转）
- `axis=input-last-dimension`：旋转作用于权重输入维（每 1024 一段）
- 推理时**激活侧**必须做对应的 FWHT 变换（`gemm → ×signs → FWHT`），否则输出乱码
- `sign_values` 是逐元素 ±1 符号表，按 [5120, 6144, 17408] 三档分给不同维度的权重
- `token_embd.weight` 是特例：lookup 结果需要**逆变换**
- 官方文档明确："Q2_0 在主线上加载会无警告地输出乱码"——**旋转处理必不可少**

### TQ1_0 三元格式（已完全解包验证）

```
block_tq1_0 = qs[48] + qh[4] + ggml_half d   = 54 字节 / 256 权重 = 1.6875 bpw
  qs[48]:  5 trits/byte × 48 = 240 个三元
  qh[4]:   4 trits/byte × 4  = 16 个三元
  d:       FP16 scale（每 256 权重一个）
```

解包公式（**必须 &0xFF 截断 uint8**，否则解出非三元值）：
```
q    = (byte * pow3[n]) & 0xFF     # pow3 = {1,3,9,27,81}
xi   = (q * 3) >> 8
val  = (xi - 1) * d
```

验证结果：`scripts/verify_tq1b.py` 全量解包输出纯 {-1, 0, +1}（127 个 -1 / 63 个 +1 / 66 个 0），
与官方 `dequantize_row_tq1_0`（release `ggml-quants.c`）逐位一致。

---

## 已完成的移植工作

### 1. 源码获取（PrismML llama.cpp fork）

- **release 基线**：`prism-b10709-9a9394a`（官方 BACKEND-SUPPORT.md 审计基线）
- 完整 fork 已 clone：K200 机器 `/home/caden/bonsai2/llmfork`（206MB）
- 关键文件已单独归档到 `bonsai2-src/`：
  - `rel_src/`：release 版 `llama-model.cpp`（164KB）、`llama-graph.cpp`（144KB）、
    `ggml-cpu.c`（131KB）、`ops.cpp`（414KB）、**`rel_fwht.cu`（11.5KB，含 1024 支持）**
  - `fork-src/`：master 版源码 + 官方 Bonsai demo 文档（MODEL-FORMATS / BACKEND-SUPPORT / FAQ / KV-CACHE）

### 2. 官方实现完全解析（权威算法基准）

**CUDA FWHT（`rel_src/rel_fwht.cu`）**——支持 N ∈ {64,128,256,512,1024,2048,4096,8192}，
N=512+ 用 `fwht_cuda_block`（每行一个 256 线程 block，shared memory 蝴蝶）：

```
fwht(N=1024, 每行):
  1. 加载: reg[i] = src[i] * (1/√1024) ；若有 sign: 再 *= signs_row[i]
     (signs_row = signs + (r % n_blk) * N, 按行取 sign)
  2. warp 内蝴蝶 (h=1..warp_size):  partner = shfl_xor(reg[j], h)
     reg[j] = (lane&h)==0 ? val+val2 : val2-val
  3. 跨 warp 蝴蝶 (h=warp_size..NT):  partner 从 shared memory (tid^h)
     reg[j] = (tid&h)==0 ? val+val2 : val2-val
  4. 块上蝴蝶 (h=NT..N):  partner = reg[j + step]（同线程寄存器）
     reg[j] = x+y ; reg[j+step] = x-y
  5. 存回 dst
```

**图变换（`rel_src/llama-graph.cpp` build_lora_mm）**——旋转权重的 gemm 路径：

```
cur_mm = ggml_mul_mat(ctx, w, cur)          # 1. 普通 GEMM（权重已是旋转基）
if (t.signs)  cur_mm = ggml_mul(cur_mm, t.signs)   # 2. 逐元素 ×signs
cur_mm = llama_mul_mat_hadamard(cur_mm, t.rot)     # 3. FWHT（乘 Hadamard，hint 触发）
```

`llama_mul_mat_hadamard` = reshape 成 [n=block_size, ...] 段，每段 `mul_mat(rot[n×n], x)`，
打 `GGML_HINT_SRC0_IS_HADAMARD` hint → CUDA/CPU 端用 FWHT 替代矩阵乘。
**Hadamard 矩阵** = `ggml_gen_hadamard`：`data[0,0]=1/√n`，递归块 `[A A; A -A]`（Sylvester）。

**权重加载（`rel_src/llama-model.cpp` 1196-1330 行）**：
- 读取全部 `prism.hadamard.*` 元数据，校验（block_size 2 的幂、transform/axis/sign_mode 枚举、
  sign_values 必须 ±1、weight_names 非空）
- 每个旋转权重建 `llama_hadamard_transform {rot, signs}`，block_size 相同的共享一个 rot 张量
- `ssm_out.weight` 且 `gdn_v_grouped=1` 时额外带 `perm_hd/perm_nk/perm_rep`（GDN head 重排）
- 拒绝未验证路径的权重（防"静默跑错数学"）

### 3. 关键实验结论（K200 实测）

（来自此前 SSM 上卡测试，`k200-inference-gateway` skill 存档）
- 官方 `gemm_int8` 实测 **129.86 GB/s**（[4096,8192] m=1 → 0.258ms）
- 自写标量 kernel 仅 **2.58 GB/s**（慢 50 倍）→ **K200 自写 kernel 必须向量化/用官方算子**
- GM2LM 单次搬运上限 4KB；GM2LM_ASYNC 数据不保证到位（不可用）
- 测试 kernel 打挂设备（State=E）会连带杀生产引擎；测试后必须查 `/proc/xpu/dev0/state`

---

## 9B 引擎代码

`9b-engine/` 包含 K200 上跑 Ornith-1.5-9B（qwen35）的完整生产引擎代码：

| 文件 | 说明 |
|---|---|
| `orn_engine/orn3.cb.cpp` | **主引擎**（3235 行）：GGUF 加载、权重量化上卡、官方 gemm_int8 调用、SSM/attention 逐层推理、双芯并发 |
| `orn_engine/vis.cpp` + `vis.h` | 逐层可视化/验证 |
| `orn_engine/buildorn3cb.sh` | 引擎编译脚本 |
| `orn_engine/*.xpu` | 自写 XPU kernel（delta_net 等，SSM 上卡实验） |
| `ornc/serve2.py` | **生产推理服务**（当前线上版本，73KB） |
| `ornc/start.sh` / `restart.sh` / `status.sh` / `stop.sh` | 服务管理 |
| `ornc/DESIGN.md` / `PROGRESS.md` / `TODO.md` | 开发文档 |
| `ornc/ATN_ON_CARD.md` | attention 上卡设计 |
| `orn_backup_20260928_124808/` | 2026-09-28 完整备份（引擎+服务） |

### 9B 引擎的关键设计

- **权重路径**：GGUF Q8_0 → 主机 dequant → **每输出行 int8 + 每行一个 float scale**
  （`q8raw_to_rowi8`，官方 gemm_int8 只支持 per-tensor/row scale）→ 上卡 → 双芯按行分裂并发
- **GEMM**：`api::gemm_int8(ctx, false, true, m, n, k, 1.f, a, lda, b, 127.f, ldb, 0.f, c, ldc)`
  （SD-CDNN 官方算子；`gemm_int8_maxptr` 版本已实测会把 dev0 打进 ERROR，禁用）
- **数值**：官方 gemm_int8 内部 per-tensor 量化补偿（mbatch2.cpp 实测钉死）
- **性能**：9B Q8 ≈ 38.6ms/token 理论极限，实测 44.45ms（87% 带宽墙）

---

## Bonsai 2 适配 9B 引擎的方法

> 未实现（硬件已售）。以下是从源码解析得到的完整路线图。

### 核心差异（vs 9B Q8）

| 维度 | 9B (Q8_0) | Bonsai 2 27B (TQ1_0) |
|---|---|---|
| 权重格式 | Q8_0，每行一个 scale | TQ1_0，**每 256 权重一个 FP16 scale** |
| 权重字节 | 9GB | **5.95GB**（压缩态，放得下双芯 16GB） |
| GEMM | 官方 gemm_int8（int8 权重） | 三元（-1/0/+1）**必须卡上解包** |
| 激活变换 | 无 | **每 gemm 前/后 FWHT(1024) + ×signs** |
| 特殊权重 | 无 | token_embd 需要逆变换 |
| SSM 输出 | 直接 | ssm_out 需要 GDN head 重排（perm_hd/nk/rep） |

### 路线 A：卡上解包三元 GEMM + FWHT kernel（自写，性能风险）

1. **TQ1_0 保持压缩态上卡**（5.95GB），gemm 时 kernel 内解包：
   ```
   q=(byte*pow3[n])&0xFF; xi=(q*3)>>8; val=(xi-1)  # 三元值 -1/0/+1
   acc += val * x[k]                                # 只需加减
   # 每 256 个权重乘一次 block scale d
   ```
2. **FWHT kernel**（参考 `rel_fwht.cu` fwht_cuda_block，移植到 XPU）：
   - K200 无 warp shuffle → 用 shared memory 蝴蝶（K200 每核私有 L2/本地存储）
   - 每行一个 block，N=1024 = 10 级蝴蝶，256 线程
3. **激活变换插入点**（照 build_lora_mm）：
   - 每个旋转权重 gemm 后：`×signs → FWHT(1024 分段)`
   - token_embd lookup 后：**逆变换**（同样 FWHT，因为 Hadamard 自逆）
   - ssm_out gemm 后：GDN head 重排
4. **signs 表**：从 GGUF `sign_values` 按 width [5120, 6144, 17408] 载入，
   按 `(行 % n_blk) * 1024` 取段（n_blk = width/1024）

**性能预判**：权重每 token 读 5.95GB，双芯 127GB/s → 理论 ~47ms/token（21 tok/s）。
但自写 kernel 带宽能否达标存疑（历史自写仅 2.58GB/s）→ **先做带宽验证再全量实现**。

### 路线 B：解包成 int8 喂官方 gemm_int8（简单，但内存不够）

- TQ1_0 → 每行 int8（三元值 -1/0/+1）→ 27.36GB > 16GB ❌ **不可行**
- 除非：三值打包 2bit（每字节 4 三元）= 6.75GB ✓，但官方 gemm_int8 不认 2bit ❌

### 路线 C：host 端解包 + 分块流式（PCIe 瓶颈）

- 权重流式从 host 读 → PCIe Gen3 x8 带宽远低于 HBM，decode 必慢于 47ms ❌

**推荐路线 A**，且第一步必须是 K200 自写 kernel 的带宽可行性验证。

### 验证方法（无卡时的替代）

- CPU 版：编译 release fork（`llmfork` 或 `bonsai2-src/rel_src`），
  用 `llama-cli` 加载 GGUF 跑 few tokens 验证输出 sane → 确认数学链
- K200：`scripts/verify_tq1b.py` 已全量验证 TQ1_0 解包正确；FWHT 可用 Python
  对照 `rel_fwht.cu` 算法逐位验证

---

## 目录结构

```
bonsai-k200/
├── README.md                     # 本文件
├── bonsai2-src/
│   ├── rel_src/                  # release 分支关键源码（fwht/graph/model/ops）
│   └── fork-src/                 # master 源码 + 官方 Bonsai demo 文档
├── 9b-engine/
│   ├── orn_engine/               # 9B 引擎源码（orn3.cb.cpp + kernel + build 脚本）
│   ├── ornc/                     # 生产服务（serve2.py + 管理脚本 + 文档）
│   └── orn_backup_20260928_124808/  # 完整备份
└── scripts/                      # 27 个解析/验证/抓取脚本
    ├── parse_gguf3.py            # GGUF 全量解析（KV + tensor 统计）
    ├── verify_tq1b.py            # TQ1_0 解包验证（全量 {-1,0,+1}）
    ├── dump_sign.py              # prism.hadamard 签名表 dump
    ├── wnames.py                 # weight_names 分类统计
    └── fetch_*.py                # 源码抓取（raw/CDN/github API）
```

## 关键结论速查

1. **TQ1_0 解包公式**（&0xFF 截断）：`val = (((byte*pow3[n])&0xFF)*3>>8 - 1) * d`
2. **FWHT 权威实现**：`rel_src/rel_fwht.cu`（支持 1024，shared-memory 蝴蝶，含 sign）
3. **变换顺序**：`gemm → ×signs → FWHT`（build_lora_mm），token_embd 用逆变换
4. **Hadamard 矩阵**：Sylvester 递归 `[A A; A -A]`，scale=1/√1024
5. **sign 取法**：`signs + (行 % n_blk) * 1024`，n_blk = width/1024
6. **瓶颈**：5.95GB/token ÷ 127GB/s ≈ 47ms/token（21 tok/s）理论上限
7. **最大风险**：K200 自写 kernel 带宽（历史 2.58 vs 官方 129.86 GB/s）
8. **模型文件**：`Ternary-Bonsai-2-27B-PTQ1_0.gguf`（5.95GB，不在本仓库，
   原位于 K200 `/home/caden/bonsai2/`，可从 modelscope/huggingface `prism-ml/Ternary-Bonsai-2-27B-gguf` 获取）
