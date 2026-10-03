# Bonsai 2 27B (TQ1_0) 移植到昆仑 K200

> **状态：硬件已出售，移植暂停。项目源码归档于此。**

---

## 现状

### ✅ 已完成：Ornith-1.5-9B 生产引擎（qwen35 架构）

`9b-engine/` 目录是**已在 K200 生产运行的推理引擎**，跑 Ornith-1.5-9B（Qwen2.5-9B 变体，qwen35 架构，SSM + 稀疏 attention），实测 15.4 tok/s（4×100token 并发）。

| 文件 | 说明 |
|---|---|
| `orn_engine/orn3.cb.cpp` | 核心引擎（3235行）：GGUF加载、Q8_0量化上卡、逐层forward、双芯并发 |
| `orn_engine/vis.cpp` + `vis.h` | attention 可视化验证 |
| `orn_engine/buildorn3cb.sh` | 编译脚本 |
| `ornc/serve2.py` | **生产服务**（当前线上版本） |
| `ornc/start.sh` / `restart.sh` / `stop.sh` / `status.sh` | 服务管理脚本 |
| `ornc/DESIGN.md` / `PROGRESS.md` / `TODO.md` | 开发文档 |
| `orn_backup_20260928_124808/` | 完整备份 |

**编译方法**：在 K200 上 `cd orn_engine && bash buildorn3cb.sh`

---

## 🔄 进行中：Bonsai 2 27B 三元模型（未完成）

**目标**：把 PrismML Bonsai 2 27B（TQ1_0 三元压缩版，5.95GB，1.75bpw）跑到 K200 双芯。

### 模型规格（已实测确认）

| 参数 | 值 |
|---|---|
| 架构 | qwen35（与 Ornith-9B 完全一致）|
| 层数 | 64（每4层一次full attention，其余SSM）|
| embedding | 5120，FFN | 17408 |
| SSM state | 128，inner | 6144 |
| 权重格式 | TQ1_0（三元，-1/0/+1，每256权重1个FP16 scale）|
| 压缩后大小 | **5.95GB**（vs int8解包27GB，放得下双芯16GB）|
| 旋转 | prism.hadamard（block=1024，Sylvester-Hadamard + sign flip）|

TQ1_0 解包公式（已验证）：
```
q    = (byte * pow3[n]) & 0xFF
xi   = (q * 3) >> 8
val  = (xi - 1) * d        # d = FP16 scale
```

### 移植路线图

**核心问题**：TQ1_0 三元权重无法解包成 int8 喂官方 `gemm_int8`（解包后27GB放不下），
必须**卡上解包**自写三元 GEMM kernel。

| 步骤 | 内容 | 状态 |
|---|---|---|
| 1 | TQ1_0 解包公式验证（Python全量验证 ✅） | ✅ 完成 |
| 2 | GGUF元数据解析（hadamard/sign/weight_names）| ✅ 完成 |
| 3 | FWHT CUDA kernel（rel_fwht.cu，N=1024支持）| ✅ 源码已得 |
| 4 | 图变换逻辑（build_lora_mm：gemm→×signs→FWHT）| ✅ 源码已得 |
| 5 | K200自写三元GEMM kernel（卡上解包）| ❌ 未实现 |
| 6 | 接入orn3.cb引擎（替换Q8_0量化路径）| ❌ 未实现 |
| 7 | 端到端验证（few tokens sanity check）| ❌ 未实现 |

### 关键源码位置

```
bonsai2-src/
├── rel_src/                    # release基线 prism-b10709-9a9394a
│   ├── rel_fwht.cu            # CUDA FWHT（含1024，支持hadamard+signs）
│   ├── llama-graph.cpp        # build_lora_mm（旋转权重gemm路径）
│   ├── llama-model.cpp        # hadamard加载/校验（1196-1330行）
│   ├── ggml-cpu.c            # CPU参考实现
│   └── ops.cpp                # CPU FWHT dispatch
└── fork-src/                  # Bonsai官方demo文档
    ├── bonsaidemo_BACKEND-SUPPORT.md   # 各后端支持表
    ├── bonsaidemo_MODEL-FORMATS.md      # 格式说明
    └── bonsaidemo_FAQ.md               # FAQ
```

### 移植核心算法

**FWHT（对gemm输出做，每1024元素一段）：**
```
y = gemm(W_rot, x)           # 旋转域权重
y = y * signs                 # 逐元素±1（sign_widths=[5120,6144,17408]）
y = FWHT_1024(y)             # Sylvester-Hadamard，scale=1/√1024
```

token_embd需要逆变换（Hadamard自逆，等价操作）。

### 性能预估

- 理论带宽上限：5.95GB ÷ 127GB/s ≈ 47ms/token ≈ 21 tok/s
- 实测风险：K200自写kernel历史2.58GB/s（vs官方gemm_int8 129.86GB/s）
- 建议：先在CPU验证数学链，再测K200自写kernel带宽可行性

---

## 目录结构

```
bonsai2-k200/
├── README.md
├── 9b-engine/                    # ✅ 生产引擎（已上线）
│   ├── orn_engine/               # 引擎核心代码
│   ├── ornc/                    # serve2.py 生产服务
│   └── orn_backup_20260928_124808/   # 完整备份
├── bonsai2-src/                  # 🔄 Bonsai2 移植源码
│   ├── rel_src/                 # release基线关键文件
│   └── fork-src/               # 官方demo文档
└── scripts/                      # 解析/验证脚本
    ├── parse_gguf3.py           # GGUF全量解析
    ├── verify_tq1b.py           # TQ1_0解包验证（全量{-1,0,+1}）
    ├── dump_sign.py             # hadamard签名表dump
    ├── wnames.py                # weight_names分类
    └── fetch_*.py               # 源码抓取脚本
```
