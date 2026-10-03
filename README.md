# Bonsai 2 27B (TQ1_0) → 昆仑 K200

> **状态：硬件已出售，移植暂停。源码归档于此。**

---

## ✅ 已完成：Ornith-1.5-9B 生产引擎

**K200 生产运行中**，Ornith-1.5-9B（qwen35 架构，SSM+稀疏attention），实测 **15.4 tok/s**（4×100token 并发）。

### 硬件环境

| 项目 | 说明 |
|---|---|
| **GPU** | 昆仑 K200 双芯（2×8GB HBM，PCIe Gen3 x8） |
| **PCIe 实测** | Gen3 x8 理论 8GT/s ≈ **7.88 GB/s 双向** |
| **PCIe 4.0 预期** | Gen4 x8 理论 16GT/s ≈ **15.76 GB/s 双向**（当前硬件 2× 带宽）|
| **编译工具链** | XTDK clang++，XPU SDK 4.33.0 |
| **推理服务** | `ornc/serve2.py`（Python + XPU API）|

> ⚠️ **当前机器跑在 PCIe 3.0**。升级到 PCIe 4.0 后 H2D/D2H 带宽翻倍，decode 性能预计同步提升。

### 核心文件

| 文件 | 说明 |
|---|---|
| `9b-engine/orn_engine/orn3.cb.cpp` | 引擎主体（3235行）：GGUF加载→Q8_0量化→逐层forward→双芯并发 |
| `9b-engine/orn_engine/vis.cpp` + `vis.h` | attention 可视化验证 |
| `9b-engine/orn_engine/buildorn3cb.sh` | 编译脚本 |
| `9b-engine/ornc/serve2.py` | **生产服务**（线上运行） |
| `9b-engine/ornc/start.sh` / `restart.sh` / `stop.sh` | 服务管理 |
| `9b-engine/ornc/DESIGN.md` / `PROGRESS.md` | 开发文档 |

**编译**：K200 上 `cd 9b-engine/orn_engine && bash buildorn3cb.sh`

### K200 实测性能数据

| 测试项 | 结果 |
|---|---|
| 官方 `gemm_int8` 带宽 | **129.86 GB/s**（m=1, [4096,8192]，0.258ms） |
| 自写 SSM kernel（delta_net） | **3.38ms/层 × 24层 ≈ 81ms** |
| CPU SSM 递推（baseline）| ~48ms（K200 自写 kernel 比 CPU 慢） |
| decode 极限 | **44.45ms ≈ 22.5 tok/s**（87% 带宽墙） |
| 并发 4×100token | **26s ≈ 15.4 tok/s** |
| GM2LM 单次上限 | **4KB**（16KB 打挂设备 State=E） |
| GM2LM_ASYNC | **不可用**（异步数据不保证到位） |

### 驱动下载

K200 驱动和 SDK（CentOS 发行版源码，Ubuntu 20.04 编译，XTDK/XTCL 工具链）：

**夸克网盘**：`/~d0fe3bEVWG~/`  
**链接**：https://pan.quark.cn/s/3a97917a8c23

**GitHub LFS 备份**（与仓库同一份，已 push）：
`drivers/k200-drivers.tar.gz`（376MB）  
`drivers/k200-runtime.tar.gz`（3MB）  
`drivers/k200-sdk-clean.tar.gz`（219MB）

解压后目录结构：
```
xpu_sdk_v2.0.0.61/
├── XTCL/               # 编译工具链（clang++、xpu2-crt.xpu 等）
├── XTDK/               # 驱动 headers
├── bin/                # elfconv / crt
├── include/xpu/       # runtime API
└── lib64/              # libxpurt.so / libxpuml.so
```

编译命令参考：
```bash
X=/path/to/xpu_sdk_v2.0.0.61/XTCL
RT=/usr/local/xpu-4.33.0
$X/bin/clang++ -target xpu-none-none -xpu-device-only ...
```

---

## 🔄 进行中：Bonsai 2 27B 三元模型

**目标**：把 PrismML Bonsai 2 27B（TQ1_0 三元，1.75bpw，**5.95GB**）跑在 K200 双芯。

### 模型规格

| 参数 | 值 |
|---|---|
| 架构 | qwen35（与 Ornith-9B **完全一致**）|
| 层数 | 64（每4层一次full attention，其余SSM）|
| embedding | 5120，FFN | 17408，SSM inner | 6144 |
| 权重格式 | **TQ1_0**（-1/0/+1，每256权重1个FP16 scale）|
| 压缩后大小 | **5.95GB**（解压成int8=27GB，放不下双芯16GB）|
| 旋转 | prism.hadamard（block=1024，Sylvester-Hadamard + sign flip）|

TQ1_0 解包公式（Python 全量验证 ✅）：
```
q    = (byte * pow3[n]) & 0xFF
xi   = (q * 3) >> 8
val  = (xi - 1) * d        # d = FP16 scale
```
验证结果：全量输出 {-1: 127, 0: 66, +1: 63}（纯三元）。

### 移植路线图

| 步骤 | 内容 | 状态 |
|---|---|---|
| 1 | TQ1_0 解包公式验证 | ✅ 完成 |
| 2 | GGUF元数据（hadamard/sign/weight_names）解析 | ✅ 完成 |
| 3 | FWHT CUDA kernel（rel_fwht.cu，N=1024）| ✅ 源码已得 |
| 4 | 图变换逻辑（build_lora_mm）| ✅ 源码已得 |
| 5 | K200自写三元GEMM kernel（卡上解包）| ❌ 未实现 |
| 6 | 接入orn3.cb引擎 | ❌ 未实现 |
| 7 | 端到端验证 | ❌ 未实现 |

**核心瓶颈**：TQ1_0 无法解包成int8（27GB），必须**卡上解包自写三元GEMM**。
历史自写kernel仅2.58GB/s vs 官方gemm_int8 129.86GB/s，**带宽差距50×**，需先验证K200自写kernel可行性。

### 移植核心算法

**FWHT（对gemm输出做，每1024元素一段）：**
```
y = gemm(W_rot, x)           # 旋转域权重
y = y * signs                 # 逐元素±1（sign_widths=[5120,6144,17408]）
y = FWHT_1024(y)             # Sylvester-Hadamard，scale=1/√1024
```
token_embd 需逆变换（Hadamard 自逆，等价操作）。

### 性能预估

| 条件 | 计算 |
|---|---|
| 理论带宽上限（PCIe 3.0） | 5.95GB ÷ 127GB/s ≈ **47ms/token ≈ 21 tok/s** |
| 理论带宽上限（PCIe 4.0） | 5.95GB ÷ 127GB/s ≈ **47ms/token**（HBM墙不变，带宽翻倍的是 PCIe，非 HBM） |
| 实际风险 | 自写三元kernel历史 2.58GB/s（需接近127GB/s才有用）|

> PCIe 4.0 提升的是主板↔GPU 搬运带宽，对 decode HBM 计算瓶颈无影响（除非激活传输成瓶颈）。

---

## 目录结构

```
bonsai2-k200/
├── README.md
│
├── 9b-engine/                          # ✅ 生产引擎
│   ├── orn_engine/                     # 引擎核心（orn3.cb.cpp）
│   ├── ornc/                           # serve2.py 生产服务
│   └── orn_backup_20260928_124808/     # 完整备份
│
├── bonsai2-src/                         # 🔄 Bonsai 2 移植源码
│   ├── rel_src/                        # release 基线（prism-b10709-9a9394a）
│   │   ├── rel_fwht.cu               # CUDA FWHT（含1024块）
│   │   ├── llama-graph.cpp            # build_lora_mm（图变换）
│   │   ├── llama-model.cpp            # hadamard 加载/校验
│   │   ├── ggml-cpu.c / ops.cpp       # CPU 参考实现
│   └── fork-src/                      # Bonsai 官方 demo 文档
│       ├── BACKEND-SUPPORT.md          # 各后端支持表
│       ├── MODEL-FORMATS.md            # 格式说明
│       └── FAQ.md
│
└── scripts/                            # 解析/验证脚本
    ├── parse_gguf3.py                  # GGUF 全量解析
    ├── verify_tq1b.py                  # TQ1_0 解包验证
    ├── dump_sign.py                     # hadamard sign 表 dump
    ├── wnames.py                       # weight_names 分类
    └── fetch_*.py                      # 源码抓取脚本
```

## 关键结论速查

1. **TQ1_0 公式**：`val = (((byte*pow3[n])&0xFF)*3>>8 - 1) * d`
2. **FWHT**：Sylvester 递归 Hadamard，scale=1/√1024，signs 取法 `signs + (行%块数)*1024`
3. **变换顺序**：`gemm → ×signs → FWHT`
4. **权重5.95GB**（双芯16GB放得下），解包int8=27GB（放不下）
5. **K200实测**：官方gemm_int8=129.86GB/s，自写kernel=2.58GB/s（差距50×）
6. **PCIe 3.0**：当前实测；PCIe 4.0 翻倍主板↔GPU 带宽（对HBM计算墙无直接影响）
