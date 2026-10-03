# ATN_ON_CARD —— 把 KV 与全注意力搬上 K200（可执行实施设计书）

> 读者 = **下一位实现代理**。本文所有内容按"照做即可"标准写：带 `文件:行号` + 该行唯一关键词（便于在任意版本重新定位）、算子签名、张量形状/dtype、显存偏移、launch 账、验收命令。
> **凡是本机没实测过的，一律标 `待探针验证` 并给出探测方法与判据；凡是没有符号/签名依据的，一律标 `✗ 无依据`。**
>
> 撰写者口径声明（重要，避免误会）：本文撰写期间**没有碰卡**——只做了只读侦察：`ssh` 读源码/头文件/日志、`nm -DC libxpuapi.so` 读导出符号（读磁盘文件）、`cat /proc/xpu/dev{0,1}/{state,reset_count}`（驱动只读查询，不发起任何设备内核）、`ps`/`crontab -l`/`ls`。**未执行任何设备内核、未跑 test_dma/pfbench、未停任何服务、未编译任何设备码。**

---

## 目录

| 章 | 你在这里能找到什么 | 什么时候读 |
|---|---|---|
| 第 0 章 | 一页速览：结论、阶段表、锚定的生产件 | **先读这个** |
| 第 1 章 | 现状核实：层型、要替换的那 40 行（含行号）、实测数字、KV/检查点/槽位的内存牵连 | 动手前 |
| 第 2 章 | **算子链设计**：KV 布局与写入偏移、解码 13 次调用逐条、预填充两种策略、缓冲清单、保留在主机的东西、launch/耗时总账、**必须探针验证的算子清单（P1~P15）**、预期收益算术 | 动手前 |
| 第 3 章 | **KV 容量账**：每芯 HBM 拆解、分芯分层、fp32/fp16 × MAXT 占用表、共存余量、切 fp16 的硬依赖链与逐项改动、卡上 KV 三条设计规则 | 决定 MAXT/dtype 前 |
| 第 4 章 | **数值风险与验收**：逐步数值变化量级、视觉塔混沌教训、照单可跑的验收命令、允许不等价的边界 | 每个阶段收尾 |
| 第 5 章 | **实施步骤与窗口计划**：阶段 ⓪~⑥ 的改动点/窗口/停机预算/回滚点/独立判据、回滚脚本模板 | 逐阶段执行 |
| 第 6 章 | **不确定点清单（U1~U13）**：先试这些，答案会改设计 | 开工第一件事 |
| 第 7 章 | 附录：关键锚点表、算子签名、禁用清单、新增文件清单、env 开关、完工定义 | 查表用 |

---

## 第 0 章 一页速览

### 0.1 结论（先看这个）

| 项 | 结论 |
|---|---|
| 能不能做 | **能**。卡上有 `qk_attention<float,float,float,float>` / `qk_v_attention<float,float,float>` / `gemm_int16`(纯 float A/B/C) 三个真实导出符号，足以承载 QK^T 与 PV 的卡上实现（`待探针验证`语义与精度）。 |
| 最小可行分解 | **只搬 O(T) 的部分**：KV 留在卡上 + QK^T/softmax/PV 在卡上。per-head RMSNorm(256)、MRoPE、sigmoid gate、残差 **留在主机不动**——它们是 O(1)/token 且是"黄金题逐字节"判据的位级锚点（§2.6）。 |
| 为什么这一步就够 | 现状的痛点是 **每 token 在主机内存上重扫整段 KV**（16 个 Q 头各自重读自己那 1 个 KV 头，读放大 4×）。卡上按 KV 头批 4 个 Q 头做 GEMM ⇒ KV 读量降到 **1/4**，带宽从 ~1.2 GB/s（主机）换成 40~130 GB/s（卡）⇒ 保守 **20~100×**。 |
| KV 容量 | **fp32 安全到 MAXT=32768**（1 GiB/芯）；**MAXT=131072 必须 fp16**（128k→2 GiB/芯），且 fp16 的可行性**完全取决于 `qk_attention<float16,...>` 能否用**（§3.4 有硬依赖链）。 |
| 关键分芯策略 | 权重保持现状（每矩阵行对半劈到两芯）；**KV 按层分芯**，且因为两芯余量不对称（chip0 少 ~1.1 GiB），推荐 **3/5 分层**（chip0 存 3 个注意层、chip1 存 5 个），128k+fp16 下两芯余量都 ≥1 GiB。 |
| 预填充能到多少 | 诚实算术：`34.5 ms/tok = 卡 gemm 7.42 + 主机 SSM/delta-net 14.27 + 注意力及层内其余 ~12 + 归一化/launch 0.36`（§1.3 实测）。本设计吃掉"注意力"那 ~12 ms/tok ⇒ 落到 **~20~22 ms/tok**。**"个位数 ms/tok" 在不动 delta-net 的前提下算术上不可达**（14.27 是硬底）——要个位数必须同时把 delta-net 搬卡，那是另一个专项（§2.9 有账）。不要把做不到的指标写进验收表。 |
| 视觉塔 | **本轮一个字节都不动 `vis.cpp`**。但视觉塔是混沌的（0.0002% 扰动 → 第 26 层放大 7% → 答案翻转，见 §4.2），任何"改数值"的加速都要重跑三图/表格 9-9。⇒ **先文本、后视觉**：本设计的全部阶段只改文本路径，视觉验收在**每个阶段都重跑一遍**当作回归网。 |
| 最大风险 | ① `qk_attention` 结尾两个参数是 `const float* max_a/max_b` —— 这正是打死过卡的 `*_maxptr` 家族特征，**只允许传 nullptr，否则直接放弃该算子**；② HBM 余量只剩 2.5~3.7 GiB/芯，KV 分配必须上线前先算账；③ 停机窗口 ≤85s（历史上有一次 416s 违规被用户当场 500 打到）。 |

### 0.2 阶段表（细节见第 5 章）

| 阶段 | 内容 | 窗口 | 停机预算 | 独立判据 | 回滚点 |
|---|---|---|---|---|---|
| ⓪ | 纯主机准备（对拍台/探针源码/离线脚本），**不碰卡** | 无 | 0 | `nm` 符号表落盘 + 探针编译通过 | — |
| ① | 极小尺寸探针（与引擎同进程，`K200_ATN=probe`） | 停→跑→起 | ≤60s | `safe_run` 退出码 0、异常计数不变、`atn_probe.json` 每 case relrms ≤ 阈值 | 不换件，无需回滚 |
| ② | **单层**（layer 3）attention 上卡 + 位级对拍 | 停→离线跑→起 | ≤120s | 逐层 relrms ≤ 1e-3、黄金题内容正确 | `orn3.preATN2.bak` |
| ③ | 8 层全上（解码 + 预填充逐 token 策略 A） | 单窗口上件 | ≤85s | 黄金题/6 条/多轮 LCP ≤3s/三图/台账全套 | `orn3.preATN3.bak` |
| ④ | KV 换 fp16（**前置：阶段① 探针 qk_attention<f16> 通过**） | 单窗口上件 | ≤85s | 同上 + KV 位级(主机 f2h vs 卡上 cast) | `orn3.preATN4.bak` |
| ⑤ | MAXT 抬到 16384 → 32768 → 131072（分三级，每级一套验收） | 每级 1 窗口 | 每级 ≤85s | 同上 + 长 prompt 实测 + 网关预算同步 | 每级各留 `.bak` |
| ⑥ | 预填充分块（策略 B，卡上掩码）—— 收益最大的一步 | 单窗口 | ≤85s | 同上 + ms/tok 分桶对比 | `orn3.preATN6.bak` |

### 0.3 本文锚定的生产件（**上线前必须重新核对**，因为今天有另一个代理在卡上做修复轮）

| 项 | 值（2026-09-23 18:0x CST 实测） |
|---|---|
| 生产源码 | `/home/caden/orn_engine/orn3.cb30.cpp`，2957 行，md5 `e6672810672a30079b59a2f836d1694f` |
| 生产二进制 | `/home/caden/orn_engine/orn3`，386392 B，md5 `9561eba8d2804ba76492324dec925c0c`，mtime 09-23 10:15 |
| 构建脚本 | `/home/caden/orn_engine/buildorn3cb30.sh`（产物 `orn3.exp`，**硬校验 `vis23.o` 的 `vis_forward` 符号大小 = 0x52f4**，不符即 exit 3） |
| 上件脚本 | `/home/caden/ornc/det_install30.sh`（抢 `.cardlock` → 证明卡真空 → 备份 → 停 → 换件 → 起 → 窗口内冒烟黄金题 → 失败自动回滚 → 恢复 crontab） |
| 网关 | `/home/caden/ornc/serve2.py`，`PROMPT_BUDGET=7800`、`ENG_MAXT=8192`（env `K200_PROMPT_BUDGET` / `K200_ENGINE_MAXT`） |
| 运行 env | `VIS_CH=96 VIS_CHA=0 VIS_FOLD=1 K200_PROF=0 K200_SLOTS=4 K200_CKPT=2 K200_MAXT=8192` |
| 旧基线源码 | `orn3.cpp`（112404 B，md5 `7d9a3e6d8c76733ff4eb71fd26bfcf81`，1974 行）——**任务书里说的 "orn3.cpp" 是这条旧线；本文行号以 `orn3.cb30.cpp` 为准** |

> ★ **所有行号都写成 `行号 + 关键词`**。若下一位代理基于 `orn3.cb.cpp`（2880 行）或别的分支，请用关键词 `grep -n` 重新定位（各版本之间差值不是常数：`g_pKc` 在 cb30 是 939、在 cb 是 934；`L.Kc.assign` 在 cb30 是 1935）。在本文里凡是关键锚点都会给"关键词"。

---

## 第 1 章 现状核实（全部有出处，不要凭记忆改）

### 1.1 模型与层型（不可变事实）

```
arch=qwen35, block_count=33(32 主干 + blk.32=MTP 跳过)
embd=4096  ff=12288  head=16  head_kv=4  head_dim=256
rope: freq_base=1e7, dim_count=64, sections=[11,11,10,0] (MRoPE)
rms_eps=1e-6  full_attention_interval=4
全注意力层 = il ∈ {3,7,11,15,19,23,27,31}（8 层）；其余 24 层 gated DeltaNet
```
对应源码常量：`orn3.cb30.cpp:385` `static const int EMB = 4096, NFF = 12288, NH = 16, NKV = 4, HD = 256, NROT = 64;`、`:389` `ROPE_BASE = 1e7f`、`:397` `is_recr(il) = ((il+1) % 4) != 0`。

### 1.2 全注意力层的主机实现（这是要被替换的那 40 行）

单 token 路径 `forward()`：`orn3.cb30.cpp:1108`（`} else {`，非 recr 分支）

| 行号 | 关键词 | 做什么 | 形状 |
|---|---|---|---|
| 1110–1112 | `kk.resize(NKV * HD)` | 从 `gy` 里切出 k[1024]、v[1024] | `kk,vv[4][256]` |
| 1113–1118 | `q5(NH * HD), g5(NH * HD)` | 从拼接输出里按 stride 512 交错切出 q 与 gate | `q5,g5[16][256]` |
| 1119–1120 | `rmsnorm(&q5[h * HD] ... L.q_norm)` | **per-head RMSNorm(256)**，q 16 头 + k 4 头 | 20 × 256 |
| 1121–1122 | `rope_apply(&q5[h * HD], pos, HD)` | **MRoPE**（前 64 维、neox 配对 `(i, i+32)`、`theta_i = pos·1e7^(-2i/64)`）；`rope_apply` 定义在 `:909` | 20 × 64 |
| 1123–1126 | `memcpy(&Kc[(size_t)pos * NKV * HD], kk.data(), NKV*HD*4)` | **KV 写入**：偏移 = `pos * 1024` 元素，K 与 V 各 4 KiB | `[MAXT][4][256]` |
| 1127–1153 | `int gpr = NH / NKV;` | **GQA 4→16 头**：头 h 用 KV 头 `kh = h/gpr`；分数 `s = q·k·(1/√256)`、`best/latch` softmax、`p·v` 累加 | `wt[16][pos+1]` |
| 1152 | `oh[i] *= sigmoidf_(g5[h*HD+i])` | **sigmoid gate 乘**（在 attention 之后、输出投影之前） | 16 × 256 |
| 1155 | `gemv(L.grp_mid, ao.data(), o.data())` | `attn_output` 投影（已在卡上，不变） | [4096,4096] |

同一段逻辑的另两份拷贝（**三处必须同步改**，否则三条路径数值不一致）：
- 预填充 `forward_batch()`：`:1330` `for (int q = 0; q < T; q++)` → `:1339-1346`（rmsnorm/rope/KV 写）、`:1349-1370`（注意力+gate）、`:1371` 回写 `g_bD`
- 多槽 `forward_multi()`：`:1610-1635`（rmsnorm/rope/KV 写到 `act[q]->Kc[l]`）、`:1639-1664`（注意力+gate）、`:1667` 回写 `g_mD`；槽结构 `struct CbSlot` 在 `:1424`（KV 字段 `std::vector<std::vector<float>> Kc, Vc;` 在 `:1438`）

★ MRoPE 的一个已核实结论（省掉一次无谓改造）：GGUF 的 `sections=[11,11,10,0]` 在**纯文本**下三段位置相同 ⇒ 与"整个 64 维做标准 neox RoPE"**逐位等价**。所以现有 `rope_apply` 就是对的，**上卡时不需要在卡上重实现分段 MRoPE**；只要保证 K 进 KV 之前已经做过宿主 RoPE（现状如此）。若将来把 RoPE 搬卡，必须按 `sections` 分段实现，否则文本也会变。

### 1.3 实测数字（用于做预算，全部有出处）

| 项 | 值 | 出处 |
|---|---|---|
| 预填充分解（1002 tok） | 总 **34.5 ms/tok**；卡 gemm **7.42**（其中真等卡仅 0.34）；主机 **31.41** = SSM/delta-net **14.27** + 注意力及层内其余 **~16.8** + 归一化/launch **0.36**；H2D 1.68 + D2H 1.78 | PROGRESS 第 25.3 章实测表 |
| 单序列解码 | 13~17 tok/s | 任务书 / PROGRESS |
| 权重常驻 HBM | **chip0=5095.8 MiB  chip1=3969.5 MiB  合计 9065.3 MiB = 8.85 GiB** | `/home/caden/ornc/serve2.log` 启动行 `权重常驻 HBM` |
| 每芯 HBM 硬件基数 | **8064 MiB (7.87 GiB)** | `references/quantized-weight-residency.md:11` |
| `api::Context` workspace | **64 MiB/芯** | PROGRESS 第 25.2 章 |
| 引擎设备 scratch（生产档） | `g_dx = BATCH_MAXT*BATCH_MAXN*4 = 3.0 MiB`，`g_dy = 64*130000*4 = 31.7 MiB`，`g_dxq 32 KiB`，`g_dxs 8 KiB` ⇒ **~34.8 MiB/芯** | `orn3.cb30.cpp:1882`（生产档分配）、`:1852`（单发档）、`:629`（`g_dx/g_dy` 声明） |
| 视觉塔卡上占用 | int8 权重 **216 MB/芯** + 折回/堆叠缓冲若干（`vis.cpp` 自己 `xpu_malloc`，**不记进 `g_hbm`**） | `references/ornith-multimodal-on-k200.md:34`；`vis.cpp:282/925-982` |
| 主机 KV 现状 | `MAXT*NKV*HD*4B = MAXT×4 KiB`/层/张量；8 层 K+V 合计 **512 MiB @ MAXT=8192**（日志实测 `KV(8 注意层) 512.0 MB`） | `orn3.cb30.cpp:1935`、启动行 `就绪. 主机常驻 ... KV(8 注意层)` |
| 官方算子主机侧固定开销 | **~16 µs/次**（gemm_int8 实测 0.016 ms） | `references/official-gemm-int8-recipe.md:71` |
| 自写 CLUSTER launch 开销 | **3.3 ms/次** ⇒ 自写内核只能"少 launch、多干活" | `orn3.cb30.cpp:398` 注释、坑 2 |
| 官方 D2D/H2D 带宽 | H2D 3.51 / D2H 3.73 GB/s（厂商 test_dma）；卡上计算路径 int8 大 M GEMM **133 GB/s** | `references/k200-kernel-launch-traps.md:109`、multimodal:181 |
| 视觉塔混沌 | 第 0 层 **0.0002%** 扰动 → 第 26 层 **7%** → 答案从 `K200`/9-9 翻成 `K2CO`/7-9 | PROGRESS 19.3 / 第 21 章 |
| 卡台账基线 | 异常 **18**（零新增）、`reset_count` **0/0**、两芯 RUNNING | PROGRESS 第 27.5 / 29.1 章 |

### 1.4 现有 KV / 检查点 / 槽位的内存牵连（抬 MAXT 前必须逐个认领）

| 机制 | 锚点（关键词） | 形状 / 代价 | 卡上 KV 后是否还需要 |
|---|---|---|---|
| `Layer::Kc,Vc` | `:430` `std::vector<float> Kc, Vc;` | 主机 `MAXT*NKV*HD*4B` | **不再需要**（仅注意层） |
| 分配 | `:1935` `L.Kc.assign((size_t)MAXT * NKV * HD, 0.f)` | MAXT×4 KiB/层 | 改为 `xpu_malloc` |
| 指针间接化 | `:939` `g_pKc[NLAYER]`、`:1957` 指向 | — | 保留（指向卡缓冲描述符） |
| 快照 `g_snap` | `:1993-2003` `g_snap.Kc[l] = g_lay[l].Kc` | 整份拷贝（+512 MiB @8192） | **丢弃**（§3.5） |
| 检查点 `g_ck` | `:2040-2100`（`:2054` 打印、`:2076-2078` 存、`:2094-2095` 还原） | `K200_CKPT` 槽 × 全 KV | **只存 host 侧（conv/S/logits/ids）**，KV 靠"截断" |
| 槽位 | `CbSlot::Kc,Vc` `:1438`；分配 `:2517-2519`；存 `:2576-2578`；还原 `:2670-2671` | **每槽 × 8 层 × MAXT×8 KiB** ⇒ `K200_SLOTS=4` @MAXT=8192 = **2 GiB 主机** | **必须改**：@MAXT=32768 会变成 8 GiB，主机 RAM 只有 7.9 GB ✗ |
| 清零 | `:2351` / `:2410` `memset(g_lay[l].Kc.data(), 0, ...)` | — | 卡上不需要（§3.5） |
| `g_bWT` | `:1207` `g_bWT.resize(NH * MAXT)` | 16×MAXT×4B = 8 MiB @131072 | 保留（主机，小） |
| `g_mWT` | `:1488` `g_mWT.resize(M * NH * MAXT)` | **M(≤64)×16×MAXT×4B = 512 MiB @131072 ✗** | **必须改**（改成分块/按 pos 分配） |
| 台账打印 | `:1947-1954`、`:2322`、`:2830` | 分母里的 `4.0` 要随 dtype 改 | 改 |

★ **结论**：主机 KV（含快照/检查点/槽位拷贝）在 MAXT≥16384 时就已经顶到主机 RAM 天花板。**"KV 上卡" 不只是提速，它是长上下文与多槽并发的先决条件。**

---

## 第 2 章 算子链设计

### 2.1 分解原则（为什么这么切）

1. **只搬 O(T) 的部分**。每 token 的注意力要扫整段 KV（O(pos)）；per-head RMSNorm/RoPE/gate/残差是 O(1)/token（20×256 + 4096 个元素），在主机上只花微秒级，搬到卡上反而要新增 5~6 次 launch 且**必然改位级**。⇒ 保它们原位（主机）不动。这是"以最小位级扰动换掉主机 KV 扫描"的关键取舍。
2. **"CPU 只调度"铁律的合规口径**（必须与用户对齐，见下方口径 A/B）。现状真正违规的是**主机逐元素重扫 65 MB/token 的 KV**；本设计消除的正是它。
   - **口径 A（本设计采用）**：占 99% FLOP、100% 权重带宽、100% KV 带宽的矩阵乘与注意力全在卡上；O(1)/token 的逐元素/归一化留在主机（与现状一致，且现状已被验收通过）。
   - **口径 B（严格派）**：连 RMSNorm/RoPE/gate/残差/silu 也搬卡 ⇒ 需要卡上 sigmoid（**表驱动**，见 §2.8 探针 P13，非位级）、pow/sqrt 等，launch 数翻倍、位级必变、黄金题判据失效。列为**可选专项，不在本次范围**。
3. **KV 设备布局 = 主机布局**：`[MAXT][NKV][HD]` 连续。理由见 §2.2——GEMM 用 `ldb = NKV*HD = 1024` + `B 基址 = buf + kh*256` 就能表达"某个 KV 头的 [pos+1, 256] 视图"，**零转置、零重排**。同一个布局让主机兜底路径和现有落盘对拍工具原样可用。
4. **每处改动都能单层开、能逐层对拍**。开关 `K200_ATN=0/1/2/3` + `K200_ATN_DUMP`（§5.2）。

### 2.2 KV 显存布局与写入偏移（**定死，后续所有设计依赖它**）

```
设备侧（每个"拥有者芯片" d 上，只放它负责的那些注意层）：
  g_dKc[d][l] : float*   长度 MAXT*NKV*HD   = MAXT*1024 个元素   fp32 → MAXT×4 KiB
  g_dVc[d][l] : float*   同上
  布局与主机完全一致：元素 (t, kh, i) 位于  base + (t*NKV + kh)*HD + i
  ⇒ t 行、kv 头 kh 的 [HD] 向量 =  base + t*1024 + kh*256      （连续 256 个 float）
  ⇒ 某一个 kv 头的整段 [pos+1, 256] 视图 = 基址 base + kh*256，行步长(ldb) = 1024

KV 写入（K 与 V 各一次，均为 1 KiB 连续拷贝）：
  H2D  src = &kk[0] (主机 float[1024]，已做 per-head RMSNorm + MRoPE)
       dst = g_dKc[d][l] + (size_t)pos * 1024      （字节偏移 = pos*4096）
  H2D  src = &vv[0] → dst = g_dVc[d][l] + (size_t)pos * 1024
  预填充批量（T 个 token，pos0..pos0+T-1 在缓冲里连续）：
       一次 H2D 搬 T*1024 个 float（T×4 KiB），dst = base + pos0*1024
```

★ **偏置公式必须逐处复核**：现状主机代码是 `Kc[(size_t)pos * NKV * HD]`（`:1124` / `:1345` / `:1631`），搬到卡上**保持同一个公式**，只是基址换成设备指针。任何"顺手优化成 `pos*HD`"都会静默错位。

★ **`ldb=1024`（非连续 B 的行步长）是否被官方 GEMM 接受，是必须探针验证的第一号问题**（P2b）。如果不支持非连续行步长，退路是**每层每张量开 4 个 per-kh 连续缓冲** `g_dKc[d][l][kh] : [MAXT][256]`，代价是 KV 写入从 2 次 H2D 变成 `1 次 H2D(8 KiB 暂存) + 8 次 D2D(1 KiB)`（每层每 token 多 7 次 launch ≈ +0.11 ms/token/layer，可接受）。

### 2.3 解码（T=1）链路 —— 逐步：算子 / 形状 dtype / launch / 预期耗时

设：当前 token 绝对位置 `pos`（`pos ≥ 0`），拥有者芯片 `d`，注意层 `l`。
`sc = 1/√256 = 0.0625`（可以直接折进 GEMM 的 `alpha`，**零额外代价**）。

| # | 步骤 | 算子（首选 → 兜底） | 形状 / dtype | launch | 备注 |
|---|---|---|---|---|---|
| A0 | `grp_a` 投影 | `api::gemm_int8`（**现状，不改**） | A[1,4096]f32 · B[10240,4096]i8 · C[10240]f32 | 1（×2 芯并发）+ H2D/D2H | `:773` `gemv_i8` |
| A1 | 切 q5/g5/kk/vv | 主机（**不改**） | `q5,g5[16][256]` `kk,vv[4][256]` | 0 | `:1113-1118` |
| A2 | per-head RMSNorm + MRoPE | 主机（**不改**，位级锚点） | 20×256 | 0 | `:1119-1122` |
| **B1** | **KV 写入** | `xpu_memcpy` H2D ×2 | K 1 KiB + V 1 KiB | **2** | 偏移 `pos*1024` 元素 |
| **B2** | **Q 上传** | `xpu_memcpy` H2D | `q5[16][256]` f32 = 16 KiB | **1** | 到 `g_dSt[d]` |
| **C1** | **打分（每 KV 头一次，共 4 次）** | ① `api::qk_attention<float,float,float,float>` ② `api::gemm_int16` ③ `api::gemm_int31` ④ `api::qk_v_attention<float,float,float>` | A=`g_dSt+kh*4*256` **[4,256]** f32；B=`g_dKc + kh*256` **[pos+1,256]** f32, ldb=1024, trans_b=true；C=`g_dScore + kh*4*(pos+1)` **[4, pos+1]** f32, ldc=pos+1；`alpha=0.0625`、`beta=0` | **4** | ≈等价于 `scores(h) = sc · q_h · K_{h/gpr}^T`；GQA 的 4 个 Q 头**天然拼成 m=4 的一次 GEMM** |
| **C2** | **softmax** | `api::softmax2d_forward(g_ctx[d], g_dScore, g_dProb, rows=16, cols=pos+1, use_sdcdnn=false)` | `[16][pos+1]` f32 | **1** | ★ 解码**不需要因果掩码**（t ∈ [0,pos] 全合法）；每行最大值项 `exp(0)=1` ⇒ Σ≥1 ⇒ **不会除零** |
| **C3** | **PV（每 KV 头一次，共 4 次）** | ① `api::gemm_int16` ② `api::qk_v_attention<float,float,float>` | A=`g_dProb + kh*4*(pos+1)` **[4, pos+1]** f32, lda=pos+1；B=`g_dVc + kh*256` **[pos+1,256]** f32, ldb=1024, **trans_b=false**；C=`g_dAO + kh*4*256` **[4,256]** f32, ldc=256；`alpha=1`、`beta=0` | **4** | B 按 `[k,n]` 解释 ⇒ 行优先的 V 直接可用，无需转置 |
| **C4** | 结果回传 | `xpu_memcpy` D2H | `g_dAO[16][256]` f32 = 16 KiB → 主机 `ao` | **1** | 之后 gate 在主机做 |
| C5 | sigmoid gate + `attn_output` 投影 | 主机 + `gemm_int8`（**不改**） | 16×256；[4096,4096] | 1 | `:1152`、`:1155` |

**每层每 token 合计 = 2+1+4+1+4+1 = 13 次调用**（×8 层 = **104 次/token**）

**耗时预算（每 token）**

| 项 | 计算式 | @pos=1024 | @pos=8192 | @pos=32768 |
|---|---|---|---|---|
| 主机侧 launch 开销 | 104 × 16 µs | 1.66 ms | 1.66 ms | 1.66 ms |
| PCIe 字节 | 104 次里 H2D 24 KiB + D2H 16 KiB ≈ 40 KiB/层 ⇒ 320 KiB/token | 0.09 ms | 0.09 ms | 0.09 ms |
| 卡上 KV 读（**核心收益**；两芯分摊，每芯 4 个注意层） | 每层每 token `(pos+1)×8 KiB`（K: 4 kh×(pos+1)×1 KiB，V 同）；8 层两芯合计 `(pos+1)×64 KiB` ⇒ **每芯 `(pos+1)×32 KiB`** | 32 MiB/芯 ÷ 60 GB/s = **0.53 ms** | 256 MiB/芯 ÷ 60 GB/s = **4.3 ms** | 1 GiB/芯 ÷ 60 GB/s = **17 ms** |
| 对照：主机路径同口径（一份内存，无分摊） | `16 头 × (pos+1) × 1 KiB × 8 层` = `(pos+1)×128 KiB` ÷ 1.2 GB/s | **107 ms** ✗ | **873 ms** ✗✗ | **3.5 s** ✗✗✗ |

（60 GB/s 是**保守**假设：薄 GEMM（m=4）未必吃得满 133 GB/s，见探针 P4c 的实测要求；上面已把折损算进去。1.2 GB/s 是实测主机带宽。）

### 2.4 预填充链路

**策略 A（阶段③，最小改动、位级最可控）**：把 `forward_batch()` 的 `:1330` `for (int q = 0; q < T; q++)` 循环体里 `:1349-1370` 的主机注意力段换成 §2.3 的 B1..C4（`pos = pos0 + q`）。
- 因果性：本 token 的 K/V 在读取前**已经写进卡上缓存**（B1 在 C1 之前），且读范围严格 `[0, pos]` ⇒ 无需掩码 ✓
- launch 账：`8 层 × T × 13`。T=1002 ⇒ 104,208 次 × 16 µs = **1.67 s 主机 CPU 时间**（占 ~7%）+ 卡上读 `Σ_pos×64 KiB/层…` 平均 pos=500 ⇒ 32 MiB/token/芯 ÷ 60 GB/s ≈ 0.55 ms/token
- 判据：**这是能上线的版本**（主机 launch 开销可接受），且逐 token 的算子序列与解码完全一致 ⇒ 数值路径与解码同构，位级对拍好做。
- 对比现状：省掉的是每 token 每层 `16×pos×1 KiB` 的主机读取（平均 pos=500 时 8 MiB/层 ⇒ 64 MiB/token ⇒ 在 1.2 GB/s 下 ~53 ms/token 量级）。

**策略 B（阶段⑥，收益最大）**：按 chunk（C=64）批处理 + 卡上掩码。
- KV 写入：整 chunk 一次 H2D（T×8 KiB）+ 0 次 D2D ⇒ **1~2 次/层/chunk**
- 打分：`m=T` 的一次 GEMM ×4(kh) ⇒ C=[T, pos_max+1]，`pos_max = pos0+(c+1)C-1`；**需要因果掩码**
- 掩码在卡上生成（**全部用已验通算子**，4 次 launch/层/chunk）：
  1. `elementwise_sub_2d`：`pos_idx[1, cols]`（列号 j） −  `p_i[rows, 1]`（该行 query 的绝对位置，逐行不同） ⇒ `d[i,j] = j - p_i`（广播：m1=1 或 m2=1 混合）
  2. `clip(d, 0, 1)`（`func_dec.h:508` `clip(ctx, x, y, len, min, max)`）⇒ `1` 表示"未来位置"
  3. `scale(-1e30, beta=0)` 或 `elementwise_mul_f32(d, -1e30 向量)` ⇒ 掩码值
  4. `elementwise_add_2d(scores, mask_bcast)` ⇒ 未来位置变成 -1e30
- ★ 安全保证：行 i 的列 `[0, p_i]` 全部非掩 ⇒ 每行至少一个有效列 ⇒ `Σexp ≥ 1` ⇒ **永不除零**（这是卡上除零打死 session 的唯一入口，见 §2.8 陷阱 T1）
- launch 账：`8 层 × ⌈T/C⌉ × (2 + 4 + 1 + 4 + 4 + 1) = 8×16×16 = 2048 次` ⇒ **33 ms**，比策略 A 的 1.67 s 好 50×
- 代价：数值顺序变了（掩码加法 + 行并批），**位级必变** ⇒ 必须走"文本宽容"判据（§4.4）。

**多槽 `forward_multi()`（阶段③起适用）**：M 条槽各自不同的 `pos` ⇒ **策略 A**（逐 token、逐槽）；策略 B 需要"每行自己的 p_i"，正好 §2.4 的掩码生成天然支持逐行不同的 p_i，所以将来也能推广（列为阶段⑥+ 的可选）。

### 2.5 卡上缓冲清单（新增，记进容量账）

| 缓冲 | 大小（MAXT=131072 时的上界） | 用途 |
|---|---|---|
| `g_dSt[d]` | 16 KiB(q5) + 8 KiB(kk,vv 暂存) = **32 KiB** | H2D 落点 |
| `g_dScore[d]` | `NH × MAXT × 4B` = 16×131072×4 = **8 MiB** | 分数 [16][pos+1] |
| `g_dProb[d]` | **8 MiB** | softmax 输出（若 `softmax2d_forward` 不支持原地，就必须分开；P7c 验证） |
| `g_dAO[d]` | 16 KiB | PV 输出 |
| `g_dMask[d]`（阶段⑥） | `BATCH_MAXT × MAXT × 4B` = 64×131072×4 = **32 MiB** | 掩码切片 |
| `g_dPosIdx[d]`（阶段⑥） | `MAXT × 4B` = **512 KiB** | 列号/行位置素材 |
| KV（见第 3 章） | 4 层 × MAXT×8 KiB (fp32) | — |
| **小计（不含 KV）** | **≈ 49 MiB/芯** | 已在 §3.3 的 reserve 里计入 |

### 2.6 有意保留在主机（位级锚点）的清单

| 保留项 | 为什么保留 | 若搬卡的代价（不建议） |
|---|---|---|
| `attn_norm` / `post_norm` RMSNorm(4096) | 已有主机实现，且是每 token 的 O(1) | 卡上 `layer_norm` **n>1024 直接 INVALID_PARAM**（实测），需 reduce+elementwise 自拼，位级必变 |
| **per-head RMSNorm(256)** ×20 | 5120 个元素，主机 ~2 µs | 卡上等价物 = `mul`（平方）+ `reduce(MEAN dim=1)` + `elementwise_pow(-0.5)` + `mul(w 广播)` ⇒ **5 次 launch/层/token**（+40 次/token）且位级变 |
| **MRoPE**（前 64 维，20 头） | 1280 次 sin/cos，主机 ~10 µs；且 `sections=[11,11,10,0]` 在文本下与现实现逐位等价（§1.2） | 卡上无 RoPE 算子 ⇒ 需 sin/cos 表 + `elementwise_mul_2d` 交错，~8 次 launch 且位级变 |
| **sigmoid gate** ×4096 | 4096 个 `expf`，主机 ~20 µs | 卡上只有 `activation_forward(SIGMOID)`，而符号表里有 `xpuapi_get_sdcdnn_sigmoid_table` / `xpuapi_get_cluster_sigmoid_table` ⇒ **它很可能是查表实现，不是位级等价**（P13） |
| 残差加法 / silu / conv1d / delta-net | 本次范围外 | delta-net 搬卡是另一个专项（§2.9） |

### 2.7 launch / 耗时总账（照这张表分配工作量）

| 场景 | 调用数 | 主机 launch 开销 | 卡上 KV 读 | PCIe |
|---|---|---|---|---|
| 解码 1 token（8 层） | 104 | 1.66 ms | 两芯合计 `(pos+1)×64 KiB`（每芯一半）÷ 60 GB/s | 320 KiB H2D + 128 KiB D2H |
| 预填充 1002 tok（策略 A） | 104,208 | **1.67 s** | 平均 pos 500 ⇒ 32 MiB/token/芯 | 320 MiB |
| 预填充 1002 tok（策略 B，C=64） | 2,048 | **33 ms** | 同策略 A | 320 MiB + 掩码 H2D 可忽略 |
| 多槽 M=4 一步（8 层） | 4×104 = 416 | 6.6 ms/步 | 4 条槽各自 pos | ~1.3 MiB |

⇒ **阶段③（策略 A）就能拿到绝大部分收益；阶段⑥（策略 B）是"把主机 launch 开销从 1.67 s 压到 33 ms"的一步**，对"个位数 ms/tok"目标必要（但仍受 delta-net 限制，见 §2.9）。

### 2.8 【必须先用极小尺寸探针验证的算子清单】

**纪律（不可协商）**：每个用例都跑在 `safe_run.sh` 里；**任何一次卡异常（退出码 3）立即停手、本阶段判失败、严禁重试**；每次跑前后对比 `grep -ac "Exception in kernel execution" /var/log/kern.log` 与 `/proc/xpu/dev{0,1}/reset_count`。

**探针载体**：**不写独立进程**，而是在 `orn3` 里加一个 `K200_ATN=probe` 模式：`main()`（`:1788`）在**加载权重之前**就建好 `api::Context`（复用 `:1886` 的同款构造 `new api::Context(api::Device(api::DeviceType::XPU1, dv))`）、跑完探针、`exit(r)`。好处：①窗口最短（不花 12~25 s 灌权重）；②与生产件**同一个二进制、同一套 Context/scratch 建立代码**，避免"探针环境与生产不同"的假象；③探针输出 JSON 落盘（`/home/caden/ornc/atn_probe.json`）。

**符号存在性（已核实，2026-09-23，`nm -DC /home/caden/xtdk/xtdk-x86_64/shlib/libxpuapi.so`）**

| 符号 | 类型 | 结论 |
|---|---|---|
| `qk_attention<float,float,float,float>` | W（弱模板实例） | **存在**，可链接；语义 `待探针验证` |
| `qk_attention<float16,float16,float16,float>` | W | **存在**（fp16 KV 路线的前提，见 §3.4） |
| `qk_v_attention<float,float,float>` | W | **存在** |
| `qk_v_attention<float16,float16,float16>` | W | **存在** |
| `gemm_int16(Context*,bool,bool,int,int,int,float,const float*,int,const float*,int,float,float*,int)` | **T（强）** | **存在，纯 float A/B/C** —— 最有希望的"无损"GEMM |
| `gemm_int31(...,float max_a,float max_b)` | **T** | **存在**，标量 max（不是指针）⇒ 比 `*_maxptr` 安全得多 |
| `gemm_int8` 的 **float-B 重载**（无 max_b） | T | 存在，但**内部必然要自己 findmax**；而 `api::fc<float,float,float,int>` 触发 `findmax_sdcdnn` 已打死过 dev1 ⇒ **直接禁用，不进探针队列** |
| `block_gemm_int8` | **库内无任何符号** | ✗ **不可用（link 会失败）** —— 头文件里有声明不代表库里有实现 |
| `gemm_strided_batched_int31` | T | 尾部是 `const float* max_a, max_b` **指针数组** ⇒ 属 `*_maxptr` 家族 ⇒ **禁用** |
| `softmax2d_forward` | T | 存在，已验通（0.00000~0.00001%） |
| `reduce` / `elementwise_*_2d` / `add` / `sub` / `scale` / `transpose` / `clip` / `elementwise_div_no_nan(_2d)` / `memcpy_device` / `memset` | T | 存在（`reduce(MAX)` **有符号**、`layer_norm` n≤1024、`xpu_memset` 不存在而 `api::memset` 存在） |

#### 探针清单（附用例与判据）

| ID | 目标 | 用例（极小尺寸） | 判据 | 失败怎么办 |
|---|---|---|---|---|
| **P1** | `cast<float,float16>` / `cast<float16,float>` 往返 | len=1024，值分布 = 真实 K/V 采样 + 含 `0`、`±1e-8`(denormal)、`1e30`、`-1e30` | ① `r==0`；② 往返 relrms ≤ 1e-3；③ **与主机 `f2h()` 逐位相同**（决定 fp16 转换放主机还是卡上） | 主机转换保底（0 launch） |
| **P2a** | `qk_attention<f32,f32,f32,f32>` **转置语义** | A=[2,3]、B=[3,4] 已知整数阵；`(TransA,TransB)` 四种组合各跑一次，m=2,n=4,k=3 | **恰好一种组合** relrms < 1e-6 且等于 `A·B` 的转置语义（把四种输出全打出来，人工比对） | 换 `gemm_int16` |
| **P2b** | `gemm_int16` **非连续 B 的行步长**（本设计的第一号依赖） | M=4,N=64,K=256；B 布局 `[64][1024]` 取 `base+kh*256`、`ldb=1024`；对照：把同一段数据拷成连续 `[64][256]` 再算一次 | **两次输出逐位相同**（relrms = 0 或 ≤1e-7） | 退到 per-kh 连续缓冲（+7 launch/层/token） |
| **P2c** | `qk_attention` 的 **max_a/max_b 语义** | `max_a=nullptr, max_b=nullptr, res_max_ptr=nullptr, use_max_ptr_4=false`；再试 `max_a=&one, max_b=&one` | ★ **只允许 nullptr 通过**。若必须传非空指针才可用 ⇒ **放弃该算子**（`*_maxptr` 家族已实测把 dev0 打进 RBRESP/ERROR） | 换 `gemm_int16` |
| **P2d** | `qk_attention` 的 **`is_asr_mask`** | 造 A·B 上三角为 +1e6 的情形，`is_asr_mask=0/1` 各跑一次 | 输出是否等于"因果掩码"语义（打印两种输出，人工判）；**我们不依赖它**，只在能确认是标准因果掩码时才可省掉 P2e 的掩码步骤 | 自己上掩码（§2.4） |
| **P2e** | `qk_attention` 的 **bias / bias_ldn / is_tf_bias / is_onnx_bias** | `bias=nullptr, bias_ldn=0, is_tf_bias=false, is_onnx_bias=false` | `r==0` 且输出与"无 bias"一致 | 保持 false 恒值 |
| **P2f** | `qk_attention` × `gemm_int16` **真实量级精度** | A=[4,256] 用**真实 q5 分布**、B=[512,256] 用**真实 K 分布**（从一次真实 forward 用 `K200_ATN_DUMP` 落盘取得）；主机 fp64 参考 | relrms **≤ 1e-5**（合格）/ ≤1e-3（勉强可用，需走"文本宽容"）/ >1e-3（弃用） | 换另一个；或转 `gemm_int31` |
| **P3** | `qk_v_attention<float,float,float>` | 同上（无 bias 参数） | 同上 | 只作备选 |
| **P4a** | `gemm_int16` 基础正确性 | M=4,N=64,K=256 随机 + 真实量级 | relrms ≤ 1e-5 | 弃用 |
| **P4b** | `gemm_int16` 边界 | N=1（pos=0 的第一次解码）、K=1、M=1 | `r==0`、无异常 | — |
| **P4c** | `gemm_int16` **实测有效带宽（薄矩阵）** | M=4, N={1024,4096,16384}, K=256；循环 50 次计时 | 记录 GB/s；**这是 §2.3 里 60 GB/s 假设的验证点**，低于 20 GB/s 就要重新做预算 | 调 M（把 4 个 kh 的 Q 头合批成 m=1 多次 / 或按 16 头一次但不同 B 不行） |
| **P5** | `gemm_int31`（标量 max） | M=4,N=512,K=256；`max_a=max|A|`、`max_b=max|K|`；**特别测 `max_a=0`（全零 A）** | relrms ≤ 1e-2；`max_a=0` 时 `r==0` 且输出全 0（**不能除零**） | 弃用 |
| **P6** | `gemm_int8`（float-B 重载） | — | **✗ 不进探针队列**（内部 findmax 风险，已知会打死 dev1） | 禁用 |
| **P7a** | `softmax2d_forward` **薄形状**正确性 | rows=16, cols={1, 2, 64, 1024}、以及 rows=4/16/64 | relrms ≤ 1e-5；cols=1 ⇒ 输出恒 1.0 | 用 reduce+div 自拼 |
| **P7b** | `softmax2d_forward` **薄形状耗时**（任务书问的"64KB 级张量是否划算"） | rows=16, cols={1024, 4096, 16384, 32768}，各 20 次 | 记录 µs；**判据：耗时 ≤ 卡上读该张量时间的 3 倍**（读 `16×cols×4B` @100 GB/s），否则不划算 | 阶段⑥ 改用"按 kh 分 4 次 rows=4"或自拼 |
| **P7c** | `softmax2d_forward` **原地** | `y == x` | `r==0` 且结果正确 | 分开缓冲（已按分开设计） |
| **P7d** | `softmax2d_forward` **全 -1e30 行**（★ 危险用例，**放在最后跑**） | 一整行填 -1e30 | 记录结局（`FP_DIV0` 打死 session / inf·nan / 被内部夹住）。此用例的目的只是确认"我们的设计里永不产生这种行"。**若它真的打死 session，这是预期内的**；跑之前必须确认 `safe_run.sh` + 看门狗在跑 | 设计约束：永远不产生全掩行 |
| **P8** | 自拼 softmax 兜底 | `elementwise_pow(e=exp?)` 无 exp 算子 ⇒ 只能用 `softmax2d_forward`；兜底 = `reduce(SUM, dim=1)` + `elementwise_div_2d` + **分母夹逼** | 与 P7a 同 | 只能用官方 softmax |
| **P9** | **卡上除法的分母夹逼**（任务书点名） | ① `elementwise_div_2d(x, y)` 且 `y` 含 `0`（★ 危险用例，**倒数第二个跑**）；② 同输入但先用 `elementwise_max_2d(y, eps_vec)` 把 `y` 抬到 `1e-30` 再除；③ `elementwise_div_no_nan_2d` | ① 记录三种可能结局：**FP_DIV0 打死 session** / 输出 inf·nan / 输出被内部夹到有限值。**前两种不可接受 ⇒ 必须夹逼**（第三种也要在报告里写明"官方算子自己夹了"，但仍按夹逼写法实现，因为规则要与主机同式）。② `r==0` 且结果 = `x/max(y,1e-30)`；③ 若 `r==0` 且 `y=0` 处输出 0 ⇒ **优先用它（省一次 launch）** | 一律走 ② |
| **P10** | `transpose`（仅在走 per-kh 布局时用） | shape `[1,256]`→`[256,1]`、`[4,256]`→`[256,4]`，`permute` 语义 | 与主机转置逐位相同 | 主机转置 |
| **P11** | `memcpy_device` / `xpu_memcpy` D2D | 1 KiB、8 KiB、1 MiB | 内容逐字节相同；**记录 µs/次**（决定 KV 写入用 D2D 还是直接 H2D） | 每 kh 直接 H2D |
| **P12** | `api::memset` | 4 KiB 缓冲填 0 | `r==0` 且全 0（`xpu_memset` 不存在） | 只写不读 ⇒ 不需要 |
| **P13** | `activation_forward(SIGMOID)`（**仅当考虑把 gate 搬卡**） | len=4096，值域 [-20,20] | 与主机 `sigmoidf_` 的 relrms **预期不是 0**（表驱动）；**≤1e-3 才可考虑，且必须走"文本宽容"判据** | **保持 gate 在主机**（本设计默认） |
| **P14** | H2D/D2H **小传输固定开销** | 1 KiB / 8 KiB / 16 KiB / 64 KiB，各 100 次 | 记录 µs/次；**这是 §2.7 里 16 µs/次 假设的验证点** | 调块大小 |
| **P15** | `slice_forward`（可选，用于按 pos 切片） | [1, 1024] 取前 256 | 与主机切片逐位相同 | 直接给基址偷懒（推荐） |

#### 全局陷阱（写进注释，别踩第二遍）

- **T1 分母夹逼（会打死 session）**：自写内核里除零实测是**真卡异常 + session error + 设备掉出 RUNNING**（`vis.cpp` 第 21 层 `FP_DIV0`）。**任何上卡的除法，分母必须先 `elementwise_max_2d` 抬到 `1e-30`（与主机同式），或用 `elementwise_div_no_nan_2d`**（官方算子的具体行为由 P9 确认，但规则一样，不因"官方可能自己夹了"而省掉）。**本设计里 softmax 是唯一的除法点，且解码/掩码路径已保证 `Σexp ≥ 1`；但如果将来动了掩码逻辑，必须重新审这一条。**
- **T2 `reduce(MAX)` 是【有符号】max**：要 `max|A|`（比如给 `gemm_int31` 准备 max_a）必须 `max(reduce(MAX,A), -reduce(MIN,A))`，其中取负用 `scale(-1, beta=0)`。**拿 `reduce(MAX)` 当 `max|A|` 用，会对全负张量给出错误的归一化系数 ⇒ 激活被量化到饱和 ⇒ 输出错**；而且它会静默发生（不报错）。
- **T3 `*_maxptr` 家族禁止**：`gemm_int8_maxptr` / `gemm_int16_maxptr` / `gemm_int31_maxptr` / `gemm_strided_batched_int31`（尾部指针）——已实测把 dev0 打进 `RBRESP/RBRESP→ERROR`。**只允许 `gemm_int31` 的标量 max 版本。**
- **T4 `co ≤ 16`（HW_CORE=16 硬编码）**：自写内核若把并行度压到 co=16 以下会慢 3~5×；内核里用 `core_num()` 会静默漏写行（用哨兵法查）。
- **T5 本地内存 8712 B/核**：只有自写内核涉及；官方算子不管。
- **T6 HBM 指针不能解引用**：设备指针只能 `xpu_memcpy`；`xpu_memset` 不存在（用 `api::memset`）。
- **T7 设备码编译必须用绝对路径** `/home/caden/xtdk/xtdk-x86_64/bin/clang`（本次不需要编译任何设备码）。
- **T8 缓冲区在自检/降级路径里也必须分配**：NULL 设备指针会把卡打异常。

### 2.9 预期收益的诚实算术（不要承诺做不到的事）

```
现状（1002 tok 预填充，实测）：     34.5 ms/tok
  ├─ 卡 gemm（权重读，已在卡上）      7.42   （真等卡仅 0.34）
  ├─ 主机 SSM/delta-net             14.27   ← 本设计【不动】
  ├─ 主机 注意力及层内其余          ~12     ← 本设计把它压到 ~1~2
  └─ 主机 归一化/launch              0.36
本设计后（策略 A）：              ≈ 7.42 + 14.27 + ~1 + 0.36 + launch 1.2  ≈ 24 ms/tok
本设计后（策略 B）：              ≈ 7.42 + 14.27 + ~1 + 0.36 + launch 0.03 ≈ 23 ms/tok
⇒ 预填充 34.5 → ~23 ms/tok（**提升 ~1.5×**）
⇒ 而"个位数 ms/tok" 的算术下界 = 7.42(卡 gemm) + 14.27(delta-net) ≈ 21.7 ms/tok
   ★ 所以：**要个位数，必须另立专项把 delta-net 搬卡**（PROGRESS 29.5(b) 已判定它是主机内存带宽硬底，
     S 状态 2 MB/层/token 往返）。不要在本次验收表里写"个位数"。
```
长上下文/解码侧的收益更直观（§2.3 表）：@pos=8192 每 token 的 KV 扫描从 **873 ms → 8.5 ms**（~100×）。
---

## 第 3 章 KV 布局与容量账

### 3.1 每芯 HBM 账（实测 + 逐项拆解）

```
每芯硬件 HBM 基数                      8064 MiB   (7.87 GiB)  ← references/quantized-weight-residency.md:11
两芯合计                              15750 MiB   (15.7 GiB)

chip0 实测已用：
  文本权重（g_hbm 记账）              5095.8 MiB  ← serve2.log 启动行「权重常驻 HBM」
  api::Context workspace                64.0 MiB  ← PROGRESS 25.2
  引擎设备 scratch                      34.8 MiB  ← :1882（dx 3.0 + dy 31.7 + dxq/dxs 0.04）
  视觉塔 int8 权重                     216.0 MiB  ← multimodal 文档（vis.cpp 自己 malloc，不记进 g_hbm）
  视觉塔折回/堆叠缓冲                  ~20 MiB    待实测（vis.cpp:925-982 一串 xpu_malloc）
  本设计新增（不含 KV）                ~49.0 MiB  ← §2.5
  ─────────────────────────────────────────────
  合计                                5479.6 MiB ⇒ 余量 2584.4 MiB

chip1 实测已用：
  文本权重                            3969.5 MiB
  其余同上（3979.8 − 5095.8 的差额按 §3.2 的分层策略再分）
  ─────────────────────────────────────────────
  合计                                4353.3 MiB ⇒ 余量 3710.7 MiB
```

★ **两芯余量不对称（chip0 比 chip1 少约 1.1 GiB）**，这是"每矩阵行对半劈"的字节平衡造成的，不要假设两芯一样。**所有容量判定必须按 chip0 来卡。**

★★ **上线前必须做的两件事**（不能沿用我这里的读数）：
1. 重读启动行：`grep -a "权重常驻 HBM" /home/caden/ornc/serve2.log | tail -1`
2. 交叉核对：`xpu_smi -m`（读 "已用 MB/芯"）。两组数不一致时**取较大的那个**做判定。
3. 预留安全垫 ≥ 300 MiB/芯（KV 分配失败会在 `:497`/`:555` 直接 `FATAL` 退出，半路死引擎比拒绝启动更糟）。

### 3.2 分芯分层策略（KV 归谁）

**权重不动**（保持现有"每矩阵行对半劈两芯"的字节平衡，`upload_i8_host` `:539-564`）；**只有 KV 按层分芯**：

| 方案 | chip0 负责的注意层 | chip1 负责的注意层 | 说明 |
|---|---|---|---|
| 4/4 | {3,7,11,15} | {19,23,27,31} | 对称好记；但受 chip0 余量 2584 MiB 制约 |
| **3/5（推荐）** | {3,7,11} | {15,19,23,27,31} | 吃住两芯余量不对称；fp16+128k 下两芯各留 ≥1 GiB 垫 |

★ 归谁不影响数值（attention 只在拥有者芯片上跑，Q/K/V 通过 PCIe 过去/回来，§2.3 的 B1/B2/C4 已经算进账）。
★ 每层的 K/V 是**独立分配**的，所以 3/5 与 4/4 可以做成 env 开关（`K200_ATN_SPLIT=3:5`），出错时可只切分配不改代码。
★ 拥有者芯片的 KV 缓冲**放它自己的 HBM**；Q 上传与结果回传走 PCIe（每层每 token 28 KiB，见 §2.3）。

### 3.3 KV 占用表（**逐格算出来的**，不是估的）

单个注意层、单个张量的公式：`MAXT × NKV × HD × sizeof(dtype)` = `MAXT × 1024 × {4,2} B` = `MAXT × 4 KiB`（fp32）/ `MAXT × 2 KiB`（fp16）。
**一层 = K + V = `MAXT × 8 KiB`（fp32）/ `MAXT × 4 KiB`（fp16）。**

| MAXT | 单层 fp32 | 单层 fp16 | 4层/芯 fp32 | 4层/芯 fp16 | 3层/芯 fp32 | 3层/芯 fp16 | 5层/芯 fp32 | 5层/芯 fp16 |
|---|---|---|---|---|---|---|---|---|
| 8192 | 64 MiB | 32 MiB | 256 MiB | 128 MiB | 192 MiB | 96 MiB | 320 MiB | 160 MiB |
| 16384 | 128 MiB | 64 MiB | 512 MiB | 256 MiB | 384 MiB | 192 MiB | 640 MiB | 320 MiB |
| 32768 | 256 MiB | 128 MiB | **1024 MiB** | 512 MiB | 768 MiB | 384 MiB | 1280 MiB | 640 MiB |
| 131072 | 1024 MiB | 512 MiB | 4096 MiB | **2048 MiB** | 3072 MiB | **1536 MiB** | 5120 MiB | **2560 MiB** |

**共存余量判定（余量：chip0 2584 MiB / chip1 3711 MiB，含 §2.5 的 49 MiB 新增缓冲）**

| 方案 | MAXT | 每芯需求 | chip0 判定 | chip1 判定 | 结论 |
|---|---|---|---|---|---|
| fp32 + 4/4 | 8192 | 256 | ✓ (余 2328) | ✓ | **✓ 可上线** |
| fp32 + 4/4 | 16384 | 512 | ✓ (余 2072) | ✓ | ✓ |
| fp32 + 4/4 | 32768 | 1024 | ✓ (余 1560) | ✓ | **✓ fp32 的天花板** |
| fp32 + 4/4 | 131072 | 4096 | ✗ 超 1512 | ✗ 超 385 | **✗ 不可能** |
| fp16 + 4/4 | 32768 | 512 | ✓ | ✓ | ✓ |
| fp16 + 4/4 | 131072 | 2048 | ✓ 但**余量只剩 536 MiB**（< 300 垫 + 视觉塔波动） | ✓ | ⚠ 风险大 |
| **fp16 + 3/5** | **131072** | chip0 1536 / chip1 2560 | ✓ 余 1048 | ✓ 余 1151 | **✓ 128k 的正解** |
| fp16 + 3/5 | 65536 | chip0 768 / chip1 1280 | ✓ 余 1816 | ✓ 余 2431 | ✓ 舒服 |
| fp32 + 3/5 | 65536 | chip0 1536 / chip1 2560 | ✓ 余 1048 | ✓ | ✓（fp32 也能到 64k） |

**结论**：
- **fp32 可以安全上到 MAXT=32768**（1 GiB/芯），这是**不依赖任何未验证算子**的稳妥档 —— 建议阶段⑤的默认目标。
- **MAXT=131072（128k）必须 fp16 + 3/5 分层**，且**必须**满足 §3.4 的硬依赖。
- 中间档 65536 两种 dtype 都行；推荐 fp16+3/5 的 65536 作为 128k 之前的过渡验收档。

### 3.4 切 fp16 KV 需要改什么（**含一条硬依赖链，先读这条**）

**硬依赖链（这是整个 fp16 方案的门）**：
```
fp16 KV 要省的是"读 KV 的字节数" ⇒ 打分/PV 的 GEMM 必须【直接吃 fp16】
  ⇒ 只能用 qk_attention<float16,float16,float16,float> / qk_v_attention<float16,float16,float16>
      （符号存在 ✓，语义与精度 待探针验证）
  ⇒ gemm_int16 / gemm_int31 【只有 float 重载】（nm 实测，T 符号只有 float 版）
      ⇒ 若走它们，fp16 KV 得先 cast 回 fp32：读 2B + 写 4B + 再读 4B = 10 B/元素
         > 直接存 fp32 的 4 B/元素 ⇒ ✗ 亏，方案作废
  ⇒ gemm_int8_maxptr<float16,signed char,float16> 属 *_maxptr 家族 ⇒ ✗ 禁用
⇒ 结论：**没有 qk_attention<f16> 就没有 fp16 KV；没有 fp16 KV 就没有 128k。**
   ⇒ 所以阶段①的探针必须把 qk_attention<float16,...> 单列一组用例（P1 的 f16 版 + P2a~P2f 全跑一遍）
```

**需要改的类型 / 算子 / 行号**

| # | 位置（关键词） | fp32 版 | fp16 版 | 备注 |
|---|---|---|---|---|
| 1 | 设备 KV 缓冲类型 | `float*`，`MAXT*1024*4` B | `float16*`，`MAXT*1024*2` B | 布局不变（§2.2） |
| 2 | KV 写入 `:1124`/`:1345`/`:1631` | H2D `1024*4` B | 主机 `f2h()` 转换后 H2D `1024*2` B，**或**卡上 `cast<float,float16>(staging → KV[pos])`（+2 launch/层/token） | ★ 主机的 `f2h` 必须与卡上 `cast` 同为 RNE 才位级等价（P1 判据③） |
| 3 | 打分 GEMM | `gemm_int16`（float） | `qk_attention<float16,float16,float16,float>`：A=q5(fp16) B=K(fp16) C=fp16 | `alpha=0.0625` 仍可折进 |
| 4 | softmax | `softmax2d_forward`（**只有 float**，T 符号） | scores 需 `cast<float16,float>` → softmax → `cast<float,float16>`（**+2 launch/层/token**，流量 `(pos+1)×16×4B×2`） | cols=32768 时 = 4 MiB/层/token 的额外 PCIe? **不 —— cast 在卡上做，不过 PCIe** ✓ |
| 5 | PV | `gemm_int16`（float） | `qk_v_attention<float16,float16,float16>` | A=prob(fp16) B=V(fp16) |
| 6 | 评分精度 | f32 输入 | **Q 与 K 都被舍入到 fp16（11 位尾数，相对 ~5e-4）**；softmax 仍在 f32（因为用了 cast） | 这是**本次改动里最大的数值变化**，见 §4.1 |
| 7 | `gemm_int8` 兼容性 | A(staging)=`const float*`、C=`float*` | **不兼容**：`gemm_int8` 收 float A 与 float C ⇒ 它**永远不能直接吃 fp16 KV**；`grp_a`/`grp_mid` 那两组权重投影与 dtype 无关（激活始终是 host f32），**不受影响** ✓ | `:773`/`:836` 两处调用**不用改** |
| 8 | 台账打印 | `:1947-1954` 分母里的 `4.0`、`:2054` | 乘 `sizeof(dtype)` | 不改的话台账会虚报一倍 |
| 9 | 主机兜底路径 | 主机 KV `vector<float>` | **必须整段删除**（不是改类型）：128k 下主机副本来就是 4 GiB ⇒ 不可能存在两份 | §3.5 |

### 3.5 卡上 KV 的三条设计规则（省掉的内存与复杂度）

1. **"截断即回滚"**：KV 是**只追加**的，位置 `pos` 的内容只由 `(token, pos)` 决定。检查点/LCP 复用只要求"前缀完全相同"（`PROGRESS` 第 27.2 章已把命中条件收紧为"LCP == 检查点全长"）⇒ **回溯 = 只把 `pos` 改回去，KV 一个字节都不用搬、不用清**。
   - 直接后果：`g_snap.Kc/Vc`（`:1993-2003`）、`g_ck[].Kc/Vc`（`:2076-2095`）、`CbSlot` 的 KV 拷贝（`:2576-2578`/`:2670-2671`）**全部删掉**，只保留 host 侧的 `conv / S / logits / ids / imgused`。
   - 收益：省掉 512 MiB 快照 + 每检查点 512 MiB + 每槽 512 MiB（@8192）。
2. **永不清零**：读到的最远位置就是本槽/本会话自己的 `pos`（因果性保证）。唯一要注意的是"槽被另一个会话复用"—— 此时新会话的 `pos` 从 0 开始，写入会覆盖，读取不会越界 ⇒ **也不需要 `memset`**。若为了保守还是要清，用 `api::memset`（`xpu_memset` 不存在）；128k 清 4 层的代价 = 2 GiB/芯 ÷ 130 GB/s ≈ 16 ms/次，可接受但不必要。
3. **读范围硬约束**：任何卡上调用里的 `n`（或 `cols`/`k`）**必须 ≤ 该槽自己的 `pos+1`**，禁止用 `MAXT` 当长度去"多算一段"。多算不仅浪费 HBM 带宽（128k 时是把 8.5 ms 变成 34 ms），还会读到别的槽的旧数据 ⇒ 输出错。**在代码里把这个约束写成 assert（`pos+1 ≤ n_used`）。**

---

## 第 4 章 数值风险与验收判据

### 4.1 每一步引入的数值变化（逐项定性 + 量级）

| 改动 | 引入的变化 | 量级（预期） | 对文本答案的影响 | 对视觉答案的影响 |
|---|---|---|---|---|
| KV 搬到卡上（**fp32**，GEMM 若为 `qk_attention<f32>` 真 fp32） | 求和顺序改变（卡上按分块/线程树形归约 vs 主机按 t 升序单累加器） | ~1e-7 相对（1~2 ulp） | **可能翻转个别 token**（因为存在 argmax 恰好接近的情形），但一般"答案内容仍正确" | 视觉塔是混沌的 ⇒ **任何 ulp 级扰动都可能翻转表格/文字读数**；但我们**不动视觉路径**，所以它只受"文本路径改动后的间接影响"（无直接耦合） |
| 同上（若用 `gemm_int16` 内部 int16 量化） | 激活/权重被内部量化到 int16 | ~3e-5 相对 | 一般仍在"内容正确"档 | 同上 |
| 同上（若被迫用 `gemm_int31` 每张量 int8） | 激活与 KV 各一个全局 max ⇒ **小幅度行被大 outlier 拖累** | 0.2%~5%（取决于张量动态范围；视觉塔 FFN 实测过 7.19%） | 有实质风险 ⇒ 只作最后兜底 | ✗ 不可接受（若将来动视觉） |
| **fp16 KV**（阶段④） | Q/K/V 舍入到 11 位尾数（相对 5e-4 ≈ 2^-11）；scores/probs 内部仍 f32 | **5e-4 相对**（是 fp32 方案的 5000 倍） | ★ **这是本次最大的数值变化**：长上下文下分数误差 ~5e-4×|q||k|，softmax 后概率误差同量级 ⇒ 文本答案"可能改措辞/改个别数字" | 不动视觉路径，但它提醒我们：**不要在这个档位上再叠加任何视觉改动** |
| 预填充分块（阶段⑥策略 B） | ① 行并批 ⇒ 每 token 的 GEMM 形状从 m=1 变 m=T（**同一个 per-tensor 量化的边界不同**，若用 int 路径影响更大）；② 加掩码（-1e30 加法，精确无害） | 视算子而定；float 路径 ~1e-7 | 可接受（但**必然不再逐字节**） | — |
| 保留在主机的一切（§2.6） | **零变化** | 0 | 黄金题逐字节的锚点 | — |

**一句话**：位级等价的唯一期望来自"**fp32 KV + 真 fp32 GEMM + 主机侧 O(1) 算子不动**"这一档；其余档位都按"允许不等价"验收（§4.4）。

### 4.2 视觉塔混沌教训（**必须完整继承，不能打折**）

实测证据（`PROGRESS` 19.3 / 第 21 章 / multimodal 文档）：
```
第 0 层 0.0002% 扰动 → layer26 放大到 7%（layer26 有 maxabs≈5000 的离群通道）→ 答案从 K200/表格 9-9 翻成 K2CO/7-9
0.00027%（vgemm_ffn_dn）/ 0.368%（卡上 gelu）同样翻转 ⇒ 行为判据不是灵敏度指标，"relrms 更小" ≠ "答案更好"
```
因此：
1. **本轮不动 `vis.cpp` 一个字节**，也不改任何 `VIS_*` 开关默认值。
2. **每个阶段都要重跑视觉验收**（三图 + 表格 9/9 + `K200` 完整读出 + 换图不复读），当作"文本路径改动的间接影响探测器"。
3. **先文本、后视觉的分批上线**：本设计的 6 个阶段**全部只作用于文本路径**；视觉路径的任何"跟着一起改"的想法（比如顺手把视觉塔的 attention 也搬卡）**一律推迟到文本路径稳定之后单独立项**，并且必须先建"全链路 13 阶段字节对拍台"再动。
4. **验收协议（单次通过不算通过）**：同一批用例**多种请求顺序 × 每个 case ≥10 次**，要求 (a) 内容正确 且 (b) 同输入重发输出 sha1 逐字节相同。只报"某次 9/9"是没有说服力的。

### 4.3 逐阶段验收命令（照单可跑，全部是 VM 上已存在的工具）

> 前置一次性动作（每个窗口开始前）：
> ```bash
> cd /home/caden/ornc
> mkdir .cardlock 2>/dev/null || { echo "卡锁被占（有别的代理在跑）⇒ 停手"; exit 9; }
> cat /proc/xpu/dev0/state /proc/xpu/dev1/state          # 必须 RUNNING / RUNNING
> A0=$(grep -ac 'Exception in kernel execution' /var/log/kern.log)
> R0="$(cat /proc/xpu/dev0/reset_count)/$(cat /proc/xpu/dev1/reset_count)"
> echo "台账基线: 异常=$A0 reset=$R0"                    # 期望 异常=18  reset=0/0
> ```

**(1) 卡状态 / 台账**
```bash
cat /proc/xpu/dev0/state /proc/xpu/dev1/state            # RUNNING/RUNNING
cat /proc/xpu/dev0/reset_count /proc/xpu/dev1/reset_count # 0 / 0（不得新增）
echo "异常累计 = $(grep -ac 'Exception in kernel execution' /var/log/kern.log)"   # 必须仍 = 18
ping -c 3 192.168.66.2                                    # NAS/宿主存活（网关）
```

**(2) 服务健康**
```bash
curl -s --max-time 8 http://127.0.0.1:8090/health
```

**(3) 黄金题（逐字节）** —— 与 `det_install30.sh` 的窗口内冒烟同款
```bash
G=$(curl -s -m 180 http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":32}')
echo "$G"
echo "$G" | grep -q '你好！有什么我可以帮你的吗' && echo "黄金题 OK" || echo "黄金题 ✗"
```

**(4) 6 条文本 + 逐字节对拍**
```bash
python3 /home/caden/ornc/accept6.py 8090 /home/caden/ornc/accept.postATN/accept6.log 64
python3 /home/caden/orn_engine/cmp_txt21.py \
        /home/caden/ornc/accept.postATN/accept6.log \
        /home/caden/ornc/accept.post23/accept6.log
# 判据：逐字节一致（位级档）；不一致时按 §4.4 降级判据（内容正确）
```

**(5) 多轮 LCP（第 2~6 轮 ≤3s）**
```bash
python3 /home/caden/orn_engine/mtsession.py /home/caden/ornc/accept.postATN/mtsession.json
python3 /home/caden/sdnn/mt27.py           /home/caden/ornc/accept.postATN/mt27.json   # 抗"另一个客户端打断"
# 判据：★ LCP 行每轮可见；第 2~6 轮 预填充 ≤3s
```

**(6) 位级等价（冷启动 vs LCP 复用）**
```bash
python3 /home/caden/sdnn/lcp_bit27.py /home/caden/ornc/accept.postATN/lcp_bit27.json
# 判据：★ GEN 三指纹（token ids / 输出字节 / prompt）在冷/复用两路完全相同
```

**(7) 视觉回归（三图 + 表格 9/9 + 换图不复读 + 顺序确定性）**
```bash
bash /home/caden/ornc/accept23.sh          # 三图 + 表格 + 6 条 + 逐字节
python3 /home/caden/sdnn/verify21.py        # 三图 + 黄金题 + 卡状态
python3 /home/caden/sdnn/det_vis.py --order T,G,B   --tag atnN
python3 /home/caden/sdnn/det_vis.py --order B,BS,TB --tag atnNswap
python3 /home/caden/sdnn/det_vis.py --order G,G     --tag atnNrep
python3 /home/caden/sdnn/det_report.py /home/caden/sdnn/det_vis.jsonl
python3 /home/caden/sdnn/det_myverify.py   # 表格图 → 同问句换形状图（必须不复读）
# 判据：三图形色位置全对；表格 9/9；"K200" 完整读出；BS 不得复述 A/1/7…
```

**(8) 长 prompt（DSH 形态）与性能分桶**
```bash
python3 /home/caden/sdnn/longtest.py       # 5742 字符/14 条 ⇒ 必须答 7
# 性能：分桶对比（改前/改后各跑一次，同一批 prompt）
#   env K200_PROF=1 起引擎后，抓 [orn] 的 bucket 行（g_p_* 台账，:2322 附近打印）
K200_PROF=1  ... && grep -a 'ms/tok\|★ 权重量' /home/caden/ornc/serve2.log | tail -20
```

**(9) 收尾（每个窗口必须做）**
```bash
crontab -l | wc -l            # 必须 = 3（svc_guard 必须恢复）
curl -s http://127.0.0.1:8090/health   # ready:true
rmdir /home/caden/ornc/.cardlock
```

**判据汇总（每阶段都要全过，缺一不可）**

| 判据 | 目标 | 硬/软 |
|---|---|---|
| `safe_run` 退出码 | 0（3 = 卡异常 ⇒ 本阶段失败，**严禁重试**） | 硬 |
| 异常计数 | 18 → **18**（零新增） | 硬 |
| `reset_count` | 0/0 → **0/0** | 硬 |
| 两芯 state | RUNNING / RUNNING | 硬 |
| NAS 存活 | `ping 192.168.66.2` 通 | 硬 |
| 黄金题 | 内容正确（位级档要求逐字节） | 硬（内容）/ 软（逐字节） |
| 6 条 | 全过（位级档要求逐字节） | 硬（内容）/ 软（逐字节） |
| 多轮 LCP | 第 2~6 轮预填充 ≤3s | 硬 |
| 三图 / 表格 9-9 / K200 | 不退化 | 硬 |
| 停机窗口 | ≤85s（硬上限 120s） | 硬 |

### 4.4 "允许不等价"的边界（写清楚，免得下一位代理自己发挥）

**文本 = 宽容**（本设计必然改位级，因为注意力求和顺序变了）：
- **必须满足**：黄金题内容正确、6 条答案内容正确且切题、多轮 ≤3s、长 prompt 答 7、无乱码/无重复退化。
- **努力满足（加分项，不作为否决项）**：黄金题与前 8 个 token 逐字节一致。
- **验收口径**：`cmp_txt21.py` 报"逐字节不一致"时，**不得**直接判失败；要看差异是否只是措辞/格式差异（人工读一遍），只要内容正确即通过。**但必须把差异逐条记录进 PROGRESS**（"6 条里第 N 条从 X 变成 Y"），不许只写"通过"。
- **不允许**：答案变短/变空、开始输出思考链、复读前一条回答、明显错答。

**视觉 = 严格**：
- 三图形色位置全对；**表格 9/9**；`K200` 完整读出；换图不复读；同输入重发 sha1 相同。
- **只要有一项退化 ⇒ 立即回滚本阶段**（因为视觉塔数值路径本轮一字节未动，退化只能来自"文本改动污染了共享状态"⇒ 是真 bug，不是噪声）。

**长上下文 = 分档**：MAXT 每次抬高后，除了上面全套，**加一条**：同一段 >8k token 的长上下文下，答案不得退化（`longtest.py` 的长档 + `mtsession.py` 的多轮）；并且 `KV(8 注意层)` 行为 0 MiB（主机不占）。
---

## 第 5 章 实施步骤与窗口计划

### 5.0 硬规矩（抄自用户铁律，违反即事故）

1. **改动/调参/测带宽期间把 8090 停掉，让卡独占**。固定顺序：**停服务 → 改动与测量 → 全改完 → 拉起服务 → 验收**。
2. **停机单次硬上限 120s**（用户铁律），**目标 ≤85s**。历史上有一次 416s 违规，用户当场被 500 打到 ⇒ **严禁在窗口里跑长基准**。
3. **停完必须自己证明"卡真空"**：不信停止脚本自述：
   ```bash
   for d in /proc/[0-9]*/fd/*; do t=$(readlink "$d" 2>/dev/null); case "$t" in /dev/xpu*) echo "$d -> $t";; esac; done
   ```
   有输出 = 还有进程占卡 ⇒ 按 PID 精确 `kill -9`（**绝不用进程名模式**，会漏杀）。
4. **抢卡锁**：`mkdir /home/caden/ornc/.cardlock`（`det_install30.sh` 已有这套），失败就停手（今天已有一次并发上机废卡的先例）。
5. **每个设备测试都包 `safe_run.sh`**：`bash /home/caden/orn_engine/safe_run.sh -n <名字> -t <秒> -- <命令>`；退出码 **3 = 卡异常 ⇒ 判失败且严禁重试**。
6. **一次异常即停**、尺寸**极小 → 中等 → 全量**、**设备内核越界会把宿主 NAS 拖死**（宿主机是用户的生产 NAS）。
7. **两芯 HBM 不能各跑一个引擎**：换件必须先杀旧引擎 + 起新引擎，二选一。
8. **不许编数字**：每轮必须明写"本次卡上异常数"（0 也要写）；没测到的一律写"未测"。

### 5.1 阶段 ⓪：纯主机准备（**不碰卡，0 停机**）

产出（都在 `/home/caden/orn_engine/` 与 `/home/caden/ornc/`）：
| 文件 | 内容 | 独立验收 |
|---|---|---|
| `atn_probe.h` / `atn_probe.cpp` | 探针实现（§2.8 的 P1~P15），编进 `orn3` 的 `K200_ATN=probe` 分支 | 主体机编译通过（`g++ -c`，**不产生设备码**） |
| `atn_cmp.py` | 逐层对拍台：读 `K200_ATN_DUMP` 落盘的每层 `scores / probs / ao / Kc[pos] / Vc[pos]`，与主机路径同层输出比 relrms/maxabs/首个不同位置 | 用**人工构造**的两份 dump 自测（一份完全相同 ⇒ 报 0；一份注入 1e-6 扰动 ⇒ 报出正确 relrms） |
| `atn_install<N>.sh` | 每阶段的单窗口上件脚本（复制 `det_install30.sh` 改名，改 `NEWENG_MD5` 与 `.bak` 名） | `bash -n` 语法检查 + 干跑（`--dry`，只打印不动作） |
| `atn_probe_list.md` | 探针用例清单与阈值（就是 §2.8 表） | — |
| 符号基线 | `nm -DC libxpuapi.so > /home/caden/ornc/shlib_syms_20260923.txt` | 与本文 §2.8 表逐行对照 |

**此阶段不许跑任何 xpu 二进制。**

### 5.2 阶段 ①：极小尺寸探针（1 个窗口，停机 ≤60s）

```bash
# 0) 抢锁 + 台账基线（见 §4.3 前置）
# 1) 停服务（窗口开始计时）
T0=$(date +%s); bash /home/caden/ornc/stop.sh; sleep 3
#    卡真空自检（见 §5.0-3）
# 2) 跑探针（不加载权重 ⇒ 秒级；仍然包 safe_run）
bash /home/caden/orn_engine/safe_run.sh -n atnprobe -t 120 -- \
     env K200_ATN=probe /home/caden/orn_engine/orn3 --atnprobe-json /home/caden/ornc/atn_probe.json
RC=$?; echo "safe_run 退出码=$RC  (0=通过 / 3=卡异常⇒停手)"
# 3) 起服务
bash /home/caden/ornc/start.sh; curl -s -m 5 http://127.0.0.1:8090/health; echo
echo "窗口 = $(( $(date +%s) - T0 ))s  (目标 ≤60s)"
# 4) 台账（异常/reset 必须不变）+ 释放锁
```
**判据**：`RC==0`；异常计数 18→18；reset 0/0→0/0；`atn_probe.json` 里 §2.8 表的每个 case 都达到判据；**P2b（ldb=非连续）与 P2c（max_a/b 只允许 nullptr）的结论决定了后面所有阶段的技术路线**。
**回滚**：不换生产件 ⇒ 无需回滚；探针代码留在 `orn3` 里（`K200_ATN=0` 时完全不执行）。

### 5.3 阶段 ②：单层上卡 + 位级对拍（1 个窗口，停机 ≤120s）

改动（**只改 layer 3**，由 `K200_ATN=1` 打开）：
| 位置（关键词） | 改动 |
|---|---|
| `:1108-1156`（`forward()` 非 recr 分支） | 在 `:1126` 之后插入 `if (atn_on(l)) { atn_decode_layer(...); }`：KV 写卡（B1）、Q 上传（B2）、C1/C2/C3/C4，然后 **skip 掉 `:1132-1151` 的主机注意力循环**，`ao` 由 D2H 结果填充；**`:1152` 的 gate 与 `:1155` 的 gemv 原样执行** |
| 新增 `atn_ctx.h/.cpp` | 卡上缓冲分配（§2.5）、KV 描述符、`atn_decode_layer()` |
| `:1852-1883`（scratch 分配） | 追加 §2.5 的缓冲分配（**必须与生产档同一段代码**，别只在单发档分配 —— T8） |
| `:1935`（`L.Kc.assign`） | 当 `atn_on(l)` 时**不分配主机 KV**，改为 `xpu_malloc` 设备 KV + 打印人类可读的容量账 |
| 新增 `K200_ATN_DUMP` | 每层把 `scores/probs/ao/Kc[pos]/Vc[pos]` 落盘（主机侧 `fwrite`） |

```bash
# 窗口内（离线模式，不占 8090 但占卡 ⇒ 仍需停机窗口）
K200_ATN=1 K200_ATN_DUMP=/tmp/atn/card  ./orn3 --gen "北京的省会" --n 16   # 卡上档
K200_ATN=0 K200_ATN_DUMP=/tmp/atn/host  ./orn3 --gen "北京的省会" --n 16   # 主机档
python3 /home/caden/ornc/atn_cmp.py /tmp/atn/host /tmp/atn/card --layers 3
```
**判据**：layer 3 的 `Kc[pos]/Vc[pos]` **逐位相同**（KV 写入无错位）；`scores` relrms ≤ 1e-5；`probs` relrms ≤ 1e-5；`ao` relrms ≤ 1e-3；**output token ids** 与主机档一致（不一致时按 §4.4 降级判"内容正确"）。
**回滚点**：`cp -a orn3 orn3.preATN2.bak; md5sum > .md5`；回滚一行 `cp -a orn3.preATN2.bak orn3 && bash /home/caden/ornc/restart.sh`。

### 5.4 阶段 ③：8 层全上（单窗口上件，停机 ≤85s）

改动：`K200_ATN=2`；`atn_on(l)` = 所有注意层；**三处路径同步**（`:1108`、`:1330`、`:1610`，策略 A）；`CbSlot` 的 KV 改为设备缓冲（`:1438`/`:2517-2519`）；**删掉 §3.5 列的 KV 拷贝**。
```bash
bash /home/caden/ornc/atn_install3.sh      # 单窗口：抢锁→备份→停→换件→起→窗口内黄金题→失败自动回滚
# 窗口内：黄金题（见 §4.3-3）
# 窗口外（服务在线）：§4.3 的 (4)(5)(6)(7)(8) 全套
```
**判据**：§4.3 判据汇总全过；额外要求 `主机常驻 ... KV(8 注意层) 0.0 MB`；`K200_PROF=1` 分桶里"注意力"桶相对改前明显下降（记录实测数字）。
**回滚点**：`orn3.preATN3.bak`（md5 记录在脚本里，脚本自动回滚）。

### 5.5 阶段 ④：KV 换 fp16（**前置：阶段① 的 `qk_attention<f16>` 探针通过**；单窗口，≤85s）

改动：`K200_KVD=f16`；§3.4 表的 9 项；`cast` 的 2 次额外 launch（softmax 前后）。
**额外的独立判据**：
```bash
# KV 位级：写 1024 个随机 f32 进卡，卡上 cast→f16，D2H 回来与主机 f2h 逐位比
bash /home/caden/orn_engine/safe_run.sh -n atnf16 -t 60 -- env K200_ATN=probe-f16 ./orn3 --atnprobe-json /home/caden/ornc/atn_probe_f16.json
```
判据：主机 `f2h` 与卡上 `cast<float,float16>` **逐位相同**（否则记录 ULP 差异，并按"文本宽容"验收）；§4.3 全套。
**回滚点**：`orn3.preATN4.bak`。

### 5.6 阶段 ⑤：抬 MAXT（分三级，每级 1 窗口，各 ≤85s）

顺序 **16384 → 32768 → 131072**（131072 只在阶段④ 通过后才做）。
每级同窗口内要改两处 env（**必须同步，否则长上下文根本进不来**）：
```bash
K200_MAXT=32768            # 引擎侧（serve2.py:515 传给引擎）
K200_ENGINE_MAXT=32768     # 网关侧（serve2.py:50，必须与引擎一致）
K200_PROMPT_BUDGET=32000   # 网关裁剪预算（serve2.py:49）★ 默认 7800 会把长上下文【丢头裁掉】
```
★ 关于预算：网关 `_TRIM`（`serve2.py:215`）按"估算 token"裁，**默认 7800 估算 tok**。要真正吃到 32k/128k，预算必须同步抬高；但**预算抬高的代价是预填充变慢**（ms/tok × tok 数 vs 客户端 30s 超时）⇒ 必须用"实测 ms/tok × 预算 ≤ 客户端超时"来定，**不要一次拍 128k**。建议：32768 档预算 25000；131072 档先只做"守门"（预算仍 25000，只验证能力），等阶段⑥（分块预填充）把 ms/tok 压下来再放开。
每级额外验收：`longtest.py` 长档 + `mtsession.py` 多轮 + **同一段 >8k 上下文下答案不退化**。
**回滚点**：每级各留 `orn3.preATN5_<MAXT>.bak` + `serve2.preATN5_<MAXT>.bak`。

### 5.7 阶段 ⑥：预填充分块（策略 B，单窗口，≤85s）

改动：`K200_ATN=3`；`forward_batch()` 的 `:1330` 循环改为"按 chunk 处理"；新增掩码生成链（§2.4，4 次 launch/层/chunk）；`g_mWT`（`:1488`，@131072 是 512 MiB ✗）改成分块/按需分配。
**判据**：§4.3 全套 + `K200_PROF=1` 分桶里"launch/归一化"桶显著下降、ms/tok 相对阶段③明显下降。
**回滚点**：`orn3.preATN6.bak`。

### 5.8 回滚脚本模板（每阶段复制一份，只改 `<N>` 与新件 md5）

```bash
#!/bin/bash
# atn_rollback<N>.sh —— 回滚到本阶段之前的生产件（幂等，随时可跑）
set -u
DD=/home/caden/orn_engine; OO=/home/caden/ornc
bash $OO/stop.sh >/dev/null 2>&1; sleep 2
cp -a $DD/orn3.preATN<N>.bak $DD/orn3
cp -f $OO/serve2.preATN<N>.bak $OO/serve2.py
bash $OO/start.sh
sleep 2
echo "回滚后 orn3=$(md5sum $DD/orn3|cut -d' ' -f1) 网关=$(md5sum $OO/serve2.py|cut -d' ' -f1)"
curl -s -m 5 http://127.0.0.1:8090/health; echo
```
★ **备份必须含 md5 记录**（`md5sum orn3.preATN<N>.bak | tee .md5.preATN<N>`），验收报告里逐条列出"本阶段回滚点 = md5"。

---

## 第 6 章 不确定点与需实验回答的问题（交给实现代理先试）

> 按"不回答就不能继续"的先后排序。每条都给了**怎么试**和**答案会影响什么决策**。

| # | 问题 | 怎么试 | 答案影响的决策 |
|---|---|---|---|
| **U1** | `qk_attention<float,float,float,float>` 到底能不能用？（语义 + 数值） | P2a（四种转置组合打表）→ P2c（max 指针只传 nullptr）→ P2d（is_asr_mask）→ P2f（真实量级 relrms vs fp64 参考） | 决定打分/PV 用 `qk_attention` 还是 `gemm_int16`。**若两者都不能用 ⇒ 整个方案要重新设计（见 U7）** |
| **U2** | `qk_attention<float16,float16,float16,float>` 能不能用、精度多少？ | 把 U1 的全套用例在 f16 上重跑一遍 + 与 f32 版对拍 | **决定 128k 能不能做**（§3.4 硬依赖链）。不能用 ⇒ 128k 作废，最高档停在 32768(fp32) |
| **U3** | `gemm_int16` 精度与 **非连续 B 的行步长（ldb=1024）** | P4a / P4b / **P2b** | P2b 失败 ⇒ KV 要改成 per-kh 连续缓冲（+7 launch/层/token，且 §2.2 布局规则要改写） |
| **U4** | 卡上 softmax 对"64 KB 级张量"（rows=16, cols=1024~32768）划不划算？ | P7b（计时）+ 判据"耗时 ≤ 3× 读该张量时间" | 不划算 ⇒ 阶段⑥ 改"按 kh 分 4 次 rows=4"，或把 softmax 也纳入自写内核的候选（见 U7） |
| **U5** | fp16 KV 的舍入对**文本答案**的实际影响 | 阶段④：同一批 6 条 + 黄金题，f32 KV 档 vs f16 KV 档逐题对比；并落盘 `scores/probs` 的 relrms | 影响"128k 能不能上线"和"是否要走 fp16 但把 Q 保留 f32"（后者需要 A 通道 f32、B 通道 f16 的混合算子—— **库里没有** ⇒ 若 f16 影响不可接受，128k 就要放弃） |
| **U6** | `reduce(MAX)` 有符号陷阱在本链里有没有被踩到 | 审代码：本设计里 `max|A|` 只在（若走）`gemm_int31` 时需要。用 P5 的 `max_a=0` 用例验证"全零 A 不除零" | 决定是否走 `gemm_int31` 兜底；**也决定将来若把 RMSNorm 搬卡时怎么写（T2）** |
| **U7** | 要不要自写 attention 内核（兜底方案） | 只有当 U1 **且** U3 **都失败**时才启动。写法：fp32 `scores = q·K^T`，`cl=8, co=16`，每 cluster 处理一批 t，**每线程 4~8 条独立累加链**（坑 2 的 ILP 要求），本地内存预算 **8712 B/核**（用 `gm2lm` 把 K 的一块搬进本地）；**launch 前做完整边界检查**，从小尺寸逐级放大 | 自写内核是历史事故（宿主死机）的直接原因 ⇒ **优先级最低**，且必须先有 `safe_run` + 看门狗 + 卡锁三件套 |
| **U8** | 官方算子全不可用时，能否用**已验通算子**拼出 attention | ★ **可以，但只在短上下文可行**：`matrix_vector_mul`（T 符号，`func_dec.h:207`）能一次算一个 KV 头的 `K_kh[pos+1,256]·q[256]` ⇒ 16~20 次 launch/层/token 拿到全部 scores（不随 pos 增加 launch 数 ✓）；PV 用 `gemm_int16` 或 `matrix_vector_mul` 配 `V^T`（**V 需转置存储**，否则 `transpose` 一次 `[256,pos+1]`）。scores 要 softmax 仍靠 `softmax2d_forward`。⇒ **这条兜底链的 launch 数从 13 升到 ~40/层/token**（≈ 5.1 ms/token），**不满足长上下文目标但能保证方案不"无解"** | 作为 U7 之前的第二兜底；必须先探针 `matrix_vector_mul` 的语义（矩阵是 `[M,K]` 还是 `[K,M]`） |
| **U9** | `layer_norm` n>1024 不能用（已实测），**per-head RMSNorm(256) 若要搬卡**怎么拼 | `mul(x,x)` → `reduce(MEAN, dim=1)` → `elementwise_pow(-0.5)` 或 `scale+sqrt` → `mul(w 广播)`：**5 次 launch / 20 头可并成 5 次**（按 [20,256] 批） | 影响"口径 B（严格 = 全部数值在卡上）"的可行性；**默认不做** |
| **U10** | 卡上 sigmoid（`activation_forward`）是不是表驱动、能不能用 | P13 | 决定 gate 是否可能搬卡。**默认不搬**（主机 gate 是位级锚点） |
| **U11** | 主机 `f2h()` 与卡上 `cast<float,float16>` 是否同为 RNE | P1 判据③ | 决定 fp16 KV 的转换放主机（0 launch）还是卡上（+2 launch/层/token） |
| **U12** | 官方算子在**多线程同时调用**（OpenMP 多槽并行）下是否安全 | 现状 `forward_multi` 是 `#pragma omp parallel for` 并行多槽（`:1639`）。卡上一个 `api::Context` 是否可并发提交？ | 影响多槽路径的实现（可能要改成串行提交、批量等一次，或每槽一个 Context）。**这也是与"并发"验收直接相关的一条** |
| **U13** | 128k 时卡上 KV 读带宽在**薄 GEMM（m=4）**下实际能到多少 | P4c（N=16384/32768 档） | 决定 §2.3 的 60 GB/s 假设是否成立；低于 20 GB/s ⇒ 解码要改成"多 token 合批"才能用满 |

---

## 第 7 章 附录

### 7.1 关键锚点表（`orn3.cb30.cpp`，2957 行，md5 `e6672810672a30079b59a2f836d1694f`）

| 行号 | 关键词（用它重新定位） | 内容 |
|---|---|---|
| 385–389 | `static const int EMB = 4096` | 形状常量 / `ROPE_BASE` |
| 395 | `static int MAXT = 8192` | 上下文上限（默认值；运行期 `K200_MAXT` 可覆盖） |
| 397 | `is_recr` | 层型判定 |
| 417–418 | `g_ctx[2]` / `g_hbm[2]` | api::Context / HBM 记账 |
| 420–432 | `struct Layer` | `:430` `std::vector<float> Kc, Vc;` |
| 497 / 555 | `FATAL: HBM 分配` | 分配失败的报错点（会 exit） |
| 539–564 | `upload_i8_host` | 权重行分裂上卡 |
| 629 | `g_dx[2]` | 设备 scratch 指针 |
| 738 / 773 / 836 | `api::gemm_int8` | 三处官方 GEMM 调用（单发 / 解码 / 批量） |
| 764–799 | `gemv_i8` | 解码用双芯 GEMM 包装 |
| 814–860 | `gemv_batch` | 预填充批量 GEMM（含 M0 行归一化） |
| 862–882 | `static void gemv(` | 分发 |
| 904–919 | `rope_init` / `rope_apply` | MRoPE |
| 924–929 | `struct State` | conv / S / Kc / Vc |
| 938–942 | `g_pKc[NLAYER]` | KV 指针间接化 |
| 954–960 | `AA_Q` / `AA_K` / `AA_V` / `AA_TOT` | 注意力拼接段偏移 |
| 1108–1156 | `} else {` + `kk.resize(NKV * HD)` | **单 token 全注意力（要改）** |
| 1195–1422 | `forward_batch` | 预填充（`:1330` 循环、`:1349-1370` 注意力） |
| 1424–1461 | `struct CbSlot` | 槽位（`:1438` KV 字段） |
| 1467–1780 | `forward_multi` | 多槽（`:1610-1668` 注意力、`:1488` `g_mWT`） |
| 1788 | `int main(` | 入口（探针模式挂这里） |
| 1852 / 1882 | `xpu_malloc(&g_dx[dv]` | 设备 scratch 分配（单发档 / 生产档） |
| 1886 | `new api::Context` | Context 构造 |
| 1935 | `L.Kc.assign` | KV 分配（**要改**） |
| 1947–1954 | `权重常驻 HBM` | 启动台账打印（**要改 4.0 分母**） |
| 1957 | `g_pKc[l] = &g_lay[l].Kc` | 指针指向 |
| 1993–2003 | `g_snap.Kc[l]` | 快照（**要删 KV 部分**） |
| 2040–2100 | `g_ck` / `ck_restore` | 检查点（`:2054`、`:2076-2078`、`:2094-2095`） |
| 2351 / 2410 | `memset(g_lay[l].Kc.data()` | KV 清零 |
| 2517–2539 | `S.Kc.resize` / `S.Kc[(size_t)l].assign` | 槽位 KV 分配 |
| 2576–2578 / 2670–2671 | `C.Kc[(size_t)l].assign` / `memcpy(S.Kc` | 槽位检查点存取 |
| 2322 / 2830 | `权重量 %.2f GiB/token` | 台账打印 |

### 7.2 关键算子签名（**全部从 `nm -DC` + 头文件核实，不是猜的**）

```cpp
// ---- 打分 / PV 首选 ----
// include/xpu/func_dec.h:2483
template<typename T_a, typename T_b, typename T_c, typename T_bias>
int qk_attention(Context* ctx, bool TransA, bool TransB,
                 int batch_size0, int batch_size1, int M, int N, int K,
                 float alpha, const T_a* A, const T_b* B,
                 float beta, T_c* C, const T_bias* bias,
                 const float* max_a, const float* max_b,
                 float* res_max_ptr, bool use_max_ptr_4,
                 bool is_asr_mask, int bias_ldn,
                 bool is_tf_bias = false, bool is_onnx_bias = false);
// 实例化符号（实测存在）：<float,float,float,float>、<float16,float16,float16,float>
// ★ max_a/max_b/res_max_ptr 是【指针】⇒ 属 *_maxptr 家族 ⇒ 只允许传 nullptr；否则放弃本算子

// func_dec.h:2492
template<typename T_a, typename T_b, typename T_c>
int qk_v_attention(Context* ctx, bool TransA, bool TransB,
                   int bs0, int bs1, int m, int n, int k,
                   float alpha, const T_a* A, const T_b* B,
                   float beta, T_c* C,
                   const float* gm_max_a, const float* gm_max_b,
                   float* res_max_ptr, bool use_max_ptr_4);
// 实例化符号：<float,float,float>、<float16,float16,float16>

// ---- 纯 float GEMM（T 强符号，最有望"无损"）----
// func_dec.h:741  /  nm 实测
int gemm_int16(Context* ctx, bool TransA, bool TransB, int M, int N, int K,
               float alpha, const float* A, int lda, const float* B, int ldb,
               float beta, float* C, int ldc);

// func_dec.h:1050（★ max_a/max_b 是【标量】，不是指针 ⇒ 比 *_maxptr 安全得多）
int gemm_int31(Context* ctx, bool TransA, bool TransB, int M, int N, int K,
               float alpha, const float* A, int lda, const float* B, int ldb,
               float beta, float* C, int ldc, float max_a, float max_b);

// ---- 逐元素 / 归约（已极小尺寸验通）----
int softmax2d_forward(Context* ctx, const float* x, float* y, int rows, int cols, bool use_sdcdnn = false);  // :1387
int reduce(Context* ctx, const float* x, float* y, const int* xdims, int ndim,
           const int* reduce_dims, int rdim, ReduceOp op);                                                   // :539
int elementwise_mul_2d(Context* ctx, const float* x, const float* y, float* z,
                       int m1, int n1, int m2, int n2);   // 广播规则：m1==m2 或其一为 1；n1==n2 或其一为 1  // :266
int elementwise_add_2d(...); int elementwise_sub_2d(...); int elementwise_div_2d(...);
int elementwise_div_no_nan_2d(...);   // y=0 时输出 0   ← 分母夹逼的替代（P9③）
int elementwise_max_2d(...);          // ★ 用于【夹逼】是对的（逐元素）；不要拿它当"max|A|"
int elementwise_max(Context*, const float*, const float*, float*, int);
int scale(Context* ctx, int len, float alpha, float beta, int bias_after_scale, const float* x, float* y);      // :61
int clip(Context* ctx, const float* x, float* y, int len, float min_val, float max_val);                         // :508
int transpose(Context* ctx, const float* x, float* y, const int* shape, const int* permute, int ndim);           // :1639
template<typename In,typename Out> int cast(Context* ctx, const In* in, Out* out, int len);                      // :526
int memcpy_device(Context* ctx, void* dst, const void* src, int len);
int memset(Context* ctx, void* ptr, int value, size_t size);        // xpu_memset 不存在，只有这个  // :41
int matrix_vector_mul(Context* ctx, const float* matrix, ...);      // :207  U8 兜底用，语义待探针
int slice_forward(Context* ctx, const int* ...);                    // :1763
int l2_normalize(Context* ctx, const float* in, float* out, float* norm, float eps, int m, int t, int n, bool is_infer); // :2499
```

### 7.3 禁用清单（本机实测过后果，别再试）

| 禁用项 | 后果 |
|---|---|
| `gemm_int8_maxptr` / `gemm_int16_maxptr` / `gemm_int31_maxptr` / 任何 `*_maxptr`（含 `gemm_strided_batched_int31` 的尾部指针） | SD 侧 `reason=RBRESP` ⇒ dev0 掉 ERROR |
| `api::fc<float,float,float,int>` | 触发 `findmax_sdcdnn` ⇒ dev1 ERROR |
| `gemm_int8` 的 **float-B 重载**（无 max_b 参数那个） | 内部必然 findmax ⇒ 同 `findmax_sdcdnn` 风险 ⇒ **不进探针队列** |
| `block_gemm_int8` | **库里没有符号**（只有头文件声明）⇒ link 失败 |
| 卡上 `api::gelu` | 与 libm `erff` 不位级一致（0.01196%），会让视觉塔翻转 |
| `layer_norm` 且 `n > 1024` | 实测 `r=1 INVALID_PARAM` |
| 自写探针内核测带宽 | 历史宿主机死机的直接原因（14 次异常 → 宿主重启） |
| `xpu_memset` | 不存在（用 `api::memset`） |
| 解引用 HBM 指针 | 段错误 / 卡异常 |
| 拿 `bw`/`br`/`rw` 当带宽工具 | 它们是寄存器读写工具（会被读成错误带宽） |

### 7.4 本设计新增/修改文件清单（供下一位代理建目录时对照）

**新增**
```
/home/caden/orn_engine/atn_probe.cpp / atn_probe.h   §2.8 的 P1~P15（编进 orn3 的 K200_ATN=probe 分支）
/home/caden/orn_engine/atn_ctx.cpp   / atn_ctx.h     设备缓冲与 KV 描述符；atn_decode_layer() / atn_prefill_chunk()
/home/caden/ornc/atn_cmp.py                          逐层对拍台（读 K200_ATN_DUMP）
/home/caden/ornc/atn_install<N>.sh                   每阶段单窗口上件（复制 det_install30.sh）
/home/caden/ornc/atn_rollback<N>.sh                  每阶段回滚（§5.8 模板）
/home/caden/ornc/atn_probe.json                      P1~P15 结论（阶段①产物，务必入库）
/home/caden/ornc/shlib_syms_20260923.txt             nm 符号基线
```
**修改**：`orn3.cb30.cpp`（→ 建议另存 `orn3.atn1.cpp`，别原地改生产源码）；`serve2.py`（仅 env 与预算，**不要动 `_TRIM` 语义**）；`buildorn3cb30.sh`（改名 `buildorn3atn.sh`，保留 `vis23.o` 的 `vis_forward=0x52f4` 硬校验 —— **这条校验必须留，它挡过一次视觉塔整体退化**）。

### 7.5 环境开关（全部默认关 ⇒ 不带 env 启动时行为与现在逐位相同）

| env | 取值 | 含义 |
|---|---|---|
| `K200_ATN` | `0`(默认) / `1` / `2` / `3` / `probe` / `probe-f16` | 0=主机路径；1=只 layer 3 上卡；2=8 层全上；3=+预填充分块；probe=只跑探针 |
| `K200_KVD` | `f32`(默认) / `f16` | KV 数据类型 |
| `K200_ATN_SPLIT` | `4:4`(默认) / `3:5` | chip0:chip1 的注意层分配数 |
| `K200_ATN_DUMP` | 目录路径 | 每层落盘 `scores/probs/ao/Kc/Vc`（对拍用；**默认关**，落盘量随 MAXT 线性增长） |
| `K200_ATN_MASK` | `0`/`1` | 阶段⑥ 的卡上掩码（1 = 开） |

### 7.6 完工定义（Definition of Done）

- [ ] `atn_probe.json` 里 §2.8 表**每一行**都有结论（含"不可用"的明确结论），且异常计数 18→18、reset 0/0→0/0。
- [ ] 阶段③ 上线后：§4.3 判据汇总全过；`主机常驻 ... KV(8 注意层) 0.0 MB`；`K200_PROF=1` 分桶数字**改前/改后**都记进 PROGRESS。
- [ ] 阶段⑤ 每一级 MAXT 都有独立验收记录；`K200_MAXT` 与 `K200_ENGINE_MAXT` 一致；长 prompt 实测记录。
- [ ] 所有回滚点的 md5 写进报告；每个停机窗口的秒数写进停机台账（目标 ≤85s，硬上限 120s）。
- [ ] **诚实边界**：报告里明写"本次未达成项"（例如"个位数 ms/tok 需要 delta-net 搬卡，未做"），以及"卡上异常数"（0 也要写）。
