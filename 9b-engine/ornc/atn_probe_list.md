# atn_probe 用例清单与判据（阶段① 施工表）

> 载体 = `/home/caden/orn_engine/atn_probe`（独立主机二进制，**不加载权重、不替换任何生产件**）
> 构建 = `bash /home/caden/orn_engine/build_atn_probe.sh`（纯主机编译，**不产生设备码**）
> 执行 = `bash /home/caden/ornc/atn_probe_window.sh`（单窗口，含全部守卫；`--dry` 先干跑）
> 结果 = `/home/caden/ornc/atn_probe.json`

## 纪律（不可协商）

1. 每个用例都跑在 `safe_run.sh` 里；**任何一次卡异常（退出码 3）→ 立即停手、本阶段判失败、严禁重试**。
2. 尺寸阶梯 **极小 → 中等 → 目标**；任一档失败**立即停**，绝不放大尺寸重试。
3. **危险用例默认不跑**，必须 `ATN_DANGER=1` 且**单独窗口、排在最后**。
4. 每次跑前后对照 `grep -ac "Exception in kernel execution" /var/log/kern.log` 与 `/proc/xpu/dev{0,1}/reset_count`。
5. 开窗口前先 `bash atn_probe_window.sh --selftest`（验证安全网"检测→立即杀→判失败"闭环，完全不碰卡）。

## 用例表

| ID | 目标 | 尺寸阶梯 | 判据 | 失败怎么办 |
|---|---|---|---|---|
| P1 | `cast<float,float16>` / `cast<float16,float>` 往返 | 1024 → 65536 → 1048576 | ①`r==0` ②往返 relrms ≤0.1% ③**卡上 cast 的 f16 位型与主机 RNE 逐位相同**（决定 fp16 转换放主机还是卡上） | 主机 `f2h` 转换（0 launch） |
| P2a | `qk_attention<f32>` **转置语义** | A[2,3]·B[3,4]，四种 `(TransA,TransB)` 全打表 | **恰好一种**组合 relrms<1e-4% 且等于 `A·B` | 换 `gemm_int16` |
| P2b | `gemm_int16` **非连续 B 行步长 ldb=1024**（本设计一号依赖） | N=64 → 512 → 4096 | 非连续版与连续版**逐位相同**（relrms ≤1e-5%） | 退 per-kh 连续缓冲（+7 launch/层/token） |
| P2c | `qk_attention` 的 `max_a/max_b` | 先 `nullptr`（安全）；非空版本归 `danger` | ★**只允许 nullptr 通过**；必须非空才可用 ⇒ 放弃该算子 | 换 `gemm_int16` |
| P2d | `is_asr_mask` 语义 | 上三角置 +1e6，`0/1` 各跑 | 信息性，两种输出打表人工判；**我们不用它** | 自己上掩码 |
| P2e | `bias=nullptr / is_tf_bias / is_onnx_bias` 全 false | 同 P2c | `r==0` | 保持 false 恒值 |
| P2f | `qk_attention<f32>` **真实量级精度** | P=64 → 512 → 8192 | vs fp64 主机 relrms **≤1e-3%** 合格 | 换 `gemm_int16`/`gemm_int31` |
| P3 | `qk_v_attention<f32>`（PV） | P=64 → 512 → 8192 | 同 P2f | 只作备选 |
| P4a | `gemm_int16` 基础 | M=4,N=64,K=256 | relrms ≤1e-3% | 弃用 |
| P4b | `gemm_int16` 边界 | N=1（pos=0）/ K=1 / M=1 | `r==0` 且值正确 | — |
| P4c | `gemm_int16` **薄矩阵有效带宽** | N=1024 → 4096 → 16384 → 32768，各 50 次 | 记录 GB/s；**<20 GB/s ⇒ §2.3 的 60 GB/s 假设不成立** | 改为多 token 合批 |
| P5 | `gemm_int31`（标量 max） | N=512 → 4096 | relrms ≤1% | 弃用 |
| P5b | `gemm_int31` **max_a=0**（危险） | N=512 | `r==0` 且输出全 0（**不能出 inf/nan**） | 一律改夹逼 |
| P7a | `softmax2d` 薄形状正确性 | cols=1 → 2 → 64 → 1024；rows=4/16/64 | relrms ≤1e-3%；cols=1 恒 1.0 | 用 `reduce(SUM)+div` 自拼 |
| P7b | `softmax2d` **耗时**（64KB~2MB 级） | cols=1024 → 4096 → 16384 → 32768 ×20 | **耗时 ≤ 3× 读该张量时间**（@100GB/s） | 改"按 kh 分 4 次 rows=4" |
| P7c | `softmax2d` **原地** y==x | rows=16, cols=1024 | `r==0` 且结果正确 | 分开缓冲（已按分开设计） |
| P7d | `softmax2d` **全 -1e30 行**（★危险，最后跑） | rows=4, cols=1024 | 记录结局（FP_DIV0 打死 session 是**预期内**）。设计约束：永远不产生全掩行 | — |
| P9-raw | `elementwise_div_2d` 分母含 0（★危险） | len=4096 | 记录三种结局；前两种不可接受 | — |
| P9-clamp | 同输入**先夹逼** `max(y,1e-30)` 再除 | len=4096 | `r==0`、无 nan/inf、结果 == `x/max(y,1e-30)` | 一律走这条 |
| P9-nonan | `elementwise_div_no_nan_2d` | len=4096 | `r==0` 且 y=0 处输出 0 ⇒ **优先用它（省一次 launch）** | 走 clamp |
| P10 | `transpose` | [1,256]、[4,256] | 与主机转置**逐位相同** | 主机转置 |
| P11 | `memcpy_device` D2D | 1 KiB / 8 KiB / 1 MiB | 逐字节相同 + 记 µs/次 | 每 kh 直接 H2D |
| P12 | `api::memset`（`xpu_memset` 不存在） | 4 KiB | `r==0` 且全 0 | — |
| P13 | `activation_forward(SIGMOID)`（**仅当考虑把 gate 搬卡**） | len=4096，值域 [-20,20] | 与主机 `sigmoid` 的最大绝对差；**表驱动 ⇒ 预计非 0** | **保持 gate 在主机**（本设计默认） |
| P14 | H2D/D2H 小传输固定开销 | 1 / 8 / 16 / 64 KiB ×100 | 记 µs/次（16µs/次 假设的验证点） | 调块大小 |
| P15 | `slice_forward`（可选） | [1,1024] 取前 256 | 与主机切片逐位相同 | **直接给基址偷懒（推荐）** ⇒ 本项如实记"未采用" |
| **P16a** | **`qk_attention<float16,float16,float16,float>`**（fp16 KV 的**硬依赖**） | A[2,3]·B[3,4] 四种转置 | 存在组合等价 `A·B` ⇒ fp16 路线成立 | **作废 ⇒ MAXT 上限停在 32768（fp32）** |
| **P17** | `matrix_vector_mul` **语义核实**（本轮新增） | matrix[2,3]·vec[3] | 与 `out=m*v 广播` 逐位相同？ | 若确认是广播乘 ⇒ **设计书 §2.9 的 U8 兜底链作废** |

## 本轮（阶段⓪）已完成的静态结论（不依赖卡）

| 项 | 结论 | 证据 |
|---|---|---|
| 全部目标算子的签名 | 与 `func_dec.h` 逐字一致 | 探针**首次编译+链接即通过**（`build_atn_probe.sh`） |
| `qk_attention<f32>` / `<f16>`、`qk_v_attention<f32>` / `<f16>` | 符号存在且**可链接** | `nm -DC atn_probe` 中均为 `U int baidu::xpu::api::qk_attention<...>` |
| `gemm_int16` / `gemm_int31` | 强符号 `T`，可链接 | 同上 |
| `block_gemm_int8` | 库里**只有 `cpu_mock::block_gemm_int8`**，真实 `api::` 版无符号 ⇒ **链接会失败** | `grep block_gemm shlib_syms_20260923.txt` = 1 条，且是 `cpu_mock::` |
| 探针未引用任何禁用项 | ✓ 无 `*_maxptr`、无 `findmax`、无卡上 `gelu`、无 `layer_norm` | `nm -DC atn_probe | grep -E '_maxptr|findmax|block_gemm|api::gelu|layer_norm'` 为空 |
| 探针不产生设备码 | ✓ 无自写内核 | `nm -C atn_probe | grep -E 'xpu_module|kernels|\.xpu'` 为空 |
| `matrix_vector_mul` 语义 | ★**头文件原文 `out[i*n+j] = matrix[i*n+j] * vec[j]` ⇒ 是逐元素广播乘，不是矩阵乘** ⇒ 设计书 U8 兜底链作废 | `func_dec.h:203-207` |
| 设计书符号表 | 逐条复核**全部成立**（仅 `block_gemm_int8` 一条被细化） | `/home/caden/ornc/shlib_syms_20260923.txt` |

## 尚未执行（需要卡窗口）

P1~P17 的**全部运行期结论**（可用性 / 耗时 / 真实数值误差）**均未取得** —— 因为窗口打开时卡被占（见 PROGRESS 第31章）。
拿到窗口后按 `atn_probe_window.sh`（safe 一窗、danger 一窗）执行，结果回填本表。
