# DESIGN —— Ornith-1.5-9B (Q8_0) 在昆仑 K200 上的推理（一页）

> 现行目标（用户钉死）：**端到端跑通、出正确的话**。速度暂缓。
> 纪律顺序：①宿主稳定 ②正确性 ③速度。任何设备运行必须过 `safe_run.sh`；出一次卡异常就整体停。

## 0. 现状一句话

8090 单端口已在跑 Ornith（引擎在安全网下、强制关思考），黄金测试通过：
`你好` → **`你好！有什么我可以帮你的吗？😊`**（与 llama.cpp CPU 基准一字不差），耗时 71.4s（0.126 tok/s）。

## 1. 机器与资产

| 项 | 值 |
|---|---|
| 卡 | 昆仑 K200 = Kunlun1(kl1)，**双芯**（dev0/dev1），每芯实测可用 HBM 7488MB |
| 驱动/SDK | 4.33.0，`/usr/local/xpu-4.33.0`；XTDK clang: `/home/caden/xtdk/xtdk-x86_64/bin/clang` |
| 引擎二进制（已验证正确） | `/home/caden/orn_engine/orn`（内核 `kq8.xpu`） |
| 服务 | `/home/caden/ornc/serve.py` → 8090（OpenAI 兼容 + `/health` + `/v1/models`） |
| 安全网 | `/home/caden/orn_engine/safe_run.sh`（自测通过；见 §5） |
| 模型 | `/home/caden/orn/Ornith-1.5-9B-Q8_0.gguf`（9.79GB） |
| 机制文档 | skills: `devops/kunlun-k200-setup/references/ornith-qwen35-mechanism.md` |

## 2. 架构（qwen35 混合）

- 33 层（32 主干 + 1 层 MTP 跳过）；`full_attention_interval=4` ⇒ **24 层 gated-delta-net 线性注意力 + 8 层全注意力**
- n_embd 4096 / n_ff 12288 / heads 16 / kv 4 / head_dim 256 / rope 64 / theta 1e7
- 442 张量 = 258 Q8_0 + 184 F32；`add_bos=0`，`eos=248046`
- 每层**没有 ffn_norm**，有**两个残差**；Q/gate **按头交错**；per-head RMSNorm；
  MRoPE sections `[11,11,10,0]`；GDN 递推带 conv1d(kernel=4) + delta 状态

## 3. 张量与量化布局

- Q8_0 = `{fp16 d; int8 qs[32]}` = **34B / 32 权重**，内存里就是 `[out, in]` 行优先
  ⇒ **零重排可直接常驻 HBM**（不要重打包，除非速度优化阶段）
- 双芯分层：按层字节均衡，chip0 ≈ 5096MB / chip1 ≈ 3970MB（合计 8.85GiB 常驻）
- 每 token 读 9.2GB 权重 ⇒ 读带宽是瓶颈（不是算力：18 GFLOP ≈ 18ms）

## 4. 铁律（踩过的坑，必须沿袭）

1. **HW_CORE = 16**：`core_num()` 是"申请值"，硬件每 cluster 只有 16 个核真执行
   ⇒ `co > 16` 时大部分输出行**从不写入**（静默错误，实测 7/8 行是旧的哨兵值）。
   修法：内核里线程号用 `cluster_id()*16 + (core_id()&15)`，启动 `co ≤ 16`。
2. 设备内核**必须在主机侧 launch 前做完边界检查**（addr/ksz/行列数/缓冲区长度 逐一对分配尺寸），下标自己先算清。
3. 尺寸逐级放大：极小 → 中等 → 全量；**不要一步到位**。
4. **不要再写探针内核**（历史卡异常元凶：mbench/mb3/probe_new/probe4/m_lm32/m_lm16 = 14 次异常）。
   测带宽只用厂商工具或已验证内核。
5. 旧 `orn.cpp`(56KB，多代代理叠加) 与 `mb/` 探针 = **只当参考/反面教材**，不再往里加东西。

## 5. 安全网 safe_run.sh（为什么这么做）

- 前置：两芯 `state` 必须 RUNNING；有别的进程打开 `/dev/xpu*` 则**拒绝启动**（防并发抢卡）
- 运行中：0.5s 扫 `dmesg`（`[sec]`=自启动秒数与 `/proc/uptime` 同原点做时间窗），命中
  `Exception in kernel execution | exception token= | session error` 或 state 掉出 RUNNING
  ⇒ **立即 `kill -9 -PGID`**（检测→杀 ≈0.5~0.7s）
- 结束：以 `kern.log` 字节偏移增量为**权威口径**统计本次窗口异常数；**>0 ⇒ 判失败(退出3)，严禁重试**
- 目标 stdout **不重定向**（它是数据通道）；需要留档用 `SAFE_TEE_STDOUT=1`
- 退出时 trap 杀掉目标进程组，**不留持有卡的孤儿进程**
- 自测：`./safe_run.sh --selftest` → 正例 0 / 反例被杀且判 3 ✓

## 6. 服务设计（serve.py）

1. 后端 = **已验证正确**的 `orn` 二进制，内核用 kq8 + HW_CORE=16 + co≤16。不做性能花样。
2. 引擎整个跑在 `safe_run.sh` 下：卡上一出异常 ⇒ 引擎被杀 ⇒ 服务判 FAILED；
   **不自动重启、不重试、绝不 soft_reset**。
3. 请求经 **FIFO** 喂入（`mkfifo` + 两端 `O_RDWR` 常开）——实测经 safe_run 的子进程 stdin 会立刻 EOF，
   FIFO 绕开这个坑（详见 PROGRESS.md §1 坑 2）。
4. **强制关思考**：assistant 回合写死 `<think>\n\n</think>\n\n`（与 CPU 基准 `--reasoning off` 等价）。
5. 特殊 token / chat_template 来自 GGUF（模型无关）。

## 7. 黄金测试（正确性的唯一判据）

固定 13-token 提示：
```
<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```
卡上必须输出：**`你好！有什么我可以帮你的吗？😊`**（与 llama.cpp CPU 完全一致）。
- 触发：`curl -s localhost:8090/v1/chat/completions -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":24}'`
- 证据链：`/home/caden/ornc/golden_raw.json` + `raw_tok.log`（逐 token hex）
- 结果：2026-09-21 13:09 **PASS**，71.4s，9 tok，0.126 tok/s

## 8. 交付物清单

| 文件 | 作用 |
|---|---|
| `/home/caden/orn_engine/safe_run.sh` | 安全网（含 `--selftest`） |
| `/home/caden/ornc/serve.py` | 8090 服务 |
| `/home/caden/ornc/start.sh` / `recycle.sh` | 起 / 干净停 |
| `/home/caden/ornc/autostart.sh` | 开机自拉起（crontab `@reboot`） |
| `/home/caden/ornc/accept6.py` + `accept.log` | 6 条验收题驱动与记录 |
| `/home/caden/ornc/golden_raw.json` + `raw_tok.log` | 黄金测试证据 |
| `/home/caden/ornc/PROGRESS.md` | 进度与踩坑记录 |

## 9. 暂缓（用户明确要求先不管）

16B 对齐重打包 / 多累加链 / 每 32 块一次 scale / 主机数学搬上卡 / 读带宽实测靶定。
（背景账：旧引擎有效带宽 ~1 GB/s，卡的有效读带宽约 133~140 GB/s ⇒ 纯代码问题，余量巨大。）
