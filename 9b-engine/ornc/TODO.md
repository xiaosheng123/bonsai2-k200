# Ornith on K200 —— 任务状态（任何代理/我重启后从这里接上）
更新时间: 2026-09-21 15:2x

## 铁律（顺序不可变）
0. **【改动前先停服务】** 任何改动（改脚本 / 改服务 / 测带宽 / 调参数）**之前先 `bash /home/caden/ornc/stop.sh`**
   把 8090 + 引擎停掉，卡保持空闲；全部改完、确认无误后**最后一步**才 `bash /home/caden/ornc/start.sh` 拉起做验收。
   带宽类测量必须**卡独占**（有别的进程占 /dev/xpu* 时数字不准，safe_run 也会直接拒绝启动）。
   验收完**保留服务运行**。拉起服务后若出现卡异常 ⇒ **立即停服务并报告**，不许硬上。
1. 宿主稳定（NAS 192.168.66.26 是生产机，vfio 直通；卡上异常会拖死宿主）
2. **绝不写/编译/运行新的自写设备内核**（历史事故就是探针内核越界拖死宿主）。卡上只许跑
   (a) 已验证正确的引擎 `orn_engine/orn3`（旧版 `orn`）、(b) 厂商工具（xre-driver/.../tools/）。
   非写新内核不可 ⇒ **停手报告**，不要自己试。每台设备作业都要包在 `safe_run.sh` 里。
3. 正确性
4. 速度

## 安全措施（已就位，不要绕过）
- 看门狗 v2: /home/caden/orn_engine/watchdog.py（在跑, pidfile /tmp/watchdog.pid, cron 每分钟自愈）
  作用: tail kern.log, 一旦出现 "Exception in kernel execution"/"exception token="/"session error" 立即 SIGKILL **所有占 /dev/xpu* 的进程**
- 每次设备运行必须过: /home/caden/orn_engine/safe_run.sh -n <名> -t <秒> -- <cmd>
- 取证: caden 在 adm 组 ⇒ 无需 sudo 读 /var/log/kern.log 与 dmesg；`grep -ac "Exception in kernel execution"` 就是异常计数
- 尺寸纪律: 极小 → 中等 → 全量；出一次异常就整体停，严禁重试
- **服务脚本只有一个入口**（名字已统一，改任一名字必须同步改 stop.sh 的 PAT_*）:
  `start.sh`（起）/ `stop.sh`（停）/ `restart.sh`（停+起）/ `status.sh`（只读一览）
  `start2.sh`、`recycle2.sh` 已退化为兼容壳（exec 到 start.sh / stop.sh）。

## 当前状态（2026-09-21 15:2x）
- 卡: 两芯 RUNNING；**历史异常 16 次**（14 次旧探针 mbench/mb3/probe* + `ri` 13:41 + `lmprobe_16384` 14:30），**必须保持不再新增**
- 服务: 8090 **在跑** = `ornc/serve2.py` + `orn_engine/orn3`（双芯行分裂 + 同 x 矩阵拼接 + 会话级 KV/状态复用），
  引擎在 safe_run 安全网内；`/health` ok/ready；开机自拉起 = cron `@reboot ornc/autostart.sh`（它只调 start.sh）
- 模型: /home/caden/orn/Ornith-1.5-9B-Q8_0.gguf (9786060384 B, qwen35 混合架构: 24 gated-DeltaNet + 8 全注意力)
- 实测性能: **生成 0.560~0.563 tok/s（双芯，稳定）**；冷 prefill 23~57s（看 prompt 长）；快照命中 prefill 0.00s
- 实测带宽（厂商工具, 卡独占）: H2D 3.49~3.53 GB/s、D2H 3.73~3.75 GB/s（PCIe 口径 1.63/1.69）；
  设备内 **D2D 拷贝 62~66 GB/s（单向读 31~33 GB/s）**；**跨芯 peer 拷贝 rc=-807 不支持**（本 VM vfio 直通）
  ⇒ "133~140 GB/s" 在本机**不成立**
- 6 条验收: **6/6 通过**（max_tokens=192，原文见 ornc/accept192.log）

## 下一步（按顺序，每步做完写 PROGRESS.md）
1. ~~包出 8090 服务~~ ✅ / ~~黄金测试~~ ✅ / ~~6 条验收~~ ✅ / ~~开机自拉起~~ ✅
2. ~~把生成速度从 0.308 提到 0.64 tok/s~~ ✅（v2 引擎 + KV 复用，端到端 4.5x）
3. **遗留 (需维护窗口 / 需新内核 ⇒ 先停手报告)**: 现有 GM2LM 原语实测每核 ~43 MB/s ⇒ 双芯 5.4 GB/s 已是该原语上限；
   要冲 2~3 tok/s 必须换读数据机制（SD-CDNN 引擎，官方 `xpu_create_sd_func`，属独立项目）。
   **在动手之前：先 stop.sh，再评估，绝不在带服务的卡上试新内核。**
4. 未定位项: 验收 Q3/Q6 墙钟比引擎自报各多 ~39.1s（引擎两计时器之外，非卡异常、非内存压力），未查因

## 机制与资产（重写时照抄，别自己猜）
- qwen35 机制: ~/.hermes/skills/devops/kunlun-k200-setup/references/ornith-qwen35-mechanism.md
- 内核坑与天花板: 同目录 k200-kernel-launch-traps.md（坑0=宿主死机, 坑1=HW_CORE=16, 坑2=延迟受限, 附2=带宽基准别用错）
- 带宽实测原文与台账: ornc/PROGRESS.md 第 8/9/10 节；卡异常逐条台账在 9.4 / 10
