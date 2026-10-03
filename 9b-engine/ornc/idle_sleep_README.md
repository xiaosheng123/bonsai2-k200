# K200 空闲休眠 (第28章) — 使用说明 / 开关 / 代价

## 一句话
卡的空闲功耗 **≈38.9W(驱动在线、无引擎)** 与 **≈40.1W(引擎常驻、空闲)** —— 实测差值只有 **~1.2W(约3%)**。
真正的大头(38.9W)是"驱动在线"本身的静态功耗;所以"保驱动、停引擎"这条 L1 能省的就是这 1.2W。
**默认关闭** (`K200_IDLE_SLEEP_MIN=0`),要不要开由你决定。

## 文件
| 文件 | 作用 |
|---|---|
| `k200_idle_sleep.conf` | 开关(默认 `K200_IDLE_SLEEP_MIN=0` = 关) |
| `idle_sleep.sh` | 执行器: cron `*/1` 调用, 空闲 N 分钟后停服务 |
| `wake.sh` | 唤醒: 清标记 -> start.sh -> 等 ready -> 打印耗时(实测暖机 70~90s) |
| `idle_sleep.log` | 每次休眠/唤醒一行 |
| `pwr_log.sh` | 功耗采样器(只读), 用来自己复测: `bash pwr_log.sh 300 30 标签` |

## 怎么开
```bash
# 1) 别让 svc_guard 在睡眠期间把服务拉起来
patch -p0 -d / < svc_guard_idle_sleep.patch      # 见同目录 (加 3 行: 有标记就退出)
# 2) 打开开关 (例如空闲 15 分钟休眠)
sed -i 's/^K200_IDLE_SLEEP_MIN=.*/K200_IDLE_SLEEP_MIN=15/' /home/caden/ornc/k200_idle_sleep.conf
# 3) 挂 cron
crontab -l | { cat; echo '*/1 * * * * /bin/bash /home/caden/ornc/idle_sleep.sh'; } | crontab -
```

## 唤醒
- 手动: `bash /home/caden/ornc/wake.sh` → 打印 `wake ready in NN s`
- 自动(需给网关加 3 行钩子, 见 `serve2_wake_hook.patch`): 休眠期间 `serve2.py` 继续监听 8090,
  收到请求走既有 `_wait_ready()` 排队路径; 只有加了这个钩子, 第一个请求才会真的把引擎拉起来。
  **不装钩子 = 休眠期间 8090 无人接听(客户端会连接被拒), 必须手动/外部调用 wake.sh。**

## 代价(必须先知道)
1. 睡眠期间 **8090 完全下线**(服务进程也被停掉了): 用户发消息会连接失败, 除非装了上面那个钩子。
2. 唤醒要重新灌 7.4GiB 权重到双芯 HBM: **实测 70~90s** 才 ready (你已明确接受"刚运行慢")。
3. 省的电只有 **~1.2W**;如果只是想让卡"热一点/凉一点", 停引擎能让温度从 72/73℃ 降到 70/71℃。

## 开关的边界(为什么只做到 L1)
- **L2 频率/PLL 旋钮存在但未测**: `kunlun1/freq <dev> static {900,800,700,600,500}`、`freq <dev> dynamic up|down`、`hbm_debug_tool --freq`。
  在跑着引擎时改 PLL 有把卡打挂的风险(第28章 L2 未做, 需独占卡窗口 + 完整回滚方案)。
- **L3 运行时 PM / D3hot: 当前不具备条件**(不是"没试", 是硬件+驱动两层都缺):
  驱动 `kunlun.ko` 没有任何 PCI PM 回调(`/sys/.../power/runtime_enabled = forbidden`,
  `xpu0/xpu1` 虚拟设备 `runtime_status = unsupported`);设备 PM 能力位里
  `PME(D0-,D1-,D2-,D3hot-,D3cold-)` —— **所有电源状态都不支持 PME 唤醒**。
  vfio 直通下进 D3hot 后没有任何唤醒路径, 且会连累宿主 ⇒ 按安全底线不做。
