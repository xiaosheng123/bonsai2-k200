#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cb_pwr.py — K200 功耗采样器 (只读 xpu_smi, 不碰卡)
#   用法: cb_pwr.py <时长秒> [采样间隔秒] > out.csv
#   输出: 每行 "epoch_s,dev0_power_W,dev0_tempC,dev1_power_W,dev1_tempC"
#   ★ xpu_smi -m 字段: [4]=温度, [8]=功耗(mW), [9..14]=freq_0..5
#   ★ 口径: dev0/dev1 两行给出的 power 到 mW 都完全相同 ⇒ 更可能是【整卡一个传感器】;
#     本表两个字段都记录, 均值<整卡口径>取 dev0, 均值<两芯累加口径>取 dev0+dev1 (仅供参考)。
import os, subprocess, sys, time

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
IV  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = "/usr/local/xpu-4.33.0/lib64:/home/caden/xtdk/xtdk-x86_64/shlib"
BIN = "/usr/local/xpu-4.33.0/bin/xpu_smi"

t_end = time.time() + DUR
print("epoch_s,dev0_W,dev0_C,dev1_W,dev1_C", flush=True)
while time.time() < t_end:
    try:
        o = subprocess.run([BIN, "-m"], capture_output=True, text=True, env=ENV, timeout=10).stdout
        d0 = d1 = None
        for ln in o.splitlines():
            f = ln.split()
            if len(f) < 10:
                continue
            try:
                pw = int(f[8]) / 1000.0
                tp = int(f[4])
            except Exception:
                continue
            if f[2] == "0":
                d0 = (pw, tp)
            elif f[2] == "1":
                d1 = (pw, tp)
        if d0 and d1:
            print("%.3f,%.2f,%d,%.2f,%d" % (time.time(), d0[0], d0[1], d1[0], d1[1]), flush=True)
    except Exception:
        pass
    time.sleep(IV)
