// lmp.cpp — LM 容量探针驱动: 一次只探一个 LBUF (编译期固定), <<<1,16>>> 单簇 16 核。
// 打印: LBUF / launch 返回码 / wait 返回码 / LM2GM 回读值 / 与期望值是否一致。
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <xpu/runtime.h>

void lmp_run(float *o, int cl, int co);

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    int cl = argc > 2 ? atoi(argv[2]) : 1;
    int co = argc > 3 ? atoi(argv[3]) : 16;
    xpu_set_device(dev);
    printf("[lmp] dev=%d launch <<<%d,%d>>> LBUF=%d bytes/core  簇内合计=%d bytes\n",
           dev, cl, co, LBUF, LBUF * co);
    float *o = nullptr;
    if (xpu_malloc((void **)&o, 64)) { printf("[lmp] malloc fail\n"); return 1; }
    float h = -1.f;
    xpu_memcpy(o, &h, 4, XPU_HOST_TO_DEVICE);
    xpu_wait();
    lmp_run(o, cl, co);
    int w = xpu_wait();
    if (w) { printf("[lmp] ✗ wait=%d (内核异常, 固件会记 WR_OVER_LM)\n", w); return 3; }
    h = -1.f;
    xpu_memcpy(&h, o, 4, XPU_DEVICE_TO_HOST);
    xpu_wait();
    int expect = (int)((unsigned char)(0 * 7 + 1)) + (int)((unsigned char)((LBUF - 8) * 7 + 1));
    printf("[lmp] ✓ 内核跑通 回读=%d 期望=%d %s\n", (int)h, expect, ((int)h == expect) ? "一致" : "不一致(可疑)");
    return (int)h == expect ? 0 : 4;
}
