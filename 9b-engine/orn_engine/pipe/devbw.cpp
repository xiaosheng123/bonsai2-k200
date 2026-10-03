// devbw.cpp — 官方 refactor 算子 在 K200 上的 带宽 / 每次调用开销 实测
//   纯官方算子 + xpu_malloc, 不含任何自写设备内核 => 零卡风险
//   目的: 判定"把主机侧数学搬上卡"到底值不值 (卡上逐元素/归一化的实际 GB/s 与 launch 开销)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <xpu/runtime.h>
#include "xpu/refactor/math.h"
#include "xpu/refactor/nn.h"
#include "xpu/refactor/quantization.h"
#include "xpu/refactor/core/device.h"
#include "xpu/refactor/context/newcontext.h"
namespace api = baidu::xpu::api;

static inline double NOWS() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char **argv) {
    int dev = 0;
    if (argc > 1) dev = atoi(argv[1]);
    if (xpu_set_device(dev)) { printf("set_device FAIL\n"); return 1; }
    api::Context *ctx = new api::Context(api::Device(api::DeviceType::XPU1, dev));
    printf("== devbw: 官方 refactor 算子实测 (dev%d) ==\n", dev);

    // 最大 64MB fp32
    size_t MAXN = 16u * 1024 * 1024;
    float *a = nullptr, *b = nullptr, *c = nullptr, *sc = nullptr, *mn = nullptr, *vr = nullptr;
    float *mean2 = nullptr, *var2 = nullptr;
    if (xpu_malloc((void **)&mean2, 4096 * 4) || xpu_malloc((void **)&var2, 4096 * 4)) return 1;
    if (xpu_malloc((void **)&a, MAXN * 4) || xpu_malloc((void **)&b, MAXN * 4) ||
        xpu_malloc((void **)&c, MAXN * 4)) { printf("malloc FAIL\n"); return 1; }
    std::vector<float> h(MAXN);
    for (size_t i = 0; i < MAXN; i++) h[i] = 0.5f + 0.001f * (float)(i % 97);
    xpu_memcpy(a, h.data(), MAXN * 4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(b, h.data(), MAXN * 4, XPU_HOST_TO_DEVICE);
    xpu_wait();

    size_t n_w = 4096;                        // layer_norm 的 scale/bias
    std::vector<float> hw(n_w, 1.0f);
    if (xpu_malloc((void **)&sc, n_w * 4) || xpu_malloc((void **)&mn, n_w * 4) || xpu_malloc((void **)&vr, n_w * 4)) { printf("malloc2 FAIL\n"); return 1; }
    xpu_memcpy(sc, hw.data(), n_w * 4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(mn, hw.data(), n_w * 4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(vr, hw.data(), n_w * 4, XPU_HOST_TO_DEVICE);
    xpu_wait();

    struct S { const char *nm; size_t n; };
    S sizes[] = { {"4KB",1024}, {"16KB",4096}, {"48KB",12288}, {"4MB",1024*1024}, {"16MB",4*1024*1024}, {"64MB",16*1024*1024} };

    // ---- 1) 逐元素算子: add(2R+1W) / mul / swish(1R+1W) / rsqrt / scale ----
    for (int si = 0; si < 6; si++) {
        size_t n = sizes[si].n;
        int reps = (int)(MAXN / n); if (reps > 50) reps = 50; if (reps < 1) reps = 1;
        printf("\n---- 向量长度 %s (%zu floats) reps=%d ----\n", sizes[si].nm, n, reps);
        struct { const char *nm; int traffic; } ops[] = { {"add",3}, {"mul",3}, {"swish",2}, {"rsqrt",2}, {"scale",2}, {"exp",2}, {"sigmoid",2} };
        for (int oi = 0; oi < 7; oi++) {
            double t0 = NOWS(); int r = 0;
            for (int rp = 0; rp < reps; rp++) {
                if (!strcmp(ops[oi].nm, "add"))      r |= api::add<float>(ctx, a, b, c, (int)n);
                else if (!strcmp(ops[oi].nm, "mul")) r |= api::mul<float>(ctx, a, b, c, (int)n);
                else if (!strcmp(ops[oi].nm, "swish")) r |= api::swish<float>(ctx, a, c, (int)n);
                else if (!strcmp(ops[oi].nm, "rsqrt")) r |= api::rsqrt<float>(ctx, a, c, (int)n);
                else if (!strcmp(ops[oi].nm, "exp"))   r |= api::exp<float>(ctx, a, c, (int)n);
                else if (!strcmp(ops[oi].nm, "sigmoid")) r |= api::sigmoid<float>(ctx, a, c, (int)n);
                else                                  r |= api::scale<float>(ctx, a, c, (int)n, true, 1.0f, 0.0f);
            }
            if (xpu_wait()) r |= 1;
            double dt = (NOWS() - t0) / reps;
            double gb = (double)n * 4 * ops[oi].traffic / 1e9 / dt;
            printf("  %-8s %8.1f us/次  %8.2f GB/s (口径=%d 流)  r=%d\n", ops[oi].nm, dt * 1e6, gb, ops[oi].traffic, r);
        }
    }

    // ---- 2) 链式: N 次小算子 只 wait 一次 => 每次调用真正的主机开销 ----
    printf("\n---- launch 开销: 1000 次 len=4096 swish, 只 xpu_wait 一次 ----\n");
    {
        double t0 = NOWS();
        for (int i = 0; i < 1000; i++) api::swish<float>(ctx, a, c, 4096);
        double tmid = NOWS();
        if (xpu_wait()) printf("  wait FAIL\n");
        double t1 = NOWS();
        printf("  1000 次提交 %.3f ms (%.1f us/次) | 最后一次 wait %.3f ms\n", (tmid - t0) * 1e3, (tmid - t0) * 1e6 / 1000, (t1 - tmid) * 1e3);
    }
    printf("\n---- 同步开销: 1000 次 len=4096 swish + 每次 wait 一次 ----\n");
    {
        double t0 = NOWS();
        for (int i = 0; i < 1000; i++) { api::swish<float>(ctx, a, c, 4096); xpu_wait(); }
        double t1 = NOWS();
        printf("  %.3f ms (%.1f us/次 含同步)\n", (t1 - t0) * 1e3, (t1 - t0) * 1e6 / 1000);
    }

    // ---- 3) layer_norm: m 行 n 列 (用户提示 n<=1024) ----
    printf("\n---- layer_norm (m 行 x n 列) ----\n");
    {
        int cases[][2] = { {32,128}, {32,1024}, {1,1024}, {1,4096}, {1,8192}, {4,4096} };
        for (int ci = 0; ci < 6; ci++) {
            int m = cases[ci][0], n = cases[ci][1];
            if ((size_t)m * n > MAXN) continue;
            int reps = 200;
            double t0 = NOWS(); int r = 0;
            for (int rp = 0; rp < reps; rp++) r |= api::layer_norm<float>(ctx, a, c, m, n, 1e-5f, sc, mn, mean2, var2);
            if (xpu_wait()) r |= 1;
            double dt = (NOWS() - t0) / reps;
            printf("  m=%-5d n=%-6d %9.1f us/次  r=%d\n", m, n, dt * 1e6, r);
        }
    }

    // ---- 4) reduce_sum / findmax ----
    printf("\n---- reduce_sum / findmax ----\n");
    {
        double t0 = NOWS(); int r = 0;
        for (int rp = 0; rp < 200; rp++) r |= api::reduce_sum<float>(ctx, a, c, std::vector<int>(1, 4096), std::vector<int>(1, 1));
        if (xpu_wait()) r |= 1;
        printf("  reduce_sum(4096->1)  %.1f us/次  r=%d\n", (NOWS() - t0) * 1e6 / 200, r);
        t0 = NOWS(); r = 0;
        for (int rp = 0; rp < 200; rp++) r |= api::findmax<float>(ctx, a, c, 4096);
        if (xpu_wait()) r |= 1;
        printf("  findmax(4096)        %.1f us/次  r=%d\n", (NOWS() - t0) * 1e6 / 200, r);
    }

    // ---- 5) fc fp32 (alpha/beta 那种 32x4096 @ 4096) 权重常驻 ----
    printf("\n---- fc<f32,f32,f32> (权重常驻卡上) ----\n");
    {
        int M = 64, K = 4096;
        float *dw = nullptr, *dx = nullptr, *dy = nullptr;
        if (xpu_malloc((void **)&dw, (size_t)M * K * 4) || xpu_malloc((void **)&dx, K * 4) || xpu_malloc((void **)&dy, M * 4)) { printf("  malloc FAIL\n"); }
        else {
            std::vector<float> hw2((size_t)M * K, 0.01f), hx(K, 1.0f);
            xpu_memcpy(dw, hw2.data(), (size_t)M * K * 4, XPU_HOST_TO_DEVICE);
            xpu_memcpy(dx, hx.data(), K * 4, XPU_HOST_TO_DEVICE);
            xpu_wait();
            int reps = 500; double t0 = NOWS(); int r = 0;
            for (int rp = 0; rp < reps; rp++)
                r |= api::fc<float, float, float, int>(ctx, dx, dw, dy, 1, M, K, false, true, nullptr, nullptr, nullptr);
            if (xpu_wait()) r |= 1;
            double dt = (NOWS() - t0) / reps;
            printf("  fc m=1 n=%d k=%d (权重 %.2f MB): %.1f us/次  => 权重带宽 %.2f GB/s  r=%d\n",
                   M, K, M * K * 4 / 1e6, dt * 1e6, (double)M * K * 4 / 1e9 / dt, r);
            printf("  ^ 提示: fc<float,float,float,int> 若 r!=0 说明该组合不支持, 需换 TGEMM 类型\n");
        }
    }

    printf("\n== devbw done ==\n");
    return 0;
}
