// hb4.cpp — 只读源差异测试: 同样的【已验证 V2 内核】, 权重分别放在 (a) 主存 MAIN (b) 片上 L3 (16MB)
//   若 L3 -> LM 的搬运远快于 MAIN -> LM, 那"用快 DMA 把权重流水进 L3, 再从 L3 算"就是通往 140 GB/s 的路。
//   零新设备代码: 只改 xpu_malloc 的内存种类。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <xpu/runtime.h>

void run_gemv_q8v2(int cl, int co, const void *W, const void *xq, const void *xs, void *y, int M, int N);

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int bench(int dev, int kind, const char *name, int M) {
    const int N = 4096, nb = N / 32, ROWB = N + nb * 2;      // 4352
    xpu_set_device(dev);
    size_t wb = (size_t)M * ROWB;
    void *W = nullptr;
    if (xpu_malloc(&W, wb, (XPUMemoryKind)kind)) { printf("[hb4] %-4s malloc %.1f MB 失败 (kind=%d)\n", name, wb / 1048576.0, kind); return 1; }
    void *xq = nullptr, *xs = nullptr, *y = nullptr;
    if (xpu_malloc(&xq, 65536) || xpu_malloc(&xs, 8192) || xpu_malloc(&y, (size_t)M * 4 + 4096)) { printf("[hb4] scratch fail\n"); return 1; }
    std::vector<unsigned char> t(1 << 20);
    for (size_t i = 0; i < t.size(); i++) t[i] = (unsigned char)((i * 131 + 7) & 0xff);
    for (size_t off = 0; off < wb; off += t.size()) {
        size_t n = (wb - off < t.size()) ? (wb - off) : t.size();
        if (xpu_memcpy((char *)W + off, t.data(), n, XPU_HOST_TO_DEVICE)) { printf("[hb4] %-4s H2D 失败 (不支持?)\n", name); xpu_free(W); return 1; }
    }
    std::vector<unsigned char> hx(N, 3), hs(nb * 4, 0);
    { float *f = (float *)hs.data(); for (int i = 0; i < nb; i++) f[i] = 0.01f; }
    xpu_memcpy(xq, hx.data(), N, XPU_HOST_TO_DEVICE);
    xpu_memcpy(xs, hs.data(), nb * 4, XPU_HOST_TO_DEVICE);
    if (xpu_wait()) { printf("[hb4] %-4s warmup wait 失败\n", name); return 1; }
    double bytes = (double)M * ROWB;
    run_gemv_q8v2(8, 16, W, xq, xs, y, M, N);
    if (xpu_wait()) { printf("[hb4] %-4s 内核 wait 失败\n", name); return 1; }
    const int R = 5;
    double t0 = now_ms();
    for (int r = 0; r < R; r++) { run_gemv_q8v2(8, 16, W, xq, xs, y, M, N); xpu_wait(); }
    double ms = (now_ms() - t0) / R;
    printf("[hb4] %-4s (kind=%d) M=%5d 读 %6.2f MB  %8.3f ms  %6.2f GB/s\n", name, kind, M, bytes / 1048576.0, ms, bytes / ms / 1e6);
    xpu_free(y); xpu_free(xs); xpu_free(xq); xpu_free(W);
    return 0;
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    printf("[hb4] 同一 V2 内核, 权重放不同内存: MAIN(主存) vs L3(片上 16MB)\n");
    int M = 1920;                                            // 1920*4352 = 8.35MB (< 16MB L3)
    printf("--- MAIN ---\n");  bench(dev, 0 /*XPU_MEM_MAIN*/, "MAIN", M);
    printf("--- L3 ---\n");    int rc = bench(dev, 1 /*XPU_MEM_L3*/, "L3", M);
    if (rc) printf("[hb4] L3 路径不可用\n");
    printf("--- 大 M 对照 (MAIN only) ---\n"); bench(dev, 0, "MAIN", 12288);
    printf("=== hb4 done ===\n");
    return 0;
}
