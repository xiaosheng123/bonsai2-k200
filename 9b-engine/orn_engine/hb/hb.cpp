// hb.cpp — 只用【已验证】的 kq8 内核对象 (kq8.proxy.o / kq8_host.o) 做分解测量。
// 目的: 把 "每次 gemv 调用的固定开销" 与 "纯读带宽" 分开, 不写任何新设备内核 => 不引入新异常风险。
// 所有 launch 配置均为已验证的 cl<=16, co=16。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <xpu/runtime.h>

void run_gemv_q8v2(int cl, int co, const void *W, const void *xq, const void *xs, void *y, int M, int N);
void kq8_info(int *o);

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    xpu_set_device(dev);
    printf("[hb] dev=%d\n", dev);


    const int N = 4096;
    const int nb = N / 32;
    const int rowbytes = N + nb * 2;          // 4352
    const int MAXM = 16384;
    size_t wbytes = (size_t)MAXM * rowbytes;  // 71 MB
    void *W = nullptr;
    if (xpu_malloc(&W, wbytes)) { printf("malloc W fail\n"); return 1; }
    void *xq = nullptr, *xs = nullptr, *y = nullptr;
    if (xpu_malloc(&xq, 65536) || xpu_malloc(&xs, 8192) ||
        xpu_malloc(&y, (size_t)MAXM * 4 + 65536)) { printf("malloc scratch fail\n"); return 1; }
    printf("[hb] W=%.1f MB @%p xq=%p xs=%p y=%p\n", wbytes / 1048576.0, W, xq, xs, y);

    std::vector<unsigned char> hw(1 << 20);
    for (size_t i = 0; i < hw.size(); i++) hw[i] = (unsigned char)((i * 131 + 7) & 0xff);
    for (size_t off = 0; off < wbytes; off += hw.size())
        xpu_memcpy((char *)W + off, hw.data(), hw.size(), XPU_HOST_TO_DEVICE);
    std::vector<unsigned char> hx(N, 3), hs(nb * 4, 0);
    { float *f = (float *)hs.data(); for (int i = 0; i < nb; i++) f[i] = 0.01f; }
    xpu_memcpy(xq, hx.data(), N, XPU_HOST_TO_DEVICE);
    xpu_memcpy(xs, hs.data(), nb * 4, XPU_HOST_TO_DEVICE);
    if (xpu_wait()) { printf("warmup wait fail\n"); return 1; }

    // ---- T1: 主机->设备小 memcpy 延迟 (4096B 与 8B) ----
    {
        double t0 = now_ms();
        const int R = 200;
        for (int r = 0; r < R; r++) { xpu_memcpy(xq, hx.data(), N, XPU_HOST_TO_DEVICE); xpu_wait(); }
        double ms = (now_ms() - t0) / R;
        printf("[T1] H2D 4096B + wait   : %8.3f ms/次   (%.2f MB/s 等效)\n", ms, 4096.0 / ms / 1000.0);
        t0 = now_ms();
        for (int r = 0; r < R; r++) { xpu_memcpy(xq, hx.data(), 8, XPU_HOST_TO_DEVICE); xpu_wait(); }
        ms = (now_ms() - t0) / R;
        printf("[T1] H2D    8B + wait   : %8.3f ms/次\n", ms);
    }
    // ---- T2: 设备->主机 memcpy 延迟 ----
    {
        std::vector<float> hy(MAXM);
        double t0 = now_ms();
        const int R = 200;
        for (int r = 0; r < R; r++) { xpu_memcpy(hy.data(), y, 128, XPU_DEVICE_TO_HOST); xpu_wait(); }
        double ms = (now_ms() - t0) / R;
        printf("[T2] D2H  128B + wait   : %8.3f ms/次\n", ms);
        t0 = now_ms();
        for (int r = 0; r < R; r++) { xpu_memcpy(hy.data(), y, 12288 * 4, XPU_DEVICE_TO_HOST); xpu_wait(); }
        ms = (now_ms() - t0) / R;
        printf("[T2] D2H 49KB + wait    : %8.3f ms/次  (%.2f MB/s 等效)\n", ms, 12288.0 * 4 / ms / 1000.0);
    }
    // ---- T3: 极小内核 launch 延迟 (M=32, 只读 1 行 4352B) ----
    {
        const int R = 200;
        run_gemv_q8v2(8, 16, W, xq, xs, y, 32, N); if (xpu_wait()) { printf("T3 wait fail\n"); return 1; }
        run_gemv_q8v2(8, 16, W, xq, xs, y, 32, N);
        double t0 = now_ms();
        for (int r = 0; r < R; r++) { run_gemv_q8v2(8, 16, W, xq, xs, y, 32, N); xpu_wait(); }
        double ms = (now_ms() - t0) / R;
        printf("[T3] 空载内核(gemv M=32)launch+wait: %8.3f ms/次\n", ms);
        t0 = now_ms();
        for (int r = 0; r < R; r++) { run_gemv_q8v2(8, 16, W, xq, xs, y, 32, N); }
        xpu_wait();
        ms = (now_ms() - t0) / R;
        printf("[T3] 空载内核 async 发射(不等待)   : %8.3f ms/次\n", ms);
        int *o = nullptr; xpu_malloc((void **)&o, 8);
        t0 = now_ms();
        for (int r = 0; r < R; r++) { kq8_info(o); xpu_wait(); }
        ms = (now_ms() - t0) / R;
        printf("[T3] kq8_info<<<1,1>>> launch+wait : %8.3f ms/次\n", ms);
    }
    // ---- T4: M 扫描 (同一 N=4096) —— 分离固定开销 vs 每字节代价 ----
    {
        int Ms[] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384};
        printf("[T4] M 扫描 (N=%d, cl=8 co=16)  读字节=%.2f MB/行\n", N, rowbytes / 1048576.0);
        for (int k = 0; k < (int)(sizeof(Ms) / sizeof(Ms[0])); k++) {
            int M = Ms[k];
            double bytes = (double)M * rowbytes;
            run_gemv_q8v2(8, 16, W, xq, xs, y, M, N); if (xpu_wait()) { printf("T4 M=%d wait fail\n", M); break; }
            const int R = 5;
            double t0 = now_ms();
            for (int r = 0; r < R; r++) { run_gemv_q8v2(8, 16, W, xq, xs, y, M, N); xpu_wait(); }
            double ms = (now_ms() - t0) / R;
            printf("[T4] M=%6d  读%8.2f MB  %8.3f ms  %7.2f GB/s(内核口径)\n", M, bytes / 1048576.0, ms,
                   bytes / ms / 1e6);
        }
    }
    // ---- T5: 完整 gemv_sel 往返 (模拟 orn 的一次调用: 2xH2D + launch + wait + D2H) ----
    {
        std::vector<float> hy(MAXM);
        int Ms[] = {512, 4096, 12288, 16384};
        for (int k = 0; k < 4; k++) {
            int M = Ms[k];
            double bytes = (double)M * rowbytes;
            const int R = 20;
            // warmup
            run_gemv_q8v2(8, 16, W, xq, xs, y, M, N); xpu_wait();
            double t0 = now_ms();
            for (int r = 0; r < R; r++) {
                xpu_memcpy(xq, hx.data(), N, XPU_HOST_TO_DEVICE);
                xpu_memcpy(xs, hs.data(), nb * 4, XPU_HOST_TO_DEVICE);
                run_gemv_q8v2(8, 16, W, xq, xs, y, M, N);
                xpu_wait();
                xpu_memcpy(hy.data(), y, (size_t)M * 4, XPU_DEVICE_TO_HOST);
                xpu_wait();
            }
            double ms = (now_ms() - t0) / R;
            printf("[T5] 完整往返 M=%6d  %8.3f ms/次  (%7.2f GB/s 权重口径, 含全部同步)\n", M, ms, bytes / ms / 1e6);
        }
    }
    printf("=== hb done ===\n");
    return 0;
}
