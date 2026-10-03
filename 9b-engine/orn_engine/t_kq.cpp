#include <xpu/runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>
void run_gemv_q4k(int cl, int co, const void *W, const void *xq, const void *xs, void *y, int M, int N);
void run_gemv_q6k(int cl, int co, const void *W, const void *xq, const void *xs, void *y, int M, int N);
static double bench(int ty, int M, int N, int cl, int co, const char *tag) {
    int bs = (ty == 12) ? 144 : 210;
    size_t wb = (size_t)M * (N / 256) * bs;
    std::vector<unsigned char> hw(wb, 0x55);
    void *dw = nullptr, *dxq = nullptr, *dxs = nullptr, *dy = nullptr;
    if (xpu_malloc(&dw, wb) || xpu_malloc(&dxq, N) || xpu_malloc(&dxs, N/32*4) || xpu_malloc(&dy, (size_t)M*4)) {
        printf("  %s malloc 失败\n", tag); return -1;
    }
    if (xpu_memcpy(dw, hw.data(), wb, XPU_HOST_TO_DEVICE)) { printf("  h2d 失败\n"); return -1; }
    std::vector<signed char> xq(N, 1); std::vector<float> xs(N/32, 0.01f);
    xpu_memcpy(dxq, xq.data(), N, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dxs, xs.data(), N/32*4, XPU_HOST_TO_DEVICE);
    xpu_wait();
    auto t0 = std::chrono::steady_clock::now();
    const int IT = 5;
    for (int it = 0; it < IT; it++) {
        if (ty == 12) run_gemv_q4k(cl, co, dw, dxq, dxs, dy, M, N);
        else          run_gemv_q6k(cl, co, dw, dxq, dxs, dy, M, N);
    }
    if (xpu_wait()) { printf("  %s wait 失败\n", tag); return -1; }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / IT;
    printf("  %-34s W=%6.1f MB  M=%5d N=%5d cl=%d co=%d  ->  %8.2f ms/次\n",
           tag, wb/1048576.0, M, N, cl, co, ms);
    xpu_free(dw); xpu_free(dxq); xpu_free(dxs); xpu_free(dy);
    return 0;
}
int main() {
    xpu_set_device(0);
    printf("=== 单芯 (dev0) 内核 microbench ===\n");
    bench(12, 4096, 4096, 8, 16, "Q4_K  attn_gate 4096x4096");
    bench(14, 8192, 4096, 8, 16, "Q6_K  attn_qkv  8192x4096");
    bench(12, 24576, 4096, 8, 16, "Q4_K  grp_ffn  24576x4096");
    bench(14, 4096, 12288, 8, 16, "Q6_K  ffn_down  4096x12288");
    printf("--- 参考: 理论下界 = 权重字节 / HBM带宽(约 375 GB/s) ---\n");
    printf("  4096x4096 Q4_K  9.4 MB -> 0.03 ms | 24576x4096 Q4_K 56.6 MB -> 0.15 ms\n");
    bench(12, 4096, 4096, 8, 16, "复测 Q4_K (预热后)");
    return 0;
}
