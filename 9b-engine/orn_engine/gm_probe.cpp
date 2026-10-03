// gm_probe.cpp — GM2LM/LM2GM 单次调用开销曲线 (空转基线扣除 launch)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <xpu/runtime.h>

extern void run_gm_empty(long *sink, int cl, int co, int N);
extern void run_gm_copy(float *src, long *sink, int cl, int co, int sz, int N);
extern void run_gm_store(float *dst, long *sink, int cl, int co, int sz, int N);

static double NOW() {
    using namespace std::chrono;
    return std::chrono::duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main() {
    setbuf(stdout, NULL);
    if (xpu_set_device(0)) { printf("set_device fail\n"); return 2; }
    void *dsrc = 0, *ddst = 0, *dsink = 0;
    size_t src_bytes = 64 * 1024, dst_bytes = 64 * 1024;
    if (xpu_malloc(&dsrc, src_bytes) || xpu_malloc(&ddst, dst_bytes) || xpu_malloc(&dsink, 8)) {
        printf("malloc fail\n"); return 2;
    }
    std::vector<float> hsrc(16384, 3.14f);
    xpu_memcpy(dsrc, hsrc.data(), src_bytes, XPU_HOST_TO_DEVICE);
    xpu_wait();

    const int N = 20000;
    long sink = 0;
    // 预热
    run_gm_empty((long*)dsink, 8, 16, 100); xpu_wait();

    // 空转基线
    double t0 = NOW();
    run_gm_empty((long*)dsink, 8, 16, N); xpu_wait();
    double t_empty = (NOW() - t0) * 1e6 / N;   // µs / iter

    printf("GM2LM/LM2GM 单次开销 (N=%d, 8cluster x 16core, 每core串行N次)\n", N);
    printf("empty 循环基线: %.3f µs/iter (含launch摊薄)\n\n", t_empty);
    printf("%8s | %14s | %14s | %12s\n", "bytes", "GM2LM µs/call", "LM2GM µs/call", "load GB/s");
    int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
    for (int sz : sizes) {
        double tc = NOW();
        run_gm_copy((float*)dsrc, (long*)dsink, 8, 16, sz, N);
        if (xpu_wait()) { printf("copy fail sz=%d\n", sz); return 3; }
        double t_c = (NOW() - tc) * 1e6 / N;

        double ts = NOW();
        run_gm_store((float*)ddst, (long*)dsink, 8, 16, sz, N);
        if (xpu_wait()) { printf("store fail sz=%d\n", sz); return 3; }
        double t_s = (NOW() - ts) * 1e6 / N;

        double pc = t_c - t_empty, ps = t_s - t_empty;
        double gbs = pc > 0 ? (sz / (pc * 1e-6)) / 1e9 : 0;
        printf("%8d | %14.3f | %14.3f | %12.1f\n", sz, pc, ps, gbs);
    }
    return 0;
}
