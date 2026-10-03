#include <xpu/runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>

extern void run_a(const void *W, int R, void *o, int cl, int co);
extern void run_b(const void *W, int R, void *o, int cl, int co);
extern void run_d(const void *W, const void *X, int R, void *o, int cl, int co);
extern void run_e(const void *W, const void *X, int R, void *o, int cl, int co);

static double now() { using namespace std::chrono; return duration<double>(steady_clock::now().time_since_epoch()).count(); }

int main() {
    xpu_set_device(1);
    void *W = nullptr, *X = nullptr, *out = nullptr;
    if (xpu_malloc(&W, 4352) || xpu_malloc(&X, 4096) || xpu_malloc(&out, 4096 * 4)) { printf("malloc fail\n"); return 1; }
    std::vector<unsigned char> hw(4352, 3), hx(4096, 5);
    xpu_memcpy(W, hw.data(), 4352, XPU_HOST_TO_DEVICE);
    xpu_memcpy(X, hx.data(), 4096, XPU_HOST_TO_DEVICE);
    xpu_wait();
    int R = 2000;
    struct { const char *n; int k; } T[] = {
        {"A local字节读", 0}, {"B local int32读", 1}, {"D local+int32MAC", 3}, {"E global直读MAC", 4}};
    for (int t = 0; t < 4; t++) {
        if (T[t].k == 0) run_a(W, R, out, 8, 16);
        else if (T[t].k == 1) run_b(W, R, out, 8, 16);
        
        else if (T[t].k == 3) run_d(W, X, R, out, 8, 16);
        else run_e(W, X, R, out, 8, 16);
        xpu_wait();
        double t0 = now();
        if (T[t].k == 0) run_a(W, R, out, 8, 16);
        else if (T[t].k == 1) run_b(W, R, out, 8, 16);
        
        else if (T[t].k == 3) run_d(W, X, R, out, 8, 16);
        else run_e(W, X, R, out, 8, 16);
        xpu_wait();
        double dt = now() - t0;
        double unit = (T[t].k <= 1) ? (double)R * 1024 : (double)R * 1024 * 4;   // 1024 次/轮 (int 单位或 MAC)
        printf("  %-18s %8.3f ms  每核 %6.1f M/s  聚合 %6.2f G/s\n", T[t].n, dt * 1000,
               unit / 128 / dt / 1e6, unit / dt / 1e9);
    }
    printf("DONE\n");
    return 0;
}
