#include <xpu/runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>

extern void run_m1(const void *W, const void *X, int R, void *o, int cl, int co);
extern void run_m2(const void *W, const void *X, int R, void *o, int cl, int co);
extern void run_m3(const void *W, const void *X, int R, void *o, int cl, int co);
extern void run_m4(const void *W, const void *X, int R, void *o, int cl, int co);

static double now() { using namespace std::chrono; return duration<double>(steady_clock::now().time_since_epoch()).count(); }

int main() {
    xpu_set_device(1);
    void *W = nullptr, *X = nullptr, *out = nullptr;
    if (xpu_malloc(&W, 4352) || xpu_malloc(&X, 4096) || xpu_malloc(&out, 4096 * 4)) { printf("malloc fail\n"); return 1; }
    std::vector<unsigned char> hw(4352, 3), hx(4096, 5);
    xpu_memcpy(W, hw.data(), 4352, XPU_HOST_TO_DEVICE);
    xpu_memcpy(X, hx.data(), 4096, XPU_HOST_TO_DEVICE);
    xpu_wait();
    int R = 2000;   // 每核 2000*4096 = 8.2M MAC
    const char *nm[4] = {"V1 逐字节4累加", "V2 int32打包", "V3 逐字节8累加", "V4 老风格2累加"};
    for (int v = 0; v < 4; v++) {
        (v == 0 ? run_m1 : v == 1 ? run_m2 : v == 2 ? run_m3 : run_m4)(W, X, R, out, 8, 16);
        xpu_wait();
        double t0 = now();
        (v == 0 ? run_m1 : v == 1 ? run_m2 : v == 2 ? run_m3 : run_m4)(W, X, R, out, 8, 16);
        xpu_wait();
        double dt = now() - t0;
        double macs = (double)R * 4096 * 128;                 // 128 核
        printf("  %-18s %7.3f ms  每核 %.1f M MAC/s  聚合 %.1f GMAC/s  (等价 W 吞吐 %.1f GB/s)\n",
               nm[v], dt * 1000, macs / 128 / dt / 1e6, macs / dt / 1e9, macs / dt / 1e9);
    }
    printf("DONE\n");
    return 0;
}
