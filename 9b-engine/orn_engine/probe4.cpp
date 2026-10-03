#include <xpu/runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>

extern void run_b8(const void *W, int nslab, void *out, int cl, int co);
extern void run_b2x4(const void *W, int nslab, void *out, int cl, int co);
extern void run_b4x2(const void *W, int nslab, void *out, int cl, int co);
extern void run_chk12(const void *W, void *out, int cl, int co);
extern void run_chk2x8(const void *W, void *out, int cl, int co);

static double now() { using namespace std::chrono; return duration<double>(steady_clock::now().time_since_epoch()).count(); }

int main() {
    xpu_set_device(1);
    void *W = nullptr, *out = nullptr;
    size_t BYTES = 256ul << 20;
    int nslab = (int)(BYTES / 8192);
    if (xpu_malloc(&W, BYTES)) { printf("malloc W fail\n"); return 1; }
    if (xpu_malloc(&out, 4096 * 4)) { printf("malloc out fail\n"); return 1; }
    std::vector<unsigned char> hw(1 << 20, 3);
    for (int i = 0; i < 256; i++) xpu_memcpy((char *)W + (size_t)i * (1 << 20), hw.data(), 1 << 20, XPU_HOST_TO_DEVICE);
    xpu_wait();
    if (!getenv("SKIPCHK")) {
        const char *nm[2] = {"12KB数组+8KBDMA", "2x8KB数组"};
        for (int t = 0; t < 2; t++) {
            std::vector<float> o(2048, -1.f);
            xpu_memcpy(out, o.data(), 2048 * 4, XPU_HOST_TO_DEVICE); xpu_wait();
            if (t == 0) run_chk12(W, out, 8, 16); else run_chk2x8(W, out, 8, 16);
            xpu_wait();
            xpu_memcpy(o.data(), out, 2048 * 4, XPU_DEVICE_TO_HOST); xpu_wait();
            int nz = 0; for (int i = 0; i < 2048; i++) if (o[i] > 0) nz++;
            printf("  chk %-16s 线程=%d out0=%.0f (期望 6)\n", nm[t], nz, o[0]);
        }
    }
    const char *bn[3] = {"单8KB阻塞", "2x4KB异步", "4x2KB异步x4"};
    for (int t = 0; t < 3; t++) {
        for (int a = 0; a < 2; a++) {
            int cl = a ? 4 : 8;
            (t == 0 ? run_b8 : (t == 1 ? run_b2x4 : run_b4x2))(W, nslab, out, cl, 16);
            xpu_wait();
            double t0 = now();
            for (int r = 0; r < 3; r++) (t == 0 ? run_b8 : (t == 1 ? run_b2x4 : run_b4x2))(W, nslab, out, cl, 16);
            xpu_wait();
            double dt = (now() - t0) / 3.0;
            printf("  %-14s grid(%d,16): %7.2f ms  %6.1f GB/s\n", bn[t], cl, dt * 1000, (double)nslab * 8192 / dt / 1e9);
        }
    }
    printf("DONE\n");
    return 0;
}
