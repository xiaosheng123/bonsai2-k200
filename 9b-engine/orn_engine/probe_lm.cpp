#include <xpu/runtime.h>
#include <cstdio>
#include <vector>

extern void run_lm(const void *W, void *out, int cl, int co);
extern void run_map(int *o, int cl, int co);

int main() {
    xpu_set_device(1);
    int *d = nullptr; void *W = nullptr, *out = nullptr;
    if (xpu_malloc((void **)&d, 16 * 64 * 4)) { printf("malloc d fail\n"); return 1; }
    if (xpu_malloc(&W, 65536)) { printf("malloc W fail\n"); return 1; }
    if (xpu_malloc(&out, 4096 * 4)) { printf("malloc out fail\n"); return 1; }
    std::vector<unsigned char> hw(65536, 7);
    xpu_memcpy(W, hw.data(), 65536, XPU_HOST_TO_DEVICE); xpu_wait();

    std::vector<int> hi(16 * 64, 0);
    xpu_memcpy(d, hi.data(), 16 * 64 * 4, XPU_HOST_TO_DEVICE); xpu_wait();
    run_map(d, 8, 16); xpu_wait();
    xpu_memcpy(hi.data(), d, 16 * 64 * 4, XPU_DEVICE_TO_HOST); xpu_wait();
    int cnt = 0, ncl = 0;
    for (int c = 0; c < 16; c++) { bool any = false;
        for (int k = 0; k < 64; k++) if (hi[c * 64 + k]) { any = true; cnt++; }
        if (any) ncl++; }

    std::vector<float> ho(4096, -1.f);
    xpu_memcpy(out, ho.data(), 4096 * 4, XPU_HOST_TO_DEVICE); xpu_wait();
    run_lm(W, out, 8, 16); xpu_wait();
    xpu_memcpy(ho.data(), out, 4096 * 4, XPU_DEVICE_TO_HOST); xpu_wait();
    int nz = 0; for (int i = 0; i < 2048; i++) if (ho[i] > 0) nz++;
    printf("map线程=%d cluster=%d | lm核对=%d 线程=%d out0=%.0f\n", cnt, ncl, nz, cnt, ho[0]);
    return 0;
}
