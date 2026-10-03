#include <xpu/runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>

extern void run_map(int *o, int cl, int co);
extern void run_rd(const void *W, int slab, int nslab, void *out, int cl, int co);
extern void run_lm(const void *W, int kb, void *out, int cl, int co);

static double now() { using namespace std::chrono; return duration<double>(steady_clock::now().time_since_epoch()).count(); }

int main() {
    xpu_set_device(1);
    int *d = nullptr;
    if (xpu_malloc((void **)&d, 16 * 64 * 4)) { printf("malloc fail\n"); return 1; }
    std::vector<int> h(16 * 64);
    printf("=== (cluster,core) 生效映射 (marker = c*100+k+1) ===\n");
    int cls[] = {1, 4, 8, 16}; int cos[] = {8, 16, 32, 64, 128};
    for (int a = 0; a < 4; a++) for (int b = 0; b < 5; b++) {
        for (int i = 0; i < 16 * 64; i++) h[i] = 0;
        xpu_memcpy(d, h.data(), 16 * 64 * 4, XPU_HOST_TO_DEVICE); xpu_wait();
        run_map(d, cls[a], cos[b]); xpu_wait();
        xpu_memcpy(h.data(), d, 16 * 64 * 4, XPU_DEVICE_TO_HOST); xpu_wait();
        int ncl = 0, maxk = -1, cnt = 0;
        for (int c = 0; c < 16; c++) { bool any = false;
            for (int k = 0; k < 64; k++) if (h[c * 64 + k]) { any = true; cnt++; if (k > maxk) maxk = k; }
            if (any) ncl++; }
        printf("  launch(%-2d,%-3d): 生效线程=%4d 生效cluster=%2d 最大core_id=%2d\n", cls[a], cos[b], cnt, ncl, maxk);
    }
    printf("=== GM2LM 读带宽 ===\n");
    {
        size_t BYTES = 256ul << 20;
        void *W = nullptr, *out = nullptr;
        int slab = 16384, nslab = (int)(BYTES / slab);
        if (xpu_malloc(&W, BYTES)) { printf("malloc W fail\n"); return 1; }
        if (xpu_malloc(&out, 2048 * 4)) { printf("malloc out fail\n"); return 1; }
        std::vector<unsigned char> hw(1 << 20, 1);
        for (int i = 0; i < 256; i++) xpu_memcpy((char *)W + (size_t)i * (1 << 20), hw.data(), 1 << 20, XPU_HOST_TO_DEVICE);
        xpu_wait();
        int cfg[][2] = {{8, 16}, {8, 8}, {4, 16}, {16, 16}, {8, 32}};
        for (int i = 0; i < 5; i++) {
            run_rd(W, slab, nslab, out, cfg[i][0], cfg[i][1]); xpu_wait();
            double t0 = now();
            for (int r = 0; r < 3; r++) run_rd(W, slab, nslab, out, cfg[i][0], cfg[i][1]);
            xpu_wait();
            double dt = (now() - t0) / 3.0;
            printf("  grid(%d,%d): %.2f ms  %.1f GB/s\n", cfg[i][0], cfg[i][1], dt * 1000, (double)nslab * slab / dt / 1e9);
        }
    }
    printf("=== local 容量: 8/16/32KB DMA ===\n");
    {
        void *W = nullptr, *out = nullptr;
        if (xpu_malloc(&W, 32768) || xpu_malloc(&out, 2048 * 4)) { printf("lm malloc fail\n"); return 1; }
        std::vector<unsigned char> hw(32768, 7);
        xpu_memcpy(W, hw.data(), 32768, XPU_HOST_TO_DEVICE); xpu_wait();
        int kbs[3] = {8, 16, 32};
        for (int i = 0; i < 3; i++) {
            std::vector<float> o(2048, -1.f);
            xpu_memcpy(out, o.data(), 2048 * 4, XPU_HOST_TO_DEVICE); xpu_wait();
            run_lm(W, kbs[i], out, 8, 16); xpu_wait();
            xpu_memcpy(o.data(), out, 2048 * 4, XPU_DEVICE_TO_HOST); xpu_wait();
            printf("  %2dKB: out[0]=%.0f out[1]=%.0f (期望 14)\n", kbs[i], o[0], o[1]);
        }
    }
    printf("DONE\n");
    return 0;
}
