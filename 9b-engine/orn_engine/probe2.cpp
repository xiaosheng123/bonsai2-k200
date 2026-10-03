#include <xpu/runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>

extern void pmap(int *o, int cl, int co);
extern void pw4(int *o, int v, int cl, int co);
extern void rd(const void *W, int slab, int nslab, void *out, int cl, int co);

static double now() { using namespace std::chrono; return duration<double>(steady_clock::now().time_since_epoch()).count(); }

int main() {
    xpu_set_device(1);
    int *d = nullptr;
    if (xpu_malloc((void **)&d, 16 * 128 * 4)) { printf("malloc fail\n"); return 1; }
    std::vector<int> h(16 * 128);

    int cls[] = {1, 2, 4, 8, 16};
    int cos[] = {4, 8, 16, 32, 64, 128};
    printf("=== (cluster,core) 生效映射 ===\n");
    for (int a = 0; a < 5; a++) for (int b = 0; b < 6; b++) {
        int cl = cls[a], co = cos[b];
        for (int i = 0; i < 16 * 128; i++) h[i] = -1;
        xpu_memcpy(d, h.data(), 16 * 128 * 4, XPU_HOST_TO_DEVICE); xpu_wait();
        pmap(d, cl, co); xpu_wait();
        xpu_memcpy(h.data(), d, 16 * 128 * 4, XPU_DEVICE_TO_HOST); xpu_wait();
        int ncl = 0, nco = 0, firstcl = -1, cnt = 0;
        for (int c = 0; c < 16; c++) { bool any = false;
            for (int k = 0; k < 128; k++) if (h[c * 128 + k] == 1) { any = true; cnt++; if (firstcl < 0) firstcl = c; } 
            if (any) ncl++; }
        for (int k = 0; k < 128; k++) { for (int c = 0; c < 16; c++) if (h[c*128+k]==1) { nco = k + 1; break; } }
        printf("  launch(%-2d,%-3d): 生效核数=%4d 生效cluster数=%d 每cluster生效core数=%d 首个cluster=%d\n",
               cl, co, cnt, ncl, nco, firstcl);
    }
    printf("=== 同一进程连续两次 launch (cl=8,co=16) ===\n");
    for (int i = 0; i < 4096; i++) h.assign(16 * 128, -1);
    {
        std::vector<int> hh(4096, -1);
        xpu_memcpy(d, hh.data(), 4096 * 4, XPU_HOST_TO_DEVICE); xpu_wait();
        pw4(d, 111, 8, 16); xpu_wait(); xpu_memcpy(hh.data(), d, 4096 * 4, XPU_DEVICE_TO_HOST); xpu_wait();
        int c1 = 0; for (int i = 0; i < 4096; i++) if (hh[i] == 111) c1++;
        pw4(d, 222, 8, 16); xpu_wait(); xpu_memcpy(hh.data(), d, 4096 * 4, XPU_DEVICE_TO_HOST); xpu_wait();
        int c2 = 0; for (int i = 0; i < 4096; i++) if (hh[i] == 222) c2++;
        printf("  第1次写111: %d 槽  第2次写222: %d 槽 (期望各 128)\n", c1, c2);
    }
    printf("=== GM2LM 读带宽 (slab=16KB, 512MB 数据) ===\n");
    {
        size_t BYTES = 512ul << 20;
        int slab = 16384;
        int nslab = (int)(BYTES / slab);
        void *W = nullptr, *out = nullptr;
        if (xpu_malloc(&W, (size_t)nslab * slab)) { printf("malloc W fail\n"); return 1; }
        if (xpu_malloc(&out, 2048 * 4)) { printf("malloc out fail\n"); return 1; }
        std::vector<unsigned char> hw(1 << 20, 1);
        for (int i = 0; i < nslab / 64; i++)
            xpu_memcpy((char *)W + (size_t)i * 64 * slab, hw.data(), 1 << 20, XPU_HOST_TO_DEVICE);
        xpu_wait();
        int cfg[][2] = {{8, 16}, {8, 8}, {4, 16}, {1, 16}, {8, 32}};
        for (int i = 0; i < 5; i++) {
            rd(W, slab, nslab, out, cfg[i][0], cfg[i][1]); xpu_wait();
            double t0 = now();
            for (int r = 0; r < 3; r++) rd(W, slab, nslab, out, cfg[i][0], cfg[i][1]);
            xpu_wait();
            double dt = (now() - t0) / 3.0;
            printf("  grid(%d,%d): %.1f ms  %.1f GB/s\n", cfg[i][0], cfg[i][1], dt * 1000, (double)nslab * slab / dt / 1e9);
        }
    }
    printf("DONE\n");
    return 0;
}
