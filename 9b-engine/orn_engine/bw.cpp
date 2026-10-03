#include <cstdio>
#include <chrono>
#include <vector>
#include <xpu/runtime.h>
extern void run_bw(const void *W, void *o, int chunk, int niter);
int main() {
    xpu_set_device(0);
    const int NTHR = 128;
    size_t total = (size_t)NTHR * 200 * 14336;
    std::vector<unsigned char> src(14336, 3);
    void *dw, *d2, *doo;
    int r = xpu_malloc((void**)&dw, total);
    r |= xpu_malloc((void**)&d2, total);
    r |= xpu_malloc((void**)&doo, NTHR * 4);
    printf("malloc %zu MB x2 rc=%d\n", total/1048576*2, r);
    for (size_t off = 0; off < total; off += 14336) xpu_memcpy((char*)dw + off, src.data(), 14336, XPU_HOST_TO_DEVICE);
    xpu_wait();
    double best = 1e9;
    for (int t = 0; t < 5; t++) {
        auto a = std::chrono::high_resolution_clock::now();
        xpu_memcpy(d2, dw, total, XPU_DEVICE_TO_DEVICE);
        xpu_wait();
        auto b = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(b - a).count();
        if (ms < best) best = ms;
    }
    printf("DtoD memcpy %.0fMB: %.2f ms => %.1f GB/s HBM\n", total/1048576.0, best, total*2.0/(best/1000.0)/1073741824.0);
    int its[] = {50, 100, 200};
    int chunks[] = {512, 4096, 12288};
    for (int ci = 0; ci < 3; ci++) for (int ii = 0; ii < 3; ii++) {
        int chunk = chunks[ci], niter = its[ii];
        if ((size_t)NTHR * niter * chunk > total) continue;
        run_bw(dw, doo, chunk, 2); int rc = xpu_wait();
        if (rc) { printf("chunk=%d niter=%d: LAUNCH/WAIT FAIL rc=%d\n", chunk, niter, rc); break; }
        double bestk = 1e9;
        for (int t = 0; t < 5; t++) {
            auto a = std::chrono::high_resolution_clock::now();
            run_bw(dw, doo, chunk, niter); xpu_wait();
            auto b = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(b - a).count();
            if (ms < bestk) bestk = ms;
        }
        double bytes = (double)NTHR * niter * chunk;
        printf("GM2LM chunk=%5dB x%d: %7.2f ms  agg %6.1f GB/s  per-call %5.2f us\n",
               chunk, niter, bestk, bytes*1000.0/bestk/1073741824.0, bestk*1000.0/niter);
    }
    // ===== 新增: H2D / D2H 大包实测 (官方标 PCIe Gen4; 核对实测 3.34GB/s 之谜) =====
    {
        size_t bszs[] = {16u<<20, 64u<<20, 256u<<20};
        std::vector<unsigned char> hbuf(bszs[2], 7);
        for (int b = 0; b < 3; b++) {
            size_t bs = bszs[b];
            xpu_memcpy(dw, hbuf.data(), bs, XPU_HOST_TO_DEVICE); xpu_wait(); // warmup
            double bh = 1e9, bd = 1e9;
            for (int t = 0; t < 5; t++) {
                auto a = std::chrono::high_resolution_clock::now();
                xpu_memcpy(dw, hbuf.data(), bs, XPU_HOST_TO_DEVICE);
                xpu_wait();
                auto c = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(c - a).count();
                if (ms < bh) bh = ms;
            }
            for (int t = 0; t < 5; t++) {
                auto a = std::chrono::high_resolution_clock::now();
                xpu_memcpy(hbuf.data(), dw, bs, XPU_DEVICE_TO_HOST);
                xpu_wait();
                auto c = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(c - a).count();
                if (ms < bd) bd = ms;
            }
            printf("H2D %3zuMB: %6.2f ms => %5.2f GB/s | D2H %6.2f ms => %5.2f GB/s\n",
                   bs>>20, bh, bs/(bh/1000.0)/1073741824.0, bd, bs/(bd/1000.0)/1073741824.0);
        }
    }
    return 0;
}
