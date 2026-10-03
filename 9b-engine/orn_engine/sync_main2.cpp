// sync_main2.cpp — host: 同步 GM2LM 4KB vs 16KB, __global_ptr__ 显式
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <xpu/runtime.h>

extern void run_sync_verify2(int clusters, int cores, int mode, float *src, float *dst);

int main() {
    int nslots = 16 * 8;
    float *d_src = nullptr, *d_dst = nullptr;
    xpu_malloc((void**)&d_src, (size_t)nslots*8192*4);
    xpu_malloc((void**)&d_dst, (size_t)nslots*4);
    std::vector<float> h((size_t)nslots*8192);
    srand(5);
    for (auto &x : h) x = (rand()%2000-1000)/1000.0f;
    xpu_memcpy(d_src, h.data(), h.size()*4, XPU_HOST_TO_DEVICE);
    xpu_wait();
    for (int mode = 0; mode < 2; mode++) {
        std::vector<float> r(nslots);
        run_sync_verify2(8, 16, mode, d_src, d_dst);
        xpu_wait();
        xpu_memcpy(r.data(), d_dst, nslots*4, XPU_DEVICE_TO_HOST);
        xpu_wait();
        float maxerr = 0;
        for (int t = 0; t < nslots; t++) {
            double s = 0;
            for (int i = 0; i < 4096; i++) s += h[(size_t)t*8192+i];
            float expect = (float)(s * 1.0001);
            float e = fabsf(r[t] - expect);
            if (e > maxerr) maxerr = e;
        }
        printf("mode %d (%s): maxerr=%.4f %s\n", mode, mode==0?"4x4KB":"1x16KB",
               maxerr, maxerr < 0.5f ? "PASS" : "FAIL");
    }
    return 0;
}