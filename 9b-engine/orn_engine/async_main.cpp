// async_main.cpp — host main for ASYNC semantics verify
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <xpu/runtime.h>

extern void run_async_verify(int clusters, int cores, float *src, float *dst);

int main() {
    int nslots = 16 * 8;
    float *d_src = nullptr, *d_dst = nullptr;
    xpu_malloc((void**)&d_src, (size_t)nslots*1024*4);
    xpu_malloc((void**)&d_dst, (size_t)nslots*4);
    std::vector<float> h((size_t)nslots*1024);
    srand(5);
    for (auto &x : h) x = (rand()%2000-1000)/1000.0f;
    xpu_memcpy(d_src, h.data(), h.size()*4, XPU_HOST_TO_DEVICE);
    xpu_wait();
    run_async_verify(8, 16, d_src, d_dst);
    xpu_wait();
    std::vector<float> r(nslots);
    xpu_memcpy(r.data(), d_dst, nslots*4, XPU_DEVICE_TO_HOST);
    xpu_wait();
    float maxerr = 0;
    for (int t = 0; t < nslots; t++) {
        double s = 0;
        for (int i = 0; i < 1024; i++) s += h[(size_t)t*1024+i];
        float expect = (float)(s * 1.0001);
        float e = fabsf(r[t] - expect);
        if (e > maxerr) maxerr = e;
    }
    printf("ASYNC verify: maxerr=%.4f %s\n", maxerr, maxerr < 0.5f ? "PASS" : "FAIL");
    printf("样本: tid0 got=%f tid100 got=%f\n", r[0], r[100]);
    return 0;
}