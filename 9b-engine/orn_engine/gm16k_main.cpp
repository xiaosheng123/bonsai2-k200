// gm16k_main.cpp — host: 确定性 16KB GM2LM
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <xpu/runtime.h>

extern void run_gm16k(int clusters, int cores, float *src, float *dst);

int main() {
    int nslots = 16 * 8;
    // src[i] = i % 4096, 确定性
    std::vector<float> src((size_t)nslots*4096);
    for (size_t i = 0; i < src.size(); i++) src[i] = (float)(i % 4096);
    float *d_src, *d_dst;
    xpu_malloc((void**)&d_src, src.size()*4);
    xpu_malloc((void**)&d_dst, (size_t)nslots*4);
    xpu_memcpy(d_src, src.data(), src.size()*4, XPU_HOST_TO_DEVICE);
    xpu_wait();
    run_gm16k(8, 16, d_src, d_dst);
    xpu_wait();
    std::vector<float> r(nslots);
    xpu_memcpy(r.data(), d_dst, r.size()*4, XPU_DEVICE_TO_HOST);
    xpu_wait();
    // 期望: sum(i for i in 0..4095) * 1.0001
    double expect = (4095.0*4096/2) * 1.0001;
    float maxerr = 0;
    for (int i = 0; i < nslots; i++) {
        float e = fabsf(r[i] - (float)expect);
        if (e > maxerr) maxerr = e;
    }
    printf("GM16K deterministic: expect=%.0f got[0]=%.1f got[63]=%.1f maxerr=%.2f %s\n",
           expect, r[0], r[63], maxerr, maxerr < 1.0f ? "PASS" : "FAIL");
    return 0;
}