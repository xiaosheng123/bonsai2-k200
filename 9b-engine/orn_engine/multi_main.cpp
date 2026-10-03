// multi_main.cpp — host main for multi kernel overhead decomposition
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <xpu/runtime.h>

extern void run_multi_bench(int clusters, int cores, int M,
                            float *S, const float *packed, float *outT, int mode);

#define VHD 128
#define KHD 128
#define NVH 32
#define PPH (2 + KHD + KHD + VHD)

int main() {
    int M = 4;
    size_t sbytes = (size_t)M * NVH * VHD * KHD * 4;
    size_t packB = (size_t)M * NVH * PPH * 4;
    size_t outB = (size_t)M * NVH * VHD * 4;

    std::vector<float> S(sbytes/4, 0.5f), packed(packB/4, 1.0f), out(outB/4);
    float *dS, *dP, *dO;
    xpu_malloc((void**)&dS, sbytes);
    xpu_malloc((void**)&dP, packB);
    xpu_malloc((void**)&dO, outB);
    xpu_memcpy(dS, S.data(), sbytes, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dP, packed.data(), packB, XPU_HOST_TO_DEVICE);

    for (int mode = 0; mode < 3; mode++) {
        // warmup
        run_multi_bench(8, 16, M, dS, dP, dO, mode);
        xpu_wait();
        int iters = 20;
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; it++) {
            run_multi_bench(8, 16, M, dS, dP, dO, mode);
            xpu_wait();
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count() / iters;
        const char *names[] = {"copy-only", "copy+sum", "full-delta"};
        printf("mode %s (M=%d): %.3f ms/层\n", names[mode], M, ms);
    }
    xpu_free(dS); xpu_free(dP); xpu_free(dO);
    return 0;
}