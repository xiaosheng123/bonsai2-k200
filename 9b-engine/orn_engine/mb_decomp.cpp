// mb_decomp.cpp — 分解 multi kernel 开销: 真实数据 copy-only vs full
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>
#include <xpu/runtime.h>

// 复用 multi_kern.xpu (run_multi_bench): mode 0=copy, 1=copy+sum, 2=full delta
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

    std::vector<float> S(sbytes/4), packed(packB/4), out(outB/4);
    srand(7);
    for (size_t i = 0; i < S.size(); i++) S[i] = (float)((rand()%2001)-1000)/1000.f;
    for (size_t i = 0; i < packed.size(); i++) packed[i] = (float)((rand()%2001)-1000)/1000.f;

    float *dS, *dP, *dO;
    xpu_malloc((void**)&dS, sbytes);
    xpu_malloc((void**)&dP, packB);
    xpu_malloc((void**)&dO, outB);
    xpu_memcpy(dS, S.data(), sbytes, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dP, packed.data(), packB, XPU_HOST_TO_DEVICE);

    for (int mode = 0; mode < 3; mode++) {
        run_multi_bench(8, 16, M, dS, dP, dO, mode);
        xpu_wait();
        int iters = 50;
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; it++) {
            run_multi_bench(8, 16, M, dS, dP, dO, mode);
            xpu_wait();
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count() / iters;
        const char *names[] = {"copy-only", "copy+sum", "full-delta"};
        printf("M=4 mode %s: %.3f ms/层\n", names[mode], ms);
    }
    return 0;
}