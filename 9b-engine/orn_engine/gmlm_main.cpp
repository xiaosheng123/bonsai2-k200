// gmlm_main.cpp — host main for GM2LM latency benchmark
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <xpu/runtime.h>

extern void run_gm2lm_bench(int clusters, int cores, int loops, int bytes,
                            float *src, float *dst);

int main() {
    int loops = 100;
    int sizes[] = {1024, 4096, 16384, 65536};
    int nslots = 16 * 8;
    for (int b : sizes) {
        float *d_src = nullptr, *d_dst = nullptr;
        xpu_malloc((void**)&d_src, (size_t)nslots*16384*4);
        xpu_malloc((void**)&d_dst, (size_t)nslots*16384*4);
        std::vector<float> h((size_t)nslots*16384, 0.f);
        xpu_memcpy(d_src, h.data(), h.size()*4, XPU_HOST_TO_DEVICE);
        xpu_wait();
        auto t0 = std::chrono::steady_clock::now();
        run_gm2lm_bench(8, 16, loops, b, d_src, d_dst);
        xpu_wait();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        printf("GM2LM %6d B x %d loops (128 cores): %.3f ms => %.1f us/call\n",
               b, loops, ms, ms*1000/loops);
        xpu_free(d_src); xpu_free(d_dst);
    }
    return 0;
}