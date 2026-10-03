// lcl_main.cpp — host main for local memory size probe
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <xpu/runtime.h>

extern void run_lcl_touch(int clusters, int cores, int size, int loops, float *dst);

int main() {
    int sizes[] = {4096, 8192, 16384, 32768};  // floats = 16/32/64/128 KB
    int loops = 1000;
    float *d_dst = nullptr;
    xpu_malloc((void**)&d_dst, 16*8*4);
    for (int s : sizes) {
        auto t0 = std::chrono::steady_clock::now();
        run_lcl_touch(8, 16, s, loops, d_dst);
        xpu_wait();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        // 每核 2*size*loops 次读
        double flops = 2.0 * s * loops * 128;  // 128 cores
        printf("__local__ %5d floats (%5d KB): %.3f ms => %.2f GB/s/核\n",
               s, s*4/1024, ms, (2.0*s*4*loops)/(ms/1000.0)/1e9);
    }
    xpu_free(d_dst);
    return 0;
}