// calc_main.cpp — host main for compute peak benchmark
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <xpu/runtime.h>

extern void run_calc_bench(int clusters, int cores, int loops, float *dst, int mode);

int main() {
    float *d_dst = nullptr;
    xpu_malloc((void**)&d_dst, 16*8*4);
    int loops = 10000;
    // 每循环: 4 行 x 128 列 x 2 段 x 2 FLOPs(乘+加) x 8 核组? 每核 8 行? 不, 每核 4 行(模拟递推)
    // 实际: 每循环 4行*128列*2段*2 = 2048 FLOPs/核
    for (int mode = 0; mode < 2; mode++) {
        auto t0 = std::chrono::steady_clock::now();
        run_calc_bench(8, 16, loops, d_dst, mode);
        xpu_wait();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        double flops = 2048.0 * loops * 128;  // 128 cores
        printf("mode %d (%s): %.3f ms => %.2f GFLOPS (128 cores)\n",
               mode, mode==0?"scalar":"simd", ms, flops/ms/1e6);
    }
    xpu_free(d_dst);
    return 0;
}