#include <cstdio>
#include <chrono>
#include <xpu/runtime.h>
extern void run_reg(float *o, int n);
int main() {
    xpu_set_device(0);
    float *d; xpu_malloc((void**)&d, 8*8*4);
    int N = 1000000;
    run_reg(d, 1000); xpu_wait();
    double best = 1e9;
    for (int t = 0; t < 5; t++) {
        auto a = std::chrono::high_resolution_clock::now();
        run_reg(d, N); xpu_wait();
        auto b = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(b-a).count();
        if (ms < best) best = ms;
    }
    double flops = (double)128 * N * 8 * 2;
    printf("ALU FMA bench: %.2f ms for %d it x 8 FMA x 128 core => %.2f GFLOPS (%.2f flop/core/cycle @900MHz)\n",
           best, N, flops/(best/1000.0)/1e9, flops/(best/1000.0)/128.0/900e6);
    return 0;
}
