#include <cstdio>
#include <chrono>
#include <xpu/runtime.h>
extern void run_nop(float *o, int cl, int co);
int main() {
    xpu_set_device(0);
    float *d; xpu_malloc((void**)&d, 1024);
    run_nop(d,8,16); xpu_wait();
    int N = 100;
    auto a = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) run_nop(d,8,16);
    xpu_wait();
    auto b = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(b-a).count();
    printf("NOP kernel 100 queued + 1 wait: %.1f ms => %.3f ms/launch\n", ms, ms/N);
    double best=1e9;
    for (int t=0;t<50;t++){ auto c=std::chrono::high_resolution_clock::now(); run_nop(d,8,16); xpu_wait(); auto e=std::chrono::high_resolution_clock::now(); double m=std::chrono::duration<double,std::milli>(e-c).count(); if(m<best)best=m; }
    printf("NOP single launch+wait: %.3f ms\n", best);
    return 0;
}
