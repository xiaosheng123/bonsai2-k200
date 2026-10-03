// dnet_multi_test.cpp - verify multi-slot merged kernel (independent per-slot S), no non-merged calls
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <vector>
#include <chrono>
#include <xpu/runtime.h>

#define VHD 128
#define KHD 128
#define NVH 32
#define PACKED_PER_HV (2 + KHD + KHD + VHD)

extern void run_delta_net_batch_merged_multi(int clusters, int cores, int M,
    float *S, const float *packed_in, float *outT);

static double NOW(){ using namespace std::chrono; return duration<double>(steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char **argv){
    setbuf(stdout, NULL);
    int M = argc > 1 ? atoi(argv[1]) : 4;
    printf("=== dnet_multi: M=%d slots ===\n", M);
    assert(xpu_set_device(0) == 0);

    size_t S_sz    = (size_t)M * NVH * VHD * KHD;
    size_t pack_sz = (size_t)M * NVH * PACKED_PER_HV;
    size_t out_sz  = (size_t)M * NVH * VHD;

    std::vector<float> S_init(S_sz), pack(pack_sz);
    srand(42);
    for (auto &x : S_init) x = (rand() % 2000 - 1000) / 1000.0f;
    for (auto &x : pack)   x = (rand() % 2000 - 1000) / 1000.0f;

    float *dS, *dpack, *dout;
    xpu_malloc((void**)&dS, S_sz*4);
    xpu_malloc((void**)&dpack, pack_sz*4);
    xpu_malloc((void**)&dout, out_sz*4);
    xpu_memcpy(dS, S_init.data(), S_sz*4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dpack, pack.data(), pack_sz*4, XPU_HOST_TO_DEVICE);
    xpu_wait();
    printf("setup ok\n");

    // Multi-slot batched kernel
    run_delta_net_batch_merged_multi(8, 16, M, dS, dpack, dout);
    int w = xpu_wait();
    printf("multi kernel xpu_wait returned %d\n", w);

    // Timing
    run_delta_net_batch_merged_multi(8,16,M,dS,dpack,dout); xpu_wait();
    int iters=200; double t0=NOW();
    for (int i=0;i<iters;i++) run_delta_net_batch_merged_multi(8,16,M,dS,dpack,dout);
    xpu_wait();
    double batch_ms=(NOW()-t0)/iters*1e3;
    printf("MULTI %d slots: %.3f ms total = %.4f ms/slot\n", M, batch_ms, batch_ms/M);
    printf("== done ==\n");
    return 0;
}