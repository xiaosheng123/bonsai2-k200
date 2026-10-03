// gemmtop.cpp v2 -- probe K200 real INT8 compute ceiling via official api::gemm_int8
// FIX: A is full MxK matrix (not single row), test large-M batched gemm TFLOPS
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <chrono>
#include <xpu/runtime.h>
#include "xpu/refactor/nn.h"
#include "xpu/refactor/context/xpu_act_type.h"
namespace api = baidu::xpu::api;
namespace baidu { namespace xpu { namespace api {
int gemm_int8(Context* ctx, const bool trans_a, const bool trans_b, int m, int n, int k,
              float alpha, const float* a, int lda, const int8_t* b, float max_b, int ldb,
              float beta, float* c, int ldc);
}}}
static double NOW(){ using namespace std::chrono; return duration<double>(steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char **argv){
    setbuf(stdout, NULL);
    int dev = argc>1 ? atoi(argv[1]) : 0;
    printf("=== K200 gemm_int8 compute probe v2 (dev%d) ===\n", dev);
    assert(xpu_set_device(dev) == 0);
    api::Context ctx(api::kXPU1);

    const int N = 4096, K = 4096;
    std::vector<int8_t> hW((size_t)K*N, 42);
    void *dW=0; xpu_malloc(&dW, (size_t)K*N);
    xpu_memcpy(dW, hW.data(), (size_t)K*N, XPU_HOST_TO_DEVICE);

    int Ms[] = {1, 4, 16, 64, 256, 1024, 2048};
    float max_b = 127.f;
    for (int mi=0; mi<7; mi++){
        int M = Ms[mi];
        // A is MxK float
        std::vector<float> hA((size_t)M*K, 0.1f);
        void *dA=0, *dC=0;
        xpu_malloc(&dA, (size_t)M*K*4);
        xpu_malloc(&dC, (size_t)M*N*4);
        xpu_memcpy(dA, hA.data(), (size_t)M*K*4, XPU_HOST_TO_DEVICE);
        // warmup
        for (int i=0;i<2;i++)
            api::gemm_int8(&ctx, false, true, M, N, K, 1.f, (const float*)dA, K,
                           (const int8_t*)dW, max_b, N, 0.f, (float*)dC, N);
        xpu_wait();
        int iters = (M<=64) ? 30 : 10;
        double t0 = NOW();
        for (int i=0;i<iters;i++)
            api::gemm_int8(&ctx, false, true, M, N, K, 1.f, (const float*)dA, K,
                           (const int8_t*)dW, max_b, N, 0.f, (float*)dC, N);
        xpu_wait();
        double dt = (NOW()-t0)/iters;
        double flops = 2.0 * M * N * K;
        double tflops = flops/dt/1e12;
        printf("M=%5d  %8.3f ms  %8.2f TFLOPS\n", M, dt*1e3, tflops);
        xpu_free(dA); xpu_free(dC);
    }
    printf("== done ==\n");
    return 0;
}