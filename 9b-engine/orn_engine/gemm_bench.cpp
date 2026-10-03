#include <xpu/runtime.h>
#include "xpu/api.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>

using namespace std::chrono;
using namespace baidu::xpu::api;

int main() {
    Device dev(DeviceType::XPU1, 0);
    Context ctx(dev);
    xpu_set_device(0);
    
    // Typical FC shapes for Ornith-9B:
    // FFN: 4096 -> 11008 -> 4096 (two FC layers)
    // M=1 (decode), N=11008/4096, K=4096/11008
    // Prefill: M=batch (e.g. 32), same N,K
    
    const int M = 1;
    const int N = 11008;
    const int K = 4096;
    
    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[M * N];
    
    // Random init
    for (int i = 0; i < M * K; i++) A[i] = (rand() % 1000) / 1000.0f;
    for (int i = 0; i < K * N; i++) B[i] = (rand() % 1000) / 1000.0f;
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;
    
    void *dA, *dB, *dC;
    xpu_malloc(&dA, M * K * sizeof(float));
    xpu_malloc(&dB, K * N * sizeof(float));
    xpu_malloc(&dC, M * N * sizeof(float));
    
    xpu_memcpy(dA, A, M * K * sizeof(float), XPU_HOST_TO_DEVICE);
    xpu_memcpy(dB, B, K * N * sizeof(float), XPU_HOST_TO_DEVICE);
    xpu_wait();
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        search_aligned_mat_mul(&ctx, false, false, 1, M, N, K, 1.0f,
                               (const float*)dA, K, (const float*)dB, N, (float*)dC, N);
        xpu_wait();
    }
    
    // Benchmark
    const int ITERS = 50;
    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        search_aligned_mat_mul(&ctx, false, false, 1, M, N, K, 1.0f,
                               (const float*)dA, K, (const float*)dB, N, (float*)dC, N);
        xpu_wait();
    }
    auto t1 = high_resolution_clock::now();
    
    double ms = duration<double, std::milli>(t1 - t0).count() / ITERS;
    double gflops = 2.0 * M * N * K / 1e9;
    double tflops = gflops / (ms / 1000.0);
    
    printf("M=%d N=%d K=%d: %.3f ms/iter, %.2f TFLOPS\n", M, N, K, ms, tflops);
    
    // Also test M=32 (prefill batch)
    int M2 = 32;
    float *A2 = new float[M2 * K];
    float *C2 = new float[M2 * N];
    void *dA2, *dC2;
    xpu_malloc(&dA2, M2 * K * sizeof(float));
    xpu_malloc(&dC2, M2 * N * sizeof(float));
    for (int i = 0; i < M2 * K; i++) A2[i] = (rand() % 1000) / 1000.0f;
    xpu_memcpy(dA2, A2, M2 * K * sizeof(float), XPU_HOST_TO_DEVICE);
    xpu_wait();
    
    for (int i = 0; i < 3; i++) {
        search_aligned_mat_mul(&ctx, false, false, 1, M2, N, K, 1.0f,
                               (const float*)dA2, K, (const float*)dB, N, (float*)dC2, N);
        xpu_wait();
    }
    t0 = high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        search_aligned_mat_mul(&ctx, false, false, 1, M2, N, K, 1.0f,
                               (const float*)dA2, K, (const float*)dB, N, (float*)dC2, N);
        xpu_wait();
    }
    t1 = high_resolution_clock::now();
    ms = duration<double, std::milli>(t1 - t0).count() / ITERS;
    gflops = 2.0 * M2 * N * K / 1e9;
    tflops = gflops / (ms / 1000.0);
    printf("M=%d N=%d K=%d: %.3f ms/iter, %.2f TFLOPS\n", M2, N, K, ms, tflops);
    
    // Second FC: 11008 -> 4096
    int N3 = 4096, K3 = 11008;
    float *B3 = new float[K3 * N3];
    void *dB3;
    xpu_malloc(&dB3, K3 * N3 * sizeof(float));
    for (int i = 0; i < K3 * N3; i++) B3[i] = (rand() % 1000) / 1000.0f;
    xpu_memcpy(dB3, B3, K3 * N3 * sizeof(float), XPU_HOST_TO_DEVICE);
    xpu_wait();
    
    for (int i = 0; i < 3; i++) {
        search_aligned_mat_mul(&ctx, false, false, 1, M, N3, K3, 1.0f,
                               (const float*)dA, K3, (const float*)dB3, N3, (float*)dC, N3);
        xpu_wait();
    }
    t0 = high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        search_aligned_mat_mul(&ctx, false, false, 1, M, N3, K3, 1.0f,
                               (const float*)dA, K3, (const float*)dB3, N3, (float*)dC, N3);
        xpu_wait();
    }
    t1 = high_resolution_clock::now();
    ms = duration<double, std::milli>(t1 - t0).count() / ITERS;
    gflops = 2.0 * M * N3 * K3 / 1e9;
    tflops = gflops / (ms / 1000.0);
    printf("M=%d N=%d K=%d: %.3f ms/iter, %.2f TFLOPS\n", M, N3, K3, ms, tflops);
    
    return 0;
}