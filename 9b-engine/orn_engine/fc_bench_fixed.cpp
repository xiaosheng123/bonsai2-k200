#include <xpu/runtime.h>
#include "xpu/api.h"
#include "xpu/refactor/util/float16.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <algorithm>

using namespace std::chrono;
using namespace baidu::xpu::api;

void compute_max_a(const float* A, int M, int K, float* max_a) {
    // max_a[4] per doc - for simplicity use global max
    float max_val = 0.0f;
    for (int i = 0; i < M * K; i++) {
        max_val = std::max(max_val, std::abs(A[i]));
    }
    for (int i = 0; i < 4; i++) max_a[i] = max_val;
}

void bench_fc_int16_fp32A_fp16C(int M, int N, int K) {
    Device dev(DeviceType::XPU1, 0);
    Context ctx(dev);
    xpu_set_device(0);
    
    float *A = new float[M * K];
    int16_t *B = new int16_t[K * N];
    float16 *C = new float16[M * N];
    float max_b = 1.0f;
    for (int i = 0; i < M * K; i++) A[i] = (rand() % 1000) / 1000.0f;
    for (int i = 0; i < K * N; i++) B[i] = (int16_t)(rand() % 32767);
    for (int i = 0; i < M * N; i++) C[i] = float16(0.0f);
    
    // Compute max_a (required!)
    float max_a[4];
    compute_max_a(A, M, K, max_a);
    
    void *dA, *dB, *dC, *dMaxA;
    xpu_malloc(&dA, M * K * sizeof(float));
    xpu_malloc(&dB, K * N * sizeof(int16_t));
    xpu_malloc(&dC, M * N * sizeof(float16));
    xpu_malloc(&dMaxA, 4 * sizeof(float));
    xpu_memcpy(dA, A, M * K * sizeof(float), XPU_HOST_TO_DEVICE);
    xpu_memcpy(dB, B, K * N * sizeof(int16_t), XPU_HOST_TO_DEVICE);
    xpu_memcpy(dMaxA, max_a, 4 * sizeof(float), XPU_HOST_TO_DEVICE);
    xpu_wait();
    
    for (int i = 0; i < 3; i++) {
        fc_int16(&ctx, false, false, M, N, K, 1.0f,
                 (const float*)dA, (float*)dMaxA, (const int16_t*)dB, max_b,
                 0.0f, (float16*)dC, nullptr, Activation_t::LINEAR);
        xpu_wait();
    }
    
    const int ITERS = 100;
    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        fc_int16(&ctx, false, false, M, N, K, 1.0f,
                 (const float*)dA, (float*)dMaxA, (const int16_t*)dB, max_b,
                 0.0f, (float16*)dC, nullptr, Activation_t::LINEAR);
        xpu_wait();
    }
    auto t1 = high_resolution_clock::now();
    double ms = duration<double, std::milli>(t1 - t0).count() / ITERS;
    double gflops = 2.0 * M * N * K / 1e9;
    double tflops = gflops / (ms / 1000.0);
    printf("FC_INT16(fp32A,int16B,fp16C) M=%d N=%d K=%d: %.3f ms, %.2f TFLOPS\n", M, N, K, ms, tflops);
    delete[] A; delete[] B; delete[] C;
    xpu_free(dA); xpu_free(dB); xpu_free(dC); xpu_free(dMaxA);
}

void bench_fc_int16_fp16A_fp16C(int M, int N, int K) {
    Device dev(DeviceType::XPU1, 0);
    Context ctx(dev);
    xpu_set_device(0);
    
    float16 *A = new float16[M * K];
    int16_t *B = new int16_t[K * N];
    float16 *C = new float16[M * N];
    float max_b = 1.0f;
    for (int i = 0; i < M * K; i++) A[i] = float16((rand() % 1000) / 1000.0f);
    for (int i = 0; i < K * N; i++) B[i] = (int16_t)(rand() % 32767);
    for (int i = 0; i < M * N; i++) C[i] = float16(0.0f);
    
    // Compute max_a for fp16 A
    float max_a[4];
    float max_val = 0.0f;
    for (int i = 0; i < M * K; i++) {
        float v = float(A[i]);
        max_val = std::max(max_val, std::abs(v));
    }
    for (int i = 0; i < 4; i++) max_a[i] = max_val;
    
    void *dA, *dB, *dC, *dMaxA;
    xpu_malloc(&dA, M * K * sizeof(float16));
    xpu_malloc(&dB, K * N * sizeof(int16_t));
    xpu_malloc(&dC, M * N * sizeof(float16));
    xpu_malloc(&dMaxA, 4 * sizeof(float));
    xpu_memcpy(dA, A, M * K * sizeof(float16), XPU_HOST_TO_DEVICE);
    xpu_memcpy(dB, B, K * N * sizeof(int16_t), XPU_HOST_TO_DEVICE);
    xpu_memcpy(dMaxA, max_a, 4 * sizeof(float), XPU_HOST_TO_DEVICE);
    xpu_wait();
    
    for (int i = 0; i < 3; i++) {
        fc_int16(&ctx, false, false, M, N, K, 1.0f,
                 (const float16*)dA, (float*)dMaxA, (const int16_t*)dB, max_b,
                 0.0f, (float16*)dC, nullptr, Activation_t::LINEAR);
        xpu_wait();
    }
    
    const int ITERS = 100;
    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        fc_int16(&ctx, false, false, M, N, K, 1.0f,
                 (const float16*)dA, (float*)dMaxA, (const int16_t*)dB, max_b,
                 0.0f, (float16*)dC, nullptr, Activation_t::LINEAR);
        xpu_wait();
    }
    auto t1 = high_resolution_clock::now();
    double ms = duration<double, std::milli>(t1 - t0).count() / ITERS;
    double gflops = 2.0 * M * N * K / 1e9;
    double tflops = gflops / (ms / 1000.0);
    printf("FC_INT16(fp16A,int16B,fp16C) M=%d N=%d K=%d: %.3f ms, %.2f TFLOPS\n", M, N, K, ms, tflops);
    delete[] A; delete[] B; delete[] C;
    xpu_free(dA); xpu_free(dB); xpu_free(dC); xpu_free(dMaxA);
}

int main() {
    printf("=== FC_INT16 with max_a_ptr FIXED ===\n\n");
    
    printf("=== DOWN PROJECTION (11008 -> 4096) ===\n");
    bench_fc_int16_fp32A_fp16C(1, 4096, 11008);
    bench_fc_int16_fp16A_fp16C(1, 4096, 11008);
    
    printf("\n=== UP PROJECTION (4096 -> 11008) ===\n");
    bench_fc_int16_fp32A_fp16C(1, 11008, 4096);
    bench_fc_int16_fp16A_fp16C(1, 11008, 4096);
    
    printf("\n=== BATCH M=32 DOWN ===\n");
    bench_fc_int16_fp32A_fp16C(32, 4096, 11008);
    bench_fc_int16_fp16A_fp16C(32, 4096, 11008);
    
    printf("\n=== BATCH M=32 UP ===\n");
    bench_fc_int16_fp32A_fp16C(32, 11008, 4096);
    bench_fc_int16_fp16A_fp16C(32, 11008, 4096);
    
    return 0;
}