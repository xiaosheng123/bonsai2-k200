#include <xpu/runtime.h>
#include "xpu/api.h"
#include "xpu/refactor/util/float16.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>

using namespace std::chrono;
using namespace baidu::xpu::api;

void bench_gemm_int16(int M, int N, int K) {
    Device dev(DeviceType::XPU1, 0);
    Context ctx(dev);
    xpu_set_device(0);
    
    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[M * N];
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
    
    for (int i = 0; i < 3; i++) {
        gemm_int16(&ctx, false, false, M, N, K, 1.0f,
                   (const float*)dA, K, (const float*)dB, N, 0.0f, (float*)dC, N);
        xpu_wait();
    }
    
    const int ITERS = 100;
    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        gemm_int16(&ctx, false, false, M, N, K, 1.0f,
                   (const float*)dA, K, (const float*)dB, N, 0.0f, (float*)dC, N);
        xpu_wait();
    }
    auto t1 = high_resolution_clock::now();
    double ms = duration<double, std::milli>(t1 - t0).count() / ITERS;
    double gflops = 2.0 * M * N * K / 1e9;
    double tflops = gflops / (ms / 1000.0);
    printf("GEMM_INT16 M=%d N=%d K=%d: %.3f ms, %.2f TFLOPS\n", M, N, K, ms, tflops);
    delete[] A; delete[] B; delete[] C;
    xpu_free(dA); xpu_free(dB); xpu_free(dC);
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
    
    void *dA, *dB, *dC;
    xpu_malloc(&dA, M * K * sizeof(float));
    xpu_malloc(&dB, K * N * sizeof(int16_t));
    xpu_malloc(&dC, M * N * sizeof(float16));
    xpu_memcpy(dA, A, M * K * sizeof(float), XPU_HOST_TO_DEVICE);
    xpu_memcpy(dB, B, K * N * sizeof(int16_t), XPU_HOST_TO_DEVICE);
    xpu_wait();
    
    for (int i = 0; i < 3; i++) {
        fc_int16(&ctx, false, false, M, N, K, 1.0f,
                 (const float*)dA, nullptr, (const int16_t*)dB, max_b,
                 0.0f, (float16*)dC, nullptr, Activation_t::LINEAR);
        xpu_wait();
    }
    
    const int ITERS = 100;
    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        fc_int16(&ctx, false, false, M, N, K, 1.0f,
                 (const float*)dA, nullptr, (const int16_t*)dB, max_b,
                 0.0f, (float16*)dC, nullptr, Activation_t::LINEAR);
        xpu_wait();
    }
    auto t1 = high_resolution_clock::now();
    double ms = duration<double, std::milli>(t1 - t0).count() / ITERS;
    double gflops = 2.0 * M * N * K / 1e9;
    double tflops = gflops / (ms / 1000.0);
    printf("FC_INT16(fp32A,int16B,fp16C) M=%d N=%d K=%d: %.3f ms, %.2f TFLOPS\n", M, N, K, ms, tflops);
    delete[] A; delete[] B; delete[] C;
    xpu_free(dA); xpu_free(dB); xpu_free(dC);
}

void bench_gemm_int8(int M, int N, int K) {
    Device dev(DeviceType::XPU1, 0);
    Context ctx(dev);
    xpu_set_device(0);
    
    float *A = new float[M * K];
    int8_t *B = new int8_t[K * N];
    float *C = new float[M * N];
    float max_b = 1.0f;
    for (int i = 0; i < M * K; i++) A[i] = (rand() % 1000) / 1000.0f;
    for (int i = 0; i < K * N; i++) B[i] = (int8_t)(rand() % 255 - 127);
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;
    
    void *dA, *dB, *dC;
    xpu_malloc(&dA, M * K * sizeof(float));
    xpu_malloc(&dB, K * N * sizeof(int8_t));
    xpu_malloc(&dC, M * N * sizeof(float));
    xpu_memcpy(dA, A, M * K * sizeof(float), XPU_HOST_TO_DEVICE);
    xpu_memcpy(dB, B, K * N * sizeof(int8_t), XPU_HOST_TO_DEVICE);
    xpu_wait();
    
    for (int i = 0; i < 3; i++) {
        gemm_int8(&ctx, false, false, M, N, K, 1.0f,
                  (const float*)dA, K, (const int8_t*)dB, max_b, N,
                  0.0f, (float*)dC, N);
        xpu_wait();
    }
    
    const int ITERS = 100;
    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        gemm_int8(&ctx, false, false, M, N, K, 1.0f,
                  (const float*)dA, K, (const int8_t*)dB, max_b, N,
                  0.0f, (float*)dC, N);
        xpu_wait();
    }
    auto t1 = high_resolution_clock::now();
    double ms = duration<double, std::milli>(t1 - t0).count() / ITERS;
    double gflops = 2.0 * M * N * K / 1e9;
    double tflops = gflops / (ms / 1000.0);
    printf("GEMM_INT8 M=%d N=%d K=%d: %.3f ms, %.2f TFLOPS\n", M, N, K, ms, tflops);
    delete[] A; delete[] B; delete[] C;
    xpu_free(dA); xpu_free(dB); xpu_free(dC);
}

int main() {
    printf("=== DOWN PROJECTION (11008 -> 4096) ===\n");
    bench_gemm_int16(1, 4096, 11008);
    bench_fc_int16_fp32A_fp16C(1, 4096, 11008);
    bench_gemm_int8(1, 4096, 11008);
    
    printf("\n=== UP PROJECTION (4096 -> 11008) ===\n");
    bench_gemm_int16(1, 11008, 4096);
    bench_fc_int16_fp32A_fp16C(1, 11008, 4096);
    bench_gemm_int8(1, 11008, 4096);
    
    printf("\n=== BATCH M=32 DOWN ===\n");
    bench_gemm_int16(32, 4096, 11008);
    bench_fc_int16_fp32A_fp16C(32, 4096, 11008);
    bench_gemm_int8(32, 4096, 11008);
    
    printf("\n=== BATCH M=32 UP ===\n");
    bench_gemm_int16(32, 11008, 4096);
    bench_fc_int16_fp32A_fp16C(32, 11008, 4096);
    bench_gemm_int8(32, 11008, 4096);
    
    return 0;
}