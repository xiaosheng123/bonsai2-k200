#include <xpu/runtime.h>
#include "xpu/api.h"
#include "xpu/refactor/util/float16.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>

using namespace std::chrono;
using namespace baidu::xpu::api;

void bench_mvm(int M, int N) {
    Device dev(DeviceType::XPU1, 0);
    Context ctx(dev);
    xpu_set_device(0);
    
    int K = N;  // matrix_vector_mul: matrix[M][N] * vec[N] -> out[M]
    float *A = new float[M * N];
    float *B = new float[N];
    float *C = new float[M];
    for (int i = 0; i < M * N; i++) A[i] = (rand() % 1000) / 1000.0f;
    for (int i = 0; i < N; i++) B[i] = (rand() % 1000) / 1000.0f;
    for (int i = 0; i < M; i++) C[i] = 0.0f;
    
    void *dA, *dB, *dC;
    xpu_malloc(&dA, M * N * sizeof(float));
    xpu_malloc(&dB, N * sizeof(float));
    xpu_malloc(&dC, M * sizeof(float));
    xpu_memcpy(dA, A, M * N * sizeof(float), XPU_HOST_TO_DEVICE);
    xpu_memcpy(dB, B, N * sizeof(float), XPU_HOST_TO_DEVICE);
    xpu_wait();
    
    for (int i = 0; i < 3; i++) {
        matrix_vector_mul(&ctx, (const float*)dA, (const float*)dB, (float*)dC, M, N);
        xpu_wait();
    }
    
    const int ITERS = 200;
    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        matrix_vector_mul(&ctx, (const float*)dA, (const float*)dB, (float*)dC, M, N);
        xpu_wait();
    }
    auto t1 = high_resolution_clock::now();
    double ms = duration<double, std::milli>(t1 - t0).count() / ITERS;
    double gflops = 2.0 * M * N / 1e9;  // M*N mul-add
    double tflops = gflops / (ms / 1000.0);
    printf("MVM M=%d N=%d: %.3f ms, %.2f TFLOPS\n", M, N, ms, tflops);
    delete[] A; delete[] B; delete[] C;
    xpu_free(dA); xpu_free(dB); xpu_free(dC);
}

void bench_fc_int16_fp16(int M, int N, int K) {
    Device dev(DeviceType::XPU1, 0);
    Context ctx(dev);
    xpu_set_device(0);
    
    // A: float16 [M][K], B: int16 [K][N], C: float16 [M][N]
    // Need max_a_ptr (float* to 4 floats) and max_b (float)
    float16 *A = new float16[M * K];
    int16_t *B = new int16_t[K * N];
    float16 *C = new float16[M * N];
    float max_a[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float max_b = 1.0f;
    
    for (int i = 0; i < M * K; i++) A[i] = float16((rand() % 1000) / 1000.0f);
    for (int i = 0; i < K * N; i++) B[i] = (int16_t)(rand() % 32767);
    for (int i = 0; i < M * N; i++) C[i] = float16(0.0f);
    
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
    printf("FC_INT16_FP16 M=%d N=%d K=%d: %.3f ms, %.2f TFLOPS\n", M, N, K, ms, tflops);
    delete[] A; delete[] B; delete[] C;
    xpu_free(dA); xpu_free(dB); xpu_free(dC); xpu_free(dMaxA);
}

int main() {
    printf("=== DECODE (M=1) ===\n");
    bench_mvm(1, 11008);   // FC1 up: M=1, N=11008
    bench_mvm(1, 4096);    // FC2 down: M=1, N=4096
    bench_fc_int16_fp16(1, 11008, 4096);
    bench_fc_int16_fp16(1, 4096, 11008);
    
    printf("\n=== PREFILL BATCH (M=32) ===\n");
    bench_mvm(32, 11008);
    bench_mvm(32, 4096);
    bench_fc_int16_fp16(32, 11008, 4096);
    bench_fc_int16_fp16(32, 4096, 11008);
    
    return 0;
}