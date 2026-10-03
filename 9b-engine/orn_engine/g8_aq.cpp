// 官方 gemm_int8: float A 会不会被内部量化成 int8?
#include <xpu/runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include <xpu/refactor/nn.h>
namespace api = baidu::xpu::api;
namespace baidu { namespace xpu { namespace api {
int gemm_int8(Context* ctx, const bool trans_a, const bool trans_b, int m, int n, int k,
              float alpha, const float* a, int lda, const int8_t* b, float max_b, int ldb,
              float beta, float* c, int ldc);
}}}
int main() {
    setbuf(stdout, NULL);
    api::Context ctx(api::kXPU1);
    const int K = 64, N = 4;
    std::vector<float> A(K), C(N);
    // A 带离群点 (模拟激活)
    for (int k = 0; k < K; k++) A[k] = (k == K - 1) ? 1000.f : 1.f;
    std::vector<int8_t> B((size_t)N * K, 1);                 // 每行全 1
    void *dA = nullptr, *dB = nullptr, *dC = nullptr;
    xpu_malloc(&dA, K * 4); xpu_malloc(&dB, (size_t)N * K); xpu_malloc(&dC, N * 4);
    xpu_memcpy(dA, A.data(), K * 4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dB, B.data(), (size_t)N * K, XPU_HOST_TO_DEVICE);
    xpu_wait();
    api::gemm_int8(&ctx, false, true, 1, N, K, 1.f, (const float *)dA, K, (const int8_t *)dB, 127.f, K, 0.f, (float *)dC, N);
    xpu_wait();
    xpu_memcpy(C.data(), dC, N * 4, XPU_DEVICE_TO_HOST); xpu_wait();
    printf("A=[1]*63+[1000], B=全1(int8, max_b=127)\n");
    printf("  精确 float 结果应为 1063.0; 若 A 被量化成 int8(max_a=1000, step=7.87) 则为 ~1000\n");
    printf("  实测 C = %.4f %.4f %.4f %.4f\n", C[0], C[1], C[2], C[3]);

    // 中等动态范围: A 全 1, B 行 n+1, max_b=127 -> C 应 = 64*(n+1)
    for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) B[(size_t)n*K+k] = (int8_t)(n + 1);
    xpu_memcpy(dB, B.data(), (size_t)N * K, XPU_HOST_TO_DEVICE);
    for (int k = 0; k < K; k++) A[k] = 1.f;
    xpu_memcpy(dA, A.data(), K * 4, XPU_HOST_TO_DEVICE); xpu_wait();
    api::gemm_int8(&ctx, false, true, 1, N, K, 1.f, (const float *)dA, K, (const int8_t *)dB, 127.f, K, 0.f, (float *)dC, N);
    xpu_wait();
    xpu_memcpy(C.data(), dC, N * 4, XPU_DEVICE_TO_HOST); xpu_wait();
    printf("max_b=127, B 行=n+1, A=1 -> 期望 64,128,192,256: 实测 %.4f %.4f %.4f %.4f\n", C[0], C[1], C[2], C[3]);
    return 0;
}
