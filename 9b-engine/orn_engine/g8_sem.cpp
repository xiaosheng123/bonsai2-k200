// 官方 gemm_int8 (float A, int8 B) 语义与带宽测试
#include <xpu/runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <chrono>
#include <xpu/refactor/nn.h>
namespace api = baidu::xpu::api;
namespace baidu { namespace xpu { namespace api {
int gemm_int8(Context* ctx, const bool trans_a, const bool trans_b, int m, int n, int k,
              float alpha, const float* a, int lda, const int8_t* b, float max_b, int ldb,
              float beta, float* c, int ldc);
int fc_int8(Context* ctx, bool TransA, bool TransB, int M, int N, int K,
            const int8_t* A, float max_a, const int8_t* B, float max_b,
            int8_t* C, float max_c);
}}}
static double now() { using namespace std::chrono; return duration<double>(steady_clock::now().time_since_epoch()).count(); }

int main() {
    setbuf(stdout, NULL);
    api::Context ctx(api::kXPU1);
    // ---------- 1) 语义测试: C[n] = ? ----------
    const int K = 64, N = 8;
    std::vector<float> A(K), C(N, -999.f);
    std::vector<int8_t> B((size_t)N * K);
    for (int k = 0; k < K; k++) A[k] = 1.0f;                 // A 全 1
    for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) B[(size_t)n*K+k] = (int8_t)(n + 1);  // 行 n 全 = n+1
    void *dA = nullptr, *dB = nullptr, *dC = nullptr;
    xpu_malloc(&dA, K * 4); xpu_malloc(&dB, (size_t)N * K); xpu_malloc(&dC, N * 4);
    xpu_memcpy(dA, A.data(), K * 4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dB, B.data(), (size_t)N * K, XPU_HOST_TO_DEVICE);
    xpu_wait();
    for (int t = 0; t < 3; t++) {
        float mb = (t == 0) ? 1.0f : (t == 1 ? 0.5f : 2.0f);
        for (int n = 0; n < N; n++) C[n] = -999.f;
        xpu_memcpy(dC, C.data(), N * 4, XPU_HOST_TO_DEVICE); xpu_wait();
        int r = api::gemm_int8(&ctx, false, true, 1, N, K, 1.f, (const float *)dA, K,
                               (const int8_t *)dB, mb, K, 0.f, (float *)dC, N);
        xpu_wait();
        xpu_memcpy(C.data(), dC, N * 4, XPU_DEVICE_TO_HOST); xpu_wait();
        printf("[语义] max_b=%g rc=%d  C[0..%d] =", mb, r, N - 1);
        for (int n = 0; n < N; n++) printf(" %.4g", C[n]);
        printf("   (int8行值 n+1, K=%d; 若 C[n]=K*(n+1)*mb/127 则约定为 int8/127*max_b)\n", K);
    }
    // ---------- 2) 真实尺寸带宽 ----------
    struct { int n, k; } S[] = {{4096, 4096}, {12288, 4096}, {4096, 12288}, {248320, 4096}};
    for (int s = 0; s < 4; s++) {
        int n = S[s].n, k = S[s].k;
        size_t bbytes = (size_t)n * k;
        void *dW = nullptr, *dX = nullptr, *dY = nullptr;
        if (xpu_malloc(&dW, bbytes) || xpu_malloc(&dX, k * 4) || xpu_malloc(&dY, n * 4)) { printf("  n=%d k=%d malloc FAIL\n", n, k); continue; }
        std::vector<int8_t> hw(1 << 20, 3); std::vector<float> hx(k, 0.5f);
        for (size_t o = 0; o < bbytes; o += (1 << 20)) {
            size_t c = bbytes - o < (1u << 20) ? bbytes - o : (1u << 20);
            xpu_memcpy((char *)dW + o, hw.data(), c, XPU_HOST_TO_DEVICE);
        }
        xpu_memcpy(dX, hx.data(), k * 4, XPU_HOST_TO_DEVICE); xpu_wait();
        api::gemm_int8(&ctx, false, true, 1, n, k, 1.f, (const float *)dX, k, (const int8_t *)dW, 1.f, k, 0.f, (float *)dY, n);
        xpu_wait();
        double t0 = now();
        int R = 5;
        for (int r = 0; r < R; r++)
            api::gemm_int8(&ctx, false, true, 1, n, k, 1.f, (const float *)dX, k, (const int8_t *)dW, 1.f, k, 0.f, (float *)dY, n);
        xpu_wait();
        double dt = (now() - t0) / R;
        printf("[带宽] m=1 n=%6d k=%6d : %7.3f ms  %6.1f GB/s (权重 %.1f MB)\n", n, k, dt * 1000, bbytes / dt / 1e9, bbytes / 1048576.0);
        xpu_free(dW); xpu_free(dX); xpu_free(dY);
    }
    printf("DONE\n");
    return 0;
}
