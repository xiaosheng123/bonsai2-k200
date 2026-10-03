// attn31.cpp — 钉死 api::gemm_int31 的量化语义与精度 (vs CPU f32 参考)
// 用法: safe_run 包住跑; 输出各尺寸 relrms/bad, 找 max_a/max_b 正确传法
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <xpu/runtime.h>
#include <xpu/api.h>

namespace api = baidu::xpu::api;
namespace baidu { namespace xpu { namespace api {
int gemm_int31(Context* ctx, const bool TransA, const bool TransB,
        int M, int N, int K,
        float alpha, const float* A, int lda, const float* B, int ldb,
        float beta, float* C, int ldc, float max_a, float max_b);
}}}

static double NOW(){ using namespace std::chrono; return duration<double>(steady_clock::now().time_since_epoch()).count(); }

static void run_case(api::Context *ctx, int M, int N, int K, int mode) {
    // 随机 A[M×K], B[N×K] (trans_b=true => C[M×N] = A · B^T, 同生产 scores 调用)
    std::vector<float> A((size_t)M*K), B((size_t)N*K), C((size_t)M*N, 0.f);
    unsigned seed = 12345;
    auto rnd = [&seed]() { seed = seed * 1664525u + 1013904223u; return ((int)(seed >> 9) % 4000 - 2000) / 1000.f; };
    for (auto &x : A) x = rnd();          // ±2
    for (auto &x : B) x = rnd();
    float maxa = 0.f, maxb = 0.f;
    for (auto x : A) maxa = std::max(maxa, std::fabs(x));
    for (auto x : B) maxb = std::max(maxb, std::fabs(x));
    // CPU 参考: C = A · B^T  (alpha=al 在调用里)
    float al = 1.f / sqrtf((float)K);
    std::vector<float> R((size_t)M*N, 0.f);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.f;
            for (int k = 0; k < K; k++) s += A[(size_t)i*K+k] * B[(size_t)j*K+k];
            R[(size_t)i*N+j] = al * s;
        }
    void *dA=0, *dB=0, *dC=0;
    if (xpu_malloc(&dA,(size_t)M*K*4) || xpu_malloc(&dB,(size_t)N*K*4) || xpu_malloc(&dC,(size_t)M*N*4)) {
        printf("malloc fail\n"); return;
    }
    xpu_memcpy(dA, A.data(), (size_t)M*K*4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dB, B.data(), (size_t)N*K*4, XPU_HOST_TO_DEVICE);
    xpu_wait();
    float ma = maxa, mb = maxb;
    if (mode == 1) { ma = maxa; mb = maxb; }        // maxabs
    if (mode == 2) { ma = 1.f;   mb = 1.f;   }      // 1.0
    if (mode == 3) { ma = 127.f; mb = 127.f; }      // 127 (int8 习惯)
    // beta=0 清 C
    int r = api::gemm_int31(ctx, false, true, M, N, K, al,
                            (const float*)dA, K, (const float*)dB, K, 0.f,
                            (float*)dC, N, ma, mb);
    int wr = xpu_wait();
    if (r || wr) { printf("M=%d N=%d mode=%d: FAIL r=%d wait=%d (跳过)\n", M, N, mode, r, wr); return; }
    xpu_memcpy(C.data(), dC, (size_t)M*N*4, XPU_DEVICE_TO_HOST);
    xpu_wait();
    double num = 0, den = 0; long long bad = 0; double maxe = 0;
    for (size_t i = 0; i < C.size(); i++) {
        double e = (double)C[i] - R[i];
        num += e*e; den += (double)R[i]*R[i];
        if (std::fabs(C[i]-R[i]) > 1e-4 * (1.0 + std::fabs(R[i]))) bad++;
        maxe = std::max(maxe, std::fabs(e));
    }
    double relrms = sqrt(num) / (sqrt(den) + 1e-30);
    double t0 = NOW();
    api::gemm_int31(ctx, false, true, M, N, K, al, (const float*)dA, K,
                    (const float*)dB, K, 0.f, (float*)dC, N, ma, mb);
    xpu_wait();
    double ms = (NOW()-t0)*1e3;
    printf("M=%4d N=%4d K=%4d mode=%d maxa=%.3f maxb=%.3f: relrms=%.3e bad=%lld/%d maxerr=%.3e %7.2fms\n",
           M, N, K, mode, ma, mb, relrms, bad, M*N, maxe, ms);
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    if (xpu_set_device(0)) { printf("set_device fail\n"); return 2; }
    printf("=== gemm_int31 语义探针 (vs CPU f32) ===\n");
    api::Context ctx(api::kXPU1);
    int sizes[][2] = {{64,128},{256,512},{1024,2048},{2048,4096}};  // {M, K}, N=M
    for (auto &sz : sizes)
        for (int mode = 1; mode <= 3; mode++)
            run_case(&ctx, sz[0], sz[0], sz[1], mode);
    printf("== done ==\n");
    return 0;
}
