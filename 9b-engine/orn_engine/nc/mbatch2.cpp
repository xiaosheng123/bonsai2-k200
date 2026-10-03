// mbatch2.cpp —— 判定 prefill 批量化能否"位级等价"于逐 token 路径
//   背景 (mbatch 实测): 官方 gemm_int8 内部把 f32 激活 a 按【per-tensor】int8 量化
//   => 一次 m=T 调用里所有 token 共用一个 max_a, 与逐 token (m=1, 每 token 自己的 max_a) 不等价
//   本程序测试一个补偿: 对 A 的每一行先乘 f_i = M0/max_i (使每行幅度都等于 M0),
//   则算子的 per-tensor scale 变成 M0/127, 每行等效量化步长 = (M0/127)/f_i = max_i/127
//   = 恰好是逐 token 路径的 per-row 步长; 输出再乘 max_i/M0 还原.
//   判定: 归一化后 m=64 与 m=1 的逐元素差异是否为 0 (位级一致) / 差多少 ulp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
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
static uint32_t s_rng = 24681357u;
static float rnd(){ float t = 0;
    for (int k = 0; k < 3; k++){ s_rng = s_rng * 1664525u + 1013904223u;
        t += (float)((s_rng >> 8) & 0xFFFFFFu) / 16777215.f; }
    return t * 2.f / 3.f - 1.f; }
static void stats(const std::vector<float> &a, const std::vector<float> &b, const char *tag){
    size_t nd = 0; double mx = 0, se = 0, sr = 0; size_t mxulp = 0;
    for (size_t i = 0; i < a.size(); i++){
        if (a[i] != b[i]) nd++;
        double d = (double)a[i] - (double)b[i]; se += d * d; sr += (double)b[i] * b[i];
        if (fabs(d) > mx) mx = fabs(d);
    }
    printf("   %-46s 不同=%zu/%zu (%.4f%%)  relrms=%.7f%%  maxabs=%.6g\n",
           tag, nd, a.size(), 100.0 * nd / a.size(), 100.0 * sqrt(se / (sr + 1e-300)), mx);
    (void)mxulp;
}
int main(){
    api::Device dev(api::DeviceType::XPU1, 0);
    api::Context ctx(dev);
    const int M = 64, N = 4096, K = 4096;
    std::vector<float> W((size_t)N * K), A((size_t)M * K);
    for (size_t i = 0; i < W.size(); i++) W[i] = rnd();
    // 两组激活: (a) 行幅度差异大 (坏情况)  (b) 行幅度接近 (真实同层 hidden state 情况)
    std::vector<float> Abad((size_t)M * K), Aok((size_t)M * K);
    for (size_t i = 0; i < Aok.size(); i++) Aok[i] = rnd();
    for (size_t i = 0; i < Abad.size(); i++) Abad[i] = rnd() * (0.2f + 3.0f * (float)((i / K) % 7) / 6.f);
    std::vector<float> srow(N); std::vector<signed char> Q((size_t)N * K);
    for (int j = 0; j < N; j++){
        float am = 0; for (int i = 0; i < K; i++){ float a = fabsf(W[(size_t)j * K + i]); if (a > am) am = a; }
        float s = am / 127.f; if (s <= 1e-30f) s = 1e-8f; srow[j] = s;
        for (int i = 0; i < K; i++){ int v = (int)lrintf(W[(size_t)j * K + i] / s);
            if (v > 127) v = 127; else if (v < -127) v = -127; Q[(size_t)j * K + i] = (signed char)v; }
    }
    void *dW = 0, *dA = 0, *dC = 0;
    if (xpu_malloc(&dW, (size_t)N * K) || xpu_malloc(&dA, (size_t)M * K * 4) || xpu_malloc(&dC, (size_t)M * N * 4)) { printf("FATAL malloc\n"); return 1; }
    xpu_memcpy(dW, Q.data(), (size_t)N * K, XPU_HOST_TO_DEVICE);
    if (xpu_wait()) { printf("FATAL init\n"); return 1; }

    const float *Aptr[2] = { Aok.data(), Abad.data() };
    const char *Aname[2] = { "行幅度接近(真实 hidden)", "行幅度差 16 倍(坏情况)" };
    for (int ci = 0; ci < 2; ci++){
        const float *A = Aptr[ci];
        printf("==== 激活样本: %s ====\n", Aname[ci]);
        std::vector<float> y1((size_t)M * N, 0.f), yT((size_t)M * N, 0.f), yN((size_t)M * N, 0.f);
        xpu_memcpy(dA, A, (size_t)M * K * 4, XPU_HOST_TO_DEVICE); xpu_wait();
        for (int m = 0; m < M; m++){                     // m=1 逐行 (现役路径)
            if (api::gemm_int8(&ctx, false, true, 1, N, K, 1.f, (const float *)dA + (size_t)m * K, K,
                               (const int8_t *)dW, 127.f, K, 0.f, (float *)dC + (size_t)m * N, N)) { printf("FATAL\n"); return 1; }
        }
        if (xpu_wait()) { printf("FATAL\n"); return 1; }
        xpu_memcpy(y1.data(), dC, (size_t)M * N * 4, XPU_DEVICE_TO_HOST); xpu_wait();
        if (api::gemm_int8(&ctx, false, true, M, N, K, 1.f, (const float *)dA, K,
                           (const int8_t *)dW, 127.f, K, 0.f, (float *)dC, N)) { printf("FATAL\n"); return 1; }
        if (xpu_wait()) { printf("FATAL\n"); return 1; }
        xpu_memcpy(yT.data(), dC, (size_t)M * N * 4, XPU_DEVICE_TO_HOST); xpu_wait();

        // --- 归一化版本: A'[i,:] = A[i,:] * f_i, f_i = M0/max_i, M0 = max_i(max_i) ---
        std::vector<float> mx(M), f(M);
        float M0 = 0;
        for (int m = 0; m < M; m++){ float am = 0;
            for (int i = 0; i < K; i++){ float a = fabsf(A[(size_t)m * K + i]); if (a > am) am = a; }
            mx[m] = am > 0 ? am : 1e-20f; }
        for (int m = 0; m < M; m++) if (mx[m] > M0) M0 = mx[m];
        for (int m = 0; m < M; m++) f[m] = M0 / mx[m];
        std::vector<float> As((size_t)M * K);
        for (int m = 0; m < M; m++) for (int i = 0; i < K; i++) As[(size_t)m * K + i] = A[(size_t)m * K + i] * f[m];
        xpu_memcpy(dA, As.data(), (size_t)M * K * 4, XPU_HOST_TO_DEVICE); xpu_wait();
        if (api::gemm_int8(&ctx, false, true, M, N, K, 1.f, (const float *)dA, K,
                           (const int8_t *)dW, 127.f, K, 0.f, (float *)dC, N)) { printf("FATAL\n"); return 1; }
        if (xpu_wait()) { printf("FATAL\n"); return 1; }
        xpu_memcpy(yN.data(), dC, (size_t)M * N * 4, XPU_DEVICE_TO_HOST); xpu_wait();

        std::vector<float> y1s((size_t)M * N), yTs((size_t)M * N), yNs((size_t)M * N);
        for (int m = 0; m < M; m++) for (int j = 0; j < N; j++){
            y1s[(size_t)m * N + j] = y1[(size_t)m * N + j] * srow[j];               // m=1 + 行 scale
            yTs[(size_t)m * N + j] = yT[(size_t)m * N + j] * srow[j];
            yNs[(size_t)m * N + j] = yN[(size_t)m * N + j] * srow[j] * (1.f / f[m]); // 归一化 + 还原
        }
        stats(yTs, y1s, "朴素 m=64 vs m=1 (逐元素)");
        stats(yNs, y1s, "★ 行归一化 m=64 vs m=1 (逐元素)");
        // 只看第一行/最后一行, 便于判断是否个别行退化
        {
            std::vector<float> a0(y1s.begin(), y1s.begin() + N), b0(yNs.begin(), yNs.begin() + N);
            size_t nd = 0; for (int j = 0; j < N; j++) if (a0[j] != b0[j]) nd++;
            printf("   [行0] 归一化后不同元素 %zu/%d\n", nd, N);
        }
        xpu_free(dA); dA = 0;
        xpu_malloc(&dA, (size_t)M * K * 4);
    }
    xpu_free(dW); xpu_free(dA); xpu_free(dC); xpu_wait();
    printf("=== mbatch2 done ===\n");
    return 0;
}
