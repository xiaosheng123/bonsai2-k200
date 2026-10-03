// mbatch.cpp —— 官方 gemm_int8 的 m 大小带宽曲线 + m>1 与 m=1 的行级一致性
//   目的1: 实测 m=1/4/8/16/32/64 的 GB/s 曲线 (小尺寸对拍 m=1 已是满速, 需确认批量不退化)
//   目的2: 判定算子内部对 f32 激活 a 的 int8 量化是【per-row(每行独立 scale)】还是
//          【per-tensor(整个 A 矩阵一个 scale)】=> 决定 prefill 批量化能否位级精确
//   安全: 单芯 dev0, 尺寸小 (16/33/50 MB 权重), 全程包在 safe_run.sh 内
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

static double NOW(){ using namespace std::chrono; return duration<double>(steady_clock::now().time_since_epoch()).count(); }
static uint32_t s_rng = 987654321u;
static float rnd(){ float t = 0;
    for (int k = 0; k < 3; k++){ s_rng = s_rng * 1664525u + 1013904223u;
        t += (float)((s_rng >> 8) & 0xFFFFFFu) / 16777215.f; }
    return t * 2.f / 3.f - 1.f; }

static void rr(const std::vector<float> &a, const std::vector<double> &b, const char *tag, int N, int M){
    double se = 0, sr = 0, mx = 0; int nbit = 0;
    for (size_t i = 0; i < a.size(); i++){ double d = (double)a[i] - b[i];
        se += d * d; sr += b[i] * b[i]; if (fabs(d) > mx) mx = fabs(d); }
    (void)N; (void)M;
    printf("      %-44s relrms = %.3f%%  maxabs=%.6g\n", tag, 100.0 * sqrt(se / (sr + 1e-300)), mx);
    (void)nbit;
}

int main(){
    api::Device dev(api::DeviceType::XPU1, 0);
    api::Context ctx(dev);

    // ===================== 一、m>1 vs m=1 行级一致性 =====================
    const int M = 64, N = 4096, K = 4096;
    printf("==== A. 一致性判定 (M=%d 行, n=%d, k=%d) ====\n", M, N, K);
    std::vector<float> W((size_t)N * K), A((size_t)M * K);
    for (size_t i = 0; i < W.size(); i++) W[i] = rnd();
    for (size_t i = 0; i < A.size(); i++) A[i] = rnd() * (0.2f + 3.0f * (float)((i / K) % 7) / 6.f);  // 各行动态范围差异大 (逼出 per-tensor 量化误差)
    // 权重按行 int8
    std::vector<float> srow(N); std::vector<signed char> Q((size_t)N * K);
    for (int j = 0; j < N; j++){
        float am = 0; for (int i = 0; i < K; i++){ float a = fabsf(W[(size_t)j * K + i]); if (a > am) am = a; }
        float s = am / 127.f; if (s <= 1e-30f) s = 1e-8f; srow[j] = s;
        for (int i = 0; i < K; i++){ int v = (int)lrintf(W[(size_t)j * K + i] / s);
            if (v > 127) v = 127; else if (v < -127) v = -127; Q[(size_t)j * K + i] = (signed char)v; }
    }
    // 主机参考
    std::vector<double> ref1((size_t)M * N, 0.0), reft((size_t)M * N, 0.0);
    {   // per-row 激活量化
        for (int m = 0; m < M; m++){
            float xm = 0; for (int i = 0; i < K; i++){ float a = fabsf(A[(size_t)m * K + i]); if (a > xm) xm = a; }
            float xs = xm / 127.f; if (xs <= 1e-30f) xs = 1e-8f;
            std::vector<float> aq(K);
            for (int i = 0; i < K; i++){ int v = (int)lrintf(A[(size_t)m * K + i] / xs); if (v > 127) v = 127; else if (v < -127) v = -127; aq[i] = (float)v * xs; }
            for (int j = 0; j < N; j++){ double s = 0;
                for (int i = 0; i < K; i++) s += (double)aq[i] * (double)((float)Q[(size_t)j * K + i] * srow[j]);
                ref1[(size_t)m * N + j] = s; }
        }
    }
    {   // per-tensor 激活量化 (整个 A 一个 scale)
        float xm = 0; for (size_t i = 0; i < A.size(); i++){ float a = fabsf(A[i]); if (a > xm) xm = a; }
        float xs = xm / 127.f;
        std::vector<float> aq(A.size());
        for (size_t i = 0; i < A.size(); i++){ int v = (int)lrintf(A[i] / xs); if (v > 127) v = 127; else if (v < -127) v = -127; aq[i] = (float)v * xs; }
        for (int m = 0; m < M; m++)
            for (int j = 0; j < N; j++){ double s = 0;
                for (int i = 0; i < K; i++) s += (double)aq[(size_t)m * K + i] * (double)((float)Q[(size_t)j * K + i] * srow[j]);
                reft[(size_t)m * N + j] = s; }
    }
    // 卡上
    void *dW = 0, *dA = 0, *dC = 0;
    if (xpu_malloc(&dW, (size_t)N * K) || xpu_malloc(&dA, (size_t)M * K * 4) || xpu_malloc(&dC, (size_t)M * N * 4)){ printf("FATAL malloc\n"); return 1; }
    xpu_memcpy(dW, Q.data(), (size_t)N * K, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dA, A.data(), (size_t)M * K * 4, XPU_HOST_TO_DEVICE);
    if (xpu_wait()){ printf("FATAL init wait\n"); return 1; }

    std::vector<float> y1((size_t)M * N, 0.f), yT((size_t)M * N, 0.f);
    for (int m = 0; m < M; m++){    // 逐行 m=1 (现役路径)
        int r = api::gemm_int8(&ctx, false, true, 1, N, K, 1.f,
                               (const float *)dA + (size_t)m * K, K,
                               (const int8_t *)dW, 127.f, K, 0.f, (float *)dC + (size_t)m * N, N);
        if (r){ printf("FATAL m=1 r=%d\n", r); return 1; }
    }
    if (xpu_wait()){ printf("FATAL m=1 wait\n"); return 1; }
    xpu_memcpy(y1.data(), dC, (size_t)M * N * 4, XPU_DEVICE_TO_HOST); xpu_wait();
    memset(dC, 0, 0);
    {   // 一次 m=64
        int r = api::gemm_int8(&ctx, false, true, M, N, K, 1.f,
                               (const float *)dA, K, (const int8_t *)dW, 127.f, K, 0.f, (float *)dC, N);
        if (r){ printf("FATAL m=64 r=%d\n", r); return 1; }
        if (xpu_wait()){ printf("FATAL m=64 wait\n"); return 1; }
        xpu_memcpy(yT.data(), dC, (size_t)M * N * 4, XPU_DEVICE_TO_HOST); xpu_wait();
    }
    // 比对 (y1 需 *srow 折回)
    std::vector<float> y1s((size_t)M * N), yTs((size_t)M * N);
    for (int m = 0; m < M; m++) for (int j = 0; j < N; j++){
        y1s[(size_t)m * N + j] = y1[(size_t)m * N + j] * srow[j];
        yTs[(size_t)m * N + j] = yT[(size_t)m * N + j] * srow[j]; }
    { double se = 0, sr = 0, mx = 0; int nd = 0;
      for (size_t i = 0; i < y1s.size(); i++){ if (y1s[i] != yTs[i]) nd++;
          double d = (double)y1s[i] - yTs[i]; se += d * d; sr += (double)y1s[i] * y1s[i]; if (fabs(d) > mx) mx = fabs(d); }
      printf("  ★ m=64 与 m=1 逐行: 不同元素 %d/%zu  relrms=%.6f%%  maxabs=%.6g\n", nd, y1s.size(),
             100.0 * sqrt(se / (sr + 1e-300)), mx); }
    rr(y1s, ref1, "m=1   vs 主机参考(激活 per-row 量化)", N, M);
    rr(yTs, ref1, "m=64  vs 主机参考(激活 per-row 量化)", N, M);
    rr(y1s, reft, "m=1   vs 主机参考(激活 per-tensor 量化)", N, M);
    rr(yTs, reft, "m=64  vs 主机参考(激活 per-tensor 量化)", N, M);
    xpu_free(dW); xpu_free(dA); xpu_free(dC); xpu_wait();

    // ===================== 二、m 大小带宽曲线 =====================
    printf("\n==== B. m 曲线 (单芯 dev0; 权重字节 n*k, 有效带宽 = n*k/耗时) ====\n");
    struct Sz { int n, k; };
    Sz szs[] = { {4096, 4096}, {8192, 4096}, {12288, 4096} };
    int ms[] = { 1, 4, 8, 16, 32, 64 };
    for (int si = 0; si < 3; si++){
        int n = szs[si].n, k = szs[si].k;
        size_t wb = (size_t)n * k;
        void *dB = 0, *dAb = 0, *dCb = 0;
        if (xpu_malloc(&dB, wb) || xpu_malloc(&dAb, (size_t)64 * k * 4) || xpu_malloc(&dCb, (size_t)64 * n * 4)){
            printf("FATAL malloc n=%d\n", n); return 1; }
        std::vector<signed char> B(wb, 1);
        std::vector<float> Ab((size_t)64 * k, 0.05f), Cb;
        xpu_memcpy(dB, B.data(), wb, XPU_HOST_TO_DEVICE);
        xpu_memcpy(dAb, Ab.data(), (size_t)64 * k * 4, XPU_HOST_TO_DEVICE);
        if (xpu_wait()){ printf("FATAL wait\n"); return 1; }
        printf("  n=%d k=%d 权重=%.1f MB\n", n, k, wb / 1048576.0);
        for (int mi = 0; mi < 6; mi++){
            int m = ms[mi];
            int reps = (m <= 1) ? 50 : (m <= 4 ? 30 : (m <= 8 ? 16 : (m <= 16 ? 8 : 4)));
            int r = api::gemm_int8(&ctx, false, true, m, n, k, 1.f, (const float *)dAb, k,
                                   (const int8_t *)dB, 127.f, k, 0.f, (float *)dCb, n);
            if (r){ printf("FATAL m=%d r=%d\n", m, r); return 1; }
            if (xpu_wait()){ printf("FATAL m=%d wait\n", m); return 1; }
            double t0 = NOW();
            for (int q = 0; q < reps; q++)
                api::gemm_int8(&ctx, false, true, m, n, k, 1.f, (const float *)dAb, k,
                               (const int8_t *)dB, 127.f, k, 0.f, (float *)dCb, n);
            if (xpu_wait()){ printf("FATAL m=%d wait2\n", m); return 1; }
            double dt = NOW() - t0;
            double per = dt / reps;
            printf("      m=%-3d  %.3f ms/次   %7.2f GB/s (读权重口径)   %7.3f GFLOP/s\n",
                   m, per * 1000.0, (double)wb / per / 1e9, 2.0 * (double)n * k * m / per / 1e9);
        }
        xpu_free(dB); xpu_free(dAb); xpu_free(dCb); xpu_wait();
    }
    printf("=== mbatch done ===\n");
    return 0;
}
