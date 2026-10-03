// mlpipe_test.cpp — 全层流水 kernel vs 逐层 multi kernel 对拍 + 性能
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <chrono>
#include <xpu/runtime.h>

extern void run_delta_net_merged_ml_pipe(int clusters, int cores, int M, int NL,
    float *S, const float *packed_in, float *outT);
extern void run_delta_net_merged_multi(int clusters, int cores, int M,
    float *S, const float *packed_in, float *outT);

#define VHD 128
#define KHD 128
#define NVH 32
#define PPH (2 + KHD + KHD + VHD)  // 386

// CPU 参考: 逐层逐槽逐hv, S 就地更新. 与 kernel 位级同序.
static void cpu_ref(int M, int NL, std::vector<float> &S, const std::vector<float> &packed,
                    std::vector<float> &out) {
    const float qscale = 1.0f / sqrtf((float)KHD);
    for (int l = 0; l < NL; l++) {
        for (int t = 0; t < M; t++) {
            size_t base = ((size_t)t * NL + l) * NVH * VHD * KHD;
            for (int hv = 0; hv < NVH; hv++) {
                const float *pk = &packed[(((size_t)t * NL + l) * NVH + hv) * PPH];
                float eg = pk[0], bb = pk[1];
                const float *kp = pk + 2, *qp = pk + 2 + KHD, *vp = pk + 2 + KHD + KHD;
                for (int row = 0; row < VHD; row++) {
                    float *sp = &S[base + (size_t)hv * VHD * KHD + (size_t)row * KHD];
                    float sk = 0.f;
                    for (int i = 0; i < KHD; i++) { float a0 = sp[i]*eg; sp[i]=a0; sk += a0*kp[i]; }
                    float d = (vp[row]-sk)*bb;
                    float so = 0.f;
                    for (int i = 0; i < KHD; i++) { float a0 = sp[i]+kp[i]*d; sp[i]=a0; so += a0*qp[i]; }
                    out[(((size_t)t * NL + l) * NVH + hv) * VHD + row] = so * qscale;
                }
            }
        }
    }
}

int main() {
    int M = 4, NL = 24;
    size_t sl_stride = (size_t)NVH * VHD * KHD;
    size_t sbytes = (size_t)M * NL * sl_stride * 4;
    size_t packB = (size_t)M * NL * NVH * PPH * 4;
    size_t outB = (size_t)M * NL * NVH * VHD * 4;
    printf("M=%d NL=%d S=%.1fMB packed=%.1fMB out=%.1fMB\n", M, NL,
           sbytes/1048576.0, packB/1048576.0, outB/1048576.0);

    std::vector<float> S(M*NL*sl_stride), packed(M*NL*NVH*PPH), out(M*NL*NVH*VHD);
    srand(42);
    for (size_t i = 0; i < S.size(); i++) S[i] = (float)((rand()%2001)-1000)/1000.f;
    for (size_t i = 0; i < packed.size(); i++) packed[i] = (float)((rand()%2001)-1000)/1000.f;

    float *dS = nullptr, *dP = nullptr, *dO = nullptr;
    if (xpu_malloc((void**)&dS, sbytes)) { printf("FATAL alloc S\n"); return 1; }
    if (xpu_malloc((void**)&dP, packB)) { printf("FATAL alloc P\n"); return 1; }
    if (xpu_malloc((void**)&dO, outB)) { printf("FATAL alloc O\n"); return 1; }

    // ============ 正确性 (全层 pipe) ============
    std::vector<float> S2 = S;
    std::vector<float> out2(out.size());
    cpu_ref(M, NL, S2, packed, out2);

    xpu_memcpy(dS, S.data(), sbytes, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dP, packed.data(), packB, XPU_HOST_TO_DEVICE);
    xpu_wait();

    auto t0 = std::chrono::steady_clock::now();
    run_delta_net_merged_ml_pipe(8, 16, M, NL, dS, dP, dO);
    xpu_wait();
    auto t1 = std::chrono::steady_clock::now();
    printf("PIPE kernel 1st: %.3f ms (%.3f ms/层, %.3f ms/槽层)\n",
           std::chrono::duration<double,std::milli>(t1-t0).count(),
           std::chrono::duration<double,std::milli>(t1-t0).count()/NL,
           std::chrono::duration<double,std::milli>(t1-t0).count()/NL/M);

    xpu_memcpy(out.data(), dO, outB, XPU_DEVICE_TO_HOST);
    xpu_wait();
    float maxeo = 0; size_t arg1 = 0;
    for (size_t i = 0; i < out.size(); i++) { float e = fabsf(out[i]-out2[i]); if (e>maxeo){maxeo=e;arg1=i;} }
    std::vector<float> Sback(sbytes/4);
    xpu_memcpy(Sback.data(), dS, sbytes, XPU_DEVICE_TO_HOST);
    xpu_wait();
    float maxes = 0; size_t arg2 = 0;
    for (size_t i = 0; i < Sback.size(); i++) { float e = fabsf(Sback[i]-S2[i]); if (e>maxes){maxes=e;arg2=i;} }
    printf("PIPE maxerr_out=%.6f (idx %zu) maxerr_S=%.6f (idx %zu)  %s\n",
           maxeo, arg1, maxes, arg2, (maxeo<1e-5f && maxes<1e-5f) ? "PASS" : "FAIL");

    // ============ 性能对比: pipe vs 逐层 multi ============
    double psum = 0, msum = 0;
    for (int it = 0; it < 10; it++) {
        // pipe 全周期
        auto b0 = std::chrono::steady_clock::now();
        xpu_memcpy(dP, packed.data(), packB, XPU_HOST_TO_DEVICE);
        auto k0 = std::chrono::steady_clock::now();
        run_delta_net_merged_ml_pipe(8, 16, M, NL, dS, dP, dO);
        xpu_wait();
        auto k1 = std::chrono::steady_clock::now();
        xpu_memcpy(out.data(), dO, outB, XPU_DEVICE_TO_HOST);
        xpu_wait();
        auto b1 = std::chrono::steady_clock::now();
        psum += std::chrono::duration<double,std::milli>(b1-b0).count();
    }
    printf("PIPE avg 全周期: %.3f ms/步\n", psum/10);

    // 逐层 multi: 每层独立 S 块 (S 布局 [M][NL]...), 24 次调用
    for (int it = 0; it < 10; it++) {
        auto b0 = std::chrono::steady_clock::now();
        xpu_memcpy(dP, packed.data(), packB, XPU_HOST_TO_DEVICE);
        for (int l = 0; l < NL; l++) {
            float *S_l = dS + (size_t)l * sl_stride;  // 注意: multi 期望 [M][hv...], 但这里 S 是 [M][NL][...]
            // multi kernel 的 S 是 [M][NVH][VHD][KHD] (单层), 所以每层传 S + t*NL... 需要按层取
            // 简化: multi 的 S 布局是 [M][NVH*VHD*KHD], 每层独立调用, 用 dS + (t*NL+l) 不行
            // 这里直接用每层: S_l 布局 [M][NL][NVH...] 中第 l 层所有槽 = stride NL 交错, 不连续
            // 跳过 — 逐层 multi 的正确布局是 [M][1][NVH...], 单独测
        }
        auto b1 = std::chrono::steady_clock::now();
        msum += std::chrono::duration<double,std::milli>(b1-b0).count();
    }
    printf("(逐层 multi 布局不匹配, 跳过)\n");
    return 0;
}