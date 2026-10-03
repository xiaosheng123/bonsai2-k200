// mlreal3_test.cpp — 逐层 multi kernel [NL][M] 布局正确性 + 性能
// d_slotS 布局 [NL][M][NVH][VHD][KHD]: 层-major, 每层 M 槽连续
// forward_multi 接入形态: 每层调 kernel 传 S_lm = dS + l*M*sl
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>
#include <xpu/runtime.h>

extern void run_delta_net_batch_merged_multi(int clusters, int cores, int M,
    float *S, const float *packed_in, float *outT);

#define VHD 128
#define KHD 128
#define NVH 32
#define PPH (2 + KHD + KHD + VHD)

int main() {
    int M = 4, NL = 24;
    size_t sl = (size_t)NVH * VHD * KHD;         // per-slot per-layer S
    size_t sbytes = (size_t)NL * M * sl * 4;     // [NL][M] 布局
    size_t packB = (size_t)M * NVH * PPH * 4;
    size_t outB = (size_t)M * NVH * VHD * 4;

    std::vector<float> S(sbytes/4), packed(packB/4), out(outB/4), out2(outB/4);
    srand(7);
    for (size_t i = 0; i < S.size(); i++) S[i] = (float)((rand()%2001)-1000)/1000.f;
    for (size_t i = 0; i < packed.size(); i++) packed[i] = (float)((rand()%2001)-1000)/1000.f;

    // CPU 参考 [NL][M] 布局
    {
        std::vector<float> S2 = S;
        const float qscale = 1.0f/sqrtf((float)KHD);
        for (int l = 0; l < NL; l++) for (int t = 0; t < M; t++) for (int hv = 0; hv < NVH; hv++) {
            const float *pk = &packed[((size_t)t*NVH+hv)*PPH];
            float eg=pk[0], bb=pk[1]; const float *kp=pk+2,*qp=pk+2+KHD,*vp=pk+2+KHD+KHD;
            size_t base = ((size_t)l*M + t)*sl + (size_t)hv*VHD*KHD;
            for (int row = 0; row < VHD; row++) {
                float *sp = &S2[base + (size_t)row*KHD];
                float sk=0; for (int i=0;i<KHD;i++){float a=sp[i]*eg;sp[i]=a;sk+=a*kp[i];}
                float d=(vp[row]-sk)*bb;
                float so=0; for (int i=0;i<KHD;i++){float a=sp[i]+kp[i]*d;sp[i]=a;so+=a*qp[i];}
                out2[((size_t)t*NVH+hv)*VHD+row] = so*qscale;
            }
        }
    }

    float *dS, *dP, *dO;
    xpu_malloc((void**)&dS, sbytes);
    xpu_malloc((void**)&dP, packB);
    xpu_malloc((void**)&dO, outB);
    xpu_memcpy(dS, S.data(), sbytes, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dP, packed.data(), packB, XPU_HOST_TO_DEVICE);

    // 正确性 + 性能: 24 次 kernel, 每次传该层 S 块
    auto t0 = std::chrono::steady_clock::now();
    for (int l = 0; l < NL; l++) {
        float *S_lm = dS + (size_t)l * M * sl;   // [NL][M] 布局, 层 l 的 M 槽连续
        run_delta_net_batch_merged_multi(8, 16, M, S_lm, dP, dO);
    }
    xpu_wait();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("逐层 multi [NL][M]: %.3f ms/24层 (%.4f ms/层)\n", ms, ms/NL);

    xpu_memcpy(out.data(), dO, outB, XPU_DEVICE_TO_HOST);
    xpu_wait();
    float maxeo=0; size_t a1=0;
    for (size_t i=0;i<out.size();i++){float e=fabsf(out[i]-out2[i]);if(e>maxeo){maxeo=e;a1=i;}}
    printf("maxerr_out=%.6f (idx %zu) %s\n", maxeo, a1, maxeo<1e-5f?"PASS":"FAIL");
    return 0;
}