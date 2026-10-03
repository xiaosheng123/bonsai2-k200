// mlreal_test.cpp — 逐层 multi kernel 真实数据性能 + 正确性
// S 布局 [M][NVH][VHD][KHD] (单层), 每层单独调用, 24 层循环
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
    size_t sbytes = (size_t)M * sl * 4;          // one layer's S for M slots
    size_t packB = (size_t)M * NVH * PPH * 4;
    size_t outB = (size_t)M * NVH * VHD * 4;

    std::vector<float> S(sbytes/4), packed(packB/4), out(outB/4), out2(outB/4);
    srand(7);
    for (size_t i = 0; i < S.size(); i++) S[i] = (float)((rand()%2001)-1000)/1000.f;
    for (size_t i = 0; i < packed.size(); i++) packed[i] = (float)((rand()%2001)-1000)/1000.f;

    float *dS, *dP, *dO;
    xpu_malloc((void**)&dS, sbytes);
    xpu_malloc((void**)&dP, packB);
    xpu_malloc((void**)&dO, outB);
    xpu_memcpy(dS, S.data(), sbytes, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dP, packed.data(), packB, XPU_HOST_TO_DEVICE);

    // CPU 参考 (位级同序)
    {
        std::vector<float> S2 = S;
        const float qscale = 1.0f/sqrtf((float)KHD);
        for (int t = 0; t < M; t++) for (int hv = 0; hv < NVH; hv++) {
            const float *pk = &packed[((size_t)t*NVH+hv)*PPH];
            float eg=pk[0], bb=pk[1]; const float *kp=pk+2,*qp=pk+2+KHD,*vp=pk+2+KHD+KHD;
            for (int row = 0; row < VHD; row++) {
                float *sp = &S2[(size_t)t*sl + (size_t)hv*VHD*KHD + (size_t)row*KHD];
                float sk=0; for (int i=0;i<KHD;i++){float a=sp[i]*eg;sp[i]=a;sk+=a*kp[i];}
                float d=(vp[row]-sk)*bb;
                float so=0; for (int i=0;i<KHD;i++){float a=sp[i]+kp[i]*d;sp[i]=a;so+=a*qp[i];}
                out2[((size_t)t*NVH+hv)*VHD+row] = so*qscale;
            }
        }
    }

    // 逐层调 multi kernel (真实数据)
    auto t0 = std::chrono::steady_clock::now();
    for (int l = 0; l < NL; l++) {
        run_delta_net_batch_merged_multi(8, 16, M, dS, dP, dO);  // S 驻卡单层块
    }
    xpu_wait();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("逐层 multi M=%d: %.3f ms/24层 (%.4f ms/层)\n", M, ms, ms/NL);

    xpu_memcpy(out.data(), dO, outB, XPU_DEVICE_TO_HOST);
    xpu_wait();
    float maxeo=0; size_t a1=0;
    for (size_t i=0;i<out.size();i++){float e=fabsf(out[i]-out2[i]);if(e>maxeo){maxeo=e;a1=i;}}
    printf("maxerr_out=%.6f (idx %zu) %s\n", maxeo, a1, maxeo<1e-5f?"PASS":"FAIL");
    return 0;
}