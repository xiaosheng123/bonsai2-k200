// mldbg_test.cpp — 调试 multi kernel inf 来源: 打印具体差异
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <xpu/runtime.h>

extern void run_delta_net_batch_merged_multi(int clusters, int cores, int M,
    float *S, const float *packed_in, float *outT);

#define VHD 128
#define KHD 128
#define NVH 32
#define PPH (2 + KHD + KHD + VHD)

int main() {
    int M = 4, NL = 1;   // 先最小: 1 槽 1 层
    size_t sl = (size_t)NVH * VHD * KHD;
    size_t sbytes = (size_t)M * sl * 4;
    size_t packB = (size_t)M * NVH * PPH * 4;
    size_t outB = (size_t)M * NVH * VHD * 4;

    std::vector<float> S(sbytes/4), packed(packB/4), out(outB/4), out2(outB/4);
    srand(7);
    for (size_t i = 0; i < S.size(); i++) S[i] = (float)((rand()%2001)-1000)/1000.f;
    for (size_t i = 0; i < packed.size(); i++) packed[i] = (float)((rand()%2001)-1000)/1000.f;

    // CPU 参考
    {
        std::vector<float> S2 = S;
        const float qscale = 1.0f/sqrtf((float)KHD);
        for (int t = 0; t < M; t++) for (int hv = 0; hv < NVH; hv++) {
            const float *pk = &packed[((size_t)t*NVH+hv)*PPH];
            float eg=pk[0], bb=pk[1]; const float *kp=pk+2,*qp=pk+2+KHD,*vp=pk+2+KHD+KHD;
            for (int row = 0; row < VHD; row++) {
                float *sp = &S2[(size_t)t*sl + (size_t)hv*VHD*KHD + (size_t)row*KHD];
                float sk=0; 
                for (int i=0;i<KHD;i++){float a=sp[i]*eg;sp[i]=a;sk+=a*kp[i];}
                float d=(vp[row]-sk)*bb;
                float so=0; 
                for (int i=0;i<KHD;i++){float a=sp[i]+kp[i]*d;sp[i]=a;so+=a*qp[i];}
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
    run_delta_net_batch_merged_multi(8, 16, M, dS, dP, dO);
    xpu_wait();
    xpu_memcpy(out.data(), dO, outB, XPU_DEVICE_TO_HOST);
    xpu_wait();

    int bad = 0, nan_count = 0, inf_count = 0;
    float maxeo = 0; size_t a1 = 0;
    for (size_t i = 0; i < out.size(); i++) {
        if (std::isnan(out[i])) { nan_count++; if (bad<10) printf("NAN @ %zu (ref %f)\n", i, out2[i]); bad++; }
        else if (std::isinf(out[i])) { inf_count++; if (bad<10) printf("INF @ %zu (ref %f)\n", i, out2[i]); bad++; }
        float e = fabsf(out[i]-out2[i]);
        if (e > maxeo) { maxeo = e; a1 = i; }
    }
    printf("M=%d: nan=%d inf=%d maxerr=%.6f (idx %zu)\n", M, nan_count, inf_count, maxeo, a1);

    // 打印几个 hv 的头部对比
    for (int hv = 0; hv < 3; hv++) {
        printf("hv%d out[0..3]: GPU %.6f %.6f %.6f %.6f | CPU %.6f %.6f %.6f %.6f\n",
               hv, out[hv*VHD+0], out[hv*VHD+1], out[hv*VHD+2], out[hv*VHD+3],
               out2[hv*VHD+0], out2[hv*VHD+1], out2[hv*VHD+2], out2[hv*VHD+3]);
    }
    return 0;
}