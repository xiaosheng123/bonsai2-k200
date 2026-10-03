// mlcore_main.cpp — core-blocked multi kernel 正确性 + 性能
// S [NL][M][CL=8][CID=16][HV=4][RPC=8][KHD] 布局
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>
#include <xpu/runtime.h>

extern void run_delta_net_merged_ml_core(int clusters, int cores, int M, int NL,
    float *S, const float *packed_in, float *outT);

#define VHD 128
#define KHD 128
#define NVH 32
#define NC 16
#define RPC 8
#define HVCL 4
#define PPH (2 + KHD + KHD + VHD)

#define CORE_S (HVCL * RPC * KHD)
#define CORE_P (HVCL * PPH)
#define CORE_O (HVCL * RPC)

int main() {
    int M = 4, NL = 24, NCL = 8;
    size_t slotS = (size_t)NCL * NC * CORE_S;
    size_t slotP = (size_t)NCL * NC * CORE_P;
    size_t slotO = (size_t)NCL * NC * CORE_O;

    // 总数据
    std::vector<float> S((size_t)NL*M*slotS), packed((size_t)NL*M*slotP), out((size_t)NL*M*slotO);
    std::vector<float> out2((size_t)NL*M*slotO);
    srand(7);
    for (size_t i = 0; i < S.size(); i++) S[i] = (float)((rand()%2001)-1000)/1000.f;
    for (size_t i = 0; i < packed.size(); i++) packed[i] = (float)((rand()%2001)-1000)/1000.f;

    // CPU 参考: 用 [NL][M][cl][cid][hv][rr][k] 布局
    {
        std::vector<float> S2 = S;
        const float qscale = 1.0f/sqrtf((float)KHD);
        for (int l = 0; l < NL; l++) for (int t = 0; t < M; t++)
        for (int cl = 0; cl < NCL; cl++) for (int cid = 0; cid < NC; cid++)
        {
            size_t so = ((size_t)l*M + t)*slotS + ((size_t)cl*NC + cid)*CORE_S;
            size_t po = ((size_t)l*M + t)*slotP + ((size_t)cl*NC + cid)*CORE_P;
            size_t oo = ((size_t)l*M + t)*slotO + ((size_t)cl*NC + cid)*CORE_O;
            for (int hv = 0; hv < HVCL; hv++) {
                // 全局 hv 索引 = cl*HVCL + hv (8 cl * 4 hv = 32)
                int gv = cl * HVCL + hv;
                const float *pk = &packed[po + (size_t)hv*PPH];
                float eg=pk[0], bb=pk[1]; const float *kp=pk+2,*qp=pk+2+KHD,*vp=pk+2+KHD+KHD;
                for (int rr = 0; rr < RPC; rr++) {
                    int row = cid * RPC + rr;   // 全局行
                    float *sp = &S2[so + (size_t)hv*RPC*KHD + (size_t)rr*KHD];
                    float sk=0; for (int i=0;i<KHD;i++){float a=sp[i]*eg;sp[i]=a;sk+=a*kp[i];}
                    float d=(vp[row]-sk)*bb;
                    float so2=0; for (int i=0;i<KHD;i++){float a=sp[i]+kp[i]*d;sp[i]=a;so2+=a*qp[i];}
                    out2[oo + (size_t)hv*RPC + rr] = so2*qscale;
                }
            }
        }
    }

    // GPU 分配 + 逐层调用
    float *dS, *dP, *dO;
    size_t sbytes = S.size()*4, pbytes = packed.size()*4, obytes = out.size()*4;
    xpu_malloc((void**)&dS, sbytes);
    xpu_malloc((void**)&dP, pbytes);
    xpu_malloc((void**)&dO, obytes);
    xpu_memcpy(dS, S.data(), sbytes, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dP, packed.data(), pbytes, XPU_HOST_TO_DEVICE);

    auto t0 = std::chrono::steady_clock::now();
    for (int l = 0; l < NL; l++) {
        float *S_l = dS + (size_t)l * M * slotS;
        float *P_l = dP + (size_t)l * M * slotP;
        float *O_l = dO + (size_t)l * M * slotO;
        run_delta_net_merged_ml_core(8, 16, M, 1, S_l, P_l, O_l);
    }
    xpu_wait();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("core-blocked M=%d: %.3f ms/24层 (%.4f ms/层)\n", M, ms, ms/NL);

    xpu_memcpy(out.data(), dO, obytes, XPU_DEVICE_TO_HOST);
    xpu_wait();
    float maxeo=0; size_t a1=0;
    for (size_t i=0;i<out.size();i++){float e=fabsf(out[i]-out2[i]);if(e>maxeo){maxeo=e;a1=i;}}
    printf("maxerr_out=%.6f (idx %zu) %s\n", maxeo, a1, maxeo<1e-5f?"PASS":"FAIL");
    return 0;
}