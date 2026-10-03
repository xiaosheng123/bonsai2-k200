// mlcore_safe_dbg.cpp — safe core-blocked kernel M=1 单层验证
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <xpu/runtime.h>

extern void run_delta_net_merged_ml_core_safe(int clusters, int cores, int M, int NL,
    float *S, const float *packed_in, float *outT);

#define VHD 128
#define KHD 128
#define NVH 32
#define NC 16
#define RPC 8
#define HVCL 4
#define PPH (2 + KHD + KHD + VHD)
#define NCL 8
#define CORE_S (HVCL * RPC * KHD)
#define CORE_P (HVCL * PPH)
#define CORE_O (HVCL * RPC)

int main() {
    int M = 1, NL = 1;
    size_t slotS = (size_t)NCL * NC * CORE_S;
    size_t slotP = (size_t)NCL * NC * CORE_P;
    size_t slotO = (size_t)NCL * NC * CORE_O;

    std::vector<float> S(slotS), packed(slotP), out(slotO), out2(slotO);
    srand(7);
    for (size_t i = 0; i < S.size(); i++) S[i] = (float)((rand()%2001)-1000)/1000.f;
    for (size_t i = 0; i < packed.size(); i++) packed[i] = (float)((rand()%2001)-1000)/1000.f;

    // CPU 参考 [NL][M][cl][cid][hv][rr][k]
    {
        std::vector<float> S2 = S;
        const float qscale = 1.0f/sqrtf((float)KHD);
        for (int cl = 0; cl < NCL; cl++) for (int cid = 0; cid < NC; cid++) {
            size_t so = ((size_t)cl*NC + cid)*CORE_S;
            size_t po = ((size_t)cl*NC + cid)*CORE_P;
            size_t oo = ((size_t)cl*NC + cid)*CORE_O;
            for (int hv = 0; hv < HVCL; hv++) {
                int gv = cl*HVCL+hv;
                const float *pk = &packed[po + (size_t)hv*PPH];
                float eg=pk[0], bb=pk[1]; const float *kp=pk+2,*qp=pk+2+KHD,*vp=pk+2+KHD+KHD;
                for (int rr = 0; rr < RPC; rr++) {
                    int row = cid*RPC+rr;
                    float *sp = &S2[so + (size_t)hv*RPC*KHD + (size_t)rr*KHD];
                    float sk=0; for (int i=0;i<KHD;i++){float a=sp[i]*eg;sp[i]=a;sk+=a*kp[i];}
                    float d=(vp[row]-sk)*bb;
                    float so2=0; for (int i=0;i<KHD;i++){float a=sp[i]+kp[i]*d;sp[i]=a;so2+=a*qp[i];}
                    out2[oo + (size_t)hv*RPC + rr] = so2*qscale;
                }
            }
        }
    }

    float *dS, *dP, *dO;
    xpu_malloc((void**)&dS, S.size()*4);
    xpu_malloc((void**)&dP, packed.size()*4);
    xpu_malloc((void**)&dO, out.size()*4);
    xpu_memcpy(dS, S.data(), S.size()*4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dP, packed.data(), packed.size()*4, XPU_HOST_TO_DEVICE);

    run_delta_net_merged_ml_core_safe(8, 16, M, NL, dS, dP, dO);
    xpu_wait();
    xpu_memcpy(out.data(), dO, out.size()*4, XPU_DEVICE_TO_HOST);
    xpu_wait();

    float maxeo=0; size_t a1=0;
    for (size_t i=0;i<out.size();i++){float e=fabsf(out[i]-out2[i]);if(e>maxeo){maxeo=e;a1=i;}}
    printf("core_safe M=%d: maxerr=%.6f (idx %zu) %s\n", M, maxeo, a1, maxeo<1e-5f?"PASS":"FAIL");
    return 0;
}