// cpu_vs_dn.cpp — CPU delta 段 vs dn_k(T=1) 位级对拍: 一步看 ULP, 100 步看漂移
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <xpu/runtime.h>

#define VHD 128
#define KHD 128
#define NVH 32
#define NKH 16
#define EPS 1e-5f

extern void run_delta_net_batch(int clusters, int cores, int T,
    float *S, const float *kT, const float *vT, const float *qT,
    const float *gT, const float *bT, float *outT);

static double NOW(){ using namespace std::chrono; return duration<double>(steady_clock::now().time_since_epoch()).count(); }

static long long ulp_diff(float a, float b) {
    if (a == b) return 0;
    int ia, ib; memcpy(&ia, &a, 4); memcpy(&ib, &b, 4);
    if ((ia < 0) != (ib < 0)) return 999999999;
    long long ua = ia, ub = ib;
    if (ia < 0) ua = 2147483648LL - ia;
    if (ib < 0) ub = 2147483648LL - ib;
    return (long long)llabs(ua - ub);
}

// CPU 参考 = orn3 forward() 的 [pipe] 单累加器段 (位级同 [pipe] 4行版)
static void cpu_delta(float *S, const float *kc, const float *qc, const float *vc,
                      const float *egv, const float *bv, float *dout) {
    const float qscale = 1.f / sqrtf((float)KHD);
    for (int hv = 0; hv < NVH; hv++) {
        int hk = hv % NKH;
        float eg = egv[hv], b_ = bv[hv];
        float *M = &S[(size_t)hv * VHD * KHD];
        const float *kk = &kc[hk * KHD];
        const float *qq = &qc[hk * KHD];
        const float *vv = &vc[hv * VHD];
        for (int j = 0; j < VHD; j++) {
            float *r = M + (size_t)j * KHD;
            float sk = 0.f;
            for (int i = 0; i < KHD; i++) { float a0 = r[i] * eg; r[i] = a0; sk += a0 * kk[i]; }
            float d = (vv[j] - sk) * b_;
            float so = 0.f;
            for (int i = 0; i < KHD; i++) { float a0 = r[i] + kk[i] * d; r[i] = a0; so += a0 * qq[i]; }
            dout[hv * KHD + j] = so * qscale;
        }
    }
}

int main() {
    setbuf(stdout, NULL);
    printf("=== cpu vs dn_k 位级对拍 ===\n");
    // 随机输入
    unsigned seed = 777;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return ((int)(seed >> 9) % 2000 - 1000) / 1000.f; };
    std::vector<float> kc(NKH*KHD), qc(NKH*KHD), vc(NVH*VHD), g(NVH), b(NVH), eg(NVH);
    for (auto &x : kc) x = rnd();
    for (auto &x : qc) x = rnd();
    for (auto &x : vc) x = rnd();
    for (auto &x : g)  { x = rnd() * 0.5f; }          // g_ 范围 ~±0.5
    for (auto &x : b)  { x = rnd(); }
    for (int i = 0; i < NVH; i++) eg[i] = expf(g[i]);  // host expf (同生产)

    std::vector<float> S((size_t)NVH*VHD*KHD);
    for (auto &x : S) x = rnd() * 0.1f;                // 非零初始 (生产从 0)
    std::vector<float> Sd = S;                          // device 用同一份

    // device 缓冲
    void *dS=0, *dIO=0;
    size_t sbytes = S.size()*4;
    size_t iobytes = (NKH*KHD + NKH*KHD + NVH*VHD + NVH + NVH + NVH*KHD) * 4;
    if (xpu_set_device(0) || xpu_malloc(&dS, sbytes) || xpu_malloc(&dIO, iobytes)) { printf("malloc fail\n"); return 2; }
    xpu_memcpy(dS, Sd.data(), sbytes, XPU_HOST_TO_DEVICE);
    xpu_wait();

    const int STEPS = 100;
    long long max_ulp1 = 0; double max_abs1 = 0;
    long long max_ulpN = 0; double max_absN = 0, relN = 0;
    std::vector<float> dout_c(NVH*KHD), dout_d(NVH*KHD);
    for (int step = 0; step < STEPS; step++) {
        // 新 k/v/q/g 每步 (随步变), S 递推
        for (auto &x : kc) x = rnd();
        for (auto &x : qc) x = rnd();
        for (auto &x : vc) x = rnd();
        for (int i = 0; i < NVH; i++) { g[i] = rnd()*0.5f; eg[i] = expf(g[i]); b[i] = rnd(); }

        cpu_delta(S.data(), kc.data(), qc.data(), vc.data(), eg.data(), b.data(), dout_c.data());

        // pack io: [q 2048|k 2048|v 4096|eg 32|b 32|out 4096]
        float *ioh = (float *)malloc(iobytes);
        memcpy(ioh, qc.data(), NKH*KHD*4);
        memcpy(ioh + NKH*KHD, kc.data(), NKH*KHD*4);
        memcpy(ioh + NKH*KHD*2, vc.data(), NVH*VHD*4);
        memcpy(ioh + NKH*KHD*2 + NVH*VHD, eg.data(), NVH*4);
        memcpy(ioh + NKH*KHD*2 + NVH*VHD + NVH, b.data(), NVH*4);
        float *dout_h = ioh + NKH*KHD*2 + NVH*VHD + NVH;
        xpu_memcpy(dIO, ioh, iobytes - NVH*KHD*4, XPU_HOST_TO_DEVICE);   // 除 out 段
        xpu_wait();
        // kernel 期望: kT, vT, qT, gT, bT, outT (device)
        float *io = (float*)dIO;
        run_delta_net_batch(8, 16, 1, (float*)dS,
            io + NKH*KHD,               // kT
            io + NKH*KHD*2,             // vT
            io,                         // qT
            io + NKH*KHD*2 + NVH*VHD,   // gT
            io + NKH*KHD*2 + NVH*VHD + NVH, // bT
            io + NKH*KHD*2 + NVH*VHD + NVH*2); // outT
        if (xpu_wait()) { printf("step%d kernel fail\n", step); return 3; }
        xpu_memcpy(dout_d.data(), io + NKH*KHD*2 + NVH*VHD + NVH*2, NVH*KHD*4, XPU_DEVICE_TO_HOST);
        xpu_wait();
        free(ioh);

        long long mu = 0; double ma = 0, den = 0;
        for (int i = 0; i < NVH*KHD; i++) {
            mu = std::max(mu, ulp_diff(dout_c[i], dout_d[i]));
            ma = std::max(ma, (double)fabsf(dout_c[i]-dout_d[i]));
            den = std::max(den, (double)fabsf(dout_c[i]));
        }
        if (step == 0) { max_ulp1 = mu; max_abs1 = ma; }
        if (step == STEPS-1) { max_ulpN = mu; max_absN = ma; relN = ma/(den+1e-30); }
        if (step < 3 || step == STEPS-1 || mu > 0 && step == 4)
            printf("step%3d: maxULP=%lld maxAbs=%.3e rel=%.3e\n", step, mu, ma, ma/(den+1e-30));
    }
    printf("\n== 汇总 ==\nstep1: ULP=%lld abs=%.3e  (0 = 位级)\nstep%d: ULP=%lld abs=%.3e rel=%.3e\n",
           max_ulp1, max_abs1, STEPS, max_ulpN, max_absN, relN);
    if (max_ulp1 == 0 && max_ulpN == 0) printf("VERDICT: 位级一致! 残余另有所在 (atn?)\n");
    else if (max_ulpN <= 8) printf("VERDICT: 1-8 ULP 级漂移 => FMA/位型案\n");
    else printf("VERDICT: 结构差异 => 逻辑 bug\n");
    return 0;
}
