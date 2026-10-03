// dnet_T.cpp -- verify batched dn_k_T (shared-per-token k/q layout) vs per-token dn_k
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <chrono>
#include <xpu/runtime.h>

#define VHD 128
#define KHD 128
#define NVH 32

extern void run_delta_net(int clusters, int cores,
    float *S, const float *k, const float *v, const float *q,
    const float *g, const float *b, float *out);
extern void run_delta_net_batch(int clusters, int cores, int T,
    float *S, const float *kT, const float *vT, const float *qT,
    const float *gT, const float *bT, float *outT);

static double NOW(){ using namespace std::chrono; return duration<double>(steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char **argv){
    setbuf(stdout, NULL);
    int T = argc > 1 ? atoi(argv[1]) : 64;
    printf("=== dnet_T: batch(shared k/q) vs per-token(expanded k/q) T=%d ===\n", T);
    assert(xpu_set_device(0) == 0);

    // shared-per-token layout for dn_k_T: kS/qS [T][16][KHD]
    size_t ks_sz = (size_t)T * 16 * KHD;
    // expanded layout for dn_k: kE/qE [T][NVH][16][KHD] where E[t][hv][j][i] = S[t][j][i]
    size_t ke_sz = (size_t)T * NVH * 16 * KHD;
    size_t v_sz  = (size_t)T * NVH * VHD;
    size_t gb_sz = (size_t)T * NVH;
    size_t S_sz  = (size_t)NVH * VHD * KHD;

    std::vector<float> kS(ks_sz), qS(ks_sz), kE(ke_sz), qE(ke_sz);
    std::vector<float> vT(v_sz), gT(gb_sz), bT(gb_sz);
    srand(42);
    for (auto &x : kS) x = (rand() % 2000 - 1000) / 1000.0f;
    for (auto &x : qS) x = (rand() % 2000 - 1000) / 1000.0f;
    // expand: E[t][hv][16][128] = S[t][16][128] (shared across heads)
    for (int t = 0; t < T; t++)
        for (int hv = 0; hv < NVH; hv++) {
            memcpy(&kE[((size_t)t*NVH + hv)*16*KHD], &kS[(size_t)t*16*KHD], 16*KHD*4);
            memcpy(&qE[((size_t)t*NVH + hv)*16*KHD], &qS[(size_t)t*16*KHD], 16*KHD*4);
        }
    for (auto &x : vT) x = (rand() % 2000 - 1000) / 1000.0f;
    for (auto &x : gT) x = (rand() % 200 - 100) / 1000.0f;
    for (auto &x : bT) x = (rand() % 1000) / 1000.0f;

    std::vector<float> S_init(S_sz);
    for (auto &x : S_init) x = (rand() % 2000 - 1000) / 1000.0f;

    float *dS1, *dS2, *dkS, *dqS, *dkE, *dqE, *dvT, *dgT, *dbT, *doutT;
    xpu_malloc((void**)&dS1, S_sz*4);
    xpu_malloc((void**)&dS2, S_sz*4);
    xpu_malloc((void**)&dkS, ks_sz*4);
    xpu_malloc((void**)&dqS, ks_sz*4);
    xpu_malloc((void**)&dkE, ke_sz*4);
    xpu_malloc((void**)&dqE, ke_sz*4);
    xpu_malloc((void**)&dvT, v_sz*4);
    xpu_malloc((void**)&dgT, gb_sz*4);
    xpu_malloc((void**)&dbT, gb_sz*4);
    xpu_malloc((void**)&doutT, (size_t)T*NVH*VHD*4);
    xpu_memcpy(dS1, S_init.data(), S_sz*4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dS2, S_init.data(), S_sz*4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dkS, kS.data(), ks_sz*4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dqS, qS.data(), ks_sz*4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dkE, kE.data(), ke_sz*4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dqE, qE.data(), ke_sz*4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dvT, vT.data(), v_sz*4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dgT, gT.data(), gb_sz*4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dbT, bT.data(), gb_sz*4, XPU_HOST_TO_DEVICE);
    xpu_wait();

    // ---- 1) correctness: per-token dn_k (dS1, expanded k/q) vs batched dn_k_T (dS2, shared k/q) ----
    for (int t = 0; t < T; t++)
        run_delta_net(8, 16, dS1,
                      dkE + (size_t)t*NVH*16*KHD, dvT + (size_t)t*NVH*VHD,
                      dqE + (size_t)t*NVH*16*KHD, dgT + (size_t)t*NVH,
                      dbT + (size_t)t*NVH, doutT + (size_t)t*NVH*VHD);
    xpu_wait();
    std::vector<float> out1((size_t)T*NVH*VHD);
    xpu_memcpy(out1.data(), doutT, (size_t)T*NVH*VHD*4, XPU_DEVICE_TO_HOST);
    xpu_wait();

    run_delta_net_batch(8, 16, T, dS2, dkS, dvT, dqS, dgT, dbT, doutT);
    xpu_wait();
    std::vector<float> out2((size_t)T*NVH*VHD);
    xpu_memcpy(out2.data(), doutT, (size_t)T*NVH*VHD*4, XPU_DEVICE_TO_HOST);
    std::vector<float> Sf1(S_sz), Sf2(S_sz);
    xpu_memcpy(Sf1.data(), dS1, S_sz*4, XPU_DEVICE_TO_HOST);
    xpu_memcpy(Sf2.data(), dS2, S_sz*4, XPU_DEVICE_TO_HOST);
    xpu_wait();

    double max_err = 0; int bad = 0;
    for (size_t i = 0; i < out1.size(); i++) {
        double e = fabs(out1[i] - out2[i]);
        if (e > max_err) max_err = e;
        if (e > 1e-3) bad++;
    }
    double s_err = 0;
    for (size_t i = 0; i < S_sz; i++) {
        double e = fabs(Sf1[i] - Sf2[i]);
        if (e > s_err) s_err = e;
    }
    printf("out maxerr=%.3e bad=%d/%zu  S maxerr=%.3e\n", max_err, bad, out1.size(), s_err);

    // ---- 2) timing ----
    run_delta_net_batch(8, 16, T, dS2, dkS, dvT, dqS, dgT, dbT, doutT);
    xpu_wait();
    int iters = 50;
    double t0 = NOW();
    for (int i = 0; i < iters; i++)
        run_delta_net_batch(8, 16, T, dS2, dkS, dvT, dqS, dgT, dbT, doutT);
    xpu_wait();
    double batch_ms = (NOW()-t0)/iters*1e3;

    t0 = NOW();
    for (int i = 0; i < iters; i++) {
        for (int t = 0; t < T; t++)
            run_delta_net(8, 16, dS1, dkE + (size_t)t*NVH*16*KHD, dvT + (size_t)t*NVH*VHD,
                          dqE + (size_t)t*NVH*16*KHD, dgT + (size_t)t*NVH,
                          dbT + (size_t)t*NVH, doutT + (size_t)t*NVH*VHD);
        xpu_wait();
    }
    double per_ms = (NOW()-t0)/iters*1e3;
    printf("BATCH  %d tok: %.3f ms total = %.4f ms/tok\n", T, batch_ms, batch_ms/T);
    printf("PERTOK %d tok: %.3f ms total = %.4f ms/tok  (speedup %.2fx)\n", T, per_ms, per_ms/T, per_ms/batch_ms);
    printf("== done ==\n");
    return 0;
}