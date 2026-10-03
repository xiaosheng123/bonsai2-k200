// dnet_merged_T.cpp - verify merged dn_k_T_merged packed-layout batch T>1
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
#define PACKED_PER_HV (2 + KHD + KHD + VHD)

extern void run_delta_net_batch_merged(int clusters, int cores, int T,
    float *S, const float *packed_in, float *outT);
extern void run_delta_net(int clusters, int cores,
    float *S, const float *k, const float *v, const float *q,
    const float *g, const float *b, float *out);

static double NOW(){ using namespace std::chrono; return duration<double>(steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char **argv){
    setbuf(stdout, NULL);
    int T = argc > 1 ? atoi(argv[1]) : 6;
    printf("=== dnet_merged_T: packed-layout T=%d ===\n", T);
    assert(xpu_set_device(0) == 0);

    size_t S_sz    = (size_t)NVH * VHD * KHD;
    size_t pack_sz = (size_t)T * NVH * PACKED_PER_HV;
    size_t out_sz  = (size_t)T * NVH * VHD;

    std::vector<float> S_init(S_sz), pack(pack_sz);
    srand(42);
    for (auto &x : S_init) x = (rand() % 2000 - 1000) / 1000.0f;
    for (auto &x : pack)   x = (rand() % 2000 - 1000) / 1000.0f;

    std::vector<float> ref_k(NVH*16*KHD), ref_q(NVH*16*KHD), ref_v(NVH*VHD), ref_g(NVH), ref_b(NVH);

    float *dS, *dS2, *dpack, *dout;
    xpu_malloc((void**)&dS,  S_sz*4);
    xpu_malloc((void**)&dS2, S_sz*4);
    xpu_malloc((void**)&dpack, pack_sz*4);
    xpu_malloc((void**)&dout, out_sz*4);
    xpu_memcpy(dS,  S_init.data(), S_sz*4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dS2, S_init.data(), S_sz*4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dpack, pack.data(), pack_sz*4, XPU_HOST_TO_DEVICE);
    xpu_wait();
    printf("alloc+copy ok, S_sz=%zu pack_sz=%zu out_sz=%zu\n", S_sz, pack_sz, out_sz);

    for (int t = 0; t < T; t++) {
        for (int hv = 0; hv < NVH; hv++) {
            const float *dst = pack.data() + ((size_t)t*NVH + hv) * PACKED_PER_HV;
            ref_g[hv] = dst[0]; ref_b[hv] = dst[1];
            memcpy(&ref_k[hv*16*KHD], dst+2,          16*KHD*4);
            memcpy(&ref_q[hv*16*KHD], dst+2+KHD,      16*KHD*4);
            memcpy(&ref_v[hv*VHD],    dst+2+KHD+KHD,  VHD*4);
        }
        run_delta_net(8, 16, dS, ref_k.data(), ref_v.data(), ref_q.data(),
                      ref_g.data(), ref_b.data(), dout + (size_t)t*NVH*VHD);
    }
    xpu_wait();
    std::vector<float> out1(out_sz);
    xpu_memcpy(out1.data(), dout, out_sz*4, XPU_DEVICE_TO_HOST);
    xpu_wait();

    run_delta_net_batch_merged(8, 16, T, dS2, dpack, dout);
    int w = xpu_wait();
    printf("merged batch xpu_wait returned %d\n", w);
    std::vector<float> out2(out_sz);
    xpu_memcpy(out2.data(), dout, out_sz*4, XPU_DEVICE_TO_HOST);
    std::vector<float> Sf1(S_sz), Sf2(S_sz);
    xpu_memcpy(Sf1.data(), dS, S_sz*4, XPU_DEVICE_TO_HOST);
    xpu_memcpy(Sf2.data(), dS2, S_sz*4, XPU_DEVICE_TO_HOST);
    xpu_wait();

    double max_err=0; int bad=0;
    for (size_t i=0;i<out1.size();i++){ double e=fabs(out1[i]-out2[i]); if(e>max_err)max_err=e; if(e>1e-3)bad++; }
    double s_err=0;
    for (size_t i=0;i<S_sz;i++){ double e=fabs(Sf1[i]-Sf2[i]); if(e>s_err)s_err=e; }
    printf("out maxerr=%.3e bad=%d/%zu  S maxerr=%.3e\n", max_err, bad, out1.size(), s_err);

    run_delta_net_batch_merged(8,16,T,dS2,dpack,dout); xpu_wait();
    int iters=50; double t0=NOW();
    for (int i=0;i<iters;i++) run_delta_net_batch_merged(8,16,T,dS2,dpack,dout);
    xpu_wait();
    double batch_ms=(NOW()-t0)/iters*1e3;
    printf("MERGED BATCH %d tok: %.3f ms total = %.4f ms/tok\n", T, batch_ms, batch_ms/T);
    printf("== done ==\n");
    return 0;
}