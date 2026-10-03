// dnet_bench.cpp — verify delta-net XPU kernel correctness + speed
// compile: g++ -O2 -I<xtdk>/include dnet_bench.cpp delta_net.host.o delta_net.proxy.o -o dnet_bench -L<xtdk>/runtime/shlib -lxpurt
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

extern void run_delta_net(int clusters, int cores,
    float *S, const float *k, const float *v, const float *q,
    const float *g, const float *b, float *out);

int main() {
    std::vector<float> S(NVH * VHD * KHD);      // state [32][128][128]
    std::vector<float> k(NVH * 16 * KHD);       // key   [32][16][128]  (NKH=16)
    std::vector<float> v(NVH * VHD);            // value [32][128]
    std::vector<float> q(NVH * 16 * KHD);       // query [32][16][128]
    std::vector<float> g(NVH), b(NVH);          // FINAL g,b (already softplus/sigmoid by host)
    std::vector<float> out(NVH * VHD);
    srand(42);
    for (auto &x : S) x = (rand() % 2000 - 1000) / 1000.0f;
    for (auto &x : k) x = (rand() % 2000 - 1000) / 1000.0f;
    for (auto &x : v) x = (rand() % 2000 - 1000) / 1000.0f;
    for (auto &x : q) x = (rand() % 2000 - 1000) / 1000.0f;
    // g_final ~ N(0, 0.1) so eg stays in sane range; b_final in [0,1]
    for (int i = 0; i < NVH; i++) g[i] = (rand() % 200 - 100) / 1000.0f;
    for (int i = 0; i < NVH; i++) b[i] = (rand() % 1000) / 1000.0f;

    // CPU reference (bit-exact match to orn3.cb.cpp single-row path, using FINAL g,b)
    std::vector<float> ref(NVH * VHD);
    std::vector<float> Sref = S;
    const float qscale = 1.f / sqrtf((float)KHD);
    for (int hv = 0; hv < NVH; hv++) {
        int hk = hv % 16;
        float g_ = g[hv];
        float b_ = b[hv];
        float eg = expf(g_);
        float *M = &Sref[(size_t)hv * VHD * KHD];
        const float *kk = &k[(size_t)hv * 16 * KHD + hk * KHD];
        const float *qq = &q[(size_t)hv * 16 * KHD + hk * KHD];
        const float *vv = &v[(size_t)hv * VHD];
        for (int j = 0; j < VHD; j++) {
            float *row = M + (size_t)j * KHD;
            float sk = 0;
            for (int i = 0; i < KHD; i++) { float rv = row[i] * eg; row[i] = rv; sk += rv * kk[i]; }
            float d = (vv[j] - sk) * b_;
            float so = 0;
            for (int i = 0; i < KHD; i++) { float rv = row[i] + kk[i] * d; row[i] = rv; so += rv * qq[i]; }
            ref[hv * VHD + j] = so * qscale;
        }
    }

    // device copies
    float *dS, *dk, *dv, *dq, *dg, *db, *dout;
    float *hS = (float*)malloc(S.size()*4), *hout = (float*)malloc(out.size()*4);
    memcpy(hS, S.data(), S.size()*4);
    xpu_set_device(0);
    xpu_malloc((void**)&dS, S.size()*4);
    xpu_malloc((void**)&dk, k.size()*4);
    xpu_malloc((void**)&dv, v.size()*4);
    xpu_malloc((void**)&dq, q.size()*4);
    xpu_malloc((void**)&dg, g.size()*4);
    xpu_malloc((void**)&db, b.size()*4);
    xpu_malloc((void**)&dout, out.size()*4);
    xpu_memcpy(dS, hS, S.size()*4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dk, k.data(), k.size()*4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dv, v.data(), v.size()*4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dq, q.data(), q.size()*4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(dg, g.data(), g.size()*4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(db, b.data(), b.size()*4, XPU_HOST_TO_DEVICE);
    xpu_wait();

    // CPU single-step timing
    double cbest = 1e9;
    for (int t = 0; t < 20; t++) {
        std::vector<float> S2 = S;
        auto c0 = std::chrono::high_resolution_clock::now();
        for (int hv = 0; hv < NVH; hv++) {
            int hk = hv % 16;
            float eg = expf(g[hv]), bb = b[hv];
            float *M = &S2[(size_t)hv * VHD * KHD];
            const float *kk = &k[(size_t)hv * 16 * KHD + hk * KHD];
            const float *qq = &q[(size_t)hv * 16 * KHD + hk * KHD];
            const float *vv = &v[(size_t)hv * VHD];
            for (int j = 0; j < VHD; j++) {
                float *row = M + (size_t)j * KHD;
                float skk = 0, so = 0;
                for (int i = 0; i < KHD; i++) { float r = row[i] * eg; row[i] = r; skk += r * kk[i]; }
                float d = (vv[j] - skk) * bb;
                for (int i = 0; i < KHD; i++) { float r = row[i] + kk[i] * d; row[i] = r; so += r * qq[i]; }
                ref[hv * VHD + j] = so * qscale;
            }
        }
        auto c1 = std::chrono::high_resolution_clock::now();
        double ms2 = std::chrono::duration<double, std::milli>(c1 - c0).count();
        if (ms2 < cbest) cbest = ms2;
    }
    printf("CPU single-step: %.3f ms (best of 20, single-thread)\n", cbest);

    // XPU single-step timing (reset S each iter, like real use)
    {
        xpu_memcpy(dS, S.data(), S.size()*4, XPU_HOST_TO_DEVICE); xpu_wait();
        run_delta_net(2, 16, dS, dk, dv, dq, dg, db, dout); xpu_wait();
        double xbest = 1e9, xsum = 0; int NI = 100;
        for (int t = 0; t < NI; t++) {
            auto x0 = std::chrono::high_resolution_clock::now();
            run_delta_net(2, 16, dS, dk, dv, dq, dg, db, dout); xpu_wait();
            auto x1 = std::chrono::high_resolution_clock::now();
            double ms2 = std::chrono::duration<double, std::milli>(x1 - x0).count();
            xsum += ms2; if (ms2 < xbest) xbest = ms2;
        }
        printf("XPU single-step: best=%.3f ms avg=%.3f ms (%d iters, S in-place)\n", xbest, xsum/NI, NI);
    }

    // SINGLE-STEP verification first
    xpu_memcpy(dS, S.data(), S.size()*4, XPU_HOST_TO_DEVICE); xpu_wait();
    run_delta_net(2, 16, dS, dk, dv, dq, dg, db, dout); xpu_wait();
    xpu_memcpy(hout, dout, out.size()*4, XPU_DEVICE_TO_HOST);
    xpu_memcpy(hS, dS, S.size()*4, XPU_DEVICE_TO_HOST);
    xpu_wait();
    {
        double maxerr = 0; int bad = 0;
        double smerr = 0; int sbad = 0;
        for (size_t i = 0; i < out.size(); i++) {
            double e = fabs(hout[i] - ref[i]);
            if (e > maxerr) maxerr = e;
            if (e > 1e-3) bad++;
        }
        for (size_t i = 0; i < S.size(); i++) {
            double e = fabs(hS[i] - Sref[i]);
            if (e > smerr) smerr = e;
            if (e > 1e-3) sbad++;
        }
        printf("SINGLE-STEP: out maxerr=%.2e bad=%d/%zu | S maxerr=%.2e sbad=%d/%zu\n",
               maxerr, bad, out.size(), smerr, sbad, S.size());
    }
    // reset S on device for the timing loop
    xpu_memcpy(dS, hS, S.size()*4, XPU_HOST_TO_DEVICE); xpu_wait();
    memcpy(hS, S.data(), S.size()*4);
    xpu_memcpy(dS, hS, S.size()*4, XPU_HOST_TO_DEVICE); xpu_wait();

    // BATCHED: 32 layer-steps queued on stream, ONE wait (real decode shape)
    {
        xpu_memcpy(dS, S.data(), S.size()*4, XPU_HOST_TO_DEVICE); xpu_wait();
        run_delta_net(2, 16, dS, dk, dv, dq, dg, db, dout); xpu_wait();
        xpu_memcpy(dS, S.data(), S.size()*4, XPU_HOST_TO_DEVICE); xpu_wait();
        double best = 1e9; int NI = 50;
        for (int t = 0; t < NI; t++) {
            auto x0 = std::chrono::high_resolution_clock::now();
            for (int l = 0; l < 32; l++) run_delta_net(2, 16, dS, dk, dv, dq, dg, db, dout);
            xpu_wait();
            auto x1 = std::chrono::high_resolution_clock::now();
            double ms2 = std::chrono::duration<double, std::milli>(x1 - x0).count();
            if (ms2 < best) best = ms2;
        }
        printf("XPU 32-step queued+1 wait: best=%.3f ms per token\n", best);
    }

    // warmup + timed runs
    int iters = 300;
    run_delta_net(2, 16, dS, dk, dv, dq, dg, db, dout); xpu_wait();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; it++)
        run_delta_net(2, 16, dS, dk, dv, dq, dg, db, dout);
    xpu_wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    xpu_memcpy(hout, dout, out.size()*4, XPU_DEVICE_TO_HOST);
    xpu_memcpy(hS, dS, S.size()*4, XPU_DEVICE_TO_HOST);
    xpu_wait();

    double maxerr = 0; int bad = 0;
    for (size_t i = 0; i < out.size(); i++) {
        double e = fabs(hout[i] - ref[i]);
        if (e > maxerr) maxerr = e;
        if (e > 1e-3) { bad++; if (bad < 5) printf("  out[%zu] xpu=%.5f ref=%.5f\n", i, hout[i], ref[i]); }
    }
    double smerr = 0;
    for (size_t i = 0; i < S.size(); i++) {
        double e = fabs(hS[i] - Sref[i]);
        if (e > smerr) smerr = e;
    }
    printf("TIMING(300it, S diverges by design): per-iter %.3f ms\n", ms);
    return 0;
}