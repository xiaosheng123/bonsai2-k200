// ###########################################################################
// ★★ 已被隔离（CRASHED）—— 不要再运行本程序的 (c2) 段！★★
// 2026-09-21 15:24:39: 调 api::gemm_int8_maxptr<float,signed char,float> 触发卡异常
//   现场(/proc/xpu/dev0/errtask): sdcl-0..3 token=314295 reason[27]=RBRESP
//   ETASK .name=_Z10fc_int8_v2IfafEviiiiifPKT_PKT0_fPTT1_PKfiS9_S9_S9_Pfiii  (SD-CDNN 引擎)
//   kern.log: xpu0 sess209 fccmp: session error Exception in kernel execution
// 结论: 官方 *_maxptr 系列在本机(vfio 直通)会打崩 dev0。如需再用, 必须先去掉 (c2) 段并另开维护窗口。
// ###########################################################################

// fccmp.cpp —— 权重搬运吞吐对比: (a) 自写 kq8 GM2LM 内核  vs  (b) 官方 fc<f16>  vs  (c) 官方 int8 GEMM
// 产出: 每个方案在同一矩阵上的 GB/s(权重字节/耗时) + 原始返回码; 小尺寸先对拍再上大尺寸。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <vector>
#include <chrono>
#include <xpu/runtime.h>
#include "xpu/refactor/nn.h"
#include "xpu/refactor/context/xpu_act_type.h"
namespace api = baidu::xpu::api;

namespace baidu { namespace xpu { namespace api {
int gemm_int8(Context* ctx, const bool trans_a, const bool trans_b, int m, int n, int k,
              float alpha, const float* a, int lda, const int8_t* b, float max_b, int ldb,
              float beta, float* c, int ldc);
int fc_int8(Context* ctx, bool TransA, bool TransB, int M, int N, int K,
            const int8_t* A, float max_a, const int8_t* B, float max_b, int8_t* C, float max_c);
template<typename TX, typename TW, typename TY>
int gemm_int8_maxptr(Context* ctx, const bool TransA, const bool TransB, int M, int N, int K,
              float alpha, const TX* A, int lda, const TW* B, int ldb,
              float beta, TY* C, int ldc,
              const float* bias, const Activation_t act_type,
              const float* max_a, const float* max_b, float* max_c);
}}}

void run_gemv_q8v2(int cl, int co, const void *W, const void *xq, const void *xs, void *y, int M, int N);

static double NOW(){ using namespace std::chrono; return duration<double>(steady_clock::now().time_since_epoch()).count(); }

// ---------------- 合成 Q8_0 权重 (每 32 权重 = fp16 d + int8 q[32]) ----------------
struct Q8W { std::vector<signed char> q; std::vector<unsigned short> d; int M, N; };
static Q8W mk_q8(int M, int N){
    Q8W w; w.M = M; w.N = N;
    assert(N % 32 == 0);
    w.q.resize((size_t)M * N);
    w.d.resize((size_t)M * (N/32));
    for (int m = 0; m < M; m++)
        for (int b = 0; b < N/32; b++){
            w.d[(size_t)m*(N/32)+b] = (unsigned short)(0x2400 + ((m*131 + b*17) % 0x100));  // ~0.015..0.031
            for (int i = 0; i < 32; i++){
                int v = (int)((m*7 + b*3 + i) % 251) - 125;
                w.q[(size_t)m*N + b*32 + i] = (signed char)v;
            }
        }
    return w;
}
// ggml block_q8_0 原始布局 (34B/32w) -> V2 布局 (每 4096 权重: [4096 int8][128 x f16])
static std::vector<unsigned char> pack_v2(const Q8W &w){
    int M = w.M, N = w.N, nb = N/32, nch = N/4096;
    assert(nb == nch*128);
    size_t rowbytes = (size_t)N + (size_t)nb*2;
    std::vector<unsigned char> dst((size_t)M*rowbytes);
    for (int m = 0; m < M; m++)
        for (int b = 0; b < nb; b++){
            int c = b/128, bb = b%128;
            size_t off = (size_t)m*rowbytes + (size_t)c*4352;
            memcpy(&dst[off + bb*32], &w.q[(size_t)m*N + b*32], 32);
            memcpy(&dst[off + 4096 + bb*2], &w.d[(size_t)m*nb + b], 2);
        }
    return dst;
}
// 激活 x (4096 权重的块) -> q8 激活
static void quant_x(const std::vector<float> &x, std::vector<signed char> &xq, std::vector<float> &xs){
    int nb = (int)x.size()/32;
    xq.resize(x.size()); xs.resize(nb);
    for (int b = 0; b < nb; b++){
        float am = 1e-30f;
        for (int i = 0; i < 32; i++){ float a = fabsf(x[b*32+i]); if (a > am) am = a; }
        float s = am/127.f; xs[b] = s;
        for (int i = 0; i < 32; i++){
            int v = (int)lrintf(x[b*32+i]/s);
            if (v > 127) v = 127; if (v < -127) v = -127;
            xq[b*32+i] = (signed char)v;
        }
    }
}

int main(int argc, char **argv){
    setbuf(stdout, NULL);
    printf("=== fccmp: 官方算子 vs 自写 GM2LM 内核 ===\n");
    assert(xpu_set_device(0) == 0);
    printf("set_device(0)=0\n");
    api::Context ctx(api::kXPU1);

    // ================= 1) 小尺寸数值对拍 (a) =================
    {
        int M = 128, N = 4096;                        // 与生产 V2 布局一致 (N 必须 >= 4096 且 4096 整除)
        assert(N % 4096 == 0);
        assert(N % 32 == 0);
        Q8W w = mk_q8(M, N);
        std::vector<unsigned char> v2 = pack_v2(w);
        std::vector<float> x(N);
        for (int i = 0; i < N; i++) x[i] = (float)(((i%9)-4) * 0.25);
        std::vector<signed char> xq; std::vector<float> xs;
        quant_x(x, xq, xs);
        // CPU 参考
        std::vector<double> ref(M, 0.0);
        for (int m = 0; m < M; m++)
            for (int b = 0; b < N/32; b++){
                double acc = 0;
                for (int i = 0; i < 32; i++)
                    acc += (double)w.q[(size_t)m*N + b*32 + i] * (double)xq[b*32+i];
                float d = fp16_ieee_to_fp32_value(w.d[(size_t)m*(N/32)+b]);
                ref[m] += acc * (double)d * (double)xs[b];
            }
        void *dW=0,*dxq=0,*dxs=0,*dy=0;
        assert(xpu_malloc(&dW, v2.size()) == 0);
        assert(xpu_malloc(&dxq, N) == 0);
        assert(xpu_malloc(&dxs, (N/32)*4) == 0);
        assert(xpu_malloc(&dy, (size_t)M*4) == 0);
        xpu_memcpy(dW, v2.data(), v2.size(), XPU_HOST_TO_DEVICE);
        xpu_memcpy(dxq, xq.data(), N, XPU_HOST_TO_DEVICE);
        xpu_memcpy(dxs, xs.data(), (N/32)*4, XPU_HOST_TO_DEVICE);
        run_gemv_q8v2(4, 16, dW, dxq, dxs, dy, M, N);
        int wr = xpu_wait();
        std::vector<float> y(M, 0.f);
        xpu_memcpy(y.data(), dy, (size_t)M*4, XPU_DEVICE_TO_HOST);
        double mx = 0, rr = 0, rn = 0;
        for (int m = 0; m < M; m++){ double d = fabs(y[m]-ref[m]); if (d > mx) mx = d; rr += (y[m]-ref[m])*(y[m]-ref[m]); rn += ref[m]*ref[m]; }
        printf("[a-对拍 m=%d n=%d] wait=%d  maxabs=%.3e  relrms=%.4f%%  y0=%.5f ref0=%.5f\n",
               M, N, wr, mx, 100.0*sqrt(rr/rn), y[0], ref[0]);
        xpu_free(dW);xpu_free(dxq);xpu_free(dxs);xpu_free(dy);
    }

    // ================= 2) 大矩阵吞吐对比 =================
    int shapes[2][2] = {{4096, 8192}, {12288, 4096}};
    for (int s = 0; s < 2; s++){
        int M = shapes[s][0], N = shapes[s][1];
        size_t wbytes = (size_t)M * N;
        printf("\n########## 矩阵 M=%d N=%d (int8 权重 %.1f MB) ##########\n", M, N, wbytes/1048576.0);
        Q8W w = mk_q8(M, N);
        std::vector<unsigned char> v2 = pack_v2(w);
        std::vector<float> x(N);
        for (int i = 0; i < N; i++) x[i] = (float)(((i%9)-4) * 0.25);
        std::vector<signed char> xq; std::vector<float> xs;
        quant_x(x, xq, xs);

        // --- 设备缓冲 ---
        void *dWv2=0,*dW8=0,*dWf=0,*dxq=0,*dxs=0,*dxf=0,*dy=0,*dy8=0,*dci=0;
        assert(xpu_malloc(&dWv2, v2.size()) == 0);
        assert(xpu_malloc(&dW8, wbytes) == 0);
        assert(xpu_malloc(&dWf, wbytes*2) == 0);
        assert(xpu_malloc(&dxq, N) == 0);
        assert(xpu_malloc(&dxs, (N/32)*4) == 0);
        assert(xpu_malloc(&dxf, (size_t)N*4) == 0);
        assert(xpu_malloc(&dy, (size_t)M*4) == 0);
        assert(xpu_malloc(&dy8, (size_t)M) == 0);
        xpu_memcpy(dWv2, v2.data(), v2.size(), XPU_HOST_TO_DEVICE);
        xpu_memcpy(dW8, w.q.data(), wbytes, XPU_HOST_TO_DEVICE);
        { // f16 权重: 直接由 int8 值转 f16 (只关心搬运带宽, 不关心数值意义)
          std::vector<unsigned short> hf(wbytes);
          for (size_t i = 0; i < wbytes; i++){ float v = (float)w.q[i] * 0.02f; hf[i] = fp16_ieee_from_fp32_value(v); }
          xpu_memcpy(dWf, hf.data(), wbytes*2, XPU_HOST_TO_DEVICE);
        }
        xpu_memcpy(dxq, xq.data(), N, XPU_HOST_TO_DEVICE);
        xpu_memcpy(dxs, xs.data(), (N/32)*4, XPU_HOST_TO_DEVICE);
        xpu_memcpy(dxf, x.data(), (size_t)N*4, XPU_HOST_TO_DEVICE);
        assert(xpu_wait() == 0);

        std::vector<float> hy(M, 0.f); std::vector<signed char> hy8(M, 0);

        // --- (a) 自写 kq8v2 GM2LM ---
        for (int ci = 0; ci < 2; ci++){
            int cl = ci ? 8 : 4, co = 16, L = 20;
            for (int i = 0; i < 3; i++){ run_gemv_q8v2(cl, co, dWv2, dxq, dxs, dy, M, N); }
            xpu_wait();
            double t0 = NOW();
            for (int i = 0; i < L; i++) run_gemv_q8v2(cl, co, dWv2, dxq, dxs, dy, M, N);
            int wr = xpu_wait();
            double dt = (NOW()-t0)/L;
            xpu_memcpy(hy.data(), dy, (size_t)M*4, XPU_DEVICE_TO_HOST);
            printf("(a) kq8v2 自写 gm2lm   cl=%d co=%d : %8.3f ms  => 权重带宽 %6.2f GB/s  wait=%d y0=%.4f\n",
                   cl, co, dt*1000, wbytes/dt/1e9, wr, hy[0]);
        }

        // --- (b) 官方 fc<f16,f16,f16,short> ---
        {
            int L = 20; int r = 0;
            for (int i = 0; i < 3; i++) r = api::fc<float16, float16, float16, short>(
                &ctx, (const float16*)dxf, (const float16*)dWf, (float16*)dy, 1, M, N, false, true, 0, 0, 0);
            int w0 = xpu_wait();
            if (r || w0) printf("(b) fc<f16> 预热失败 r=%d wait=%d\n", r, w0);
            double t0 = NOW();
            for (int i = 0; i < L; i++) r = api::fc<float16, float16, float16, short>(
                &ctx, (const float16*)dxf, (const float16*)dWf, (float16*)dy, 1, M, N, false, true, 0, 0, 0);
            int w1 = xpu_wait();
            double dt = (NOW()-t0)/L;
            printf("(b) 官方 fc<f16,f16,f16,s16> m=1 n=%d k=%d : %8.3f ms  => 权重带宽 %6.2f GB/s  r=%d wait=%d\n",
                   M, N, dt*1000, (wbytes*2.0)/dt/1e9, r, w1);
        }

        // --- (c) 官方 gemm_int8 (float 激活, int8 权重, float 出) ---
        {
            int L = 20;
            int r = api::gemm_int8(&ctx, false, true, 1, M, N, 1.f, (const float*)dxf, N,
                                   (const int8_t*)dW8, 127.f, N, 0.f, (float*)dy, M);
            int w0 = xpu_wait();
            if (r || w0) printf("(c) gemm_int8 预热失败 r=%d wait=%d\n", r, w0);
            double t0 = NOW();
            for (int i = 0; i < L; i++) r = api::gemm_int8(&ctx, false, true, 1, M, N, 1.f, (const float*)dxf, N,
                                   (const int8_t*)dW8, 127.f, N, 0.f, (float*)dy, M);
            int w1 = xpu_wait();
            double dt = (NOW()-t0)/L;
            xpu_memcpy(hy.data(), dy, (size_t)M*4, XPU_DEVICE_TO_HOST);
            printf("(c) 官方 gemm_int8 (f32激活/int8权) m=1 n=%d k=%d : %8.3f ms => 权重带宽 %6.2f GB/s  r=%d wait=%d y0=%.4f\n",
                   M, N, dt*1000, (double)wbytes/dt/1e9, r, w1, hy[0]);
        }

        // --- (c2) 官方 gemm_int8_maxptr: max_b 传数组, 验证是否 per-row ---
        {
            // 前 8 行的权重按 2 倍强缩放: 若 max_b 是 per-row, 这些行的输出会相应变化
            std::vector<signed char> w2 = w.q;
            for (int m = 0; m < 8 && m < M; m++) for (int i = 0; i < N; i++) w2[(size_t)m*N+i] = (signed char)(w2[(size_t)m*N+i]/2);
            void *dW2=0; assert(xpu_malloc(&dW2, wbytes) == 0);
            xpu_memcpy(dW2, w2.data(), wbytes, XPU_HOST_TO_DEVICE);
            std::vector<float> mb(M, 127.f);
            for (int m = 0; m < 8 && m < M; m++) mb[m] = 63.5f;      // 若 per-row: 前 8 行 y 会翻倍
            std::vector<float> ma(1, 1.f), mc(M, 0.f);
            int r = api::gemm_int8_maxptr<float, signed char, float>(&ctx, false, true, 1, M, N, 1.f,
                        (const float*)dxf, N, (const signed char*)dW2, N, 0.f, (float*)dy, M,
                        nullptr, api::Activation_t(api::Activation_t::LINEAR), nullptr, mb.data(), nullptr);
            int w0 = xpu_wait();
            xpu_memcpy(hy.data(), dy, (size_t)M*4, XPU_DEVICE_TO_HOST);
            printf("(c2) gemm_int8_maxptr (max_b 传 per-row 数组) r=%d wait=%d  y0=%.4f y7=%.4f y8=%.4f\n",
                   r, w0, hy[0], hy[7], hy[8 < M ? 8 : M-1]);
            xpu_free(dW2);
        }

        // --- (d) 官方 fc_int8 (int8 激活/int8 权/int8 出) ---
        {
            int L = 20;
            std::vector<signed char> xq1(N, 3);
            void *dx8=0; assert(xpu_malloc(&dx8, N) == 0);
            xpu_memcpy(dx8, xq1.data(), N, XPU_HOST_TO_DEVICE);
            int r = api::fc_int8(&ctx, false, true, 1, M, N, (const int8_t*)dx8, 1.f,
                                 (const int8_t*)dW8, 1.f, (int8_t*)dy8, 1.f);
            int w0 = xpu_wait();
            if (r || w0) printf("(d) fc_int8 预热失败 r=%d wait=%d\n", r, w0);
            double t0 = NOW();
            for (int i = 0; i < L; i++) r = api::fc_int8(&ctx, false, true, 1, M, N, (const int8_t*)dx8, 1.f,
                                 (const int8_t*)dW8, 1.f, (int8_t*)dy8, 1.f);
            int w1 = xpu_wait();
            double dt = (NOW()-t0)/L;
            printf("(d) 官方 fc_int8 (i8激活/i8权/i8出) m=1 n=%d k=%d : %8.3f ms => 权重带宽 %6.2f GB/s  r=%d wait=%d\n",
                   M, N, dt*1000, (double)wbytes/dt/1e9, r, w1);
            xpu_free(dx8);
        }

        xpu_free(dWv2);xpu_free(dW8);xpu_free(dWf);xpu_free(dxq);xpu_free(dxs);xpu_free(dxf);
        xpu_free(dy);xpu_free(dy8);
    }
    printf("\n=== fccmp done ===\n");
    return 0;
}
