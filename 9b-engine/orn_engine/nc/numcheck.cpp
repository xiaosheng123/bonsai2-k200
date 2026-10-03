// numcheck.cpp —— ★ 小尺寸数值对拍 (任务硬要求, 不得跳) ★
//   同一份权重 / 同一个 x, 三路对拍逐元素 relrms:
//     (i)  自写内核 kq8v2 (V2 布局: 每 32 权重一个 f16 scale)   —— 现役引擎路径
//     (ii) 官方 gemm_int8 (B = 行优先 [out][in] int8, 每输出行一个 float scale) —— 新方案
//     (iii)主机 double 参考 (未量化 f32 权重 x)
//   另加一个"输入也量化"的参考 (iii-q): 用于把「布局/scale 是否正确」与
//   「官方算子内部把 f32 激活按 per-tensor int8 量化带来的误差」分开 —— 前者必须 ~0%,
//   后者是纯精度代价。还跑 dev1 一遍, 确认双芯路径一致。
//   尺寸刻意小 (M=256, N=4096), 无越界风险; 全程包在 safe_run.sh 内。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <vector>
#include <algorithm>
#include <chrono>
#include <xpu/runtime.h>
#include "xpu/refactor/nn.h"
#include "xpu/refactor/context/xpu_act_type.h"
namespace api = baidu::xpu::api;

namespace baidu { namespace xpu { namespace api {
int gemm_int8(Context* ctx, const bool trans_a, const bool trans_b, int m, int n, int k,
              float alpha, const float* a, int lda, const int8_t* b, float max_b, int ldb,
              float beta, float* c, int ldc);
}}}
void run_gemv_q8v2(int cl, int co, const void *W, const void *xq, const void *xs, void *y, int M, int N);

static double NOW(){ using namespace std::chrono; return duration<double>(steady_clock::now().time_since_epoch()).count(); }

static uint32_t s_rng = 12345u;
static float rnd(){                       // 三角分布 (-1,1), 确定性
    float t = 0;
    for (int k = 0; k < 3; k++){ s_rng = s_rng * 1664525u + 1013904223u;
        t += (float)((s_rng >> 8) & 0xFFFFFFu) / 16777215.f; }
    return t * 2.f / 3.f - 1.f;
}
static void wdump(const char *nm, const void *p, size_t bytes){
    char path[512]; snprintf(path, sizeof(path), "/home/caden/orn_engine/nc/%s", nm);
    FILE *f = fopen(path, "wb"); if (!f){ printf("WARN: 打不开 %s\n", path); return; }
    fwrite(p, 1, bytes, f); fclose(f);
}
static double relrms(const std::vector<float> &a, const std::vector<double> &b, const char *tag,
                     double *mxabs = nullptr){
    double se = 0, sr = 0, mx = 0;
    for (size_t i = 0; i < a.size(); i++){ double d = (double)a[i] - b[i];
        se += d * d; sr += b[i] * b[i]; if (fabs(d) > mx) mx = fabs(d); }
    double rr = 100.0 * sqrt(se / (sr + 1e-300));
    if (mxabs) *mxabs = mx;
    printf("      %-26s relrms = %.5f%%   maxabs=%.4g\n", tag, rr, mx);
    return rr;
}

// GGUF block_q8_0 原始布局 (32 权重 = 2B f16 d + 32 int8) -> V2 布局 (每 4096 权重: [4096 int8][128 x f16])
static std::vector<unsigned char> mk_v2(const std::vector<float> &W, int M, int N){
    int nb = N / 32, nch = N / 4096;
    std::vector<unsigned char> raw((size_t)M * nb * 34, 0), dst((size_t)M * ((size_t)N + (size_t)nb * 2));
    for (int m = 0; m < M; m++)
        for (int b = 0; b < nb; b++){
            float amax = 0;
            for (int i = 0; i < 32; i++){ float a = fabsf(W[(size_t)m * N + b * 32 + i]); if (a > amax) amax = a; }
            float d = amax / 127.f; if (d <= 0) d = 1e-8f;
            unsigned char *blk = &raw[((size_t)m * nb + b) * 34];
            unsigned short h = fp16_ieee_from_fp32_value(d);
            memcpy(blk, &h, 2);
            for (int i = 0; i < 32; i++){
                int v = (int)lrintf(W[(size_t)m * N + b * 32 + i] / d);
                if (v > 127) v = 127; else if (v < -128) v = -128;
                blk[2 + i] = (unsigned char)(signed char)v;
            }
        }
    (void)nch;
    for (int m = 0; m < M; m++)
        for (int b = 0; b < nb; b++){
            int c = b / 128, bb = b % 128;
            size_t off = (size_t)m * ((size_t)N + (size_t)nb * 2) + (size_t)c * 4352;
            memcpy(&dst[off + bb * 32], &raw[((size_t)m * nb + b) * 34 + 2], 32);
            memcpy(&dst[off + 4096 + bb * 2], &raw[((size_t)m * nb + b) * 34], 2);
        }
    return dst;
}
static void quant_x32(const std::vector<float> &x, std::vector<signed char> &xq, std::vector<float> &xs){
    int nb = (int)x.size() / 32;
    xq.resize(x.size()); xs.resize(nb);
    for (int b = 0; b < nb; b++){
        float am = 0; for (int i = 0; i < 32; i++){ float a = fabsf(x[b * 32 + i]); if (a > am) am = a; }
        float s = am / 127.f; if (s <= 1e-12f) s = 1e-8f; xs[b] = s;
        for (int i = 0; i < 32; i++){ int v = (int)lrintf(x[b * 32 + i] / s);
            if (v > 127) v = 127; else if (v < -128) v = -128; xq[b * 32 + i] = (signed char)v; }
    }
}

int main(){
    setbuf(stdout, NULL);
    const int M = 256, N = 4096;
    printf("=== numcheck: kq8v2  vs  官方 gemm_int8(行 int8 + 行 scale)  vs  主机 double 参考 ===\n");
    printf("形状: W[out=%d][in=%d] 行优先 (GGUF 原样, 不转置); x[%d]\n", M, N, N);

    // ---------- 1) 数据 ----------
    std::vector<float> W((size_t)M * N), x(N);
    for (size_t i = 0; i < W.size(); i++) W[i] = rnd();
    for (int i = 0; i < N; i++) x[i] = rnd();

    // ---------- 2) 每输出行 int8 + 每行一个 scale ----------
    std::vector<float> srow(M);
    std::vector<signed char> Q((size_t)M * N);
    for (int m = 0; m < M; m++){
        float am = 0;
        for (int n = 0; n < N; n++){ float a = fabsf(W[(size_t)m * N + n]); if (a > am) am = a; }
        float s = am / 127.f; if (s <= 1e-30f) s = 1e-8f; srow[m] = s;
        for (int n = 0; n < N; n++){
            int v = (int)lrintf(W[(size_t)m * N + n] / s);
            if (v > 127) v = 127; else if (v < -127) v = -127;
            Q[(size_t)m * N + n] = (signed char)v;
        }
    }

    // ---------- 3) 主机 double 参考 ----------
    std::vector<double> yref(M, 0.0), yref_qA(M, 0.0);
    for (int m = 0; m < M; m++){ double a = 0;
        for (int n = 0; n < N; n++) a += (double)W[(size_t)m * N + n] * (double)x[n];
        yref[m] = a; }
    // (iii-q) 参考: 权重按【行 int8】解量化 + 激活按【per-tensor int8】解量化 (模拟算子内部把 f32 激活量化)
    float xmax = 0; for (int i = 0; i < N; i++){ float a = fabsf(x[i]); if (a > xmax) xmax = a; }
    float xscale = xmax / 127.f;
    std::vector<signed char> xaq(N);
    for (int i = 0; i < N; i++){ int v = (int)lrintf(x[i] / xscale); if (v > 127) v = 127; else if (v < -127) v = -127; xaq[i] = (signed char)v; }
    for (int m = 0; m < M; m++){ double a = 0;
        for (int n = 0; n < N; n++) a += (double)((float)xaq[n] * xscale) * (double)((float)Q[(size_t)m * N + n] * srow[m]);
        yref_qA[m] = a; }
    printf("[数据] W 已按行 int8 量化 (每行 scale: min=%.6g max=%.6g); x per-tensor scale=%.6g\n",
           (double)*std::min_element(srow.begin(), srow.end()), (double)*std::max_element(srow.begin(), srow.end()), (double)xscale);

    // ---------- 4) 卡上 ----------
    std::vector<unsigned char> v2 = mk_v2(W, M, N);
    std::vector<signed char> xq; std::vector<float> xs;
    quant_x32(x, xq, xs);
    printf("[布局] kq8v2 V2 缓冲 = %zu 字节 (M*(N+N/32*2)); 行int8 缓冲 = %d 字节\n", v2.size(), M * N);

    std::vector<float> yi_dev[2], yii_dev[2];
    for (int dv = 0; dv < 2; dv++){
        printf("---- dev%d ----\n", dv);
        xpu_set_device(dv);
        void *dWv2 = 0, *dWq = 0, *dxf = 0, *dyf = 0, *dxq = 0, *dxs = 0, *dy = 0;
        if (xpu_malloc(&dWv2, v2.size()) || xpu_malloc(&dWq, (size_t)M * N) || xpu_malloc(&dxf, (size_t)N * 4) ||
            xpu_malloc(&dyf, (size_t)M * 4) || xpu_malloc(&dxq, N) || xpu_malloc(&dxs, (N / 32) * 4) ||
            xpu_malloc(&dy, (size_t)M * 4)) { printf("FATAL: dev%d xpu_malloc 失败\n", dv); return 1; }
        xpu_memcpy(dWv2, v2.data(), v2.size(), XPU_HOST_TO_DEVICE);
        xpu_memcpy(dWq, Q.data(), (size_t)M * N, XPU_HOST_TO_DEVICE);
        xpu_memcpy(dxf, x.data(), (size_t)N * 4, XPU_HOST_TO_DEVICE);
        xpu_memcpy(dxq, xq.data(), N, XPU_HOST_TO_DEVICE);
        xpu_memcpy(dxs, xs.data(), (N / 32) * 4, XPU_HOST_TO_DEVICE);
        if (xpu_wait()) { printf("FATAL: dev%d 初始 H2D wait 非0\n", dv); return 1; }

        // (i) 自写 kq8v2
        run_gemv_q8v2(8, 16, dWv2, dxq, dxs, dy, M, N);
        int w0 = xpu_wait();
        std::vector<float> yi(M, 0.f);
        xpu_memcpy(yi.data(), dy, (size_t)M * 4, XPU_DEVICE_TO_HOST); xpu_wait();
        printf("  (i)  kq8v2 (cl=8 co=16) wait=%d  y[0]=%.6f\n", w0, yi[0]);

        // (ii) 官方 gemm_int8: B=行优先 [out][in] int8, max_b=127 -> C = x·q (未折 scale)
        api::Device dev(api::DeviceType::XPU1, dv);
        api::Context ctx(dev);
        std::vector<float> c(M, 0.f);
        double t0 = NOW();
        int r = api::gemm_int8(&ctx, false, true, 1, M, N, 1.f, (const float *)dxf, N,
                               (const int8_t *)dWq, 127.f, N, 0.f, (float *)dyf, M);
        int w1 = xpu_wait();
        double dt = NOW() - t0;
        xpu_memcpy(c.data(), dyf, (size_t)M * 4, XPU_DEVICE_TO_HOST); xpu_wait();
        printf("  (ii) gemm_int8(trans_b=1, m=1,n=%d,k=%d) r=%d wait=%d  %.3f ms  C[0]=%.6f\n", M, N, r, w1, dt * 1000, c[0]);

        std::vector<float> yii(M);
        for (int m = 0; m < M; m++) yii[m] = srow[m] * c[m];
        // 恒定比例检测 (若算子的解量化因子不是 max_b/127, 会在这里露出来)
        double num = 0, den = 0;
        for (int m = 0; m < M; m++){ double e = 0; for (int n = 0; n < N; n++) e += (double)x[n] * (double)Q[(size_t)m * N + n];
            num += (double)c[m] * e; den += e * e; }
        printf("  [比例检测] sum(C*E)/sum(E*E) = %.6f  (期望 1.0 => 因子确为 max_b/127)\n", num / den);

        yi_dev[dv] = yi; yii_dev[dv] = yii;
        xpu_free(dWv2); xpu_free(dWq); xpu_free(dxf); xpu_free(dyf); xpu_free(dxq); xpu_free(dxs); xpu_free(dy);
        xpu_wait();
    }

    // ---------- 5) 三方对拍 ----------
    for (int dv = 0; dv < 2; dv++){
        printf("======== dev%d 对拍 ========\n", dv);
        relrms(yi_dev[dv],  yref,    "(i)  kq8v2   vs 参考");
        relrms(yii_dev[dv], yref,    "(ii) gemm_int8 vs 参考");
        relrms(yii_dev[dv], yref_qA, "(ii) gemm_int8 vs 参考(激活也量化)  <== 布局/scale 判定");
        double mx = 0, se = 0, sr = 0;
        for (int m = 0; m < M; m++){ double d = (double)yi_dev[dv][m] - yii_dev[dv][m]; se += d * d; sr += (double)yii_dev[dv][m] * (double)yii_dev[dv][m]; if (fabs(d) > mx) mx = fabs(d); }
        printf("      (i) vs (ii) relrms = %.4f%%  maxabs=%.4g\n", 100.0 * sqrt(se / (sr + 1e-300)), mx);
    }
    {
        double mx = 0, se = 0, sr = 0;
        for (int m = 0; m < M; m++){ double d = (double)yii_dev[0][m] - yii_dev[1][m]; se += d * d; sr += (double)yii_dev[0][m] * (double)yii_dev[0][m]; if (fabs(d) > mx) mx = fabs(d); }
        printf("======= dev0 vs dev1 (gemm_int8) relrms = %.6f%%  maxabs=%.4g =======\n", 100.0 * sqrt(se / (sr + 1e-300)), mx);
    }
    printf("[样例] m=0: 参考=%.6f  kq8v2=%.6f  int8row=%.6f\n", yref[0], yi_dev[0][0], yii_dev[0][0]);
    printf("[样例] m=1: 参考=%.6f  kq8v2=%.6f  int8row=%.6f\n", yref[1], yi_dev[0][1], yii_dev[0][1]);

    // ---------- 6) 落盘给 numpy 独立复算 ----------
    wdump("Wf.bin", W.data(), W.size() * 4);
    wdump("x.bin", x.data(), x.size() * 4);
    wdump("srow.bin", srow.data(), srow.size() * 4);
    wdump("Q.bin", Q.data(), Q.size());
    wdump("yi.bin", yi_dev[0].data(), yi_dev[0].size() * 4);
    wdump("yii.bin", yii_dev[0].data(), yii_dev[0].size() * 4);
    wdump("yref_cpp.bin", yref.data(), yref.size() * 8);
    printf("[落盘] /home/caden/orn_engine/nc/*.bin (供 numpy 独立复算)\n");
    printf("=== numcheck done ===\n");
    return 0;
}
