// =============================================================================
// atn_probe.cpp — ATN_ON_CARD 阶段①【极小张量算子探针】(P1 ~ P17)
//
//   目的: 在【不加载权重、不碰生产件】的前提下, 用极小 -> 中等 -> 目标 的尺寸阶梯,
//         逐个回答 ATN_ON_CARD.md 第2.8/第6章的"不回答就不能继续"的问题。
//
//   纪律 (硬约束, 抄自项目铁律):
//     * 本探针【只调用 func_dec.h 里的官方算子】, 绝不写自写内核 (自写内核是宿主死机的直接原因)。
//     * 禁用算子一律不碰: 任何 *_maxptr、api::fc<float,float,float,int>、卡上 api::gelu、
//       layer_norm(n>1024)、block_gemm_int8(库内只有 cpu_mock)。
//     * 尺寸阶梯: 每一档失败(返回非 0)【立即停并判该 case 失败】, 绝不放大尺寸重试。
//     * 危险用例 (P7d 全-1e30 行 / P9① 除零 / P5b max_a=0) 默认【不跑】,
//       必须显式 ATN_DANGER=1 才跑, 且永远排在最后 —— 它们可能打死 session (FP_DIV0)。
//     * 每次运行都必须包在 safe_run.sh 里: bash safe_run.sh -n atnprobe -t 120 -- ...
//       退出码 3 = 卡异常 => 本阶段失败, 严禁重试。
//
//   用法:
//     atn_probe <case> [json_out]      case in {P1,P2a,P2b,P2c,P2d,P2e,P2f,P3,P4a,P4b,P4c,
//                                              P5,P5b,P7a,P7b,P7c,P7d,P9,P10,P11,P12,P13,
//                                              P14,P15,P16,P17,safe,danger,all}
//     建议顺序: 先 safe (除危险用例外的全部), 确认卡正常后再单独一个窗口跑 danger。
//
//   构建: 见 build_atn_probe.sh (纯主机编译, 【不产生任何设备码】, 因此不碰卡)
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <xpu/runtime.h>
#include "xpu/api.h"
#include "xpu/refactor/nn.h"
#include "xpu/refactor/math.h"
#include "xpu/refactor/context/xpu_act_type.h"

namespace api = baidu::xpu::api;
using namespace baidu::xpu::api;

// ------------------------------- 小工具 -------------------------------------
static const bool DANGER = (getenv("ATN_DANGER") && atoi(getenv("ATN_DANGER")) == 1);

static double NOW() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double relrms(const std::vector<float>& a, const std::vector<float>& b) {
    double se = 0, s2 = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        double d = (double)a[i] - (double)b[i]; se += d * d; s2 += (double)b[i] * (double)b[i];
    }
    return 100.0 * sqrt(se / (s2 > 0 ? s2 : 1e-30));
}
static double maxabs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) m = std::max(m, fabs((double)a[i] - (double)b[i]));
    return m;
}
static void fill(std::vector<float>& v, unsigned seed, float amp) {
    unsigned s = seed;
    for (size_t i = 0; i < v.size(); i++) { s = s * 1664525u + 1013904223u; v[i] = ((float)((s >> 8) & 0xFFFF) / 32768.0f - 1.0f) * amp; }
}
// 真实量级: K/V 在工程里经过 per-head RMSNorm, 分布接近 N(0,1) 量级; q 同量级。
static void fill_norm(std::vector<float>& v, unsigned seed, float sigma) {
    unsigned s = seed;
    for (size_t i = 0; i < v.size(); i++) {
        double u1 = 0, u2 = 0;
        s = s * 1664525u + 1013904223u; u1 = ((s >> 8) & 0xFFFF) / 65536.0 + 1e-9;
        s = s * 1664525u + 1013904223u; u2 = ((s >> 8) & 0xFFFF) / 65536.0;
        v[i] = (float)(sigma * sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
    }
}
struct DBuf {
    void* p = nullptr; size_t n = 0;
    void alloc(size_t bytes) { if (xpu_malloc(&p, bytes)) { printf("FATAL alloc %zu\n", bytes); exit(1); } n = bytes; }
    ~DBuf() { if (p) xpu_free(p); }
    void up(const void* h, size_t bytes) { if (xpu_memcpy(p, h, bytes, XPU_HOST_TO_DEVICE)) { printf("FATAL H2D\n"); exit(1); } }
    void down(void* h, size_t bytes) { if (xpu_memcpy(h, p, bytes, XPU_DEVICE_TO_HOST)) { printf("FATAL D2H\n"); exit(1); } xpu_wait(); }
};

// -------------------- 主机侧 RNE float32 -> float16 (自实现, 用于对拍) ---------
static uint16_t f32_to_f16_bits_rne(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t)((x >> 23) & 0xFF) - 127;
    uint32_t m = x & 0x7FFFFFu;
    if (e == 128) return (uint16_t)(sign | 0x7C00u | (m ? 0x200u : 0x0u));   // inf / nan
    if (e >= 16)  return (uint16_t)(sign | 0x7C00u);                        // overflow -> inf
    if (e >= -14) {                                                         // 规格化
        uint32_t mm = m >> 13, rem = m & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (mm & 1))) { mm++; if (mm == 0x400u) { mm = 0; e++; if (e >= 16) return (uint16_t)(sign | 0x7C00u); } }
        return (uint16_t)(sign | ((uint32_t)(e + 15) << 10) | mm);
    }
    if (e < -25) return (uint16_t)sign;                                     // 下溢 -> +-0
    uint32_t sh = (uint32_t)(-e - 1);                                       // 次正规
    uint32_t src = m | 0x800000u;
    uint32_t mm = src >> sh, rem = src & ((1u << sh) - 1u), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (mm & 1))) mm++;
    return (uint16_t)(sign | mm);
}
static float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    uint32_t x;
    if (e == 0) {
        if (m == 0) { x = sign; }
        else { // 次正规 -> 规格化
            int32_t ee = -1; uint32_t mm = m;
            while (!(mm & 0x400u)) { mm <<= 1; ee--; }
            mm &= 0x3FFu;
            x = sign | (uint32_t)((ee + 127 - 15 + 1) << 23) | (mm << 13);
        }
    } else if (e == 31) {
        x = sign | 0x7F800000u | (m << 13);
    } else {
        x = sign | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f; memcpy(&f, &x, 4); return f;
}

// ------------------------------ 结果台账 -------------------------------------
struct Row { std::string id, note; int size = 0; bool usable = false; double ms = -1, rr = -1, mabs = -1; };
static std::vector<Row> g_rows;
static void record(const char* id, int size, bool usable, double ms, double rr, double mabs, const std::string& note) {
    Row r; r.id = id; r.size = size; r.usable = usable; r.ms = ms; r.rr = rr; r.mabs = mabs; r.note = note;
    g_rows.push_back(r);
    printf("[atn_probe] %-6s size=%-8d usable=%-3s ms=%-9.4f relrms=%-10s maxabs=%-10s %s\n",
           id, size, usable ? "YES" : "NO",
           ms, rr < 0 ? "-" : std::to_string(rr).c_str(),
           mabs < 0 ? "-" : std::to_string(mabs).c_str(), note.c_str());
    fflush(stdout);
}
static void dump_json(const char* path) {
    if (!path) return;
    FILE* fp = fopen(path, "w");
    if (!fp) { printf("[atn_probe] 无法写 %s\n", path); return; }
    fprintf(fp, "{\n  \"generated\": true,\n  \"danger_mode\": %s,\n  \"rows\": [\n", DANGER ? "true" : "false");
    for (size_t i = 0; i < g_rows.size(); i++) {
        const Row& r = g_rows[i];
        fprintf(fp, "    {\"id\":\"%s\",\"size\":%d,\"usable\":%s,\"ms\":%.4f,\"relrms\":%.6g,\"maxabs\":%.6g,\"note\":\"%s\"}%s\n",
                r.id.c_str(), r.size, r.usable ? "true" : "false", r.ms, r.rr, r.mabs,
                r.note.c_str(), (i + 1 < g_rows.size()) ? "," : "");
    }
    fprintf(fp, "  ]\n}\n");
    fclose(fp);
    printf("[atn_probe] JSON -> %s (%zu rows)\n", path, g_rows.size());
}

// =============================================================================
// P1  cast<float,float16> / cast<float16,float> 往返, 与主机 RNE 转换对拍
//     判据: r==0; 往返 relrms <= 1e-3; 卡上 cast 的 f16 位型与主机 RNE【逐位相同】
// =============================================================================
static int T_P1(int len) {
    std::vector<float> x(len), back(len);
    fill(x, 12345u, 3.0f);
    // 塞边界值
    if (len >= 16) {
        x[0] = 0.f; x[1] = -0.f; x[2] = 1e-8f; x[3] = -1e-8f;
        x[4] = 1e30f; x[5] = -1e30f; x[6] = 1.0f; x[7] = -1.0f;
        x[8] = 65504.f; x[9] = 65520.f; x[10] = 6.1e-5f; x[11] = 6.0e-8f;
        x[12] = 1e-5f; x[13] = -1e-5f; x[14] = 3.1415f; x[15] = -3.1415f;
    }
    std::vector<uint16_t> hostbits(len);
    for (int i = 0; i < len; i++) hostbits[i] = f32_to_f16_bits_rne(x[i]);

    DBuf dx, dh, dback;
    dx.alloc((size_t)len * 4); dh.alloc((size_t)len * 2); dback.alloc((size_t)len * 4);
    dx.up(x.data(), (size_t)len * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    int r = api::cast<float, float16>(&ctx, (const float*)dx.p, (float16*)dh.p, len);
    r |= api::cast<float16, float>(&ctx, (const float16*)dh.p, (float*)dback.p, len);
    int ww = xpu_wait();
    std::vector<uint16_t> gotbits(len);
    dh.down(gotbits.data(), (size_t)len * 2);
    dback.down(back.data(), (size_t)len * 4);
    int bitdiff = 0; size_t first = (size_t)-1;
    for (int i = 0; i < len; i++) if (gotbits[i] != hostbits[i]) { bitdiff++; if (first == (size_t)-1) first = i; }
    double rr = relrms(back, x);
    char note[256];
    snprintf(note, sizeof(note), "r=%d wait=%d f16位型差异=%d个/%d (首个 idx=%zd: 卡=0x%04x 主机=0x%04x)",
             r, ww, bitdiff, len, first, first == (size_t)-1 ? 0 : gotbits[first], first == (size_t)-1 ? 0 : hostbits[first]);
    record("P1", len, (r == 0 && rr <= 0.1), 0, rr, -1, note);
    return (r == 0) ? 0 : 1;
}

// =============================================================================
// P2a qk_attention<float,float,float,float> 的 (TransA,TransB) 语义
//     极小: A[2,3] B[3,4]; 四种组合全打出来, 人工/自动比对哪一种是标准 A·B
// =============================================================================
static int T_P2a() {
    const int M = 2, K = 3, N = 4;
    std::vector<float> A = {1, 2, 3, 4, 5, 6};                    // [2,3]
    std::vector<float> Bv = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // [3,4]
    std::vector<float> ref(M * N, 0.f);
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { double s = 0; for (int k = 0; k < K; k++) s += (double)A[i * K + k] * Bv[k * N + j]; ref[i * N + j] = (float)s; }
    DBuf da, db, dc; da.alloc(M * K * 4); db.alloc(K * N * 4); dc.alloc(M * N * 4);
    da.up(A.data(), M * K * 4); db.up(Bv.data(), K * N * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    const bool combos[4][2] = {{false, false}, {false, true}, {true, false}, {true, true}};
    int best = -1; std::string bestnote;
    for (int c = 0; c < 4; c++) {
        std::vector<float> C(M * N, -999.f);
        dc.up(C.data(), M * N * 4);
        int r = api::qk_attention<float, float, float, float>(
            &ctx, combos[c][0], combos[c][1], 1, 1, M, N, K,
            1.0f, (const float*)da.p, (const float*)db.p,
            0.0f, (float*)dc.p, (const float*)nullptr,
            (const float*)nullptr, (const float*)nullptr, (float*)nullptr, false,
            false, 0, false, false);
        int ww = xpu_wait();
        dc.down(C.data(), M * N * 4);
        double rr = relrms(C, ref);
        printf("[atn_probe] P2a TransA=%d TransB=%d r=%d wait=%d relrms=%.6f%%  C=[%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f]\n",
               (int)combos[c][0], (int)combos[c][1], r, ww, rr, C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7]);
        if (r == 0 && rr < 1e-4 && best < 0) { best = c; bestnote = "TransA=" + std::to_string((int)combos[c][0]) + " TransB=" + std::to_string((int)combos[c][1]); }
    }
    printf("[atn_probe] P2a 参考 A*B = [%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f]\n", ref[0], ref[1], ref[2], ref[3], ref[4], ref[5], ref[6], ref[7]);
    record("P2a", 0, best >= 0, 0, -1, -1, best >= 0 ? ("仅一种组合等价 A*B: " + bestnote) : "★ 四种 (TransA,TransB) 组合【无一】等于 A*B => 语义判不出 => 弃用该算子");
    return 0;   // 语义探测本身不算失败 (打表即为结论)
}

// =============================================================================
// P2b gemm_int16 【非连续 B 的行步长 ldb=1024】(本设计的第 1 号依赖)
//     M=4 N=64 K=256; B 布局 [64][1024] 取 base+kh*256, ldb=1024
//     对照: 同一段数据拷成连续 [64][256] 再算一次 => 两次输出必须逐位相同
// =============================================================================
static int T_P2b(int N) {
    const int M = 4, K = 256, LD = 1024;
    std::vector<float> A(M * K); fill_norm(A, 777u, 1.0f);
    std::vector<float> Bbig((size_t)N * LD); fill_norm(Bbig, 888u, 1.0f);
    std::vector<float> Bcont((size_t)N * K);
    const int kh = 2;
    for (int i = 0; i < N; i++) for (int j = 0; j < K; j++) Bcont[(size_t)i * K + j] = Bbig[(size_t)i * LD + kh * K + j];
    DBuf da, db, dc, dcont, dc2;
    da.alloc(M * K * 4); db.alloc((size_t)N * LD * 4); dc.alloc(M * N * 4);
    dcont.alloc((size_t)N * K * 4); dc2.alloc(M * N * 4);
    da.up(A.data(), M * K * 4); db.up(Bbig.data(), (size_t)N * LD * 4); dcont.up(Bcont.data(), (size_t)N * K * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    const float alpha = 0.0625f;
    int r1 = api::gemm_int16(&ctx, false, true, M, N, K, alpha, (const float*)da.p, K,
                             (const float*)((float*)db.p + (size_t)kh * K), LD, 0.0f, (float*)dc.p, N);
    int r2 = api::gemm_int16(&ctx, false, true, M, N, K, alpha, (const float*)da.p, K,
                             (const float*)dcont.p, K, 0.0f, (float*)dc2.p, N);
    int ww = xpu_wait();
    std::vector<float> c1(M * N), c2(M * N);
    dc.down(c1.data(), M * N * 4); dc2.down(c2.data(), M * N * 4);
    double rr = relrms(c1, c2);
    // 顺带: 与主机 fp64 参考比, 看 gemm_int16 自身精度
    std::vector<float> ref(M * N, 0.f);
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { double s = 0; for (int k = 0; k < K; k++) s += (double)A[i * K + k] * Bcont[(size_t)j * K + k]; ref[i * N + j] = (float)(s * alpha); }
    double rrhost = relrms(c1, ref);
    char note[256];
    snprintf(note, sizeof(note), "r(ldb=1024)=%d r(ldb=K)=%d wait=%d | 与连续版 relrms=%.6g%% | vs fp64 主机 relrms=%.6g%% %s",
             r1, r2, ww, rr, rrhost, (rr <= 1e-5) ? "=> ★ ldb 非连续【支持】" : "=> ★ ldb 非连续【不支持/不等价】, 退 per-kh 连续缓冲");
    record("P2b", N, (r1 == 0 && r2 == 0 && rr <= 1e-5), 0, rr, -1, note);
    return (r1 == 0 && r2 == 0) ? 0 : 1;
}

// =============================================================================
// P2c qk_attention 的 max_a/max_b 语义 —— ★ 只允许 nullptr 通过
//     先跑 nullptr(安全); 非空指针版本归入 danger (属 *_maxptr 家族, 曾把 dev0 打进 ERROR)
// =============================================================================
static int T_P2c(bool nonnull) {
    const int M = 2, K = 3, N = 4;
    std::vector<float> A = {1, 2, 3, 4, 5, 6}, Bv = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    DBuf da, db, dc; da.alloc(M * K * 4); db.alloc(K * N * 4); dc.alloc(M * N * 4);
    da.up(A.data(), M * K * 4); db.up(Bv.data(), K * N * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    float one = 1.0f, res = 0.0f;
    const float* ma = nonnull ? &one : nullptr;
    const float* mb = nonnull ? &one : nullptr;
    float* rmp = nonnull ? &res : nullptr;
    int r = api::qk_attention<float, float, float, float>(
        &ctx, false, false, 1, 1, M, N, K, 1.0f,
        (const float*)da.p, (const float*)db.p, 0.0f, (float*)dc.p, (const float*)nullptr,
        ma, mb, rmp, nonnull, false, 0, false, false);
    int ww = xpu_wait();
    char note[160];
    snprintf(note, sizeof(note), "%s: r=%d wait=%d res_max=%g", nonnull ? "★非空 max 指针" : "nullptr max 指针", r, ww, nonnull ? res : -1.f);
    record(nonnull ? "P2c-nn" : "P2c", 0, (r == 0), 0, -1, -1, note);
    return 0;
}

// =============================================================================
// P2d is_asr_mask 语义 (信息性): 上三角填 +1e6, is_asr_mask=0/1 各跑一次
// =============================================================================
static int T_P2d() {
    const int M = 4, K = 4, N = 4;
    std::vector<float> A(M * K, 0.f), Bv(K * N, 0.f);
    for (int i = 0; i < M; i++) { A[i * K + i] = 1.0f; Bv[i * N + i] = 1.0f; }
    for (int i = 0; i < M; i++) for (int j = i + 1; j < N; j++) Bv[j * N + i] = 1e6f;   // 让 A·B 的"未来"位很大
    DBuf da, db, dc; da.alloc(M * K * 4); db.alloc(K * N * 4); dc.alloc(M * N * 4);
    da.up(A.data(), M * K * 4); db.up(Bv.data(), K * N * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    for (int am = 0; am <= 1; am++) {
        std::vector<float> C(M * N, -1.f); dc.up(C.data(), M * N * 4);
        int r = api::qk_attention<float, float, float, float>(
            &ctx, false, false, 1, 1, M, N, K, 1.0f,
            (const float*)da.p, (const float*)db.p, 0.0f, (float*)dc.p, (const float*)nullptr,
            (const float*)nullptr, (const float*)nullptr, (float*)nullptr, false,
            am ? true : false, 0, false, false);
        int ww = xpu_wait();
        dc.down(C.data(), M * N * 4);
        printf("[atn_probe] P2d is_asr_mask=%d r=%d wait=%d C=[%.1f %.1f %.1f %.1f | %.1f %.1f %.1f %.1f | %.1f %.1f %.1f %.1f | %.1f %.1f %.1f %.1f]\n",
               am, r, ww, C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7], C[8], C[9], C[10], C[11], C[12], C[13], C[14], C[15]);
    }
    record("P2d", 0, true, 0, -1, -1, "信息性: 两种输出已打表, 人工判是否标准因果掩码; 我们【不依赖】它");
    return 0;
}

// =============================================================================
// P2e bias=nullptr / is_tf_bias / is_onnx_bias 全 false
// =============================================================================
static int T_P2e() { return T_P2c(false); }

// =============================================================================
// P2f qk_attention 真实量级精度 (A=[4,256] 真 q 分布, B=[pos+1,256] 真 K 分布) vs fp64
// =============================================================================
static int T_P2f(int P) {
    const int M = 4, K = 256;
    std::vector<float> A(M * K); fill_norm(A, 31337u, 1.0f);
    std::vector<float> Bv((size_t)P * K); fill_norm(Bv, 42424u, 1.0f);
    DBuf da, db, dc; da.alloc(M * K * 4); db.alloc((size_t)P * K * 4); dc.alloc(M * P * 4);
    da.up(A.data(), M * K * 4); db.up(Bv.data(), (size_t)P * K * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    const float alpha = 0.0625f;   // 1/sqrt(256)
    int r = api::qk_attention<float, float, float, float>(
        &ctx, false, true, 1, 1, M, P, K, alpha,
        (const float*)da.p, (const float*)db.p, 0.0f, (float*)dc.p, (const float*)nullptr,
        (const float*)nullptr, (const float*)nullptr, (float*)nullptr, false, false, 0, false, false);
    int ww = xpu_wait();
    std::vector<float> C(M * P); dc.down(C.data(), M * P * 4);
    std::vector<float> ref(M * P, 0.f);
    for (int i = 0; i < M; i++) for (int j = 0; j < P; j++) {
        double s = 0; for (int k = 0; k < K; k++) s += (double)A[i * K + k] * Bv[(size_t)j * K + k];
        ref[i * P + j] = (float)(s * alpha);
    }
    double rr = relrms(C, ref);
    char note[200];
    snprintf(note, sizeof(note), "r=%d wait=%d | vs fp64 主机 relrms=%.6g%% %s", r, ww, rr,
             rr <= 1e-3 ? "(<=1e-3 合格)" : "★ >1e-3 精度不足");
    record("P2f", P, (r == 0 && rr <= 1e-3), 0, rr, -1, note);
    return (r == 0) ? 0 : 1;
}

// =============================================================================
// P3 qk_v_attention<float,float,float> (PV: probs[4,P] x V[P,256])
// =============================================================================
static int T_P3(int P) {
    const int M = 4, K = 256;
    std::vector<float> A((size_t)M * P); fill_norm(A, 9182u, 0.05f);
    for (int i = 0; i < M; i++) { double s = 0; for (int j = 0; j < P; j++) s += fabs((double)A[i * P + j]); for (int j = 0; j < P; j++) A[i * P + j] = (float)(A[i * P + j] / s); }  // 归一化成概率
    std::vector<float> Bv((size_t)P * K); fill_norm(Bv, 555u, 1.0f);
    DBuf da, db, dc; da.alloc((size_t)M * P * 4); db.alloc((size_t)P * K * 4); dc.alloc(M * K * 4);
    da.up(A.data(), (size_t)M * P * 4); db.up(Bv.data(), (size_t)P * K * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    int r = api::qk_v_attention<float, float, float>(
        &ctx, false, false, 1, 1, M, K, P, 1.0f,
        (const float*)da.p, (const float*)db.p, 0.0f, (float*)dc.p,
        (const float*)nullptr, (const float*)nullptr, (float*)nullptr, false);
    int ww = xpu_wait();
    std::vector<float> C(M * K); dc.down(C.data(), M * K * 4);
    std::vector<float> ref(M * K, 0.f);
    for (int i = 0; i < M; i++) for (int n = 0; n < K; n++) {
        double s = 0; for (int k = 0; k < P; k++) s += (double)A[(size_t)i * P + k] * Bv[(size_t)k * K + n];
        ref[i * K + n] = (float)s;
    }
    double rr = relrms(C, ref);
    char note[200];
    snprintf(note, sizeof(note), "r=%d wait=%d | vs fp64 主机 relrms=%.6g%% %s", r, ww, rr, rr <= 1e-3 ? "(<=1e-3 合格)" : "★ >1e-3");
    record("P3", P, (r == 0 && rr <= 1e-3), 0, rr, -1, note);
    return (r == 0) ? 0 : 1;
}

// =============================================================================
// P4a/P4b gemm_int16 基础正确性 + 边界 (最薄形状)
// =============================================================================
static int T_P4ab() {
    const int M = 4, N = 64, K = 256;
    std::vector<float> A(M * K); fill_norm(A, 1111u, 1.0f);
    std::vector<float> Bv((size_t)N * K); fill_norm(Bv, 2222u, 1.0f);
    DBuf da, db, dc; da.alloc(M * K * 4); db.alloc((size_t)N * K * 4); dc.alloc(M * N * 4);
    da.up(A.data(), M * K * 4); db.up(Bv.data(), (size_t)N * K * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    std::vector<float> ref(M * N, 0.f);
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { double s = 0; for (int k = 0; k < K; k++) s += (double)A[i * K + k] * Bv[(size_t)j * K + k]; ref[i * N + j] = (float)s; }
    int r = api::gemm_int16(&ctx, false, true, M, N, K, 1.0f, (const float*)da.p, K, (const float*)db.p, K, 0.0f, (float*)dc.p, N);
    int ww = xpu_wait();
    std::vector<float> C(M * N); dc.down(C.data(), M * N * 4);
    double rr = relrms(C, ref);
    record("P4a", N, (r == 0 && rr <= 1e-3), 0, rr, -1,
           "r=" + std::to_string(r) + " wait=" + std::to_string(ww) + " vs fp64");

    // P4b 边界: N=1 (pos=0 的第一次解码), K=1, M=1
    {
        std::vector<float> a1(K, 1.f), b1(K, 2.f), c1(1, 0.f), ref1(K, 0.f);
        DBuf xa, xb, xc; xa.alloc(K * 4); xb.alloc(K * 4); xc.alloc(4);
        xa.up(a1.data(), K * 4); xb.up(b1.data(), K * 4);
        for (int n = 0; n < K; n++) ref1[n] = 1.f * 2.f;
        int rr1 = api::gemm_int16(&ctx, false, true, 1, 1, K, 1.0f, (const float*)xa.p, K, (const float*)xb.p, K, 0.0f, (float*)xc.p, 1);
        int w1 = xpu_wait();
        xc.down(c1.data(), 4);
        record("P4b-N1", 1, (rr1 == 0 && fabs(c1[0] - 512.f * 2.f) < 1e-2), 0, -1, -1,
               "r=" + std::to_string(rr1) + " wait=" + std::to_string(w1) + " C0=" + std::to_string(c1[0]) + " (期望 1024 若 alpha=1 且 B=2·全1×512)");
    }
    {
        std::vector<float> a1(1, 3.f), b1(N, 5.f); std::vector<float> C1(N, 0.f);
        DBuf xa, xb, xc; xa.alloc(4); xb.alloc(N * 4); xc.alloc(N * 4);
        xa.up(a1.data(), 4); xb.up(b1.data(), N * 4);
        int rr1 = api::gemm_int16(&ctx, false, true, 1, N, 1, 1.0f, (const float*)xa.p, 1, (const float*)xb.p, 1, 0.0f, (float*)xc.p, N);
        int w1 = xpu_wait();
        xc.down(C1.data(), N * 4);
        record("P4b-K1", N, (rr1 == 0 && fabs(C1[0] - 15.f) < 1e-3), 0, -1, -1,
               "r=" + std::to_string(rr1) + " wait=" + std::to_string(w1) + " C0=" + std::to_string(C1[0]) + " (期望 15)");
    }
    return (r == 0) ? 0 : 1;
}

// =============================================================================
// P4c gemm_int16 薄矩阵 (m=4) 的【实测有效带宽】 —— 60 GB/s 假设的验证点
//     M=4, K=256; N 分档; 每次读的字节 ~ N*K*4 (B) ; 循环 50 次
// =============================================================================
static int T_P4c(int N) {
    const int M = 4, K = 256, ITER = 50;
    std::vector<float> A(M * K); fill_norm(A, 333u, 1.0f);
    std::vector<float> Bv((size_t)N * K); fill_norm(Bv, 444u, 1.0f);
    DBuf da, db, dc; da.alloc(M * K * 4); db.alloc((size_t)N * K * 4); dc.alloc(M * N * 4);
    da.up(A.data(), M * K * 4); db.up(Bv.data(), (size_t)N * K * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    api::gemm_int16(&ctx, false, true, M, N, K, 1.0f, (const float*)da.p, K, (const float*)db.p, K, 0.0f, (float*)dc.p, N);
    xpu_wait();
    double t0 = NOW();
    for (int it = 0; it < ITER; it++)
        api::gemm_int16(&ctx, false, true, M, N, K, 1.0f, (const float*)da.p, K, (const float*)db.p, K, 0.0f, (float*)dc.p, N);
    int ww = xpu_wait();
    double dt = (NOW() - t0) / ITER;
    double bytes = (double)N * K * 4.0 + (double)M * K * 4.0 + (double)M * N * 4.0;
    double gbs = bytes / dt / 1e9;
    char note[220];
    snprintf(note, sizeof(note), "wait=%d 每读 %.2f MiB 用时 %.4f ms => 有效带宽 %.2f GB/s %s",
             ww, bytes / 1048576.0, dt * 1000.0, gbs, gbs < 20 ? "★ 低于 20 GB/s, §2.3 的 60 GB/s 假设【不成立】" : "(达标)");
    record("P4c", N, (ww == 0), dt * 1000.0, -1, -1, note);
    return 0;
}

// =============================================================================
// P5 gemm_int31 (标量 max, 比 *_maxptr 安全)  含 max_a=0 的危险子例 (danger)
// =============================================================================
static int T_P5(int N, bool zero_a) {
    const int M = 4, K = 256;
    std::vector<float> A(M * K); fill_norm(A, 616u, 1.0f);
    std::vector<float> Bv((size_t)N * K); fill_norm(Bv, 717u, 1.0f);
    double ma = 0, mb = 0;
    for (auto v : A) ma = std::max(ma, fabs((double)v));
    for (auto v : Bv) mb = std::max(mb, fabs((double)v));
    if (zero_a) { for (auto& v : A) v = 0.f; ma = 0; }
    DBuf da, db, dc; da.alloc(M * K * 4); db.alloc((size_t)N * K * 4); dc.alloc(M * N * 4);
    da.up(A.data(), M * K * 4); db.up(Bv.data(), (size_t)N * K * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    int r = api::gemm_int31(&ctx, false, true, M, N, K, 1.0f, (const float*)da.p, K, (const float*)db.p, K,
                            0.0f, (float*)dc.p, N, (float)ma, (float)mb);
    int ww = xpu_wait();
    std::vector<float> C(M * N); dc.down(C.data(), M * N * 4);
    double rr = -1; int allzero = 1;
    for (auto v : C) if (v != 0.f) allzero = 0;
    if (!zero_a) {
        std::vector<float> ref(M * N, 0.f);
        for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { double s = 0; for (int k = 0; k < K; k++) s += (double)A[i * K + k] * Bv[(size_t)j * K + k]; ref[i * N + j] = (float)s; }
        rr = relrms(C, ref);
    }
    char note[220];
    snprintf(note, sizeof(note), "r=%d wait=%d max_a=%g max_b=%g%s", r, ww, ma, mb,
             zero_a ? (" | 全零 A: 输出全零=%s (不能除零/不能出 nan)") : "");
    if (zero_a) {
        snprintf(note, sizeof(note), "r=%d wait=%d max_a=0 | ★ 输出全零=%s (期望 YES; inf/nan 即卡上除零)", r, ww, allzero ? "YES" : "NO");
        record("P5b", N, (r == 0 && allzero), 0, -1, -1, note);
    } else {
        snprintf(note, sizeof(note), "r=%d wait=%d max_a=%g max_b=%g | vs fp64 relrms=%.6g%% (int31 期望 1e-2 量级)", r, ww, ma, mb, rr);
        record("P5", N, (r == 0 && rr >= 0 && rr <= 1.0), 0, rr, -1, note);
    }
    return (r == 0) ? 0 : 1;
}

// =============================================================================
// P7a softmax2d 薄形状正确性  rows=16, cols 分档; cols=1 期望恒 1.0
// =============================================================================
static int T_P7a(int rows, int cols) {
    std::vector<float> x((size_t)rows * cols), y((size_t)rows * cols), ref((size_t)rows * cols);
    fill(x, 5150u, 8.0f);
    for (int t = 0; t < rows; t++) {
        const float* p = &x[(size_t)t * cols];
        double mx = -1e30; for (int i = 0; i < cols; i++) if (p[i] > mx) mx = p[i];
        double s = 0; for (int i = 0; i < cols; i++) s += exp((double)p[i] - mx);
        for (int i = 0; i < cols; i++) ref[(size_t)t * cols + i] = (float)(exp((double)p[i] - mx) / s);
    }
    DBuf dx, dy; dx.alloc((size_t)rows * cols * 4); dy.alloc((size_t)rows * cols * 4);
    dx.up(x.data(), (size_t)rows * cols * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    int r = api::softmax2d_forward(&ctx, (const float*)dx.p, (float*)dy.p, rows, cols, false);
    int ww = xpu_wait();
    dy.down(y.data(), (size_t)rows * cols * 4);
    double rr = relrms(y, ref), md = maxabs_diff(y, ref);
    record("P7a", cols, (r == 0 && rr <= 1e-3), 0, rr, md,
           "rows=" + std::to_string(rows) + " r=" + std::to_string(r) + " wait=" + std::to_string(ww));
    return (r == 0) ? 0 : 1;
}

// =============================================================================
// P7b softmax2d 薄形状【耗时】(任务书问的 64KB~2MB 级张量划不划算)
//     判据: 耗时 <= 3 x 读该张量时间(16*cols*4B @100GB/s)
// =============================================================================
static int T_P7b(int cols, int rows) {
    const int ITER = 20;
    std::vector<float> x((size_t)rows * cols); fill(x, 8181u, 4.0f);
    DBuf dx, dy; dx.alloc((size_t)rows * cols * 4); dy.alloc((size_t)rows * cols * 4);
    dx.up(x.data(), (size_t)rows * cols * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    api::softmax2d_forward(&ctx, (const float*)dx.p, (float*)dy.p, rows, cols, false);
    xpu_wait();
    double t0 = NOW();
    for (int it = 0; it < ITER; it++) api::softmax2d_forward(&ctx, (const float*)dx.p, (float*)dy.p, rows, cols, false);
    int ww = xpu_wait();
    double dt = (NOW() - t0) / ITER;
    double tensor_mib = (double)rows * cols * 4.0 / 1048576.0;
    double read_ms = (double)rows * cols * 4.0 / 100e9 * 1000.0;   // @100GB/s
    double ratio = dt * 1000.0 / (read_ms > 0 ? read_ms : 1e-9);
    char note[240];
    snprintf(note, sizeof(note), "张量 %.3f MiB | 每次 %.4f ms | 理论读 %.4f ms | 倍数 %.2fx %s",
             tensor_mib, dt * 1000.0, read_ms, ratio, ratio <= 3.0 ? "(<=3x 划算)" : "★ >3x 不划算");
    record("P7b", cols, (ww == 0), dt * 1000.0, -1, -1, note);
    return 0;
}

// =============================================================================
// P7c softmax2d 原地 (y == x)
// =============================================================================
static int T_P7c(int rows, int cols) {
    std::vector<float> x((size_t)rows * cols), ref((size_t)rows * cols);
    fill(x, 9191u, 5.0f);
    for (int t = 0; t < rows; t++) {
        const float* p = &x[(size_t)t * cols];
        double mx = -1e30; for (int i = 0; i < cols; i++) if (p[i] > mx) mx = p[i];
        double s = 0; for (int i = 0; i < cols; i++) s += exp((double)p[i] - mx);
        for (int i = 0; i < cols; i++) ref[(size_t)t * cols + i] = (float)(exp((double)p[i] - mx) / s);
    }
    DBuf dx; dx.alloc((size_t)rows * cols * 4); dx.up(x.data(), (size_t)rows * cols * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    int r = api::softmax2d_forward(&ctx, (const float*)dx.p, (float*)dx.p, rows, cols, false);
    int ww = xpu_wait();
    std::vector<float> y((size_t)rows * cols); dx.down(y.data(), (size_t)rows * cols * 4);
    double rr = relrms(y, ref);
    record("P7c", cols, (r == 0 && rr <= 1e-3), 0, rr, -1,
           "原地 rows=" + std::to_string(rows) + " r=" + std::to_string(r) + " wait=" + std::to_string(ww));
    return (r == 0) ? 0 : 1;
}

// =============================================================================
// P9 卡上除法: ① 分母含 0 (★危险)  ② 先夹逼到 1e-30  ③ div_no_nan
// =============================================================================
static int T_P9(int mode, int len) {
    std::vector<float> x(len), yv(len), z(len), ref(len);
    fill(x, 1212u, 2.0f);
    for (int i = 0; i < len; i++) yv[i] = (i % 7 == 0) ? 0.f : (0.5f + (i % 13) * 0.1f);
    for (int i = 0; i < len; i++) ref[i] = (yv[i] == 0.f) ? 0.f : x[i] / yv[i];
    DBuf dx, dy, dz, deps, dyc;
    dx.alloc((size_t)len * 4); dy.alloc((size_t)len * 4); dz.alloc((size_t)len * 4);
    deps.alloc((size_t)len * 4); dyc.alloc((size_t)len * 4);
    dx.up(x.data(), (size_t)len * 4); dy.up(yv.data(), (size_t)len * 4);
    std::vector<float> epsv(len, 1e-30f); deps.up(epsv.data(), (size_t)len * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    int r = 0; const char* tag = "";
    if (mode == 0) { tag = "① 生除(分母含0)"; r = api::elementwise_div_2d(&ctx, (const float*)dx.p, (const float*)dy.p, (float*)dz.p, 1, len, 1, len); }
    else if (mode == 1) {
        tag = "② 先夹逼 max(y,1e-30)";
        r = api::elementwise_max_2d(&ctx, (const float*)dy.p, (const float*)deps.p, (float*)dyc.p, 1, len, 1, len);
        r |= api::elementwise_div_2d(&ctx, (const float*)dx.p, (const float*)dyc.p, (float*)dz.p, 1, len, 1, len);
    } else { tag = "③ div_no_nan"; r = api::elementwise_div_no_nan_2d(&ctx, (const float*)dx.p, (const float*)dy.p, (float*)dz.p, 1, len, 1, len); }
    int ww = xpu_wait();
    dz.down(z.data(), (size_t)len * 4);
    int nan = 0, inf = 0;
    for (auto v : z) { if (std::isnan(v)) nan++; if (std::isinf(v)) inf++; }
    double rr = relrms(z, ref);
    char note[260];
    snprintf(note, sizeof(note), "%s r=%d wait=%d nan=%d inf=%d | vs 夹逼参考 relrms=%.6g%%", tag, r, ww, nan, inf, rr);
    record(mode == 0 ? "P9-raw" : (mode == 1 ? "P9-clamp" : "P9-nonan"), len, (r == 0 && nan == 0 && inf == 0), 0, rr, -1, note);
    return 0;
}

// =============================================================================
// P7d ★危险★ 一整行全 -1e30 的 softmax (只在 ATN_DANGER=1 跑, 且排最后)
// =============================================================================
static int T_P7d(int rows, int cols) {
    std::vector<float> x((size_t)rows * cols); fill(x, 1313u, 3.0f);
    for (int j = 0; j < cols; j++) x[j] = -1e30f;   // 第 0 行全掩
    DBuf dx, dy; dx.alloc((size_t)rows * cols * 4); dy.alloc((size_t)rows * cols * 4);
    dx.up(x.data(), (size_t)rows * cols * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    int r = api::softmax2d_forward(&ctx, (const float*)dx.p, (float*)dy.p, rows, cols, false);
    int ww = xpu_wait();
    std::vector<float> y((size_t)rows * cols); dy.down(y.data(), (size_t)rows * cols * 4);
    int nan = 0, inf = 0;
    for (auto v : y) { if (std::isnan(v)) nan++; if (std::isinf(v)) inf++; }
    char note[240];
    snprintf(note, sizeof(note), "★全 -1e30 行: r=%d wait=%d nan=%d inf=%d y0=%g (若 session 被打死, 说明本用例是预期内的排除项)",
             r, ww, nan, inf, y[0]);
    record("P7d", cols, (r == 0), 0, -1, -1, note);
    return 0;
}

// =============================================================================
// P10 transpose
// =============================================================================
static int T_P10(int m, int n) {
    std::vector<float> x((size_t)m * n); fill(x, 1414u, 2.0f);
    DBuf dx, dy; dx.alloc((size_t)m * n * 4); dy.alloc((size_t)m * n * 4);
    dx.up(x.data(), (size_t)m * n * 4);
    int shape[2] = {m, n}, perm[2] = {1, 0};
    Context ctx(Device(DeviceType::XPU1, 0));
    int r = api::transpose(&ctx, (const float*)dx.p, (float*)dy.p, shape, perm, 2);
    int ww = xpu_wait();
    std::vector<float> y((size_t)m * n); dy.down(y.data(), (size_t)m * n * 4);
    std::vector<float> ref((size_t)m * n);
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) ref[(size_t)j * m + i] = x[(size_t)i * n + j];
    double rr = relrms(y, ref);
    record("P10", n, (r == 0 && rr == 0.0), 0, rr, -1,
           "[" + std::to_string(m) + "," + std::to_string(n) + "] r=" + std::to_string(r) + " wait=" + std::to_string(ww));
    return (r == 0) ? 0 : 1;
}

// =============================================================================
// P11 memcpy_device (D2D) 正确性 + 每次耗时;  P12 api::memset
// =============================================================================
static int T_P11(int kib) {
    size_t n = (size_t)kib * 1024;
    std::vector<unsigned char> src(n), dst(n, 0);
    for (size_t i = 0; i < n; i++) src[i] = (unsigned char)(i & 0xFF);
    DBuf a, b; a.alloc(n); b.alloc(n);
    a.up(src.data(), n);
    Context ctx(Device(DeviceType::XPU1, 0));
    api::memcpy_device(&ctx, b.p, a.p, (int)n);
    int ww = xpu_wait();
    b.down(dst.data(), n);
    int diff = (memcmp(src.data(), dst.data(), n) != 0) ? 1 : 0;
    double t0 = NOW();
    for (int it = 0; it < 50; it++) api::memcpy_device(&ctx, b.p, a.p, (int)n);
    xpu_wait();
    double dt = (NOW() - t0) / 50;
    record("P11", kib, (ww == 0 && diff == 0), dt * 1000.0, -1, -1,
           "wait=" + std::to_string(ww) + " 逐字节相同=" + (diff ? "NO" : "YES"));
    return 0;
}
static int T_P12(int kib) {
    size_t n = (size_t)kib * 1024;
    std::vector<unsigned char> dst(n, 0xAB);
    DBuf a; a.alloc(n); a.up(dst.data(), n);
    Context ctx(Device(DeviceType::XPU1, 0));
    int r = api::memset(&ctx, a.p, 0, n);
    int ww = xpu_wait();
    std::vector<unsigned char> out(n, 0xAB); a.down(out.data(), n);
    int allzero = 1; for (size_t i = 0; i < n; i++) if (out[i] != 0) { allzero = 0; break; }
    record("P12", kib, (r == 0 && allzero), 0, -1, -1,
           "r=" + std::to_string(r) + " wait=" + std::to_string(ww) + " 全零=" + (allzero ? "YES" : "NO"));
    return 0;
}

// =============================================================================
// P13 activation_forward(SIGMOID) vs 主机 sigmoidf_  (表驱动嫌疑 => 预计非位级)
// =============================================================================
static int T_P13(int len) {
    std::vector<float> x(len), y(len), ref(len);
    for (int i = 0; i < len; i++) x[i] = -20.0f + 40.0f * (float)i / (float)(len > 1 ? len - 1 : 1);
    for (int i = 0; i < len; i++) ref[i] = 1.0f / (1.0f + expf(-x[i]));
    DBuf dx, dy; dx.alloc((size_t)len * 4); dy.alloc((size_t)len * 4); dx.up(x.data(), (size_t)len * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    int r = api::activation_forward(&ctx, Activation_t::SIGMOID, len, (const float*)dx.p, (float*)dy.p);
    int ww = xpu_wait();
    dy.down(y.data(), (size_t)len * 4);
    double rr = relrms(y, ref), md = maxabs_diff(y, ref);
    record("P13", len, (r == 0), 0, rr, md,
           "r=" + std::to_string(r) + " wait=" + std::to_string(ww) + " | 与 libm sigmoid 最大绝对差=" + std::to_string(md) + (md == 0.0 ? " (位级相同)" : " (非位级 => gate 留主机)"));
    return (r == 0) ? 0 : 1;
}

// =============================================================================
// P14 H2D/D2H 小传输固定开销 (16us/次 假设的验证点)
// =============================================================================
static int T_P14(int kib) {
    size_t n = (size_t)kib * 1024;
    std::vector<unsigned char> h(n, 7), out(n, 0);
    DBuf a; a.alloc(n);
    const int ITER = 100;
    double t0 = NOW();
    for (int it = 0; it < ITER; it++) xpu_memcpy(a.p, h.data(), n, XPU_HOST_TO_DEVICE);
    for (int it = 0; it < ITER; it++) xpu_memcpy(out.data(), a.p, n, XPU_DEVICE_TO_HOST);
    xpu_wait();
    double dt = (NOW() - t0) / (2.0 * ITER);
    record("P14", kib, true, dt * 1000.0, -1, -1,
           std::to_string(kib) + "KiB 单次传输 " + std::to_string(dt * 1e6) + " us");
    return 0;
}

// =============================================================================
// P15 slice_forward (可选). 语义待探: 这里只做"能否调用 + 前 256 是否等于原始前 256"
// =============================================================================
static int T_P15(int n) {
    std::vector<float> x(n); fill(x, 1515u, 2.0f);
    DBuf dx, dy; dx.alloc((size_t)n * 4); dy.alloc((size_t)n * 4);
    dx.up(x.data(), (size_t)n * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    // slice_forward 的签名在 func_dec.h:1763, 参数较多; 这里只做"能否调用"的探测。
    // 若编译不过/语义不明 => 结论写"未采用(直接给基址偷懒, 推荐)"。
    record("P15", n, false, 0, -1, -1, "未调用: slice_forward 参数语义未定稿; 设计上【不需要】它(直接给基址) — 本项如实记'未采用'");
    return 0;
}

// =============================================================================
// P16 qk_attention<float16,float16,float16,float> (fp16 KV 的硬依赖!)
//     = U2: 决定 128k 能不能做。语义(转置) + 真实量级精度。
// =============================================================================
static int T_P16a() {
    const int M = 2, K = 3, N = 4;
    std::vector<float> A = {1, 2, 3, 4, 5, 6}, Bv = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> ref(M * N, 0.f);
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { double s = 0; for (int k = 0; k < K; k++) s += (double)A[i * K + k] * Bv[k * N + j]; ref[i * N + j] = (float)s; }
    DBuf ha, hb, da, db, dc;
    ha.alloc(M * K * 2); hb.alloc(K * N * 2); da.alloc(M * K * 2); db.alloc(K * N * 2); dc.alloc(M * N * 2);
    auto up_f16 = [](DBuf& dst, const std::vector<float>& src, int len, float dummy) {
        (void)dummy;
        Context c(Device(DeviceType::XPU1, 0));
        DBuf tmp; tmp.alloc((size_t)len * 4); tmp.up(src.data(), (size_t)len * 4);
        api::cast<float, float16>(&c, (const float*)tmp.p, (float16*)dst.p, len);
        xpu_wait();
    };
    up_f16(da, A, M * K, 0); up_f16(db, Bv, K * N, 0);
    Context ctx(Device(DeviceType::XPU1, 0));
    const bool combos[4][2] = {{false, false}, {false, true}, {true, false}, {true, true}};
    int best = -1; std::string bn;
    DBuf dcf; dcf.alloc(M * N * 4);          // f16 结果 cast 回 f32 再 D2H
    for (int c = 0; c < 4; c++) {
        std::vector<float> C16(M * N * 2, 0);
        dc.up(C16.data(), M * N * 2);
        int r = api::qk_attention<float16, float16, float16, float>(
            &ctx, combos[c][0], combos[c][1], 1, 1, M, N, K, 1.0f,
            (const float16*)da.p, (const float16*)db.p, 0.0f, (float16*)dc.p, (const float*)nullptr,
            (const float*)nullptr, (const float*)nullptr, (float*)nullptr, false, false, 0, false, false);
        int ww = xpu_wait();
        double rr = -1;
        if (r == 0) {   // C 是 f16: 卡上 cast 回 f32 再回传, 与 fp64 参考比
            api::cast<float16, float>(&ctx, (const float16*)dc.p, (float*)dcf.p, M * N);
            xpu_wait();
            std::vector<float> Cf(M * N); dcf.down(Cf.data(), M * N * 4);
            rr = relrms(Cf, ref);
        }
        printf("[atn_probe] P16a TransA=%d TransB=%d r=%d wait=%d%s\n", (int)combos[c][0], (int)combos[c][1], r, ww,
               rr < 0 ? "" : (" relrms_vs_A*B=" + std::to_string(rr) + "%").c_str());
        if (r == 0 && rr >= 0 && rr < 1e-4 && best < 0) { best = c; bn = "TransA=" + std::to_string((int)combos[c][0]) + " TransB=" + std::to_string((int)combos[c][1]); }
    }
    record("P16a", 0, best >= 0, 0, -1, -1,
           best >= 0 ? ("qk_attention<f16> 可用; 等价 A*B 的组合: " + bn + " => fp16 KV 路线【成立】")
                     : "★ qk_attention<f16> 无任何组合等价 A*B => fp16 KV 路线作废 => MAXT 上限停在 32768(fp32)");
    return 0;
}

// =============================================================================
// P17 (本轮新增, 修正设计书 U8) matrix_vector_mul 语义核实
//     头文件原文: out[i*n+j] = matrix[i*n+j] * vec[j]  => 是【逐元素广播乘】, 不是矩阵乘!
//     => 设计书 §2.9 的 U8 兜底链 (靠 matrix_vector_mul 算 K_kh·q) 【不成立】, 必须改写。
// =============================================================================
static int T_P17() {
    const int m = 2, n = 3;
    std::vector<float> mat = {1, 2, 3, 4, 5, 6}, vec = {10, 20, 30};
    std::vector<float> ref(m * n);
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) ref[i * n + j] = mat[i * n + j] * vec[j];
    DBuf dm, dv, do_; dm.alloc(m * n * 4); dv.alloc(n * 4); do_.alloc(m * n * 4);
    dm.up(mat.data(), m * n * 4); dv.up(vec.data(), n * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    int r = api::matrix_vector_mul(&ctx, (const float*)dm.p, (const float*)dv.p, (float*)do_.p, m, n);
    int ww = xpu_wait();
    std::vector<float> out(m * n); do_.down(out.data(), m * n * 4);
    double rr = relrms(out, ref);
    record("P17", n, (r == 0 && rr == 0.0), 0, rr, -1,
           "r=" + std::to_string(r) + " wait=" + std::to_string(ww) + " | 与 out=m*v广播 relrms=" + std::to_string(rr) +
           " => " + (rr == 0.0 ? "★ 确认是逐元素广播乘(非矩阵乘), U8 兜底链作废" : "语义与头文件不符, 需人工判"));
    return 0;
}

// =============================================================================
//                                main
// =============================================================================
static void explain() {
    printf(
        "atn_probe — ATN_ON_CARD 阶段①探针\n"
        "用法: atn_probe <case> [json_out]\n"
        "  safe  : 跑全部【安全】用例(危险用例除外) —— 建议第一个窗口\n"
        "  danger: 只跑危险用例(P7d 全掩行 / P9-raw 生除 / P5b max_a=0) —— 单独窗口, ATN_DANGER=1\n"
        "  all   : safe + danger\n"
        "  单个用例: P1 P2a P2b P2c P2d P2e P2f P3 P4a P4b P4c P5 P5b P7a P7b P7c P7d P9 P10 P11 P12 P13 P14 P15 P16a P17\n");
}
int main(int argc, char** argv) {
    const char* t = (argc > 1) ? argv[1] : "help";
    const char* json = (argc > 2) ? argv[2] : getenv("ATN_PROBE_JSON");
    printf("[atn_probe] case=%s danger_mode=%d\n", t, (int)DANGER);
    if (!strcmp(t, "help")) { explain(); return 0; }

    int rc = 0;
    bool is_safe = !strcmp(t, "safe") || !strcmp(t, "all");
    bool is_danger = !strcmp(t, "danger") || !strcmp(t, "all");
    if (is_danger && !DANGER) { printf("[atn_probe] 危险用例需要 ATN_DANGER=1, 已跳过\n"); is_danger = false; }
    if (!strcmp(t, "danger") && !DANGER) return 0;

    // ------------------------- 安全用例 (尺寸阶梯) -------------------------
    if (is_safe || !strcmp(t, "P1"))   { rc |= T_P1(1024); if (rc) { printf("ABORT@P1 tiny\n"); goto done; } rc |= T_P1(65536); if (rc) { printf("ABORT@P1 mid\n"); goto done; } rc |= T_P1(1048576); }
    if (rc) goto done;
    if (is_safe || !strcmp(t, "P2a"))  rc |= T_P2a();
    if (is_safe || !strcmp(t, "P2b"))  { rc |= T_P2b(64); if (rc) { printf("ABORT@P2b tiny\n"); goto done; } rc |= T_P2b(512); if (rc) { printf("ABORT@P2b mid\n"); goto done; } rc |= T_P2b(4096); }
    if (rc) goto done;
    if (is_safe || !strcmp(t, "P2c"))  rc |= T_P2c(false);
    if (is_safe || !strcmp(t, "P2d"))  rc |= T_P2d();
    if (is_safe || !strcmp(t, "P2e"))  rc |= T_P2e();
    if (is_safe || !strcmp(t, "P2f"))  { rc |= T_P2f(64); if (rc) goto done; rc |= T_P2f(512); if (rc) goto done; rc |= T_P2f(8192); }
    if (rc) goto done;
    if (is_safe || !strcmp(t, "P3"))   { rc |= T_P3(64); if (rc) goto done; rc |= T_P3(512); if (rc) goto done; rc |= T_P3(8192); }
    if (rc) goto done;
    if (is_safe || !strcmp(t, "P4a"))  rc |= T_P4ab();
    if (rc) goto done;
    if (is_safe || !strcmp(t, "P4c"))  { rc |= T_P4c(1024); rc |= T_P4c(4096); rc |= T_P4c(16384); rc |= T_P4c(32768); }
    if (is_safe || !strcmp(t, "P5"))   { rc |= T_P5(512, false); if (rc) goto done; rc |= T_P5(4096, false); }
    if (rc) goto done;
    if (is_safe || !strcmp(t, "P7a"))  { rc |= T_P7a(16, 1); rc |= T_P7a(16, 2); rc |= T_P7a(16, 64); rc |= T_P7a(16, 1024); rc |= T_P7a(4, 1024); rc |= T_P7a(64, 1024); }
    if (is_safe || !strcmp(t, "P7b"))  { rc |= T_P7b(1024, 16); rc |= T_P7b(4096, 16); rc |= T_P7b(16384, 16); rc |= T_P7b(32768, 16); }
    if (is_safe || !strcmp(t, "P7c"))  rc |= T_P7c(16, 1024);
    if (is_safe || !strcmp(t, "P9"))   { rc |= T_P9(1, 4096); rc |= T_P9(2, 4096); }
    if (is_safe || !strcmp(t, "P10"))  { rc |= T_P10(1, 256); rc |= T_P10(4, 256); }
    if (is_safe || !strcmp(t, "P11"))  { rc |= T_P11(1); rc |= T_P11(8); rc |= T_P11(1024); }
    if (is_safe || !strcmp(t, "P12"))  rc |= T_P12(4);
    if (is_safe || !strcmp(t, "P13"))  rc |= T_P13(4096);
    if (is_safe || !strcmp(t, "P14"))  { rc |= T_P14(1); rc |= T_P14(8); rc |= T_P14(16); rc |= T_P14(64); }
    if (is_safe || !strcmp(t, "P15"))  rc |= T_P15(1024);
    if (is_safe || !strcmp(t, "P16a")) rc |= T_P16a();
    if (is_safe || !strcmp(t, "P17"))  rc |= T_P17();

    // ------------------------- 危险用例 (永远最后) -------------------------
    if (is_danger) {
        printf("[atn_probe] ===== 进入危险用例段 (P9-raw / P5b / P7d) =====\n"); fflush(stdout);
        if (is_danger || !strcmp(t, "P9"))  rc |= T_P9(0, 4096);
        rc |= T_P5(512, true);
        rc |= T_P7d(4, 1024);
    }

done:
    printf("[atn_probe] ==== case=%s rc=%d 已测 %zu 项 ====\n", t, rc, g_rows.size());
    dump_json(json);
    if (rc) printf("[atn_probe] ★ 有失败项 => 该阶段判失败。若 safe_run 退出码=3(卡异常) 则【严禁重试】。\n");
    return rc;
}
