// ============================================================================
// vis.cpp — Ornith 视觉塔 (SigLIP-so400m-16-768 + qwen3vl_merger) 在 K200 双芯上实现
//   大矩阵乘 = 官方 api::gemm_int8 (SD-CDNN), 与文本路径完全同一套约定:
//       - 权重每输出行对称 int8 + 每行一个 float scale; max_b 传 127; 输出按行 scale 折回
//       - 激活在主机做"行归一化" (A[t,:] *= M0/max|A[t,:]|), 输出再 *max_t/M0
//   norm / GELU / softmax / M-RoPE / 残差 在主机 float (与文本引擎既有做法一致)
//   ★ 禁用 gemm_int8_maxptr / api::fc<float,float,float,int> (都曾把卡打进 ERROR)
// ============================================================================
#include "vis.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <xpu/runtime.h>
#include "xpu/refactor/nn.h"
#include "xpu/refactor/context/xpu_act_type.h"
#include "xpu/api.h"   // ★ 官方逐元素/归一/激活算子 (elementwise_mul_2d / elementwise_add / reduce)
#include "xpu/refactor/context/xpu_act_type.h"

namespace api = baidu::xpu::api;
namespace baidu { namespace xpu { namespace api {
int gemm_int8(Context* ctx, const bool trans_a, const bool trans_b, int m, int n, int k,
              float alpha, const float* a, int lda, const int8_t* b, float max_b, int ldb,
              float beta, float* c, int ldc);
}}}

#define VNE      1152        // 视觉塔 hidden
#define VNH      16          // head 数
#define VDH      72          // head dim
#define VNL      27          // 层数
#define VFFN     4304
#define VEPS     1e-6f
#define VMA        1024      // 512x512 => 32x32 patches
#define VMA_MAX    2304      // 768x768 => 48x48 patches
#define VKMAX    4608        // mm.0 的 K
#define VTMAX    4096        // 最大图像 token 数 (留足余量)

static double vnow() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static int g_vdbg = 0;

// ============================================================================
// 1. 极简 GGUF 解析 (只服务 mmproj: F32 / BF16 / F16)
// ============================================================================
struct VGT { int nd; int64_t ne[4]; int type; uint64_t off; };
struct VG {
    FILE* f = nullptr;
    uint64_t dstart = 0, align = 32;
    std::map<std::string, VGT> ts;

    static void skip_val(FILE* f, int t) {
        switch (t) {
            case 0: case 1: case 7: { int8_t x; if (fread(&x, 1, 1, f)) {} break; }
            case 2: case 3: { int16_t x; if (fread(&x, 2, 1, f)) {} break; }
            case 4: case 5: case 6: { int32_t x; if (fread(&x, 4, 1, f)) {} break; }
            case 10: case 11: case 12: { int64_t x; if (fread(&x, 8, 1, f)) {} break; }
            case 8: { uint64_t n; if (fread(&n, 8, 1, f)) {} fseek(f, (long)n, SEEK_CUR); break; }
            case 9: {
                uint32_t et; uint64_t n;
                if (fread(&et, 4, 1, f)) {} if (fread(&n, 8, 1, f)) {}
                for (uint64_t i = 0; i < n; i++) skip_val(f, (int)et);
                break;
            }
            default: printf("[vis] FATAL: 未知 KV 类型 %d\n", t); exit(1);
        }
    }
    bool open(const char* path) {
        f = fopen(path, "rb");
        if (!f) { printf("[vis] FATAL: 打不开 %s\n", path); return false; }
        uint32_t magic, ver; uint64_t nt, nkv;
        if (fread(&magic, 4, 1, f) != 1 || fread(&ver, 4, 1, f) != 1 ||
            fread(&nt, 8, 1, f) != 1 || fread(&nkv, 8, 1, f) != 1) return false;
        if (magic != 0x46554747u) { printf("[vis] FATAL: 非 GGUF (magic=%08x)\n", magic); return false; }
        for (uint64_t i = 0; i < nkv; i++) {
            uint64_t kl; if (fread(&kl, 8, 1, f) != 1) return false;
            std::vector<char> k(kl + 1); if (fread(k.data(), 1, kl, f) != kl) return false; k[kl] = 0;
            uint32_t t; if (fread(&t, 4, 1, f) != 1) return false;
            if (std::string(k.data()) == "general.alignment" && t == 4) {
                uint32_t a; if (fread(&a, 4, 1, f) != 1) return false; align = a;
            } else skip_val(f, (int)t);
        }
        for (uint64_t i = 0; i < nt; i++) {
            uint64_t nl; if (fread(&nl, 8, 1, f) != 1) return false;
            std::vector<char> nm(nl + 1); if (fread(nm.data(), 1, nl, f) != nl) return false; nm[nl] = 0;
            uint32_t nd; if (fread(&nd, 4, 1, f) != 1) return false;
            VGT t; t.nd = (int)nd; t.ne[0] = t.ne[1] = t.ne[2] = t.ne[3] = 1;
            for (uint32_t d = 0; d < nd; d++) { if (fread(&t.ne[d], 8, 1, f) != 1) return false; }
            uint32_t ty; uint64_t off;
            if (fread(&ty, 4, 1, f) != 1 || fread(&off, 8, 1, f) != 1) return false;
            t.type = (int)ty; t.off = off;
            ts[std::string(nm.data())] = t;
        }
        long p = ftell(f);
        dstart = (uint64_t)((p + (long)align - 1) / (long)align * (long)align);
        printf("[vis] mmproj GGUF: %zu tensors, align=%llu dstart=%llu\n", ts.size(),
               (unsigned long long)align, (unsigned long long)dstart);
        return true;
    }
    // 读张量 -> f32 (ggml 序, 长度 = prod(ne))
    bool read_f32(const std::string& name, std::vector<float>& out) {
        auto it = ts.find(name);
        if (it == ts.end()) { printf("[vis] FATAL: 缺张量 %s\n", name.c_str()); return false; }
        const VGT& t = it->second;
        size_t n = 1; for (int i = 0; i < t.nd; i++) n *= (size_t)t.ne[i];
        out.resize(n);
        if (fseek(f, (long)(dstart + t.off), SEEK_SET)) { printf("[vis] FATAL: seek %s\n", name.c_str()); return false; }
        if (t.type == 0) { if (fread(out.data(), 4, n, f) != n) return false; }
        else if (t.type == 1) {
            std::vector<uint16_t> h(n); if (fread(h.data(), 2, n, f) != n) return false;
            for (size_t i = 0; i < n; i++) { uint32_t u = ((uint32_t)h[i]) << 16; float v; memcpy(&v, &u, 4); out[i] = v; }
        } else if (t.type == 30) {
            std::vector<uint16_t> h(n); if (fread(h.data(), 2, n, f) != n) return false;
            for (size_t i = 0; i < n; i++) { uint32_t u = ((uint32_t)h[i]) << 16; float v; memcpy(&v, &u, 4); out[i] = v; }
        } else { printf("[vis] FATAL: %s 类型 %d 不支持\n", name.c_str(), t.type); return false; }
        return true;
    }
};

// ============================================================================
// 2. 卡上 int8 权重 (双芯按输出行分裂) + gemm 封装
// ============================================================================
struct VW {
    std::vector<signed char> qhost;    // 仅调试 (VIS_CPUGEMM): 分块打包后的 int8 [c][n][k]
    int N = 0, K = 0, CH = 0, nc = 0;  // [N,K] 行优先, 沿 K 按 CH 分块
    std::vector<int> cs, koff;         // 每块长度 / 每块在 K 上的起点
    std::vector<float> rs;             // 每块每输出行 scale: rs[c*N+n]
    std::vector<size_t> coff[2];       // 每芯每块在权重缓冲内的 int8 偏移
    int ns = 2, dev[2] = {0, 1};
    void* d[2] = {nullptr, nullptr};       // ★ 每芯一份【完整】权重 (M 行分裂 ⇒ 每芯要全 N 列)
    void* rsdev[2] = {nullptr, nullptr};   // ★ 折回列 scale 向量 (设备侧, [c][n] 布局, nc*N floats)
};static int g_vcpugemm = 0;
static int g_vonly = 27;    // 调试: 只跑前 N 层   // ★ 调试开关 (VIS_CPUGEMM=1): 用主机 double 参考替代卡上 GEMM, 仅离线核对用; 默认 0
// ★ 激活量化修正 (K 分块 + 每块各自行归一): VIS_CH = 大 gemm 的块大小 (默认 96);
//   VIS_CHA = 注意力小 K (<=128) 的块大小 (默认 24)。<=0 表示不分块 (退回旧行为)
static int g_vch = 0, g_vcha = 0;
// ★ VIS_FOLD=1 (默认): 逐块行归一 + 每块每行 scale + 主机折回 (精度最好, 但 D2H 是 nc 倍)
//   VIS_FOLD=0: 权重每块【单标量】scale (进 max_b) + 卡上 beta=1 累加 + 只 D2H 一次 M*N
//   VIS_GROW=1: VIS_FOLD=0 时再把激活按【全 K】逐行归一到 M0, 输出后每行乘回 (一次遍历, 不占 PCIe)
static int g_vfold = 1;   // ★ 本轮: 折回在卡上 (唯一路径)
// ★ 计时/计数剖析 (VIS_PROF=1): 分块开销到底花在哪
static int g_vprof = 0;
static double g_p_norm = 0, g_p_h2d = 0, g_p_gemm = 0, g_p_wait = 0, g_p_d2h = 0, g_p_fold = 0, g_p_pack = 0;
static double g_p_attn = 0, g_p_softmax = 0, g_p_rope = 0, g_p_bias = 0, g_p_gelu = 0, g_p_post = 0;
static long g_p_n_h2d = 0, g_p_n_gemm = 0, g_p_n_wait = 0, g_p_n_d2h = 0, g_p_n_chunk = 0, g_p_n_fold = 0;
static void vprof_report() {
    if (!g_vprof) return;
    printf("[vis] ★ PROF: 块迭代=%ld | H2D %.3fs/%ld次 | gemm %.3fs/%ld次 | wait %.3fs/%ld次 | D2H %.3fs/%ld次 "
           "| 归一(主机) %.3fs | 折回(卡上, %ld次算子) %.3fs | 打包 %.3fs\n",
           g_p_n_chunk, g_p_h2d, g_p_n_h2d, g_p_gemm, g_p_n_gemm, g_p_wait, g_p_n_wait,
           g_p_d2h, g_p_n_d2h, g_p_norm, g_p_n_fold, g_p_fold, g_p_pack);
    printf("[vis] ★ PROF2: 主机其它 | attn %.3fs | softmax %.3fs | rope %.3fs | bias/残差 %.3fs | gelu %.3fs | post/merger %.3fs\n",
           g_p_attn, g_p_softmax, g_p_rope, g_p_bias, g_p_gelu, g_p_post);
}static uint64_t g_vhbm[2] = {0, 0};
static api::Context* g_vctx[2] = {nullptr, nullptr};
static void* g_vxA[2] = {nullptr, nullptr};    // A f32 (M*K) — 只放本分区行
static void* g_vBd[2] = {nullptr, nullptr};    // 动态 int8 B (N*K)
static void* g_vCs[2] = {nullptr, nullptr};    // ★ 卡上折回: 分块输出堆叠 [G][Mh][N]
static void* g_vTc[2] = {nullptr, nullptr};    // ★ 卡上折回: 同容量 scratch
static void* g_vCa[2] = {nullptr, nullptr};    // ★ 卡上折回: 累加器 [Mh][N]
static void* g_vCr[2] = {nullptr, nullptr};    // ★ 卡上折回: 次 reduce 缓冲 [Mh][N]
static void* g_vFd[2] = {nullptr, nullptr};    // ★ 卡上折回: 行因子 [nc][Mh]
static void* g_vRd[2] = {nullptr, nullptr};    // ★ 动态 B 的折回列 scale [nc*N]
static size_t g_vAsz = 0, g_vBsz = 0, g_vCssz = 0, g_vCaszz = 0, g_vFdsz = 0, g_vRdsz = 0;
// ★ 第 19 轮: A 常驻卡上的分块 gemm —— 卡上"逐块逐行归一"所需的额外缓冲
static void* g_vAn[2] = {nullptr, nullptr};   // 归一后的 A ([Mh][K], 主机不再上传/下载 A)
static void* g_vFx[2] = {nullptr, nullptr};   // 逐块行 max, t-major [m][nc]  (reduce 输出)
static void* g_vFa[2] = {nullptr, nullptr};   // 归一因子 f = Mc/mx, [m][nc]
static void* g_vFb[2] = {nullptr, nullptr};   // 临时 (mx*invMc), [m][nc]
static void* g_vFe[2] = {nullptr, nullptr};   // 折回行因子, c-major [nc][m] (转置后)
static void* g_vFi[2] = {nullptr, nullptr};   // 小: [1][nc] Mc 与 1/Mc  (2*nc 个 float)
static void* g_vOn[2] = {nullptr, nullptr};   // 小: [1][nc] 全 1 (算 1/Mc 用)
static void* g_vSe[2] = {nullptr, nullptr};   // 跨组累加的次 reduce 缓冲 [Mh][N]
static size_t g_vAnsz = 0, g_vFxsz = 0, g_vSesz = 0, g_vFisz = 0;
static int g_vcarda = 1;    // VIS_CARDA=0 退回老路径(主机归一 + A 上下卡), 仅作 A/B 对照
// ★★ 第 20 轮 (全部为【实验开关】, 默认 0) ★★
//   VIS_CARDFFN: 0=关(默认)  1=(A)全卡: ffn_up 留卡 + 卡上 bias/gelu + 卡上 max|A| 归一 + 分块 gemm
//                           2=(A)半卡: 主机 bias/gelu(与老路径逐位相同) + 卡上 max|A| 归一 + 分块 gemm
//   VIS_HEAP   : 1=(B) 16 头 O gemm 输出堆叠在卡上, 每层每芯只做一次 D2H (32→2 次/层)
static int g_vcardffn = 0;
static int g_vheap = 0;
static int g_vkpad = 0;      // FFN padding 布局 (Kp = 45*96 = 4320): 与老的 96+80 非均匀分块逐位等价
static void* g_vUpP[2] = {nullptr, nullptr};   // (A) 卡上 FFN 激活 (padding 布局 [m][Kp])
static void* g_vBup[2] = {nullptr, nullptr};   // (A) ffn_up.bias 的 padding 副本 [VNL][Kp] (尾部 0)
static void* g_vSlO[2] = {nullptr, nullptr};   // (B) 16 头 O gemm 输出堆叠 [16][Mh][VDH]
static size_t g_vUpPsz = 0, g_vSlOsz = 0, g_vBupsz = 0;
static void* g_vMg[2] = {nullptr, nullptr};    // ★ 全 -1 的 [1][nc] 向量 (算 -minA 用; api::neg 不在 api.h 里)
// ★★ 第 21 轮 ★★
//   g_vEp      : 全 1e-30 的 [m][nc] 常量张量 —— 卡上【所有除法】的分母夹逼 (官方 elementwise_max_2d)
//                ★ 通用教训: 主机除零只给 inf/nan 不报错, 卡上直接 FP_DIV0 打死 session ⇒ 任何上卡的除法先夹分母
//   VIS_HEAPD2H: 1(默认)=每芯先 device 同步(xpu_wait)再做那一次大 D2H; 2=逐头小块 D2H (诊断/兜底)
//                3=两芯都先同步, 再各做一次大 D2H
//   VIS_HEAPDBG: 1=层0 头循环前用哨兵填 g_vSlO 并落盘 + 头循环后把原始堆叠缓冲整块落盘
static void* g_vEp[2] = {nullptr, nullptr};
static int g_vheapd2h = 1;
static int g_vheapdbg = 0;
static int g_vkpadn = VFFN;                    // padding 后的 FFN 维度 (4320)
static double g_t_gemm = 0, g_t_host = 0;// K 分块表: cs = 每块长度, koff = 每块起点 (<=0 或 >K 表示不分块)
static void vchunks(int K, int CH, std::vector<int>& cs, std::vector<int>& koff) {
    cs.clear(); koff.clear();
    if (CH <= 0 || CH > K) CH = K;
    int o = 0;
    while (o < K) { int n = K - o; if (n > CH) n = CH; cs.push_back(n); koff.push_back(o); o += n; }
}

// 把 W 的第 n0..n0+nrows 行按 [c][n][k] 打包量化到 out (长度 nrows*K)
//   rs[c*N+n0+row] = 该块该行 scale;  coff[c] = 该块在 out 内的偏移
static void vpack_rows(const float* W, int N, int K, int n0, int nrows,
                       const std::vector<int>& cs, const std::vector<int>& koff,
                       std::vector<float>& rs, std::vector<size_t>& coff, signed char* out,
                       const std::vector<float>* csc = NULL) {
    int nc = (int)cs.size();
    coff.resize(nc);
    size_t o = 0;
    for (int c = 0; c < nc; c++) {
        int cs_ = cs[c], k0 = koff[c];
        coff[c] = o;
        float sg = csc ? (*csc)[c] : 0.f;
        #pragma omp parallel for schedule(static)
        for (int nn = 0; nn < nrows; nn++) {
            const float* r = W + (size_t)(n0 + nn) * K + k0;
            float s;
            if (csc) s = sg;
            else {
                float am = 0;
                for (int j = 0; j < cs_; j++) { float a = fabsf(r[j]); if (a > am) am = a; }
                s = am / 127.f; if (s <= 1e-30f) s = 1e-8f;
                rs[(size_t)c * N + n0 + nn] = s;
            }
            if (s <= 1e-30f) s = 1e-8f;
            signed char* dq = out + o + (size_t)nn * cs_;
            for (int j = 0; j < cs_; j++) { int v = (int)lrintf(r[j] / s); if (v > 127) v = 127; else if (v < -127) v = -127; dq[j] = (signed char)v; }
        }
        o += (size_t)nrows * cs_;
    }
}

// 每块一个单标量 scale: csc[c] = max|W[:, 块 c]| / 127  (全局 N 行, 两分区共用)
static void vblocksc(const float* W, int N, int K, const std::vector<int>& cs,
                     const std::vector<int>& koff, std::vector<float>& csc) {
    int nc = (int)cs.size();
    csc.assign(nc, 1e-8f);
    for (int c = 0; c < nc; c++) {
        int cs_ = cs[c], k0 = koff[c];
        double am = 0;
        #pragma omp parallel for schedule(static) reduction(max:am)
        for (int n = 0; n < N; n++) {
            const float* r = W + (size_t)n * K + k0;
            for (int j = 0; j < cs_; j++) { double a = fabs((double)r[j]); if (a > am) am = a; }
        }
        float s = (float)(am / 127.0);
        csc[c] = (s <= 1e-30f) ? 1e-8f : s;
    }
}

// ★ 权重沿 K 分块、每块各自行归一 (每块每行一个 scale); 激活在同一次调用里做同样的事
// ★ 权重装载 (M 行分裂): 每芯放一份完整权重 + 折回列 scale 向量 rsdev
static void vw_from_f32(const std::vector<float>& W, int N, int K, VW& w) {
    w.N = N; w.K = K;
    w.CH = (g_vch > 0) ? g_vch : K;
    vchunks(K, w.CH, w.cs, w.koff);
    w.nc = (int)w.cs.size();
    w.rs.assign((size_t)w.nc * N, 0.f);
    if (g_vcpugemm) {
        w.ns = 1; w.dev[0] = 0; w.dev[1] = -1;
        w.qhost.resize((size_t)N * K);
        vpack_rows(W.data(), N, K, 0, N, w.cs, w.koff, w.rs, w.coff[0], w.qhost.data(), NULL);
        return;
    }
    w.ns = 2; w.dev[0] = 0; w.dev[1] = 1;
    size_t by = (size_t)N * K;
    std::vector<signed char> buf(by);
    vpack_rows(W.data(), N, K, 0, N, w.cs, w.koff, w.rs, w.coff[0], buf.data(), NULL);
    for (int p = 0; p < w.ns; p++) {
        int dv = w.dev[p];
        xpu_set_device(dv);
        if (xpu_malloc(&w.d[p], by)) { printf("[vis] FATAL: HBM 分配 %.2f MB 失败 chip%d\n", by / 1048576.0, dv); exit(1); }
        if (xpu_memcpy(w.d[p], buf.data(), by, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: H2D int8\n"); exit(1); }
        w.coff[p] = w.coff[0];
        xpu_wait();
        g_vhbm[dv] += by;
        // ★ 折回列 scale 向量上卡 ([c][n] 布局, 与 rs 主机布局一致)
        {
            size_t rn = (size_t)w.nc * N;
            if (xpu_malloc(&w.rsdev[p], rn * 4)) { printf("[vis] FATAL: HBM 分配 rs %.0f KB 失败 chip%d\n", rn * 4 / 1024.0, dv); exit(1); }
            if (xpu_memcpy(w.rsdev[p], w.rs.data(), rn * 4, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: H2D rs\n"); exit(1); }
            xpu_wait();
            g_vhbm[dv] += rn * 4;
        }
    }
}

// ★ 激活量化修正核心:// ★ 本轮: 折回搬到卡上 —— "激活不出卡" 第一步 (瓶颈 ① 的 15.5s 硬下限被打掉)
//   数学: C[t,n] = Σ_c (mx_c[t]/Mc_c) · rs[c*N+n] · D_c[t,n]
//           D_c = gemm_int8(A 已按块行归一, B 块 c)   mx_c = 块 c 的逐行 |A| 最大值
//   卡上只用官方算子: elementwise_mul_2d ×2 (列 scale / 行 scale) + reduce(SUM) (+ elementwise_add)
//   分区 = 按【行 M】分裂 (每芯拿全 N 列) ⇒ D2H 可直接落进主机 C 的目标行, 不需要任何 scatter
static void vgemm_chunked(int N, int K, int nc, const std::vector<int>& cs, const std::vector<int>& koff,
                          const signed char* const* bdev, const void* const* rsdev,
                          const float* A, int M, float* C, int leave_on_card = 0,
                          float* const* accdst = nullptr) {
    static std::vector<float> mxc, An, f2;
    double t0 = vnow();
    if (N <= 0 || K <= 0 || M <= 0 || nc <= 0 || (int)cs.size() != nc) { printf("[vis] FATAL: gemm 形状非法 N=%d K=%d M=%d nc=%d\n", N, K, M, nc); exit(1); }
    size_t MK = (size_t)M * K;
    int Mh = (M + 1) / 2;
    // ---- 边界检查 (可读 assert) ----
    if (MK * 4 > g_vAsz) { printf("[vis] FATAL: A 越界 M=%d K=%d (缓冲 %zu B 需 %zu)\\n", M, K, g_vAsz, MK * 4); exit(1); }
    if ((size_t)Mh * N * 4 > g_vCaszz) { printf("[vis] FATAL: 折回累加越界 Mh=%d N=%d (缓冲 %zu B 需 %zu)\\n", Mh, N, g_vCaszz, (size_t)Mh * N * 4); exit(1); }
    if ((size_t)nc * Mh * 4 > g_vFdsz) { printf("[vis] FATAL: 行因子越界 nc=%d Mh=%d (缓冲 %zu B 需 %zu)\\n", nc, Mh, g_vFdsz, (size_t)nc * Mh * 4); exit(1); }
    if (g_vCssz == 0) { printf("[vis] FATAL: 折回堆叠缓冲未分配\n"); exit(1); }
    // ---- 1) 主机: 逐块各自行归一 (与旧实现逐元素同式同序) + 折回行因子 mx/Mc ----
    double th = vnow();
    An.resize(MK);
    mxc.resize((size_t)nc * M);
    for (int c = 0; c < nc; c++) {
        int cs_ = cs[c], k0 = koff[c];
        if (cs_ <= 0 || k0 < 0 || k0 + cs_ > K) { printf("[vis] FATAL: 块 %d 越界 cs=%d koff=%d K=%d\n", c, cs_, k0, K); exit(1); }
        float* mx = &mxc[(size_t)c * M];
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < M; t++) {
            const float* r = A + (size_t)t * K + k0; float am = 0;
            for (int j = 0; j < cs_; j++) { float a = fabsf(r[j]); if (a > am) am = a; }
            mx[t] = am;
        }
        float Mc = 0;
        for (int t = 0; t < M; t++) if (mx[t] > Mc) Mc = mx[t];
        if (Mc <= 1e-30f) Mc = 1e-30f;
        const float invMc = 1.0f / Mc;
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < M; t++) {
            float m = mx[t] > 1e-30f ? mx[t] : 1e-30f;
            mx[t] = m * invMc;                 // ★ 折回行因子 (设备侧乘回)
            float f = Mc / m;                  // 上卡前的归一因子
            const float* r = A + (size_t)t * K + k0;
            float* d = An.data() + (size_t)t * K + k0;
            for (int j = 0; j < cs_; j++) d[j] = r[j] * f;
        }
    }
    g_t_host += vnow() - th; g_p_norm += vnow() - th;
    // ---- 2) 每芯: H2D → 逐组 (gemm + 列 scale + 行 scale + reduce SUM) → D2H ----
    int G = (int)(g_vCssz / ((size_t)Mh * N * 4));
    if (G < 1) G = 1; if (G > nc) G = nc;
    for (int p = 0; p < 2; p++) {
        int dv = p;
        int r0 = p * Mh, m = Mh; if (r0 + m > M) m = M - r0;
        if (m <= 0) continue;
        xpu_set_device(dv);
        float* accp = (accdst && accdst[dv]) ? accdst[dv] : (float*)g_vCa[dv];   // ★ 第20轮落点 (第21轮: 判元素非空)
        {   // H2D 1) An 本分区行
            double t = vnow();
            if (xpu_memcpy(g_vxA[dv], An.data() + (size_t)r0 * K, (size_t)m * K * 4, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: H2D A 分区\n"); exit(1); }
            g_p_h2d += vnow() - t; g_p_n_h2d++;
        }
        {   // H2D 2) 折回行因子 (整份 nc*m, 布局 [c][t])
            f2.resize((size_t)nc * m);
            for (int c = 0; c < nc; c++) memcpy(&f2[(size_t)c * m], &mxc[(size_t)c * M + r0], (size_t)m * 4);
            double t = vnow();
            if (xpu_memcpy(g_vFd[dv], f2.data(), (size_t)nc * m * 4, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: H2D 行因子\n"); exit(1); }
            g_p_h2d += vnow() - t; g_p_n_h2d++;
        }
        float* stk = (float*)g_vCs[dv];
        float* sct = (float*)g_vTc[dv];
        for (int c0 = 0; c0 < nc; c0 += G) {
            int c1 = c0 + G; if (c1 > nc) c1 = nc; int gg = c1 - c0;
            for (int c = c0; c < c1; c++) {
                int cs_ = cs[c], k0 = koff[c];
                double t = vnow();
                int r = api::gemm_int8(g_vctx[dv], false, true, m, N, cs_, 1.f,
                                       (const float*)((const char*)g_vxA[dv] + (size_t)k0 * 4), K,
                                       (const int8_t*)bdev[(size_t)p * nc + c], 127.f, cs_,
                                       0.f, stk + (size_t)(c - c0) * (size_t)m * N, N);
                g_p_gemm += vnow() - t; g_p_n_gemm++;
                if (r) { printf("[vis] FATAL: gemm_int8 r=%d (chip%d m=%d n=%d k=%d 块%d/%d)\n", r, dv, m, N, cs_, c, nc); exit(1); }
            }
            { double t = vnow(); if (xpu_wait()) { printf("[vis] FATAL: gemm wait\n"); exit(1); } g_p_wait += vnow() - t; g_p_n_wait++; }
            // 行 scale: stack 视作 [gg*m, N] × f[gg*m, 1] → scratch  (块 c 起点在 stk + (c-c0)*m*N, ldc=N)
            { double t = vnow();
              int r = api::elementwise_mul_2d(g_vctx[dv], stk, (const float*)g_vFd[dv] + (size_t)c0 * m, sct, gg * m, N, gg * m, 1);
              if (r) { printf("[vis] FATAL: mul_2d 行 r=%d (m=%d n=%d)\n", r, gg * m, N); exit(1); }
              g_p_fold += vnow() - t; g_p_n_fold++; }
            // 列 scale: 逐块 (scratch 块 c) × vcat[1,N] → 写回 stack 同块 (x≠z, 无原地依赖)
            for (int c = c0; c < c1; c++) {
              double t = vnow();
              int r = api::elementwise_mul_2d(g_vctx[dv], sct + (size_t)(c - c0) * (size_t)m * N,
                                              (const float*)rsdev[p] + (size_t)c * N,
                                              stk + (size_t)(c - c0) * (size_t)m * N, m, N, 1, N);
              if (r) { printf("[vis] FATAL: mul_2d 列 r=%d (块 %d, m=%d n=%d)\n", r, c, m, N); exit(1); }
              g_p_fold += vnow() - t; g_p_n_fold++;
            }
            // reduce SUM dim0: [gg,m,N] -> [m,N]
            { double t = vnow();
              int xd[3]; xd[0] = gg; xd[1] = m; xd[2] = N; int rd[1]; rd[0] = 0;
              if (c0 == 0) {
                  int r = api::reduce(g_vctx[dv], stk, accp, xd, 3, rd, 1, api::REDUCE_SUM);
                  if (r) { printf("[vis] FATAL: reduce SUM r=%d\n", r); exit(1); }
              } else {
                  int r = api::reduce(g_vctx[dv], stk, (float*)g_vCr[dv], xd, 3, rd, 1, api::REDUCE_SUM);
                  if (r) { printf("[vis] FATAL: reduce SUM(2) r=%d\n", r); exit(1); }
                  r = api::elementwise_add(g_vctx[dv], (const float*)accp, (const float*)g_vCr[dv], (float*)accp, m * N);
                  if (r) { printf("[vis] FATAL: 折回累加 r=%d\n", r); exit(1); }
              }
              g_p_fold += vnow() - t; g_p_n_fold++; }
            g_p_n_chunk += gg;
        }
        if (!leave_on_card) {   // D2H: 直接落进主机 C 的本分区行
            double t = vnow();
            if (xpu_memcpy(C + (size_t)r0 * N, accp, (size_t)m * N * 4, XPU_DEVICE_TO_HOST)) { printf("[vis] FATAL: D2H C 分区\n"); exit(1); }
            xpu_wait();
            g_p_d2h += vnow() - t; g_p_n_d2h++;
        }
    }
    g_t_gemm += vnow() - t0;
}

// ★ VIS_FOLD=0 (权重每块单标量 scale + 卡上 beta=1 累加) 精度不够 (34%), 已否决:
//   见 PROGRESS.md 第 17 章 17.6 —— 本轮把它整体删除, 只保留卡上折回这一条正确路径。
static void vgemm(const VW& w, const float* A, int M, float* C, int leave_on_card = 0,
                  float* const* accdst = nullptr) {
    int K = w.K, N = w.N;
    if (w.N <= 0 || w.K <= 0 || M <= 0) { printf("[vis] FATAL: 形状非法\n"); exit(1); }
    if (g_vcpugemm) {   // 主机 double 参考: 精确激活 × 与卡上同一份分块量化权重
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < M; t++) { float* c = C + (size_t)t * N; for (int n = 0; n < N; n++) c[n] = 0.f; }
        for (int c = 0; c < w.nc; c++) {
            int cs_ = w.cs[c], k0 = w.koff[c];
            const signed char* qb = w.qhost.data() + w.coff[0][c];
            const float* rsc = &w.rs[(size_t)c * N];
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < M; t++) {
                const float* a = A + (size_t)t * K + k0; float* d = C + (size_t)t * N;
                for (int n = 0; n < N; n++) {
                    const signed char* qr = qb + (size_t)n * cs_;
                    double acc = 0;
                    for (int j = 0; j < cs_; j++) acc += (double)a[j] * (double)qr[j];
                    d[n] += (float)(acc * (double)rsc[n]);
                }
            }
        }
        return;
    }
    if (w.rsdev[0] == nullptr || w.rsdev[1] == nullptr) { printf("[vis] FATAL: 权重 rsdev 未上卡\n"); exit(1); }
    std::vector<const int8_t*> bd((size_t)2 * w.nc, nullptr);
    for (int p = 0; p < 2; p++)
        for (int c = 0; c < w.nc; c++)
            bd[(size_t)p * w.nc + c] = (const int8_t*)w.d[p] + w.coff[p][c];
    const void* rd[2] = {w.rsdev[0], w.rsdev[1]};
    vgemm_chunked(N, K, w.nc, w.cs, w.koff, bd.data(), rd, A, M, C, leave_on_card, accdst);
}
// 运行时矩阵当 int8 B (分块打包, 每块每行 scale): C[M,N] = A[M,K] · B[N,K]^T
//   qpack 布局 = [c][n][k] (由 vquant_chunked / vpack_T 产生); 折回同样在卡上
static void vgemm_dynB(const std::vector<signed char>& qpack, const std::vector<float>& rs,
                       int N, int K, const std::vector<int>& cs, const std::vector<int>& koff,
                       const float* A, int M, float* C, int leave_on_card = 0) {
    int nc = (int)cs.size();
    if ((int)qpack.size() != N * K) { printf("[vis] FATAL: dynB 打包尺寸 %zu != %d\n", qpack.size(), N * K); exit(1); }
    if ((int)rs.size() != nc * N) { printf("[vis] FATAL: dynB rs 尺寸 %zu != %d\n", rs.size(), nc * N); exit(1); }
    if (g_vcpugemm) {
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < M; t++) { float* c = C + (size_t)t * N; for (int n = 0; n < N; n++) c[n] = 0.f; }
        size_t base = 0;
        for (int c = 0; c < nc; c++) {
            int cs_ = cs[c], k0 = koff[c];
            const signed char* qb = qpack.data() + base;
            const float* rsc = &rs[(size_t)c * N];
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < M; t++) {
                const float* a = A + (size_t)t * K + k0; float* d = C + (size_t)t * N;
                for (int n = 0; n < N; n++) {
                    const signed char* qr = qb + (size_t)n * cs_;
                    double acc = 0;
                    for (int j = 0; j < cs_; j++) acc += (double)a[j] * (double)qr[j];
                    d[n] += (float)(acc * (double)rsc[n]);
                }
            }
            base += (size_t)N * cs_;
        }
        return;
    }
    size_t NB = (size_t)N * K;
    if (NB > g_vBsz) { printf("[vis] FATAL: Bd 越界 n=%d K=%d (%zu > %zu)\n", N, K, NB, g_vBsz); exit(1); }
    if ((size_t)nc * N * 4 > g_vRdsz) { printf("[vis] FATAL: Rd 越界 nc=%d N=%d (%zu > %zu)\n", nc, N, (size_t)nc * N * 4, g_vRdsz); exit(1); }
    std::vector<size_t> cbase(nc, 0);
    { size_t o = 0; for (int c = 0; c < nc; c++) { cbase[c] = o; o += (size_t)N * cs[c]; } }
    for (int p = 0; p < 2; p++) {
        int dv = p;
        xpu_set_device(dv);
        double t = vnow();
        if (xpu_memcpy(g_vBd[dv], qpack.data(), NB, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: H2D dynB 整块\n"); exit(1); }
        g_p_h2d += vnow() - t; g_p_n_h2d++;
        t = vnow();
        if (xpu_memcpy(g_vRd[dv], rs.data(), (size_t)nc * N * 4, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: H2D dynB rs\n"); exit(1); }
        g_p_h2d += vnow() - t; g_p_n_h2d++;
    }
    std::vector<const int8_t*> bd((size_t)2 * nc, nullptr);
    for (int p = 0; p < 2; p++)
        for (int c = 0; c < nc; c++)
            bd[(size_t)p * nc + c] = (const int8_t*)g_vBd[p] + cbase[c];
    const void* rd[2] = {g_vRd[0], g_vRd[1]};
    vgemm_chunked(N, K, nc, cs, koff, bd.data(), rd, A, M, C, leave_on_card);
}

// ============================================================================
// ★ 第 19 轮: A 常驻卡上的分块 gemm —— 卡上完成"逐块逐行归一"
//   动机(定量, 见 PROGRESS 18.4): 老路径把 A 从卡上 D2H → 主机逐块逐行归一 → 再 H2D 回卡。
//     注意力第二个 gemm 的 A = softmax(S) 恰好是 T×T: 768² 下每头 21MB,
//     16 头 × 27 层 = 340MB/层 的纯往返 (D2H 5.2s + H2D 3.3s + 主机归一 3.7s 的主因)。
//   做法(全部官方算子; 分块必须整除 K 才能用 [m*nc][cs] 连续视图):
//     1) reduce(MAX, xdims={m,nc,cs}, rdims={2})      → mxA [m][nc]  逐块逐行 max
//     2) transpose({m,nc}, permute={1,0})             → mFe [nc][m]  折回用(块主序)
//     3) elementwise_div_2d(A 视作[m*nc,cs] / mxA 视作[m*nc,1])      → An
//        ⇒ 每块每行 max 恰为 1.0 ⇒ 算子内部 max_a ≡ 1.0 (满量程、不截断)
//     4) 逐块 gemm_int8(a = An + c*cs, lda = K, k = cs) → 堆叠 [nc][m][N]
//     5) 折回: mul_2d(行因子 mFe[gg*m,1]) + mul_2d(列因子 rs[c][N]) + reduce(SUM, dim0)
//   ★ 与老路径【数学同式】, 只差归一/求和的结合律; 离线对拍(真 softmax/V 数据)证明
//     精度与老路径相同(768² 逐位同档)或更好(512² 分块更细)。
// ============================================================================
static int vunif_div(int T, int want) {   // 返回整除 T 且 <= want 的最大 cs; 0 = 找不到
    if (want > T) want = T;
    for (int d = want; d >= 8; d--) if (T % d == 0) return d;
    return 0;
}

// ★ 第 20 轮: 核心多了 3 个参数 (全部默认 = 第 19 轮行为)
//   bdev/rsdev : B 与折回列 scale 的【设备指针表】(索引 p*nc+c) —— 便于直接用常驻卡上的权重
//   accdst     : 结果落点 (默认 g_vCa); skip_d2h: 只算不下载 (供 (B) 堆叠后一次性 D2H)
//   absmax_mode: 1 = A 含负值, 用 max(reduce MAX, -reduce MIN) 求 max|A| (与主机 fabsf 逐位相同)
static void vdump(const char* dir, const char* name, const float* p, size_t n);
static void vgemm_cardA_core(int N, int K, int cs, const signed char* const* bdev,
                             const void* const* rsdev, const void* const* A_dev, int M, float* C,
                             int leave_on_card, float* const* accdst, int skip_d2h, int absmax_mode) {
    if (cs <= 0 || K % cs) { printf("[vis] FATAL: cardA 分块不整除 K=%d cs=%d\n", K, cs); exit(1); }
    int nc = K / cs;
    if ((size_t)nc * N * 4 > g_vRdsz) { printf("[vis] FATAL: cardA Rd 越界\n"); exit(1); }
    if (A_dev[0] == nullptr || A_dev[1] == nullptr) { printf("[vis] FATAL: cardA A_dev 为空\n"); exit(1); }
    int Mh = (M + 1) / 2;
    for (int p = 0; p < 2; p++) {
        int dv = p, r0 = p * Mh, m = Mh; if (r0 + m > M) m = M - r0;
        if (m <= 0) continue;
        xpu_set_device(dv);
        if ((size_t)m * K * 4 > g_vAnsz) { printf("[vis] FATAL: cardA An 越界 m=%d K=%d (%zu>%zu)\n", m, K, (size_t)m * K * 4, g_vAnsz); exit(1); }
        if ((size_t)m * nc * 4 > g_vFxsz) { printf("[vis] FATAL: cardA 行因子越界 m=%d nc=%d\n", m, nc); exit(1); }
        if ((size_t)m * N * 4 > g_vSesz) { printf("[vis] FATAL: cardA 次 reduce 越界 m=%d N=%d\n", m, N); exit(1); }
        const float* A = (const float*)A_dev[p];        // 本分区的 m 行已就位
        float* mxA = (float*)g_vFx[dv];     // [m][nc] 逐块逐行 max
        float* fAr = (float*)g_vFa[dv];     // [m][nc] 归一因子 f = Mc/mx   (与主机同式)
        float* tmp = (float*)g_vFb[dv];     // [m][nc] 临时 (mx*invMc)
        float* mFe = (float*)g_vFe[dv];     // [nc][m] 折回行因子 (转置后)
        float* mBc = (float*)g_vFi[dv];     // [1][nc] 每块全局 max Mc
        float* invM = mBc + nc;             // [1][nc] 1/Mc
        float* An  = (float*)g_vAn[dv];
        // ★★ 第 21 轮 BUGFIX: 第 20 轮把"结果落点"做成 accdst 数组后, 非堆叠路径传进来的
        //   是 `float* accd[2] = {nullptr,nullptr}` —— 数组本身非空但元素是 null, 于是
        //   acc = nullptr, 后面的 reduce 直接返回 r=1 ⇒【HEAP=0 (生产路径) 全废】。
        //   判据必须是"元素是否有效", 不是"数组是否为空"。
        float* acc = (accdst && accdst[dv]) ? accdst[dv] : (float*)g_vCa[dv];
        float* sec = (float*)g_vSe[dv];
        {   // 1) 逐块逐行 MAX: [m, nc, cs] --dim2--> [m, nc]
            int xd[3]; xd[0] = m; xd[1] = nc; xd[2] = cs;
            int rd[1]; rd[0] = 2;
            int r = api::reduce(g_vctx[dv], A, mxA, xd, 3, rd, 1, api::REDUCE_MAX);
            if (r) { printf("[vis] FATAL: cardA reduce MAX r=%d (m=%d nc=%d cs=%d)\n", r, m, nc, cs); exit(1); }
        }
        if (absmax_mode) {   // ★ 1b~1d) A 含负值: max|A| = max( maxA, -minA )  (与主机 fabsf 逐位相同)
            float* mnA = tmp;
            int xd[3]; xd[0] = m; xd[1] = nc; xd[2] = cs;
            int rd[1]; rd[0] = 2;
            int r = api::reduce(g_vctx[dv], A, mnA, xd, 3, rd, 1, api::REDUCE_MIN);
            if (r) { printf("[vis] FATAL: cardA reduce MIN r=%d\n", r); exit(1); }
            r = api::elementwise_mul_2d(g_vctx[dv], (const float*)mnA, (const float*)g_vMg[dv], fAr, m, nc, 1, nc);
            if (r) { printf("[vis] FATAL: cardA neg(-1 乘) r=%d\n", r); exit(1); }
            r = api::elementwise_max_2d(g_vctx[dv], (const float*)mxA, (const float*)fAr, mxA, m, nc, m, nc);
            if (r) { printf("[vis] FATAL: cardA elementwise_max r=%d\n", r); exit(1); }
        }
        {   // ★★ 第 21 轮: 分母夹逼 (卡上除法前必须做) —— 主机 `float m = mx[t] > 1e-30f ? mx[t] : 1e-30f;`
            //   与 `if (Mc <= 1e-30f) Mc = 1e-30f;` 的卡上等价式。**先夹 mx 再取 Mc** ⇒
            //   Mc = max_t max(mx_t,1e-30) = max(max_t mx_t, 1e-30), 与主机"先算 Mc 再夹"逐位相同;
            //   f = Mc/mxc、行因子 = mxc*(1/Mc) 也与主机夹逼后的算式逐位相同。
            //   正常数据 (所有 mx > 1e-30) 下 elementwise_max 只是"选择" ⇒ 【位级不变】(probe 有专门对照)。
            //   动机: FFN 的 A=gelu 在 float32 下 tanhf 饱和 ⇒ 某个 96 宽块整体精确 0 ⇒ mx=0 ⇒
            //   f = Mc/0 与 1/Mc 在卡上触发 FP_DIV0 打死 session (第 20 轮真实事故 17→18)。
            int r = api::elementwise_max_2d(g_vctx[dv], (const float*)mxA, (const float*)g_vEp[dv], mxA, m, nc, m, nc);
            if (r) { printf("[vis] FATAL: cardA 分母夹逼 r=%d (m=%d nc=%d)\n", r, m, nc); exit(1); }
        }
        {   // 2) 每块全局 MAX: [m,nc] --dim0--> [1,nc]
            int xd[2]; xd[0] = m; xd[1] = nc;
            int rd[1]; rd[0] = 0;
            int r = api::reduce(g_vctx[dv], mxA, mBc, xd, 2, rd, 1, api::REDUCE_MAX);
            if (r) { printf("[vis] FATAL: cardA reduce Mc r=%d (m=%d nc=%d)\n", r, m, nc); exit(1); }
        }
        {   // 3) f = Mc / mx  (与主机 float f = Mc / m 同式, 为的是【位级复现】主机路径)
            int r = api::elementwise_div_2d(g_vctx[dv], mBc, mxA, fAr, 1, nc, m, nc);
            if (r) { printf("[vis] FATAL: cardA div f r=%d\n", r); exit(1); }
        }
        {   // 4) invMc = 1/Mc  (与主机 float invMc = 1.0f/Mc 同式)
            int r = api::elementwise_div_2d(g_vctx[dv], (const float*)g_vOn[dv], mBc, invM, 1, nc, 1, nc);
            if (r) { printf("[vis] FATAL: cardA div invMc r=%d\n", r); exit(1); }
        }
        {   // 5) 折回行因子 = mx*invMc (与主机 mx[t] = m*invMc 同式)
            int r = api::elementwise_mul_2d(g_vctx[dv], mxA, invM, tmp, m, nc, 1, nc);
            if (r) { printf("[vis] FATAL: cardA mul 行因子 r=%d\n", r); exit(1); }
        }
        {   // 6) 转置 [m,nc] -> [nc,m]
            int sh[2]; sh[0] = m; sh[1] = nc;
            int pm[2]; pm[0] = 1; pm[1] = 0;
            int r = api::transpose(g_vctx[dv], tmp, mFe, sh, pm, 2);
            if (r) { printf("[vis] FATAL: cardA transpose r=%d (m=%d nc=%d)\n", r, m, nc); exit(1); }
        }
        {   // 7) An = A * f  (与主机 d[j] = r[j]*f 同式; A 视作[m*nc,cs], f 视作[m*nc,1])
            int r = api::elementwise_mul_2d(g_vctx[dv], A, fAr, An, m * nc, cs, m * nc, 1);
            if (r) { printf("[vis] FATAL: cardA mul An r=%d (m=%d n=%d)\n", r, m * nc, cs); exit(1); }
        }
        size_t per = (size_t)m * N;
        int G = (int)(g_vCssz / (per * 4)); if (G < 1) G = 1; if (G > nc) G = nc;
        float* stk = (float*)g_vCs[dv];
        float* sct = (float*)g_vTc[dv];
        for (int c0 = 0; c0 < nc; c0 += G) {
            int c1 = c0 + G; if (c1 > nc) c1 = nc; int gg = c1 - c0;
            for (int c = c0; c < c1; c++) {
                const int8_t* Bc = bdev[(size_t)p * nc + c];
                int r = api::gemm_int8(g_vctx[dv], false, true, m, N, cs, 1.f,
                                       An + (size_t)c * cs, K, Bc, 127.f, cs, 0.f,
                                       stk + (size_t)(c - c0) * per, N);
                g_p_gemm += 0; g_p_n_gemm++;
                if (r) { printf("[vis] FATAL: cardA gemm r=%d (m=%d n=%d k=%d 块%d/%d)\n", r, m, N, cs, c, nc); exit(1); }
            }
            if (xpu_wait()) { printf("[vis] FATAL: cardA wait\n"); exit(1); }
            {   // 行因子 (块主序 [nc][m]) —— 同一算子, 因子来自卡上 reduce
                int r = api::elementwise_mul_2d(g_vctx[dv], stk, mFe + (size_t)c0 * m, sct, gg * m, N, gg * m, 1);
                if (r) { printf("[vis] FATAL: cardA mul 行 r=%d\n", r); exit(1); }
            }
            for (int c = c0; c < c1; c++) {   // 列 scale (每块每行一个 scale)
                int r = api::elementwise_mul_2d(g_vctx[dv], sct + (size_t)(c - c0) * per,
                                                (const float*)rsdev[p] + (size_t)c * N,
                                                stk + (size_t)(c - c0) * per, m, N, 1, N);
                if (r) { printf("[vis] FATAL: cardA mul 列 r=%d\n", r); exit(1); }
            }
            {   // 跨块 reduce SUM -> acc (多组时再 elementwise_add 累加)
                int xd[3]; xd[0] = gg; xd[1] = m; xd[2] = N;
                int rd[1]; rd[0] = 0;
                int r;
                if (c0 == 0) r = api::reduce(g_vctx[dv], stk, acc, xd, 3, rd, 1, api::REDUCE_SUM);
                else {
                    r = api::reduce(g_vctx[dv], stk, sec, xd, 3, rd, 1, api::REDUCE_SUM);
                    if (!r) r = api::elementwise_add(g_vctx[dv], acc, sec, acc, m * N);
                }
                if (r) { printf("[vis] FATAL: cardA 折回 r=%d\n", r); exit(1); }
            }
        }
        if (!leave_on_card && !skip_d2h) {
            double t = vnow();
            if (xpu_memcpy(C + (size_t)r0 * N, acc, per * 4, XPU_DEVICE_TO_HOST)) { printf("[vis] FATAL: cardA D2H\n"); exit(1); }
            xpu_wait();
            g_p_d2h += vnow() - t; g_p_n_d2h++;
        }
    }
}

// 第 19 轮原接口 (注意力路径唯一入口) —— 行为一字未改; 第 20 轮只多了 heap_head
//   heap_head >= 0: (B) 16 头输出堆叠, 写第 heap_head 段, 不 D2H
static void vgemm_cardA(const std::vector<signed char>& qpack, const std::vector<float>& rs,
                        int N, int K, int cs, const void* const* A_dev, int M,
                        float* C, int leave_on_card = 0, int heap_head = -1) {
    if (cs <= 0 || K % cs) { printf("[vis] FATAL: cardA 分块不整除 K=%d cs=%d\n", K, cs); exit(1); }
    int nc = K / cs;
    if ((size_t)qpack.size() != (size_t)N * K) { printf("[vis] FATAL: cardA 打包尺寸 %zu != %d\n", qpack.size(), N * K); exit(1); }
    if ((size_t)rs.size() != (size_t)nc * N) { printf("[vis] FATAL: cardA rs 尺寸 %zu != %d\n", rs.size(), nc * N); exit(1); }
    if ((size_t)N * K > g_vBsz) { printf("[vis] FATAL: cardA Bd 越界 n=%d K=%d\n", N, K); exit(1); }
    if ((size_t)nc * N * 4 > g_vRdsz) { printf("[vis] FATAL: cardA Rd 越界\n"); exit(1); }
    if (A_dev[0] == nullptr || A_dev[1] == nullptr) { printf("[vis] FATAL: cardA A_dev 为空\n"); exit(1); }
    for (int p = 0; p < 2; p++) {      // B / rs 每芯一份 (与 vgemm_dynB 同约定)
        xpu_set_device(p);
        double t = vnow();
        if (xpu_memcpy(g_vBd[p], qpack.data(), (size_t)N * K, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: cardA H2D B\n"); exit(1); }
        if (xpu_memcpy(g_vRd[p], rs.data(), (size_t)nc * N * 4, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: cardA H2D rs\n"); exit(1); }
        g_p_h2d += vnow() - t; g_p_n_h2d += 2;
    }
    std::vector<const int8_t*> bd((size_t)2 * nc, nullptr);
    for (int p = 0; p < 2; p++)
        for (int c = 0; c < nc; c++)
            bd[(size_t)p * nc + c] = (const int8_t*)g_vBd[p] + (size_t)c * (size_t)N * cs;
    const void* rd[2] = {g_vRd[0], g_vRd[1]};
    float* accd[2] = {nullptr, nullptr};
    int skip = 0;
    if (heap_head >= 0) {
        int Mh = (M + 1) / 2;
        if (g_vSlO[0] == nullptr || g_vSlO[1] == nullptr) { printf("[vis] FATAL: heap 堆叠缓冲未分配\n"); exit(1); }
        if ((size_t)(heap_head + 1) * Mh * N * 4 > g_vSlOsz) { printf("[vis] FATAL: heap 越界\n"); exit(1); }
        for (int p = 0; p < 2; p++) accd[p] = (float*)g_vSlO[p] + (size_t)heap_head * Mh * N;
        skip = 1;
    }
    vgemm_cardA_core(N, K, cs, bd.data(), rd, A_dev, M, C, leave_on_card,
                     (heap_head >= 0) ? (float* const*)accd : (float* const*)nullptr, skip, 0);
}

// ★ (A) 第 20 轮: FFN 的 Wdn 分块 gemm —— A = gelu 输出(含负值) 常驻卡上
//   K = Kp = 45*96 = 4320 (padding 布局), 与老路径 96+80 非均匀分块【逐位等价】(见 PROGRESS 第 20 章)
static void vgemm_ffn_dn(const VW& w, const void* const* A_dev, int M, float* C) {
    int Kp = g_vkpadn;
    if (Kp % 96) { printf("[vis] FATAL: FFN Kp=%d 不是 96 的倍数\n", Kp); exit(1); }
    if (w.K != Kp || w.N != VNE) { printf("[vis] FATAL: Wdn 形状 N=%d K=%d != %d/%d\n", w.N, w.K, VNE, Kp); exit(1); }
    int cs = 96, nc = Kp / cs;
    std::vector<const int8_t*> bd((size_t)2 * nc, nullptr);
    for (int p = 0; p < 2; p++)
        for (int c = 0; c < nc; c++)
            bd[(size_t)p * nc + c] = (const int8_t*)w.d[p] + w.coff[p][c];
    const void* rd[2] = {w.rsdev[0], w.rsdev[1]};
    vgemm_cardA_core(w.N, Kp, cs, bd.data(), rd, A_dev, M, C, 0, nullptr, 0, 1);
}

// ★ (A) 模式 1: 卡上 bias(广播加) + gelu ([m][Kp]; padding 尾部恒 0 ⇒ gelu(0)=0 仍为 0)
static void vffn_card_bias_gelu(int T, int il) {
    int Kp = g_vkpadn, Mh = (T + 1) / 2;
    for (int p = 0; p < 2; p++) {
        int dv = p, r0 = p * Mh, m = Mh; if (r0 + m > T) m = T - r0;
        if (m <= 0) continue;
        xpu_set_device(dv);
        float* up = (float*)g_vUpP[dv];
        float* gl = (float*)g_vAn[dv];       // 暂存 (g_vAn 在归一阶段才用)
        const float* bias = (const float*)g_vBup[dv] + (size_t)il * Kp;
        double t = vnow();
        int r = api::elementwise_add_2d(g_vctx[dv], (const float*)up, bias, gl, m, Kp, 1, Kp);
        if (r) { printf("[vis] FATAL: FFN 卡上 bias r=%d (m=%d Kp=%d)\n", r, m, Kp); exit(1); }
        r = api::gelu(g_vctx[dv], (const float*)gl, (float*)up, m * Kp);
        if (r) { printf("[vis] FATAL: FFN 卡上 gelu r=%d (m=%d Kp=%d)\n", r, m, Kp); exit(1); }
        if (xpu_wait()) { printf("[vis] FATAL: FFN bias/gelu wait\n"); exit(1); }
        g_p_gelu += vnow() - t;
    }
}

// ★ (A) 模式 2: 主机 bias+gelu 后把 padding 布局的 Up 传到卡上
static void vffn_up_upload(int T, const float* Up) {
    int Kp = g_vkpadn, Mh = (T + 1) / 2;
    for (int p = 0; p < 2; p++) {
        int dv = p, r0 = p * Mh, m = Mh; if (r0 + m > T) m = T - r0;
        if (m <= 0) continue;
        xpu_set_device(dv);
        double t = vnow();
        if (xpu_memcpy(g_vUpP[dv], Up + (size_t)r0 * Kp, (size_t)m * Kp * 4, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: H2D FFN Up\n"); exit(1); }
        xpu_wait();
        g_p_h2d += vnow() - t; g_p_n_h2d++;
    }
}

// ★ (A) 调试: 卡上 g_vUpP 的 [T][Kp] 落盘 (与 VIS_CARDFFN=0 的 l0_up 对照)
static void vffn_dump_up(int T, const char* dumpdir, const char* fn) {
    if (!dumpdir || !dumpdir[0]) return;
    int Kp = g_vkpadn, Mh = (T + 1) / 2;
    std::vector<float> h((size_t)T * Kp);
    for (int p = 0; p < 2; p++) {
        int dv = p, r0 = p * Mh, m = Mh; if (r0 + m > T) m = T - r0;
        if (m <= 0) continue;
        xpu_set_device(dv);
        if (xpu_memcpy(h.data() + (size_t)r0 * Kp, g_vUpP[dv], (size_t)m * Kp * 4, XPU_DEVICE_TO_HOST)) { printf("[vis] FATAL: D2H FFN Up(dump)\n"); exit(1); }
        xpu_wait();
    }
    vdump(dumpdir, fn, h.data(), h.size());
}

// 行优先 W[N,K] -> 分块打包 int8 [c][n][k] + 每块每行 scale
static void vquant_chunked(const float* W, int N, int K, int CH,
                           std::vector<signed char>& qpack, std::vector<float>& rs,
                           std::vector<int>& cs, std::vector<int>& koff) {    vchunks(K, CH, cs, koff);
    int nc = (int)cs.size();
    rs.assign((size_t)nc * N, 0.f);
    qpack.resize((size_t)N * K);
    size_t o = 0;
    for (int c = 0; c < nc; c++) {
        int cs_ = cs[c], k0 = koff[c];
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            const float* r = W + (size_t)n * K + k0;
            float s;
            {
                float am = 0;
                for (int j = 0; j < cs_; j++) { float a = fabsf(r[j]); if (a > am) am = a; }
                s = am / 127.f; if (s <= 1e-30f) s = 1e-8f;
                rs[(size_t)c * N + n] = s;
            }
            if (s <= 1e-30f) s = 1e-8f;
            signed char* dq = qpack.data() + o + (size_t)n * cs_;
            for (int j = 0; j < cs_; j++) { int v = (int)lrintf(r[j] / s); if (v > 127) v = 127; else if (v < -127) v = -127; dq[j] = (signed char)v; }
        }
        o += (size_t)N * cs_;
    }
}

// 把 [T,D] 转置量化成 B = [D][T] 的分块打包 (每行 = 一个 d, 行内 = T 个 t) —— 用于 V
static void vpack_T(const float* M, int T, int D, int CH,
                    std::vector<signed char>& qpack, std::vector<float>& rs,
                    std::vector<int>& cs, std::vector<int>& koff) {
    static std::vector<float> tmp;
    tmp.resize((size_t)D * T);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++) tmp[(size_t)d * T + t] = M[(size_t)t * D + d];
    vquant_chunked(tmp.data(), D, T, CH, qpack, rs, cs, koff);
}

// 兼容旧接口: 分块量化 + 还原成 [N,K] 行优先 (只关心 q/rs 时用)
static void vquant_row(const float* W, int N, int K, std::vector<signed char>& q, std::vector<float>& rs) {
    std::vector<signed char> pack; std::vector<int> cs, koff;
    vquant_chunked(W, N, K, g_vcha, pack, rs, cs, koff);
    q.resize((size_t)N * K);
    size_t o = 0;
    for (size_t c = 0; c < cs.size(); c++) {
        int cs_ = cs[c];
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++)
            memcpy(&q[(size_t)n * K + koff[c]], &pack[o + (size_t)n * cs_], cs_);
        o += (size_t)N * cs_;
    }
}

// ============================================================================
// 3. 主机逐元素算子
// ============================================================================
static inline float vgelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
// X: [VNE, T] (i + VNE*t), 每列做 LayerNorm
static void vlayernorm(const float* X, const float* w, const float* b, float* Y, int T) {
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const float* x = X + (size_t)t * VNE;
        float s = 0;
        for (int i = 0; i < VNE; i++) s += x[i];
        float mean = s / VNE;
        float v = 0;
        for (int i = 0; i < VNE; i++) { float dd = x[i] - mean; v += dd * dd; }
        v /= VNE;
        float sc = 1.0f / sqrtf(v + VEPS);
        float* y = Y + (size_t)t * VNE;
        for (int i = 0; i < VNE; i++) y[i] = (x[i] - mean) * sc * w[i] + b[i];
    }
}
// token 排列: 合并链路 (permute/cont/reshape/permute/cont) 的等价显式形式
//   结果[i][t] = 输入网格 grid( w(t), h(t) )[i]
static inline void tok_xy(int t, int OW, int OH, int& w, int& h) {
    int j2 = (t / 4) % (OW / 2);
    int j1 = (t / 2) % 2;
    int j3 = t / (2 * OW);
    w = ((t % 2) + 2 * j2) % OW;
    h = 2 * j3 + j1;
    if (h >= OH) h = OH - 1;
}
// M-RoPE (vision): cache 累积方式与 ggml 一致
static void vrope(float* Q, const int* prow, const int* pcol, int T) {
    const float ts = powf(10000.f, -2.f / 36.f);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < VNH; h++) {
            float* v = Q + (size_t)(h * VDH) + (size_t)VNE * t;
            float tt = (float)prow[t], th = (float)pcol[t];
            float out[VDH];
            for (int ic = 0; ic < 36; ic++) {
                float theta;
                if (ic < 18) { theta = tt; tt *= ts; } else { theta = th; th *= ts; }
                float c = cosf(theta), s = sinf(theta);
                float a0 = v[ic], a1 = v[ic + 36];
                out[ic] = a0 * c - a1 * s;
                out[ic + 36] = a0 * s + a1 * c;
            }
            memcpy(v, out, sizeof(out));
        }
    }
}

// ============================================================================
// 4. 视觉塔前向
// ============================================================================
static VG g_vg;
static int g_vinit = 0;
static VW g_Wpatch, g_Wqkv[VNL], g_Wo[VNL], g_Wup[VNL], g_Wdn[VNL], g_Wmm0, g_Wmm2;
static std::vector<float> g_bpatch, g_bln1w[VNL], g_bln1b[VNL], g_bln2w[VNL], g_bln2b[VNL],
                          g_bqkv[VNL], g_bo[VNL], g_bup[VNL], g_bdn[VNL], g_bpostw, g_bpostb,
                          g_bmm0, g_bmm2;

int vis_loaded() { return g_vinit; }

int vis_init(const char* path) {
    if (g_vinit) return 0;
    if (getenv("VIS_CPUGEMM")) { g_vcpugemm = atoi(getenv("VIS_CPUGEMM")); printf("[vis] ★ VIS_CPUGEMM=%d (调试: 主机参考 GEMM, 不占卡)\n", g_vcpugemm); }
    if (g_vcpugemm) { g_vAsz = g_vBsz = (size_t)1 << 32; }   // 调试模式不占卡
    g_vch = getenv("VIS_CH") ? atoi(getenv("VIS_CH")) : 96;
    g_vcha = getenv("VIS_CHA") ? atoi(getenv("VIS_CHA")) : 0;   // ★ 默认 0 = 注意力小 K 不分块 (实测比 24 更准也更快)
    g_vprof = getenv("VIS_PROF") ? atoi(getenv("VIS_PROF")) : 0;
    g_vcarda = getenv("VIS_CARDA") ? atoi(getenv("VIS_CARDA")) : 1;   // ★ 第 19 轮: A 常驻卡上 (0=老路径, 仅对照)
    // ★★ 第 20 轮开关 (默认全关) ★★
    g_vcardffn = getenv("VIS_CARDFFN") ? atoi(getenv("VIS_CARDFFN")) : 0;
    g_vheap    = getenv("VIS_HEAP")    ? atoi(getenv("VIS_HEAP"))    : 1;   // ★ 第21轮: 默认开启 (位级等价已对拍; 512² 不受影响)
    g_vkpad    = getenv("VIS_PAD")     ? atoi(getenv("VIS_PAD"))     : (g_vcardffn ? 1 : 0);
    g_vheapd2h = getenv("VIS_HEAPD2H") ? atoi(getenv("VIS_HEAPD2H")) : 4;   // ★ 第21轮: 默认 4 = 每头两芯同步 (唯一数值正确的档)
    g_vheapdbg = getenv("VIS_HEAPDBG") ? atoi(getenv("VIS_HEAPDBG"))  : 0;
    {   // FFN padding 布局: 找 >= VFFN 的 (g_vch 整数倍) (4304 -> 4320 = 45*96)
        int kp = g_vkpad ? ((VFFN + g_vch - 1) / g_vch) * g_vch : VFFN;
        if (kp != VFFN && (g_vch <= 0 || kp % g_vch)) { printf("[vis] ★ FFN padding 不可用 (CH=%d), 退回老布局\n", g_vch); kp = VFFN; g_vkpad = 0; }
        g_vkpadn = kp;
        if (kp != VFFN) printf("[vis] ★★ 第20轮 FFN padding 布局: VFFN %d -> Kp %d (=%d*%d, 尾部补 0) —— 与老 96+80 分块逐位等价\n",
                                VFFN, kp, g_vch, kp / g_vch);
    }
    if (g_vch <= 0) g_vch = 1 << 20;      // <=0 => 不分块 (退回旧行为)
    if (g_vcha <= 0) g_vcha = 1 << 20;
    printf("[vis] ★ 量化/折回方案: 逐块行归一 + 【折回在卡上】(官方 elementwise_mul_2d/reduce) VIS_CH=%d VIS_CHA=%d\n",
           g_vch, g_vcha);
    printf("[vis] ★ 激活量化修正: K 分块 CH=%d (注意力小 K 用 %d)%s\n", g_vch, g_vcha,
           g_vcpugemm ? "  [CPUGEMM 调试]" : "");
    printf("[vis] ★ A 常驻卡上: %s —— 卡上 reduce(MAX)+elementwise_div_2d 逐块行归一, A 不再上下卡\n",
           g_vcarda ? "ON (第19轮)" : "OFF (老路径: 主机归一 + A 上下卡)");
    printf("[vis] ★★ 第 20 轮开关: VIS_CARDFFN=%d (%s)  VIS_HEAP=%d%s\n", g_vcardffn,
           g_vcardffn == 1 ? "全卡 bias/gelu/归一/gemm" : (g_vcardffn == 2 ? "主机 bias/gelu + 卡上归一/gemm" : "关(第19轮路径)"),
           g_vheap, g_vheap ? " (16 头输出堆叠, D2H 32->2 次/层)" : "");
    printf("[vis] ★★ 第 21 轮: 卡上除法分母夹逼(>=1e-30, 官方 elementwise_max_2d) VIS_HEAPD2H=%d VIS_HEAPDBG=%d\n",
           g_vheapd2h, g_vheapdbg);
    if (g_vheap && g_vheapd2h != 4 && g_vheapd2h != 5)
        printf("[vis] WARN: VIS_HEAPD2H=%d 是【已知数值错误】的档 (1/2/3) —— 只用于复现, 不要上线\n", g_vheapd2h);
    if (!g_vcpugemm) for (int dv = 0; dv < 2; dv++) {
        xpu_set_device(dv);
        // ★ 先分配临时缓冲 (低地址), 再灌权重 — 文本引擎同款规避策略
        if (xpu_malloc(&g_vxA[dv], (size_t)VMA_MAX * VKMAX * 4)) { printf("[vis] FATAL: alloc A chip%d\n", dv); return 1; }
        if (xpu_malloc(&g_vBd[dv], (size_t)VTMAX * 4096)) { printf("[vis] FATAL: alloc Bd chip%d\n", dv); return 1; }
        // ★ 卡上折回缓冲: 堆叠 stk + 同容量 scratch (成对分配; 失败退到更小)
        {
            static const size_t pr[4] = {(size_t)64 << 20, (size_t)32 << 20, (size_t)16 << 20, (size_t)8 << 20};
            g_vCssz = 0;
            for (int k = 0; k < 4; k++) {
                if (xpu_malloc(&g_vCs[dv], pr[k])) { g_vCs[dv] = nullptr; continue; }
                if (xpu_malloc(&g_vTc[dv], pr[k]) == 0) { g_vCssz = pr[k]; break; }
                xpu_free(g_vCs[dv]); g_vCs[dv] = nullptr;
            }
            printf("[vis] 折回堆叠缓冲 chip%d = %.0f MB (含 scratch)\n", dv, g_vCssz / 1048576.0);
            if (!g_vCs[dv] || !g_vTc[dv]) { printf("[vis] FATAL: 折回堆叠缓冲分配失败 chip%d\n", dv); return 1; }
        }
        // ★ 折回累加器 [Mh][N] 两份 (Mh = VMA_MAX/2, N = VKMAX)
        g_vCaszz = (size_t)(VMA_MAX / 2) * VKMAX * 4;
        if (xpu_malloc(&g_vCa[dv], g_vCaszz) || xpu_malloc(&g_vCr[dv], g_vCaszz)) { printf("[vis] FATAL: alloc 折回累加 chip%d\n", dv); return 1; }
        // ★ 行因子缓冲 [nc][Mh] (nc 上限按 CH>=64 估: VKMAX/64 * VMA_MAX/2)
        g_vFdsz = (size_t)(VKMAX / 64) * (VMA_MAX / 2 + 8) * 4;
        if (xpu_malloc(&g_vFd[dv], g_vFdsz)) { printf("[vis] FATAL: alloc 行因子 chip%d\n", dv); return 1; }
        // ★ 动态 B 的折回列 scale
        g_vRdsz = (size_t)1 << 22;
        if (xpu_malloc(&g_vRd[dv], g_vRdsz)) { printf("[vis] FATAL: alloc dynB rs chip%d\n", dv); return 1; }
        // ★ 第 19 轮: A 常驻卡上的分块 gemm 所需缓冲 (归一结果 / 逐块行 max / 次 reduce)
        g_vAnsz = (size_t)(VMA_MAX / 2) * VKMAX * 4;
        if (xpu_malloc(&g_vAn[dv], g_vAnsz)) { printf("[vis] FATAL: alloc An chip%d\n", dv); return 1; }
        g_vFxsz = (size_t)(VMA_MAX / 2) * (VKMAX / 8 + 8) * 4;
        if (xpu_malloc(&g_vFx[dv], g_vFxsz) || xpu_malloc(&g_vFa[dv], g_vFxsz) ||
            xpu_malloc(&g_vFb[dv], g_vFxsz) || xpu_malloc(&g_vFe[dv], g_vFxsz)) { printf("[vis] FATAL: alloc 行因子 chip%d\n", dv); return 1; }
        g_vFisz = (size_t)(VKMAX / 8 + 8) * 4;
        if (xpu_malloc(&g_vFi[dv], 2 * g_vFisz) || xpu_malloc(&g_vOn[dv], g_vFisz)) { printf("[vis] FATAL: alloc Mc/1 chip%d\n", dv); return 1; }
        {   // 全 1 数组 (算 1/Mc 用) + 全 -1 数组 (算 -minA 用, api::neg 不在 api.h 里)
            std::vector<float> ones((VKMAX / 8 + 8), 1.0f);
            if (xpu_memcpy(g_vOn[dv], ones.data(), g_vFisz, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: H2D ones chip%d\n", dv); return 1; }
            if (xpu_malloc(&g_vMg[dv], g_vFisz)) { printf("[vis] FATAL: alloc -1 向量 chip%d\n", dv); return 1; }
            std::vector<float> mgs((VKMAX / 8 + 8), -1.0f);
            if (xpu_memcpy(g_vMg[dv], mgs.data(), g_vFisz, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: H2D -1 向量 chip%d\n", dv); return 1; }
            xpu_wait();
        }
        g_vSesz = (size_t)(VMA_MAX / 2) * VKMAX * 4;
        if (xpu_malloc(&g_vSe[dv], g_vSesz)) { printf("[vis] FATAL: alloc 次 reduce chip%d\n", dv); return 1; }
        {   // ★★ 第 21 轮: 分母夹逼常量 [m][nc] = 1e-30 (与 mxA 同形, 走"完整张量×完整张量"那条已验证的算子用法)
            if (xpu_malloc(&g_vEp[dv], g_vFxsz)) { printf("[vis] FATAL: alloc eps chip%d\n", dv); return 1; }
            std::vector<float> epsv(g_vFxsz / 4, 1e-30f);
            if (xpu_memcpy(g_vEp[dv], epsv.data(), g_vFxsz, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: H2D eps chip%d\n", dv); return 1; }
            xpu_wait();
        }
        // ★★ 第 20 轮: (A) FFN 卡上激活缓冲 + bias 副本;  (B) 16 头输出堆叠
        if (g_vcardffn) {
            g_vUpPsz = (size_t)(VMA_MAX / 2) * g_vkpadn * 4;
            if (xpu_malloc(&g_vUpP[dv], g_vUpPsz)) { printf("[vis] FATAL: alloc FFN UpP chip%d\n", dv); return 1; }
            g_vBupsz = (size_t)VNL * g_vkpadn * 4;
            if (xpu_malloc(&g_vBup[dv], g_vBupsz)) { printf("[vis] FATAL: alloc FFN bias chip%d\n", dv); return 1; }
            printf("[vis] (A) FFN 卡上激活缓冲 chip%d = %.1f MB (Kp=%d, m<=%d)\n", dv, g_vUpPsz / 1048576.0, g_vkpadn, VMA_MAX / 2);
        }
        if (g_vheap) {
            g_vSlOsz = (size_t)VNH * (VMA_MAX / 2) * VDH * 4;
            if (xpu_malloc(&g_vSlO[dv], g_vSlOsz)) { printf("[vis] FATAL: alloc heap 堆叠 chip%d\n", dv); return 1; }
            printf("[vis] (B) 16 头输出堆叠缓冲 chip%d = %.1f MB\n", dv, g_vSlOsz / 1048576.0);
        }
        printf("[vis] BUF chip%d: A=%p/%zu Cs=%p Tc=%p(%zu) Ca=%p Cr=%p(%zu) An=%p(%zu) Fx=%p Fa=%p Fb=%p Fe=%p(%zu)"
               " Fd=%p(%zu) Rd=%p(%zu) Fi=%p On=%p Mg=%p Ep=%p(%zu) Se=%p(%zu) Bd=%p(%zu) UpP=%p(%zu) SlO=%p(%zu)\n",
               dv, g_vxA[dv], (size_t)VMA_MAX * VKMAX * 4, g_vCs[dv], g_vTc[dv], g_vCssz, g_vCa[dv], g_vCr[dv], g_vCaszz,
               g_vAn[dv], g_vAnsz, g_vFx[dv], g_vFa[dv], g_vFb[dv], g_vFe[dv], g_vFxsz, g_vFd[dv], g_vFdsz, g_vRd[dv], g_vRdsz,
               g_vFi[dv], g_vOn[dv], g_vMg[dv], g_vEp[dv], g_vFxsz, g_vSe[dv], g_vSesz, g_vBd[dv], (size_t)VTMAX * 4096,
               g_vUpP[dv], g_vUpPsz, g_vSlO[dv], g_vSlOsz);
        g_vctx[dv] = new api::Context(api::Device(api::DeviceType::XPU1, dv));
        printf("[vis] BUF chip%d: ctx=%p (Context 内部 workspace)\n", dv, (void*)g_vctx[dv]);
        g_vAsz = (size_t)VMA_MAX * VKMAX * 4; g_vBsz = (size_t)VTMAX * 4096;
    }    if (!g_vg.open(path)) return 1;
    std::vector<float> W;
    printf("[vis] 加载视觉塔权重 (int8 每行 + 每行 scale) ...\n");
    // patch conv: W_eff = W0 + W1  -> [1152, 768]
    {
        std::vector<float> w0, w1;
        if (!g_vg.read_f32("v.patch_embd.weight", w0)) return 1;
        if (!g_vg.read_f32("v.patch_embd.weight.1", w1)) return 1;
        if (w0.size() != w1.size()) { printf("[vis] FATAL: patch conv 形状不符\n"); return 1; }
        for (size_t i = 0; i < w0.size(); i++) w0[i] += w1[i];
        vw_from_f32(w0, VNE, 768, g_Wpatch);
        if (!g_vg.read_f32("v.patch_embd.bias", g_bpatch)) return 1;
    }
    for (int il = 0; il < VNL; il++) {
        char nm[128];
        #define VLOAD_WI(field, suffix, N_, K_) { \
            snprintf(nm, sizeof(nm), "v.blk.%d.%s.weight", il, suffix); \
            if (!g_vg.read_f32(nm, W)) return 1; \
            if ((int)W.size() != (N_) * (K_)) { printf("[vis] FATAL: %s 形状 %zu != %d*%d\n", nm, W.size(), (N_), (K_)); return 1; } \
            vw_from_f32(W, (N_), (K_), field[il]); }
        VLOAD_WI(g_Wqkv, "attn_qkv", VNE * 3, VNE);
        VLOAD_WI(g_Wo,   "attn_out", VNE, VNE);
        #define VLOAD_WI_PAD(field, suffix, N_, K_, NP_, KP_) { \
            snprintf(nm, sizeof(nm), "v.blk.%d.%s.weight", il, suffix); \
            if (!g_vg.read_f32(nm, W)) return 1; \
            if ((int)W.size() != (N_) * (K_)) { printf("[vis] FATAL: %s 形状 %zu != %d*%d\n", nm, W.size(), (N_), (K_)); return 1; } \
            if ((NP_) != (N_) || (KP_) != (K_)) { \
                std::vector<float> Wp((size_t)(NP_) * (KP_), 0.f); \
                for (int n = 0; n < (N_); n++) memcpy(&Wp[(size_t)n * (KP_)], &W[(size_t)n * (K_)], (size_t)(K_) * 4); \
                vw_from_f32(Wp, (NP_), (KP_), field[il]); \
            } else vw_from_f32(W, (N_), (K_), field[il]); }
        VLOAD_WI_PAD(g_Wup,  "ffn_up",   VFFN, VNE, g_vkpadn, VNE);
        VLOAD_WI_PAD(g_Wdn,  "ffn_down", VNE, VFFN, VNE, g_vkpadn);
        #undef VLOAD_WI_PAD
        #undef VLOAD_WI
        #define VLOAD_V(vec, suffix) { snprintf(nm, sizeof(nm), "v.blk.%d.%s", il, suffix); if (!g_vg.read_f32(nm, vec[il])) return 1; }
        VLOAD_V(g_bln1w, "ln1.weight"); VLOAD_V(g_bln1b, "ln1.bias");
        VLOAD_V(g_bln2w, "ln2.weight"); VLOAD_V(g_bln2b, "ln2.bias");
        VLOAD_V(g_bqkv, "attn_qkv.bias"); VLOAD_V(g_bo, "attn_out.bias");
        VLOAD_V(g_bup, "ffn_up.bias"); VLOAD_V(g_bdn, "ffn_down.bias");
        if (g_vkpadn != VFFN && (int)g_bup[il].size() == VFFN) g_bup[il].resize(g_vkpadn, 0.f);   // ★ padding 尾部恒 0
        #undef VLOAD_V
    }
    {
        std::vector<float> pw;
        if (!g_vg.read_f32("v.post_ln.weight", g_bpostw)) return 1;
        if (!g_vg.read_f32("v.post_ln.bias", g_bpostb)) return 1;
        if (!g_vg.read_f32("mm.0.weight", W)) return 1;
        vw_from_f32(W, 4608, 4608, g_Wmm0);
        if (!g_vg.read_f32("mm.0.bias", g_bmm0)) return 1;
        if (!g_vg.read_f32("mm.2.weight", W)) return 1;
        vw_from_f32(W, 4096, 4608, g_Wmm2);
        if (!g_vg.read_f32("mm.2.bias", g_bmm2)) return 1;
    }
    printf("[vis] 权重点卡完成: chip0 %.0f MB, chip1 %.0f MB\n", g_vhbm[0] / 1048576.0, g_vhbm[1] / 1048576.0);
    if (g_vcardffn) {   // ★ (A): ffn_up.bias 的 padding 副本 (每层一份) 上卡
        std::vector<float> bp((size_t)VNL * g_vkpadn, 0.f);
        for (int il = 0; il < VNL; il++) memcpy(&bp[(size_t)il * g_vkpadn], g_bup[il].data(), (size_t)g_bup[il].size() * 4);
        for (int dv = 0; dv < 2; dv++) {
            xpu_set_device(dv);
            if (xpu_memcpy(g_vBup[dv], bp.data(), (size_t)VNL * g_vkpadn * 4, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: H2D FFN bias\n"); return 1; }
            xpu_wait();
        }
    }
    g_vinit = 1;
    return 0;
}

static void vdump(const char* dir, const char* name, const float* p, size_t n) {
    if (!dir || !dir[0]) return;
    char path[512]; snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE* f = fopen(path, "wb");
    if (!f) { printf("[vis] WARN: 打不开 dump %s\n", path); return; }
    fwrite(p, 4, n, f); fclose(f);
}

// 头部: patch conv + 位置编码 -> X[VNE, T]
static int vis_head(const float* in, int W, int H, int ow, int oh, int T,
                    const std::vector<int>& tw, const std::vector<int>& th,
                    std::vector<float>& X, const char* dumpdir) {
    std::vector<float> cols((size_t)T * 768);
    for (int oy = 0; oy < oh; oy++)
        for (int ox = 0; ox < ow; ox++) {
            float* c = &cols[((size_t)ox + (size_t)ow * oy) * 768];
            for (int ic = 0; ic < 3; ic++)
                for (int kh = 0; kh < 16; kh++)
                    for (int kw = 0; kw < 16; kw++)
                        c[ic * 256 + kh * 16 + kw] = in[(ox * 16 + kw) + (size_t)W * (oy * 16 + kh) + (size_t)W * H * ic];
        }
    vdump(dumpdir, "cols.f32", cols.data(), cols.size());
    std::vector<float> conv((size_t)T * VNE);
    vgemm(g_Wpatch, cols.data(), T, conv.data());
    // 位置编码网格 (F32): 表 [1152, 2304], p = 48*y + x -> grid[x][y][c]
    std::vector<float> pw;
    if (!g_vg.read_f32("v.position_embd.weight", pw)) return 1;
    const int NS = 48;
    static std::vector<float> grid, g2;
    grid.assign((size_t)NS * NS * VNE, 0.f);
    for (int y = 0; y < NS; y++)
        for (int x = 0; x < NS; x++)
            memcpy(&grid[((size_t)x + (size_t)NS * y) * VNE], &pw[(size_t)(48 * y + x) * VNE], VNE * 4);
    const float* use = grid.data(); int uh = NS;
    if (ow != NS || oh != NS) {
        g2.resize((size_t)ow * oh * VNE);
        float sf0 = (float)(ow - 1) / (NS - 1), sf1 = (float)(oh - 1) / (NS - 1);
        for (int i = 0; i < ow; i++) {
            float x = i / sf0; int x0 = (int)floorf(x); int x1 = x0 + 1;
            if (x0 < 0) x0 = 0; if (x0 > NS - 1) x0 = NS - 1;
            if (x1 < 0) x1 = 0; if (x1 > NS - 1) x1 = NS - 1;
            float dx = x - x0; if (dx < 0) dx = 0; if (dx > 1) dx = 1;
            for (int j = 0; j < oh; j++) {
                float y = j / sf1; int y0 = (int)floorf(y); int y1 = y0 + 1;
                if (y0 < 0) y0 = 0; if (y0 > NS - 1) y0 = NS - 1;
                if (y1 < 0) y1 = 0; if (y1 > NS - 1) y1 = NS - 1;
                float dy = y - y0; if (dy < 0) dy = 0; if (dy > 1) dy = 1;
                const float* A = &grid[((size_t)x0 + (size_t)NS * y0) * VNE];
                const float* B = &grid[((size_t)x1 + (size_t)NS * y0) * VNE];
                const float* C = &grid[((size_t)x0 + (size_t)NS * y1) * VNE];
                const float* D = &grid[((size_t)x1 + (size_t)NS * y1) * VNE];
                float* o = &g2[((size_t)i + (size_t)ow * j) * VNE];
                for (int c = 0; c < VNE; c++)
                    o[c] = A[c] * (1 - dx) * (1 - dy) + B[c] * dx * (1 - dy) + C[c] * (1 - dx) * dy + D[c] * dx * dy;
            }
        }
        use = g2.data(); uh = oh;
    }
    (void)uh;
    X.assign((size_t)VNE * T, 0.f);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const float* cv = &conv[((size_t)tw[t] + (size_t)ow * th[t]) * VNE];
        const float* pv = &use[((size_t)tw[t] + (size_t)ow * th[t]) * VNE];
        float* x = &X[(size_t)VNE * t];
        for (int i = 0; i < VNE; i++) x[i] = cv[i] + g_bpatch[i] + pv[i];
    }
    vdump(dumpdir, "conv.f32", conv.data(), conv.size());
    vdump(dumpdir, "gridi.f32", use, (size_t)ow * oh * VNE);
    {   // 排列落盘 (w,h per token)
        static std::vector<float> mw, mh;
        mw.resize(T); mh.resize(T);
        for (int t = 0; t < T; t++) { mw[t] = (float)tw[t]; mh[t] = (float)th[t]; }
        vdump(dumpdir, "mapw.f32", mw.data(), T);
        vdump(dumpdir, "maph.f32", mh.data(), T);
    }
    vdump(dumpdir, "inp.f32", X.data(), X.size());
    return 0;
}

int vis_forward(const float* in, int W, int H, std::vector<float>& out, int& n_tok, const char* dumpdir) {
    if (!g_vinit) { printf("[vis] FATAL: 未初始化\n"); return 1; }
    double t_all = vnow();
    if (W % 16 || H % 16) { printf("[vis] FATAL: 尺寸必须是 16 的倍数 (%d %d)\n", W, H); return 1; }
    int ow = W / 16, oh = H / 16, T = ow * oh;
    if (T > VMA_MAX) { printf("[vis] FATAL: patch 数 %d > 上限 %d\n", T, VMA_MAX); return 1; }
    if (T % 4) { printf("[vis] FATAL: patch 数不是 4 的倍数\n"); return 1; }
    n_tok = T / 4;
    printf("[vis] 前向: %dx%d -> %dx%d patch = %d (合并后 %d token)\n", W, H, ow, oh, T, n_tok);
    std::vector<int> tw(T), th(T), prow(T), pcol(T);
    for (int t = 0; t < T; t++) { int w, h; tok_xy(t, ow, oh, w, h); tw[t] = w; th[t] = h; prow[t] = h; pcol[t] = w; }
    std::vector<float> X;
    double t0 = vnow();
    if (vis_head(in, W, H, ow, oh, T, tw, th, X, dumpdir)) return 1;
    printf("[vis]   head %.3fs\n", vnow() - t0);
    std::vector<float> Y((size_t)VNE * T), qkv((size_t)T * 3 * VNE), Qb((size_t)VNE * T), Kb((size_t)VNE * T), Vb((size_t)VNE * T);
    std::vector<float> Sm((size_t)T * T), C2((size_t)T * VDH), Ao((size_t)VNE * T), Up((size_t)T * g_vkpadn), Dn((size_t)T * VNE);
    std::vector<float> qt, kt, vt, rs2, rs3;
    std::vector<signed char> qq, qk, qv;
    std::vector<int> csA, koA, csB, koB;   // ★ 分块表 (注意力路径)
    if (getenv("VIS_ONLY")) { g_vonly = atoi(getenv("VIS_ONLY")); }
    for (int il = 0; il < VNL && il < g_vonly; il++) {
        double tl = vnow();
        vlayernorm(X.data(), g_bln1w[il].data(), g_bln1b[il].data(), Y.data(), T);
        vgemm(g_Wqkv[il], Y.data(), T, qkv.data());
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) {
            const float* s = &qkv[(size_t)t * 3 * VNE];
            for (int i = 0; i < VNE; i++) {
                Qb[(size_t)i + (size_t)VNE * t] = s[i] + g_bqkv[il][i];
                Kb[(size_t)i + (size_t)VNE * t] = s[VNE + i] + g_bqkv[il][VNE + i];
                Vb[(size_t)i + (size_t)VNE * t] = s[2 * VNE + i] + g_bqkv[il][2 * VNE + i];
            }
        }
        double trp = vnow();
        vrope(Qb.data(), prow.data(), pcol.data(), T);
        {   // ★ 1/sqrt(VDH) 折进 Q (归一化会自动吸收), 等价于原 softmax 前的 r[k]*sq
            const float sqq = 1.0f / sqrtf((float)VDH);
            for (size_t i = 0; i < Qb.size(); i++) Qb[i] *= sqq;
        }
        vrope(Kb.data(), prow.data(), pcol.data(), T);
        if (il == 0) { vdump(dumpdir, "l0_qb.f32", Qb.data(), Qb.size()); vdump(dumpdir, "l0_kb.f32", Kb.data(), Kb.size()); vdump(dumpdir, "l0_vb.f32", Vb.data(), Vb.size()); }
        g_p_rope += vnow() - trp;
        double tat = vnow();
        // 逐 head attention: S[qt][kt] = Q·K^T ; O[d][qt] = V·Sm ; 走 gemm_int8
        qt.resize((size_t)T * VDH); kt.resize((size_t)T * VDH); vt.resize((size_t)T * VDH);
        // ★ 第 19 轮: 卡上归一要用 [m*nc][cs] 连续视图 ⇒ 分块必须整除 T;
        //   并且只在【分块与主机路径完全一致】时才启用 —— 只有这样卡上归一链才与主机逐位等价
        //   (自检 新 vs 老 = 0.0000%), 端到端行为保持不变。分块不一致时 (如 T=1024, 96∤1024) 退回主机路径。
        //   ★ 第 21 轮: 提到头循环之前 —— (B) 堆叠路径也必须由它门控:
        //   主机路径 (useCA=0, 如 512²) 时堆叠缓冲【不会】被本层写入, 用陈旧内容重建 Ao 会全错。
        const int csWant = (g_vch > 0) ? g_vch : T;
        const int csO = vunif_div(T, csWant);
        const int useCA = (g_vcarda && csO > 0 && csO == csWant) ? 1 : 0;
        const int useHeap = (g_vheap && useCA) ? 1 : 0;
        // ★ 第21轮: (B) 堆叠读回缓冲提前声明 (逐头立即 D2H 的模式 5 要在头循环里写)
        std::vector<float> C2s;
        if (useHeap) C2s.assign((size_t)2 * VNH * ((T + 1) / 2) * VDH, 0.f);
        if (useHeap && g_vheapdbg && il == 0 && dumpdir && dumpdir[0]) {
            // ★ 第21轮诊断: 头循环前把堆叠缓冲填成哨兵 12345 —— 头循环后若某些头仍等于哨兵,
            //   就证明"那些头的写入没有落盘"(DMA/顺序问题), 而不是"算错了"。
            int Mh0 = (T + 1) / 2;
            std::vector<float> sent((size_t)VNH * Mh0 * VDH, 12345.0f), back((size_t)VNH * Mh0 * VDH);
            for (int p = 0; p < 2; p++) {
                xpu_set_device(p);
                if (xpu_memcpy(g_vSlO[p], sent.data(), sent.size() * 4, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: H2D heap 哨兵\n"); exit(1); }
                xpu_wait();
                if (xpu_memcpy(back.data(), g_vSlO[p], back.size() * 4, XPU_DEVICE_TO_HOST)) { printf("[vis] FATAL: D2H heap 哨兵回读\n"); exit(1); }
                xpu_wait();
                int nbad = 0; for (size_t i = 0; i < back.size(); i++) if (back[i] != 12345.0f) nbad++;
                printf("[vis] ★ 哨兵 chip%d: 回读非哨兵个数=%d/%zu (期望 0)\n", p, nbad, back.size());
                char dn[64]; snprintf(dn, sizeof(dn), "slO_sent%d.f32", p); vdump(dumpdir, dn, back.data(), back.size());
            }
        }
        for (int h = 0; h < VNH; h++) {
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < T; t++) {
                memcpy(&qt[(size_t)t * VDH], &Qb[(size_t)(h * VDH) + (size_t)VNE * t], VDH * 4);
                memcpy(&kt[(size_t)t * VDH], &Kb[(size_t)(h * VDH) + (size_t)VNE * t], VDH * 4);
                memcpy(&vt[(size_t)t * VDH], &Vb[(size_t)(h * VDH) + (size_t)VNE * t], VDH * 4);
            }
            vquant_chunked(kt.data(), T, VDH, g_vcha, qk, rs2, csA, koA);
            vgemm_dynB(qk, rs2, T, VDH, csA, koA, qt.data(), T, Sm.data(), 1);   // ★ 结果留在卡上 g_vCa[dv] (m,T)
            static int h_dumped = 0;
            if (il == 0 && h == 0 && dumpdir && dumpdir[0]) { vdump(dumpdir, "l0_qt0.f32", qt.data(), qt.size()); h_dumped = 1; }
            double tsf = vnow();
            {   // ★ 卡上 softmax (official softmax2d_forward): 每分区各自 softmax 自己的行块
                int Th = (T + 1) / 2;
                for (int p = 0; p < 2; p++) {
                    int dv = p, r0 = p * Th, m = Th; if (r0 + m > T) m = T - r0;
                    if (m <= 0) continue;
                    xpu_set_device(dv);
                    double t = vnow();
                    int r = api::softmax2d_forward(g_vctx[dv], (const float*)g_vCa[dv], (float*)g_vCr[dv], m, T, false);
                    if (r) { printf("[vis] FATAL: softmax2d r=%d (m=%d T=%d)\n", r, m, T); exit(1); }
                    if (xpu_wait()) { printf("[vis] FATAL: softmax wait\n"); exit(1); }
                    g_p_softmax += vnow() - t;
                    if (!useCA) {   // 老路径: A 必须回主机做逐块行归一 ⇒ 这里就得 D2H
                        t = vnow();
                        if (xpu_memcpy(Sm.data() + (size_t)r0 * T, g_vCr[dv], (size_t)m * T * 4, XPU_DEVICE_TO_HOST)) { printf("[vis] FATAL: D2H Sm\n"); exit(1); }
                        xpu_wait();
                        g_p_d2h += vnow() - t; g_p_n_d2h++;
                    }
                }
            }
            g_p_softmax += vnow() - tsf;
            if (useCA && dumpdir && dumpdir[0] && il == 0 && h < 2) {   // 仅为落盘对照
                int Th = (T + 1) / 2;
                for (int p = 0; p < 2; p++) { int r0 = p * Th, mm = Th; if (r0 + mm > T) mm = T - r0;
                    if (xpu_memcpy(Sm.data() + (size_t)r0 * T, g_vCr[p], (size_t)mm * T * 4, XPU_DEVICE_TO_HOST)) { printf("[vis] FATAL: D2H Sm(dump)\n"); exit(1); }
                    xpu_wait(); }
            }
            if (dumpdir && dumpdir[0] && il == 0 && h < 2) { char hn[64]; snprintf(hn, sizeof(hn), "l0_sm%d.f32", h); vdump(dumpdir, hn, Sm.data(), Sm.size()); }
            // O^T[qt][d] = Σ_t S[qt][t] · V[t][d]   (B = V 的行 d, 每块每行一个 scale)
            if (useCA) {   // ★ 第 19 轮: A = softmax 输出常驻卡上, 卡上逐块行归一 ⇒ 零 A 往返
                vpack_T(vt.data(), T, VDH, csO, qv, rs3, csB, koB);      // B[d][t] = V[t][d]
                const void* Ad2[2] = {g_vCr[0], g_vCr[1]};
                // ★ (B) VIS_HEAP=1: 每头结果写卡上堆叠缓冲第 h 段, 不 D2H (头循环后一次取回)
                vgemm_cardA(qv, rs3, VDH, T, csO, Ad2, T, C2.data(), 0, useHeap ? h : -1);
                if (useHeap && (g_vheapd2h == 4 || g_vheapd2h == 5)) {
                    // ★ 修复候选: 每个头结束就在两芯上做一次 device 同步 —— 把芯片上"尚未执行/尚未落盘"的
                    //   kernel 队列在每个头就抽干一次 (第19轮的逐头 D2H 事实上起到的就是这个作用)。
                    int Mn = (T + 1) / 2;
                    for (int p = 0; p < 2; p++) {
                        xpu_set_device(p);
                        if (xpu_wait()) { printf("[vis] FATAL: heap 逐头 wait\n"); exit(1); }
                        if (g_vheapd2h == 5) {   // 再立即把该头的结果取回 (顺序最强; D2H 回到 32 次/层)
                            double t = vnow();
                            if (xpu_memcpy(&C2s[((size_t)p * VNH + h) * Mn * VDH], (float*)g_vSlO[p] + (size_t)h * Mn * VDH,
                                           (size_t)Mn * VDH * 4, XPU_DEVICE_TO_HOST)) { printf("[vis] FATAL: D2H heap 逐头立即\n"); exit(1); }
                            xpu_wait();
                            g_p_d2h += vnow() - t; g_p_n_d2h++;
                        }
                    }
                }
            } else {
                vpack_T(vt.data(), T, VDH, g_vch, qv, rs3, csB, koB);      // B[d][t] = V[t][d]
                vgemm_dynB(qv, rs3, VDH, T, csB, koB, Sm.data(), T, C2.data());     // C2[qt][d]
            }
            if (!useHeap) {   // 主机路径 (或堆叠关闭): 逐头搬运 (位级同第19章)
                #pragma omp parallel for schedule(static)
                for (int t = 0; t < T; t++)
                    for (int d = 0; d < VDH; d++)
                        Ao[(size_t)(h * VDH + d) + (size_t)VNE * t] = C2[(size_t)t * VDH + d];
            }
        }
        if (useHeap) {   // ★ (B) 头循环结束: 每芯【一次】D2H 取回 16 头输出 (32 -> 2 次/层)
            int Mh = (T + 1) / 2;
            // ★★ 第 21 轮: 多档 D2H —— 目标是把"大 D2H 与芯片上尚未落盘的 kernel 抢跑"排除掉
            //   1(默认): 每芯【先 device 同步(xpu_wait) 再】做那一次大 D2H —— 顺序最强
            //   2       : 逐头小块 D2H (每头 1 次, 回到 32 次/层, 对照/兜底)
            //   3       : 两芯都先同步, 再各做一次大 D2H
            if (g_vheapd2h == 5) { /* 已在头循环里逐头同步并取回 */ }
            else if (g_vheapd2h == 2) {
                for (int h = 0; h < VNH; h++)
                    for (int p = 0; p < 2; p++) {
                        xpu_set_device(p);
                        double t = vnow();
                        if (xpu_memcpy(&C2s[((size_t)p * VNH + h) * Mh * VDH], (float*)g_vSlO[p] + (size_t)h * Mh * VDH,
                                       (size_t)Mh * VDH * 4, XPU_DEVICE_TO_HOST)) { printf("[vis] FATAL: D2H heap 逐头\n"); exit(1); }
                        xpu_wait();
                        g_p_d2h += vnow() - t; g_p_n_d2h++;
                    }
            } else {
                if (g_vheapd2h == 3)
                    for (int p = 0; p < 2; p++) { xpu_set_device(p); if (xpu_wait()) { printf("[vis] FATAL: heap 同步 wait\n"); exit(1); } }
                for (int p = 0; p < 2; p++) {
                    xpu_set_device(p);
                    if (g_vheapd2h != 3) { if (xpu_wait()) { printf("[vis] FATAL: heap 同步 wait\n"); exit(1); } }   // ★ 先同步再拷
                    double t = vnow();
                    if (xpu_memcpy(&C2s[(size_t)p * VNH * Mh * VDH], g_vSlO[p], (size_t)VNH * Mh * VDH * 4, XPU_DEVICE_TO_HOST)) { printf("[vis] FATAL: D2H heap\n"); exit(1); }
                    xpu_wait();
                    g_p_d2h += vnow() - t; g_p_n_d2h++;
                }
            }
            if (g_vheapdbg && dumpdir && dumpdir[0] && il == 0) {   // ★ 诊断: 原始堆叠缓冲整块落盘
                vdump(dumpdir, "slO_post0.f32", &C2s[0], (size_t)VNH * Mh * VDH);
                vdump(dumpdir, "slO_post1.f32", &C2s[(size_t)VNH * Mh * VDH], (size_t)VNH * Mh * VDH);
            }
            #pragma omp parallel for schedule(static)
            for (int h = 0; h < VNH; h++)
                for (int t = 0; t < T; t++) {
                    int p = (t < Mh) ? 0 : 1, i = t - p * Mh;
                    const float* sc = &C2s[((size_t)p * VNH * Mh + (size_t)h * Mh + i) * VDH];
                    for (int d = 0; d < VDH; d++) Ao[(size_t)(h * VDH + d) + (size_t)VNE * t] = sc[d];
                }
        }
        g_p_attn += vnow() - tat;
        if (il == 0) vdump(dumpdir, "l0_ao.f32", Ao.data(), Ao.size());
        std::vector<float> Wost((size_t)T * VNE), Aot((size_t)VNE * T);
        {   // attn_out = Wo · O + bo
            std::vector<float> O((size_t)T * VNE);
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < T; t++)
                for (int i = 0; i < VNE; i++) O[(size_t)t * VNE + i] = Ao[(size_t)i + (size_t)VNE * t];
            if (il == 0) vdump(dumpdir, "l0_o.f32", O.data(), O.size());
        vgemm(g_Wo[il], O.data(), T, Wost.data());
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < T; t++)
                for (int i = 0; i < VNE; i++) Aot[(size_t)i + (size_t)VNE * t] = Wost[(size_t)t * VNE + i] + g_bo[il][i];
        }
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < X.size(); i++) X[i] += Aot[i];
        vlayernorm(X.data(), g_bln2w[il].data(), g_bln2b[il].data(), Y.data(), T);
        if (il == 0 && dumpdir && dumpdir[0]) vdump(dumpdir, "ln2_0.f32", Y.data(), Y.size());
        const int Kp = g_vkpadn;   // ★ 第20轮: FFN 维度 (padding 布局, 尾部恒 0)
        if (g_vcardffn) {
            if (g_vcardffn == 1) {
                // ★ (A) 模式1: ffn_up 结果【留卡】→ 卡上 bias + gelu → 卡上 max|A| 归一 + 分块 gemm
                float* upd[2] = {(float*)g_vUpP[0], (float*)g_vUpP[1]};
                vgemm(g_Wup[il], Y.data(), T, Up.data(), 1, upd);
                if (il == 0 && dumpdir && dumpdir[0]) vffn_dump_up(T, dumpdir, "l0_up.f32");
                vffn_card_bias_gelu(T, il);
            } else {
                // ★ (A) 模式2: 主机 bias+gelu (与老路径逐位相同) → 上卡 → 卡上 max|A| 归一 + 分块 gemm
                vgemm(g_Wup[il], Y.data(), T, Up.data());
                if (il == 0 && dumpdir && dumpdir[0]) vdump(dumpdir, "l0_up.f32", Up.data(), Up.size());
                double th = vnow();
                #pragma omp parallel for schedule(static)
                for (int t = 0; t < T; t++) {
                    float* u = Up.data() + (size_t)t * Kp;
                    const float* b = g_bup[il].data();
                    for (int i = 0; i < Kp; i++) u[i] = vgelu(u[i] + b[i]);
                }
                g_p_gelu += vnow() - th;
                vffn_up_upload(T, Up.data());
            }
            const void* Ad2f[2] = {g_vUpP[0], g_vUpP[1]};
            vgemm_ffn_dn(g_Wdn[il], Ad2f, T, Dn.data());
        } else {
            // ---- 第 19 轮老路径 (默认); 只把 bias+gelu 的取模换成等价两重循环 (位级相同、更快) ----
            vgemm(g_Wup[il], Y.data(), T, Up.data());
            if (il == 0) vdump(dumpdir, "l0_up.f32", Up.data(), Up.size());
            double tge = vnow();
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < T; t++) {
                float* u = Up.data() + (size_t)t * Kp;
                const float* b = g_bup[il].data();
                for (int i = 0; i < Kp; i++) u[i] = vgelu(u[i] + b[i]);
            }
            g_p_gelu += vnow() - tge;
            vgemm(g_Wdn[il], Up.data(), T, Dn.data());
        }
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++)
            for (int i = 0; i < VNE; i++) X[(size_t)i + (size_t)VNE * t] += Dn[(size_t)t * VNE + i] + g_bdn[il][i];
        printf("[vis]   层 %2d  %.3fs (gemm 累计 %.2fs host %.2fs)\n", il, vnow() - tl, g_t_gemm, g_t_host);
        if (dumpdir && dumpdir[0]) {
            char nm[64]; snprintf(nm, sizeof(nm), "layer%d.f32", il);
            if (il == 0 || il == VNL - 1 || il + 1 == g_vonly) vdump(dumpdir, nm, X.data(), X.size());
        }
    }
    if (g_vonly < VNL) { printf("[vis] 调试: 只跑 %d 层, 到此为止\n", g_vonly); return 0; }
    double tpo = vnow();
    // post_ln
    std::vector<float> Xp((size_t)VNE * T);
    vlayernorm(X.data(), g_bpostw.data(), g_bpostb.data(), Xp.data(), T);
    vdump(dumpdir, "postln.f32", Xp.data(), Xp.size());
    // merger: [VNE,T] -> [4608, T/4] -> mm.0 -> gelu -> mm.2 -> [4096, T/4]
    int nt = T / 4;
    std::vector<float> E((size_t)nt * 4608);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < nt; j++)
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < VNE; c++)
                E[(size_t)j * 4608 + 1152 * r + c] = Xp[(size_t)c + (size_t)VNE * (4 * j + r)];
    std::vector<float> Hg((size_t)nt * 4608), Emb((size_t)nt * 4096);
    vgemm(g_Wmm0, E.data(), nt, Hg.data());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < Hg.size(); i++) Hg[i] = vgelu(Hg[i] + g_bmm0[i % 4608]);
    vgemm(g_Wmm2, Hg.data(), nt, Emb.data());
    out.resize((size_t)nt * 4096);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < nt; j++)
        for (int o = 0; o < 4096; o++) out[(size_t)o + (size_t)4096 * j] = Emb[(size_t)j * 4096 + o] + g_bmm2[o];
    if (dumpdir && dumpdir[0]) {
        char path[512]; snprintf(path, sizeof(path), "%s/emb.bin", dumpdir);
        FILE* f = fopen(path, "wb");
        if (f) { int32_t hd[2] = {nt, 4096}; fwrite(hd, 4, 2, f); fwrite(out.data(), 4, out.size(), f); fclose(f); }
    }
    g_p_post += vnow() - tpo;
    printf("[vis] 完成: %d token, 总耗时 %.2fs (gemm %.2fs, host %.2fs)\n", n_tok, vnow() - t_all, g_t_gemm, g_t_host);
    vprof_report();
    return 0;
}

// ============================================================================
// 5. 上卡前的极小尺寸 gemm_int8 自检 (把卡打进过 ERROR 的调用绝不碰)
// ============================================================================
// 单个尺寸的分块 gemm 自检: 同时报"对精确 f32"与"对精确激活×分块量化权重"
static int vselftest_one(int M, int NN, int K, int CH) {
    std::vector<float> A((size_t)M * K), B((size_t)NN * K), C((size_t)M * NN);
    for (size_t i = 0; i < A.size(); i++) A[i] = sinf(i * 0.01f) * 0.7f;
    for (size_t i = 0; i < B.size(); i++) B[i] = cosf(i * 0.003f) * (1.f + 2.f * ((i % 97) == 0));
    std::vector<signed char> Bq; std::vector<float> Brs, Bcs; std::vector<int> cs, koff;
    vquant_chunked(B.data(), NN, K, CH, Bq, Brs, cs, koff);
    std::vector<float> ref((size_t)M * NN, 0.f), ref2((size_t)M * NN, 0.f);
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++)
        for (int n = 0; n < NN; n++) {
            double acc = 0, acc2 = 0;
            size_t base = 0;
            for (size_t c = 0; c < cs.size(); c++) {
                for (int j = 0; j < cs[c]; j++) {
                    acc  += (double)A[(size_t)m * K + koff[c] + j] * (double)B[(size_t)n * K + koff[c] + j];
                    acc2 += (double)A[(size_t)m * K + koff[c] + j] * (double)Bq[base + (size_t)n * cs[c] + j]
                            * (double)Brs[c * NN + n];
                }
                base += (size_t)NN * cs[c];
            }
            ref[(size_t)m * NN + n] = (float)acc; ref2[(size_t)m * NN + n] = (float)acc2;
        }
    vgemm_dynB(Bq, Brs, NN, K, cs, koff, A.data(), M, C.data());
    double se = 0, se2 = 0, s2 = 0;
    for (size_t i = 0; i < (size_t)M * NN; i++) {
        double d = C[i] - ref[i], d2 = C[i] - ref2[i];
        se += d * d; se2 += d2 * d2; s2 += (double)ref[i] * ref[i];
    }
    double rr  = 100.0 * sqrt(se  / (s2 > 0 ? s2 : 1e-30));
    double rr2 = 100.0 * sqrt(se2 / (s2 > 0 ? s2 : 1e-30));
    printf("[vis] ★ 分块 gemm 自检 m=%d n=%d k=%d CH=%d (块数 %zu, ldb=k=cs): "
           "对精确值 %.4f%% | 对'精确激活×分块量化权重' %.4f%%  y0=%.5f ref0=%.5f\n",
           M, NN, K, CH, cs.size(), rr, rr2, C[0], ref[0]);
    return (rr < 3.0) ? 0 : 1;
}

// ★ 第 19 轮自检: 同一份 A/B, 老路径(主机逐块行归一 + A 上下卡) vs 新路径(卡上归一, A 常驻)
//   全用官方算子; 极小尺寸 (m=64 k=192 n=40) —— 先过这一关再上大尺寸
//
//   ★ 前提 (生产路径成立): A 必须【非负闭区间】—— 本轮唯一用法是注意力里的 softmax 输出。
//     卡上归一用 reduce(MAX) (有符号 max), 对 A>=0 它恒等于 max|A| ⇒ 归一后每块每行 max 恰为
//     1.0 ⇒ 算子内部 max_a ≡ 1.0 (满量程、不截断)。若将来把 gelu 输出(含负值)也搬上卡,
//     必须改成 max(mx, -mn) 形式的 max|A| 归一 —— 见下面第二个(信息性, 不判定)用例的实测差异。
static void cardA_run_pair(const std::vector<float>& A, const std::vector<float>& B,
                           int M, int K, int NN, const std::vector<signed char>& Bq,
                           const std::vector<float>& Brs, const std::vector<int>& cs,
                           const std::vector<int>& koff, int csO,
                           std::vector<float>& Cold, std::vector<float>& Cnew) {
    vgemm_dynB(Bq, Brs, NN, K, cs, koff, A.data(), M, Cold.data());     // 老路径
    int Mh = (M + 1) / 2;
    for (int p = 0; p < 2; p++) {
        int r0 = p * Mh, m = Mh; if (r0 + m > M) m = M - r0;
        if (m <= 0) continue;
        xpu_set_device(p);
        if (xpu_memcpy(g_vxA[p], A.data() + (size_t)r0 * K, (size_t)m * K * 4, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: cardA H2D A\n"); exit(1); }
        xpu_wait();
    }
    const void* Ad2[2] = {g_vxA[0], g_vxA[1]};
    vgemm_cardA(Bq, Brs, NN, K, csO, Ad2, M, Cnew.data());             // 新路径
}

static double cardA_stats(const std::vector<float>& Cnew, const std::vector<float>& Cold,
                          const std::vector<float>& Cref, const char* tag, int M, int NN, int K, int csO, size_t nc) {
    double seN = 0, seO = 0, seNO = 0, s2 = 0;
    for (size_t i = 0; i < (size_t)M * NN; i++) {
        double dN = Cnew[i] - Cref[i], dO = Cold[i] - Cref[i], dNO = Cnew[i] - Cold[i];
        seN += dN * dN; seO += dO * dO; seNO += dNO * dNO; s2 += (double)Cref[i] * Cref[i];
    }
    double d = sqrt(s2 > 0 ? s2 : 1e-30);
    double rN = 100.0 * sqrt(seN) / d, rO = 100.0 * sqrt(seO) / d, rNO = 100.0 * sqrt(seNO) / d;
    int nbit = 0;
    for (size_t i = 0; i < (size_t)M * NN; i++) if (Cnew[i] != Cold[i]) nbit++;
    printf("[vis] ★ cardA 自检[%s] m=%d n=%d k=%d cs=%d (nc=%zu): 新路径 vs 精确 %.4f%% | 老路径 vs 精确 %.4f%% | 新 vs 老 %.4f%% (位不同 %d/%d)\n",
           tag, M, NN, K, csO, nc, rN, rO, rNO, nbit, M * NN);
    return (rN > rNO) ? rN : rNO;   // 返回两者较大者作判据
}

int vis_cardA_selftest() {
    if (getenv("VIS_CPUGEMM")) { printf("[vis] cardA selftest 跳过 (CPUGEMM 模式)\n"); return 0; }
    g_vch = getenv("VIS_CH") ? atoi(getenv("VIS_CH")) : 96;
    if (g_vch <= 0) g_vch = 1 << 20;
    for (int dv = 0; dv < 2; dv++) {
        xpu_set_device(dv);
        if (xpu_malloc(&g_vxA[dv], (size_t)1 << 18) || xpu_malloc(&g_vBd[dv], (size_t)1 << 18) ||
            xpu_malloc(&g_vCs[dv], (size_t)1 << 20) || xpu_malloc(&g_vTc[dv], (size_t)1 << 20) ||
            xpu_malloc(&g_vCa[dv], (size_t)1 << 18) || xpu_malloc(&g_vCr[dv], (size_t)1 << 18) ||
            xpu_malloc(&g_vFd[dv], (size_t)1 << 18) || xpu_malloc(&g_vRd[dv], (size_t)1 << 18) ||
            xpu_malloc(&g_vAn[dv], (size_t)1 << 18) || xpu_malloc(&g_vFx[dv], (size_t)1 << 18) ||
            xpu_malloc(&g_vFa[dv], (size_t)1 << 18) || xpu_malloc(&g_vFb[dv], (size_t)1 << 18) ||
            xpu_malloc(&g_vFe[dv], (size_t)1 << 18) || xpu_malloc(&g_vFi[dv], (size_t)1 << 18) ||
            xpu_malloc(&g_vOn[dv], (size_t)1 << 18) || xpu_malloc(&g_vSe[dv], (size_t)1 << 18) ||
            xpu_malloc(&g_vEp[dv], (size_t)1 << 18)) {
            printf("[vis] cardA selftest alloc fail\n"); return 1; }
        g_vAsz = (size_t)1 << 18; g_vBsz = (size_t)1 << 18; g_vCssz = (size_t)1 << 20;
        g_vCaszz = (size_t)1 << 18; g_vFdsz = (size_t)1 << 18; g_vRdsz = (size_t)1 << 18;
        g_vAnsz = (size_t)1 << 18; g_vFxsz = (size_t)1 << 18; g_vSesz = (size_t)1 << 18; g_vFisz = (size_t)1 << 18;
        { std::vector<float> ones(1 << 12, 1.0f);
          if (xpu_memcpy(g_vOn[dv], ones.data(), (size_t)1 << 14, XPU_HOST_TO_DEVICE)) { printf("[vis] cardA ones fail\n"); return 1; }
          std::vector<float> epsv(1 << 15, 1e-30f);
          if (xpu_memcpy(g_vEp[dv], epsv.data(), (size_t)1 << 18, XPU_HOST_TO_DEVICE)) { printf("[vis] cardA eps fail\n"); return 1; }
          xpu_wait(); }
        g_vctx[dv] = new api::Context(api::Device(api::DeviceType::XPU1, dv));
    }
    int rc = 0;
    const int M = 64, K = 192, NN = 40;
    int csO = vunif_div(K, g_vch); if (csO <= 0) csO = K;
    std::vector<float> A((size_t)M * K), B((size_t)NN * K);
    // A >= 0 (忠实于生产路径: A = softmax 输出)
    for (size_t i = 0; i < A.size(); i++) A[i] = fabsf(sinf(i * 0.017f)) * (0.2f + 2.0f * (float)((i / 37) % 5));
    for (size_t i = 0; i < B.size(); i++) B[i] = cosf(i * 0.011f) * (1.f + 1.5f * ((i % 91) == 0));
    std::vector<signed char> Bq; std::vector<float> Brs; std::vector<int> cs, koff;
    vquant_chunked(B.data(), NN, K, csO, Bq, Brs, cs, koff);
    std::vector<float> Cold((size_t)M * NN), Cnew((size_t)M * NN), Cref((size_t)M * NN, 0.f);
    cardA_run_pair(A, B, M, K, NN, Bq, Brs, cs, koff, csO, Cold, Cnew);
    for (int t = 0; t < M; t++)   // 主机 double 精确参考
        for (int n = 0; n < NN; n++) {
            double acc = 0;
            for (size_t c = 0; c < cs.size(); c++)
                for (int j = 0; j < cs[c]; j++)
                    acc += (double)A[(size_t)t * K + koff[c] + j] * (double)B[(size_t)n * K + koff[c] + j];
            Cref[(size_t)t * NN + n] = (float)acc;
        }
    double gate = cardA_stats(Cnew, Cold, Cref, "A>=0 生产路径", M, NN, K, csO, cs.size());
    if (gate > 3.0) rc = 1;
    {   // 信息性(不判定): 含负数的 A —— 卡上 reduce(MAX) 是有符号 max ⇒ 归一后 max_a != 1, 误差变大
        std::vector<float> A2(A), Cold2((size_t)M * NN), Cnew2((size_t)M * NN);
        for (size_t i = 0; i < A2.size(); i++) if (i % 3 == 0) A2[i] = -A2[i];
        cardA_run_pair(A2, B, M, K, NN, Bq, Brs, cs, koff, csO, Cold2, Cnew2);
        cardA_stats(Cnew2, Cold2, Cref, "A 含负值(仅参考)", M, NN, K, csO, cs.size());
    }
    printf("[vis] ★ cardA 自检总结论: %s (判定只看 A>=0 那一行, 阈值 3%%)\n", rc ? "FAIL" : "PASS");
    return rc;
}

int vis_gemm_selftest() {
    if (getenv("VIS_CPUGEMM")) { printf("[vis] selftest 跳过 (CPUGEMM 模式)\n"); return 0; }
    g_vch = getenv("VIS_CH") ? atoi(getenv("VIS_CH")) : 96;
    g_vcha = getenv("VIS_CHA") ? atoi(getenv("VIS_CHA")) : 24;
    if (g_vch <= 0) g_vch = 1 << 20;
    if (g_vcha <= 0) g_vcha = 1 << 20;
    for (int dv = 0; dv < 2; dv++) {
        xpu_set_device(dv);
        if (xpu_malloc(&g_vxA[dv], (size_t)64 * 1024 * 4) || xpu_malloc(&g_vBd[dv], (size_t)256 * 1024) ||
            xpu_malloc(&g_vCs[dv], (size_t)1 << 20) || xpu_malloc(&g_vTc[dv], (size_t)1 << 20) ||
            xpu_malloc(&g_vCa[dv], (size_t)1 << 18) || xpu_malloc(&g_vCr[dv], (size_t)1 << 18) ||
            xpu_malloc(&g_vFd[dv], (size_t)1 << 18) || xpu_malloc(&g_vRd[dv], (size_t)1 << 18)) { printf("[vis] selftest alloc fail\n"); return 1; }
        g_vAsz = (size_t)64 * 1024 * 4; g_vBsz = (size_t)256 * 1024;
        g_vCssz = (size_t)1 << 20; g_vCaszz = (size_t)1 << 18; g_vFdsz = (size_t)1 << 18; g_vRdsz = (size_t)1 << 18;
        g_vctx[dv] = new api::Context(api::Device(api::DeviceType::XPU1, dv));
    }
    int rc = 0;
    rc |= vselftest_one(8, 64, 768, g_vch);    // 与 patch conv 的 K 同量级
    rc |= vselftest_one(8, 64, 72, g_vcha);    // ★ 注意力小 K (ldb=k=cs) —— 与塔内实际路径一致
    printf("[vis] selftest 总结论: %s\n", rc ? "FAIL" : "PASS");
    return rc;
}

