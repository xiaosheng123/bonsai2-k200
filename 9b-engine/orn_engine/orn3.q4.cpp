// orn.cpp — Ornith-1.5-9B (arch=qwen35: 24 层 gated-delta-net 线性注意力 + 8 层全注意力, 32 层主干, blk.32 为 MTP 跳过)
// 昆仑 K200 双芯:
//   * Q8_0 原生布局 (block_q8_0 = fp16 d + int8 qs[32] = 34B/32w) 零重排直接常驻两芯 HBM
//   * 全部 GEMV 在卡上 (自写 int8 内核, 每 32 权重 scale 作用于累加器), 两芯按层字节均衡分担
//   * norm / rope / softmax / silu / delta-net / conv1d 在主机 float (与官方 llama.cpp CPU 路径同序)
// 协议: stdin  "@<max_tokens>@b64:<prompt_b64>\n"  -> stdout "__BEGIN__" / "__TOK__ <hex>" / "__END__"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <functional>
#include <chrono>
#include <unistd.h>
#include <poll.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <csignal>
#include <cerrno>
#include <cstddef>
#include <xpu/runtime.h>
#include "xpu/refactor/nn.h"
#include "xpu/refactor/context/xpu_act_type.h"
namespace api = baidu::xpu::api;

// ============================================================================
// ★ VIS_TOWER_PATCHED: Ornith 视觉塔 (SigLIP-so400m-16-768 + qwen3vl_merger)
//   接法: 服务循环里新增 "!VIS <in.f32> <W> <H>" 命令; 图像 embedding 常驻主机内存,
//   prefill 时把 <|image_pad|> 位置的 token embedding 换成对应图像 embedding 行。
//   ★ 纯增量: 没有图像时 g_img_n==0, 文本路径一个字节都不变。
// ============================================================================
#include "vis.h"
static std::vector<float> g_imgemb;
static int g_img_n = 0, g_img_used = 0, g_img_pad = -1;
// ============================================================================
// ★★ DETVIS_FIX (第26轮): 图像身份指纹 ★★
//   既有缺陷 (已实测复现): 提示词级快照 g_snap / 会话复用只比 token id 序列, 而
//   <|image_pad|> 占位 token 的 id 与图像内容无关 ⇒ 同一问题配不同图 (只要
//   image_pad 个数相同) 会逐 token 相同 ⇒ 直接命中上一张图的 KV+状态, 输出
//   上一张图的答案。实测: big_table.png 之后问 big_shapes.png 同一个"逐格读表"
//   问题, 答出的是上一张表的 A/1/7 B/2/8 C/3/9 (快照 复用603 tok)。
//   修法: 对每张图算一个 64 位内容指纹 (FNV-1a, 覆盖 f32 像素 + W/H + token 数),
//   快照/会话复用必须【指纹也相等】才允许命中, 否则走全量 prefill (数值不变)。
// ============================================================================
static uint64_t g_img_id = 0;
static inline void img_mix_u64(uint64_t v) {
    uint64_t h = g_img_id ? g_img_id : 1469598103934665603ULL;
    for (int i = 0; i < 8; i++) { h ^= (unsigned char)((v >> (i * 8)) & 0xFF); h *= 1099511628211ULL; }
    g_img_id = h;
}
static inline void vis_dev_sync_all() {
    // 视觉塔前后各抽干一次两芯: 视觉塔的 H2D/大 D2H 必须与上一请求/下一请求的
    // 内核队列完全隔开 (历史上"大 D2H 与未落盘 kernel 抢跑"就是这一类 ✗)
    for (int dv = 0; dv < 2; dv++) { xpu_set_device(dv); xpu_wait(); }
}
static void vis_cmd(const char * path, int W, int H) {
    const char * mm = getenv("K200_MMPROJ");
    if (!mm) mm = "/home/caden/orn/mmproj-Ornith-1.5-9B-BF16.gguf";
    if (W <= 0 || H <= 0 || W % 16 || H % 16) { printf("__VIS__ ERR 尺寸 %dx%d\n", W, H); return; }
    if (vis_init(mm)) { printf("__VIS__ ERR 权重\n"); return; }
    std::vector<float> in((size_t)W * H * 3);
    FILE * f = fopen(path, "rb");
    if (!f) { printf("__VIS__ ERR 打不开 %s\n", path); return; }
    size_t got = fread(in.data(), 4, in.size(), f);
    fclose(f);
    if (got != in.size()) { printf("__VIS__ ERR 尺寸不符 %zu/%zu\n", got, in.size()); return; }
    // ★ DETVIS: 指纹先吃像素 (逐字节 FNV-1a) + 尺寸; 喂完再吃 token 数
    {
        uint64_t h = g_img_id ? g_img_id : 1469598103934665603ULL;
        const unsigned char * b = (const unsigned char *)in.data();
        size_t nb = in.size() * 4;
        for (size_t i = 0; i < nb; i++) { h ^= b[i]; h *= 1099511628211ULL; }
        g_img_id = h;
    }
    img_mix_u64((uint64_t)W); img_mix_u64((uint64_t)H);
    vis_dev_sync_all();
    std::vector<float> out; int ntok = 0;
    int rc = vis_forward(in.data(), W, H, out, ntok, getenv("VIS_DUMP"));
    if (rc) { printf("__VIS__ ERR 前向 rc=%d\n", rc); return; }
    vis_dev_sync_all();
    img_mix_u64((uint64_t)ntok); img_mix_u64((uint64_t)out.size());
    g_imgemb.insert(g_imgemb.end(), out.begin(), out.end());   // ★ VIS_APPEND_FIX: 多图累加
    g_img_n += ntok;
    printf("__VIS__ %d %016llx\n", ntok, (unsigned long long)g_img_id);
}

void run_gemv_q8_0(int cl, int co, const void *W, const void *x, void *y, int M, int N);

// ============================================================================
// ★★ 本轮核心: 官方 int8 GEMM (SD-CDNN 引擎) 取代自写 GM2LM 内核 ★★
//   C = alpha*(trans_a)a*(trans_b)b + beta*c ,  a=f32 激活, b=int8 权重, c=f32 出
//   权重布局: trans_b=1 ⇒ B 按 [n][k] = [输出行][输入维] 行优先 = GGUF 原样, ldb=k
//   量化语义 (已用 numcheck 实测钉死): C = sum_k A[k]*q[n][k], 即 max_b=127 时
//   解量化因子恰为 max_b/127 ⇒ 每输出行 scale s_row 由调用方在输出上折回 (y *= s_row)
//   实测带宽: [4096,8192] m=1 → 0.258 ms = 129.86 GB/s (自写 kq8v2 2.58 GB/s 的 50.3 倍)
//   ⚠ 禁用 gemm_int8_maxptr (max_b 传数组的版本): 已实测把 dev0 打进 ERROR
// ============================================================================
namespace baidu { namespace xpu { namespace api {
int gemm_int8(Context* ctx, const bool trans_a, const bool trans_b, int m, int n, int k,
              float alpha, const float* a, int lda, const int8_t* b, float max_b, int ldb,
              float beta, float* c, int ldc);
}}}

// ---- ★ 数值回归钩子: 逐位置 logits dump (K200_LDUMP=<文件>) — 纯落盘, 不参与计算 ----
static FILE *g_ldump = nullptr;

// ---- ★ 计时器 (prefill 批量化 + 主机侧瓶颈定位, K200_PROF=1 时打印) ----
static inline double NOWS_() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
// ---- ★ 中间张量 trace (K200_TRACE=1): 对比逐 token 与批量路径的同一 token 的中间量 ----
static int g_tr = 0;
static std::vector<float> g_trX[2];      // 0 = layer0 grpA 输出(该 token), 1 = 末层 x
static std::vector<std::vector<float> > g_trRows;   // ★ 逐行捕获 (batch 侧一次捕获 T 行)
static std::vector<std::vector<float> > g_trRows2;  // ★ 逐行捕获: layer0 之后的隐状态 x
static std::vector<std::vector<std::vector<float> > > g_trL;  // ★ [层][行] 隐状态 x (前 8 层)
#if defined(__SSE__)
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif


#define CL 4
#define CO 16

// ============================ GGUF ============================
struct GT { std::string name; std::vector<uint64_t> dims; uint32_t type = 0; uint64_t off = 0; };
struct GF {
    FILE *f = nullptr;
    uint64_t data_start = 0, align = 32;
    std::vector<GT> ts;  std::map<std::string, int> idx;
    std::vector<std::string> tokens, merges;
    std::vector<int32_t> ttypes;
    std::map<std::string, std::string> strs;
    std::map<std::string, uint64_t> u64s;

    uint64_t nbytes(const GT &t) const {
        uint64_t n = 1; for (size_t i = 0; i < t.dims.size(); i++) n *= t.dims[i];
        switch (t.type) {
            case 0:  return n * 4;
            case 1:  return n * 2;
            case 28: return n * 2;
            case 8:  return (n / 32) * 34;   // Q8_0
            case 2:  return (n / 32) * 18;   // Q4_0
            case 14: return (n / 256) * 210; // Q6_K
            case 13: return (n / 256) * 176; // Q5_K
            case 12: return (n / 256) * 144; // Q4_K
            default: return 0;
        }
    }
    bool read(const GT &t, void *buf) const {
        if (fseek(f, (long)(data_start + t.off), SEEK_SET)) return false;
        return fread(buf, 1, nbytes(t), f) == nbytes(t);
    }
    const GT *get(const char *n) const { auto it = idx.find(n); return it == idx.end() ? nullptr : &ts[it->second]; }

    bool open(const char *path) {
        f = fopen(path, "rb");
        if (!f) { perror("open"); return false; }
        char mg[4]; if (fread(mg, 1, 4, f) != 4 || memcmp(mg, "GGUF", 4)) { printf("not gguf\n"); return false; }
        auto r32 = [&]() { uint32_t v = 0; if (fread(&v, 4, 1, f) != 1) exit(1); return v; };
        auto r64 = [&]() { uint64_t v = 0; if (fread(&v, 8, 1, f) != 1) exit(1); return v; };
        auto rstr = [&]() { uint64_t n = r64(); std::string s; s.resize(n);
                             if (n && fread(&s[0], 1, n, f) != n) exit(1); return s; };
        r32(); uint64_t nten = r64(), nkv = r64();
        // 通用值读取; 只保留我们需要的
        std::function<void(uint32_t, const std::string &)> rd = [&](uint32_t t, const std::string &key) {
            bool wtok = (key == "tokenizer.ggml.tokens"), wmrg = (key == "tokenizer.ggml.merges");
            bool wtt  = (key == "tokenizer.ggml.token_type");
            switch (t) {
                case 0: { uint8_t v; if (fread(&v,1,1,f)!=1) exit(1); u64s[key]=v; break; }
                case 1: { int8_t v;  if (fread(&v,1,1,f)!=1) exit(1); u64s[key]=(uint64_t)(int64_t)v; break; }
                case 2: { uint16_t v; if (fread(&v,2,1,f)!=1) exit(1); u64s[key]=v; break; }
                case 3: { int16_t v; if (fread(&v,2,1,f)!=1) exit(1); u64s[key]=(uint64_t)(int64_t)v; break; }
                case 4: { u64s[key]=r32(); break; }
                case 5: { int32_t v; if (fread(&v,4,1,f)!=1) exit(1); u64s[key]=(uint64_t)(int64_t)v; break; }
                case 6: { float v; if (fread(&v,4,1,f)!=1) exit(1); u64s[key]=(uint64_t)v; break; }
                case 7: { uint8_t v; if (fread(&v,1,1,f)!=1) exit(1); u64s[key]=v; break; }
                case 8: { strs[key]=rstr(); break; }
                case 10: { u64s[key]=r64(); break; }
                case 11: { int64_t v; if (fread(&v,8,1,f)!=1) exit(1); u64s[key]=(uint64_t)v; break; }
                case 12: { double v; if (fread(&v,8,1,f)!=1) exit(1); u64s[key]=(uint64_t)v; break; }
                case 9: {
                    uint32_t et = r32(); uint64_t n = r64();
                    if (et == 8) { for (uint64_t i = 0; i < n; i++) { std::string s = rstr();
                                     if (wtok) tokens.push_back(s); else if (wmrg) merges.push_back(s); } }
                    else if (et == 4) { for (uint64_t i = 0; i < n; i++) { uint32_t v = r32(); if (wtt) ttypes.push_back((int32_t)v); } }
                    else if (et == 5) { for (uint64_t i = 0; i < n; i++) { int32_t v; if (fread(&v,4,1,f)!=1) exit(1); if (wtt) ttypes.push_back(v); } }
                    else if (et == 10) { for (uint64_t i = 0; i < n; i++) r64(); }
                    else if (et == 0 || et == 1 || et == 7) { for (uint64_t i = 0; i < n; i++) { uint8_t v; if (fread(&v,1,1,f)!=1) exit(1); } }
                    else if (et == 6) { for (uint64_t i = 0; i < n; i++) { float v; if (fread(&v,4,1,f)!=1) exit(1); } }
                    else { printf("array et=%u unsupported (%s)\n", et, key.c_str()); exit(1); }
                    u64s[key] = n; break; }
                default: printf("kv type %u key=%s\n", t, key.c_str()); exit(1);
            }
        };
        for (uint64_t i = 0; i < nkv; i++) {
            std::string k = rstr(); uint32_t t = r32();
            rd(t, k);
        }
        for (uint64_t i = 0; i < nten; i++) {
            GT t; t.name = rstr(); uint32_t nd = r32(); t.dims.resize(nd);
            for (uint32_t d = 0; d < nd; d++) t.dims[d] = r64();
            t.type = r32(); t.off = r64();
            idx[t.name] = (int)ts.size(); ts.push_back(t);
        }
        long pos = ftell(f);
        align = u64s.count("general.alignment") ? u64s["general.alignment"] : 32;
        data_start = (uint64_t)((pos + align - 1) / align * align);
        return true;
    }
};

// ============================ 基础 ============================
static inline float h2f(unsigned short h) {
    unsigned s = (unsigned)(h & 0x8000) << 16;
    int e = (h >> 10) & 0x1f; unsigned m = h & 0x3ff, u;
    if (e == 0) { if (!m) u = s; else { int sh = 0; while (!(m & 0x400)) { m <<= 1; sh++; } m &= 0x3ff;
                 u = s | ((unsigned)(127 - 15 - sh) << 23) | (m << 13); } }
    else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((unsigned)(e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}
static inline void dequant_q8_0(const unsigned char *src, float *dst, uint64_t n) {
    uint64_t nb = n / 32;
    for (uint64_t b = 0; b < nb; b++) {
        float d = h2f(*(const unsigned short *)(src + b * 34));
        const signed char *q = (const signed char *)(src + b * 34 + 2);
        for (int i = 0; i < 32; i++) dst[b * 32 + i] = d * q[i];
    }
}

// 第35轮: K-quant host decode to int8 for gemm_int8 path
void dequant_q4k_to_i8(const unsigned char *src, signed char *dst, int M, int N) {
    int nb = N / 256;
    for (int m = 0; m < M; m++) {
        const unsigned char *row = src + (long long)m * nb * 144;
        signed char *dstrow = dst + (long long)m * N;
        for (int sb = 0; sb < nb; sb++) {
            const unsigned char *b = row + (long long)sb * 144;
            const unsigned char *qs = b + 16;
            for (int i = 0; i < 128; i++) dstrow[sb*256 + i]     = (signed char)(qs[i] & 0x0F);
            for (int i = 0; i < 128; i++) dstrow[sb*256 + 128 + i] = (signed char)(qs[i] >> 4);
        }
    }
}
void dequant_q6k_to_i8(const unsigned char *src, signed char *dst, int M, int N) {
    int nb = N / 256;
    for (int m = 0; m < M; m++) {
        const unsigned char *row = src + (long long)m * nb * 210;
        signed char *dstrow = dst + (long long)m * N;
        for (int sb = 0; sb < nb; sb++) {
            const unsigned char *b = row + (long long)sb * 210;
            const unsigned char *ql = b;
            const unsigned char *qh = b + 128;
            for (int half = 0; half < 2; half++) {
                const unsigned char *qlh = ql + half * 64;
                const unsigned char *qhh = qh + half * 32;
                for (int l = 0; l < 32; l++) {
                    int idx = half * 128 + l;
                    int q1 = (int)((qlh[l]      & 0x0F) | (((qhh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((qlh[l + 32] & 0x0F) | (((qhh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((qlh[l]      >>  4) | (((qhh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((qlh[l + 32] >>  4) | (((qhh[l] >> 6) & 3) << 4)) - 32;
                    dstrow[idx]             = (signed char)q1;
                    dstrow[idx + 32]        = (signed char)q2;
                    dstrow[idx + 64]        = (signed char)q3;
                    dstrow[idx + 96]        = (signed char)q4;
                }
            }
        }
    }
}

static inline float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }
static inline float siluf(float x) { return x * sigmoidf_(x); }
static inline float softplusf_(float x) { return x > 20.f ? x : log1pf(expf(x)); }

static int g_dump = 0;
static inline float absmax(const float *v, int n) { float m = 0; for (int i = 0; i < n; i++) { float a = fabsf(v[i]); if (a > m) m = a; } return m; }
static void dumpv(const char *nm, const float *v, int n) {
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)v[i] * v[i];
    printf("[DUMP] %-20s ne=[%d,1] L2=%.6f v:", nm, n, sqrt(ss));
    for (int i = 0; i < 6; i++) printf(" %.6f", v[i]);
    printf("\n");
}
static void rmsnorm(float *o, const float *x, const float *w, int n, float eps) {
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float r = 1.f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * r * w[i];
}
static void l2norm_head(float *p, int n, float eps) {   // build_gdn_l2_norm: x / sqrt(sum x^2 + eps)
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)p[i] * p[i];
    float r = 1.f / sqrtf((float)ss + eps);
    for (int i = 0; i < n; i++) p[i] *= r;
}

// ============================ tokenizer (byte-level BPE) ============================
static const std::map<uint32_t,uint32_t> &b2u() {
    static std::map<uint32_t,uint32_t> m;
    if (m.empty()) { int n = 0;
        for (int b = 0; b < 256; b++) {
            bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            m[b] = direct ? (uint32_t)b : (uint32_t)(256 + n++);
        } }
    return m;
}
static void u8app(std::string &s, uint32_t c) {
    if (c < 0x80) s += (char)c;
    else if (c < 0x800) { s += (char)(0xC0|(c>>6)); s += (char)(0x80|(c&0x3F)); }
    else if (c < 0x10000) { s += (char)(0xE0|(c>>12)); s += (char)(0x80|((c>>6)&0x3F)); s += (char)(0x80|(c&0x3F)); }
    else { s += (char)(0xF0|(c>>18)); s += (char)(0x80|((c>>12)&0x3F)); s += (char)(0x80|((c>>6)&0x3F)); s += (char)(0x80|(c&0x3F)); }
}
static std::string bytes2uni(const std::string &raw) {
    std::string o; for (size_t i = 0; i < raw.size(); i++) u8app(o, b2u().at((unsigned char)raw[i])); return o;
}
static std::string uni2bytes(const std::string &u) {
    static std::map<uint32_t,uint32_t> rev;
    if (rev.empty()) { for (std::map<uint32_t,uint32_t>::const_iterator kv = b2u().begin(); kv != b2u().end(); ++kv) rev[kv->second] = kv->first; }
    std::string o; size_t i = 0;
    while (i < u.size()) {
        unsigned char c = u[i]; uint32_t cp = c; int len = 1;
        if (c >= 0xF0) { cp = c & 0x07; len = 4; } else if (c >= 0xE0) { cp = c & 0x0F; len = 3; }
        else if (c >= 0xC0) { cp = c & 0x1F; len = 2; }
        for (int k = 1; k < len && i + (size_t)k < u.size(); k++) cp = (cp << 6) | (u[i+k] & 0x3F);
        std::map<uint32_t,uint32_t>::iterator it = rev.find(cp);
        if (it == rev.end()) o += u.substr(i, len); else o += (char)it->second;
        i += len;
    }
    return o;
}
static std::vector<std::string> split_utf8(const std::string &s) {
    std::vector<std::string> v; size_t i = 0;
    while (i < s.size()) { unsigned char c = s[i]; int len = 1;
        if (c >= 0xF0) len = 4; else if (c >= 0xE0) len = 3; else if (c >= 0xC0) len = 2;
        v.push_back(s.substr(i, len)); i += len; }
    return v;
}
struct Tokenizer {
    std::map<std::string,int> vocab, rank;
    int bos = 0, eos = 1;
    std::vector<std::pair<std::string,int> > specials;
    bool digits_single = true;
    void init(const std::vector<std::string> &toks, const std::vector<std::string> &merges,
              int b, int e, const std::vector<int32_t> *ttypes, const std::string &pre) {
        for (size_t i = 0; i < toks.size(); i++) vocab[toks[i]] = (int)i;
        int r = 0; for (size_t i = 0; i < merges.size(); i++) { if (!rank.count(merges[i])) rank[merges[i]] = r; r++; }
        bos = b; eos = e;
        bool have_tt = ttypes && ttypes->size() == toks.size();
        for (size_t i = 0; i < toks.size(); i++) {
            bool sp = false;
            if (have_tt) { int tt = (*ttypes)[i]; sp = (tt == 2 || tt == 3 || tt == 4); }
            else { const std::string &s = toks[i];
                   sp = (s.size() >= 3 && s.size() <= 64 && s[0] == '<' && s[s.size()-1] == '>' && s.find(' ') == std::string::npos); }
            if (sp) specials.push_back(std::make_pair(toks[i], (int)i));
        }
        std::sort(specials.begin(), specials.end(),
                  [](const std::pair<std::string,int> &a, const std::pair<std::string,int> &b) { return a.first.size() > b.first.size(); });
        digits_single = (pre.find("qwen") != std::string::npos);
    }
    std::vector<int> bpe(const std::string &s) const {
        std::vector<std::string> sym = split_utf8(s);
        while (sym.size() > 1) {
            int best = -1, br = INT32_MAX;
            for (size_t k = 0; k + 1 < sym.size(); k++) {
                std::map<std::string,int>::const_iterator it = rank.find(sym[k] + " " + sym[k+1]);
                if (it != rank.end() && it->second < br) { br = it->second; best = (int)k; }
            }
            if (best < 0) break;
            sym[best] += sym[best+1];
            sym.erase(sym.begin() + best + 1);
        }
        std::vector<int> ids;
        for (size_t i = 0; i < sym.size(); i++) { std::map<std::string,int>::const_iterator it = vocab.find(sym[i]); ids.push_back(it == vocab.end() ? 0 : it->second); }
        return ids;
    }
    std::vector<int> encode(const std::string &text) const {
        std::vector<int> out; size_t i = 0, n = text.size();
        auto u8 = [](unsigned char c) { return c >= 0xF0 ? 4 : (c >= 0xE0 ? 3 : (c >= 0xC0 ? 2 : 1)); };
        auto isL = [](unsigned char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c >= 0x80; };
        auto isD = [](unsigned char c) { return c >= '0' && c <= '9'; };
        auto isS = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
        while (i < n) {
            if (text[i] == '<') {
                bool hit = false;
                for (size_t k = 0; k < specials.size(); k++) {
                    size_t L = specials[k].first.size();
                    if (i + L <= n && text.compare(i, L, specials[k].first) == 0) { out.push_back(specials[k].second); i += L; hit = true; break; }
                }
                if (hit) continue;
            }
            size_t st = i;
            if (text[i] == ' ' && i + 1 < n && !isS(text[i + 1])) i++;
            size_t j = i;
            if (i < n && text[i] == '\'') {
                static const char *sfx[7] = {"'s","'t","'re","'ve","'m","'ll","'d"};
                size_t best = 0;
                for (int k = 0; k < 7; k++) { size_t L = strlen(sfx[k]);
                    if (i + L <= n && text.compare(i, L, sfx[k]) == 0 && L > best) best = L; }
                if (best) j = i + best;
            }
            if (j == i) {
                if (i < n && isL((unsigned char)text[i])) {
                    while (j < n && isL((unsigned char)text[j])) j += u8((unsigned char)text[j]);
                } else if (i < n && isD(text[i])) {
                    j++;
                    if (!digits_single) while (j < n && isD(text[j])) j++;
                } else if (i < n && !isS(text[i])) {
                    while (j < n && !isS(text[j]) && !isL((unsigned char)text[j]) && !isD(text[j])) j += u8((unsigned char)text[j]);
                } else {
                    while (j < n && isS(text[j])) j++;
                }
            }
            if (j <= i) j = i + u8((unsigned char)text[i]);
            std::string ch = text.substr(st, j - st);
            std::vector<int> v = bpe(bytes2uni(ch));
            out.insert(out.end(), v.begin(), v.end());
            i = j;
        }
        return out;
    }
    std::string decode_token(int id, const std::vector<std::string> &toks) const {
        if (id < 0 || id >= (int)toks.size()) return "";
        return uni2bytes(toks[id]);
    }
};

// ============================ 模型常量 (Ornith-1.5-9B / qwen35) ============================
static const int EMB = 4096, NFF = 12288, NH = 16, NKV = 4, HD = 256, NROT = 64;
static const int NLAYER = 32;                 // 主干 32 层; blk.32 = MTP, 不加载不执行
static const int KHD = 128, NKH = 16, NVH = 32, VHD = 128, CVD = 8192, CVK = 4;
static const float EPS = 1e-6f;
static const float ROPE_BASE = 1e7f;
// ★★ 第30轮(问题⑥): 上下文上限 MAXT 改成【运行期可配】, 默认 4096 (原来硬编码 1024)。
//   MAXT = 提示词 + 生成 的【总】上下文上限: 1024 对 agent/工具场景太小 ——
//   system(含工具定义) + 多轮历史 + tool 结果 轻松超 1024 ⇒ 要么生成 0 token, 要么留给
//   生成的空间只剩几十 token。只在 vector 尺寸/算术里用到, 不做编译期数组界 ⇒ 可运行期改。
//   K200_MAXT 可覆盖 (512..16384)。
static int MAXT = 8192;

static bool is_recr(int il) { return ((il + 1) % 4) != 0; }   // 与 llama.cpp full_attention_interval=4 一致

// ============================ 权重 ============================
// ★★ 双芯行分裂: 每个矩阵的行按 [0,Mh) 放 dev0, [Mh,M) 放 dev1, 两次 launch 并发执行 ★★
//   依据 (实测): 单芯 GM2LM 有效带宽 2.71 GB/s; 两芯并发 = 5.38 GB/s (1.99x)
//   依据 (实测): 每次 CLUSTER launch 固定 3.3 ms 主机开销 => 必须靠"拼接同 x 矩阵"把 launch 数压回去
struct WDev {
    void *d[2] = {nullptr, nullptr};
    int   dev[2] = {-1, -1};
    int   mp[2] = {0, 0};            // 每片行数
    int   M = 0, N = 0;              // 总行数 / 输入维 (N 各张量必须相同才能拼)
    size_t bytes = 0, rowbytes = 0;
    bool  v2 = false;
    bool  i8row = false;             // ★ true = 官方 gemm_int8 路径 (d[p] = mp[p] 行 x N 字节 int8)
    std::vector<float> rs;           // ★ 每输出行 scale (长度 M; gemm_int8 后折回)
    int   nsplit() const { return d[1] ? 2 : 1; }
};
static const long long SPLIT_MIN_BYTES = 10LL << 20;   // 小于此字节数的矩阵不拆 (拆了多付 3.3ms launch 不划算)
static int g_nosplit = 0;                              // K200_NOSPLIT=1 关闭分裂 (诊断/A-B 用)
static int g_i8row = 1;                                // ★ 1=官方 gemm_int8(每行 int8+行 scale); 0=自写 kq8v2 回退
static api::Context *g_ctx[2] = {nullptr, nullptr};    // ★ 每芯一个 SD-CDNN 上下文
static uint64_t g_hbm[2] = {0, 0};

struct Layer {
    bool recr = false;
    std::vector<float> attn_norm, post_norm;
    // 线性注意力 (gated delta net)
    WDev grp_a, grp_mid;                     // grp_a = 同读 attn_norm 输出的矩阵拼接; grp_mid = ssm_out/o
    std::vector<float> alpha_w, beta_w;      // [32 x 4096] host (小, 不值得上卡)
    std::vector<float> conv1d;               // [c*4 + tap] = 4 per channel
    std::vector<float> ssm_a, ssm_dt, ssm_norm;
    // 全注意力
    std::vector<float> q_norm, k_norm;
    std::vector<float> Kc, Vc;   // KV cache (仅注意层, MAXT*NKV*HD)
    // MLP
    WDev grp_ffn;                            // {ffn_gate, ffn_up} (同读 post_norm 输出)
    WDev ffn_down;
};

static std::vector<float> load_f32_host(GF &g, const char *name) {
    const GT *t = g.get(name);
    if (!t) { printf("FATAL: 缺张量 %s\n", name); exit(1); }
    uint64_t n = 1; for (size_t i = 0; i < t->dims.size(); i++) n *= t->dims[i];
    std::vector<char> raw(g.nbytes(*t));
    if (!g.read(*t, raw.data())) { printf("FATAL: 读 %s 失败\n", name); exit(1); }
    std::vector<float> v(n);
    if (t->type == 0) memcpy(v.data(), raw.data(), n * 4);
    else if (t->type == 1) { const unsigned short *p = (const unsigned short *)raw.data(); for (uint64_t i = 0; i < n; i++) v[i] = h2f(p[i]); }
    else if (t->type == 28) { const unsigned short *p = (const unsigned short *)raw.data(); for (uint64_t i = 0; i < n; i++) { unsigned u = ((unsigned)p[i]) << 16; float f; memcpy(&f, &u, 4); v[i] = f; } }
    else { printf("FATAL: %s type=%u 不支持 (只处理 F32/F16/BF16 的标量/norm 张量)\n", name, t->type); exit(1); }
    return v;
}
// 小张量 (ssm_alpha/beta) 从 Q8_0 解到主机
static void load_q8_host(GF &g, const char *name, std::vector<float> &out, int &M, int &N) {
    const GT *t = g.get(name);
    if (!t) { printf("FATAL: 缺张量 %s\n", name); exit(1); }
    N = (int)t->dims[0]; M = (int)t->dims[1];
    std::vector<unsigned char> raw(g.nbytes(*t));
    if (!g.read(*t, raw.data())) { printf("FATAL: 读 %s 失败\n", name); exit(1); }
    if (t->type == 8) { out.resize((size_t)M * N); dequant_q8_0(raw.data(), out.data(), (uint64_t)M * N); return; }
    if (t->type == 0) { out.resize((size_t)M * N); memcpy(out.data(), raw.data(), (size_t)M * N * 4); return; }
    printf("FATAL: %s type=%u\n", name, t->type); exit(1);
}

// ---- V2 重排: 每 4096 权重 -> [4096 int8][128 x f16 scale] (16B 对齐); 行步长 = N + N/32*2 ----
static void v2_pack(const unsigned char *raw, unsigned char *dst, int M, int N) {
    int nb = N / 32;
    size_t rs = (size_t)N + (size_t)nb * 2;
    for (int m = 0; m < M; m++) {
        const unsigned char *src = raw + (size_t)m * nb * 34;
        unsigned char *drow = dst + (size_t)m * rs;
        for (int c = 0; c < N / 4096; c++) {
            unsigned char *qc = drow + (size_t)c * 4352;
            unsigned char *scc = qc + 4096;
            for (int b = 0; b < 128; b++) {
                const unsigned char *sp = src + (size_t)(c * 128 + b) * 34;
                memcpy(scc + b * 2, sp, 2);
                memcpy(qc + b * 32, sp + 2, 32);
            }
        }
    }
}
// ---- 把 V2 主机缓冲按行上卡; 自动决定是否两芯行分裂 ----
static WDev upload_v2_host(const std::vector<unsigned char> &packed, int M, int N, bool allow_split) {
    WDev w; w.N = N; w.M = M; w.bytes = packed.size(); w.v2 = true;
    int nb = N / 32;
    w.rowbytes = (size_t)N + (size_t)nb * 2;
    if ((int64_t)packed.size() != (int64_t)M * (int64_t)w.rowbytes) {
        printf("FATAL: 打包尺寸不符 M=%d N=%d bytes=%zu 期望=%lld\n", M, N, packed.size(),
               (long long)((int64_t)M * (int64_t)w.rowbytes)); exit(1);
    }
    int mh = M, ns = 1;
    if (allow_split && !g_nosplit && (long long)packed.size() >= SPLIT_MIN_BYTES && M >= 64 && nb >= 32) { mh = M / 2; ns = 2; }
    w.dev[0] = 0; w.mp[0] = mh;
    w.dev[1] = (ns == 2) ? 1 : -1; w.mp[1] = (ns == 2) ? (M - mh) : 0;
    for (int p = 0; p < 2; p++) {
        if (w.mp[p] <= 0) { w.d[p] = nullptr; continue; }
        size_t by = (size_t)w.mp[p] * w.rowbytes;
        size_t src_off = (size_t)(p == 0 ? 0 : mh) * w.rowbytes;
        xpu_set_device(w.dev[p]);
        if (xpu_malloc(&w.d[p], by)) {
            printf("FATAL: HBM 分配 %.1f MB 失败 chip%d [已用 %.0f MB]\n", by / 1048576.0, w.dev[p], g_hbm[w.dev[p]] / 1048576.0);
            exit(1);
        }
        if (xpu_memcpy(w.d[p], packed.data() + src_off, by, XPU_HOST_TO_DEVICE)) { printf("FATAL: H2D 失败 (M=%d)\n", w.mp[p]); exit(1); }
        xpu_wait();
        g_hbm[w.dev[p]] += by;
    }
    return w;
}
// ============================================================================
// ★ 权重量化方案: 每输出行 (per-output-channel) 对称 int8 + 每行一个 float scale
//   s_row = max|w_row| / 127 ;  q = round(w / s_row) ,  截断到 [-127,127]
//   为什么改成这个: 官方 int8 系列只有 max_a/max_b/max_c 三个标量, 无法表达
//   Q8_0 的"每 32 权重一个 scale"; 每行一个 scale 是官方能承受的最细粒度.
//   分块解量化 (每次 2048 行) 是为了把主机内存峰值压在 ~2.1 GB (本机只有 7 GB RAM).
// ============================================================================
static void q8raw_to_rowi8(const unsigned char *raw, int M, int N,
                           std::vector<signed char> &q, std::vector<float> &rs) {
    const int CH = 2048;
    const size_t nb = (size_t)(N / 32);
    q.resize((size_t)M * N); rs.resize(M);
    std::vector<float> tmp((size_t)CH * N);
    for (int m0 = 0; m0 < M; m0 += CH) {
        int mc = (M - m0 < CH) ? (M - m0) : CH;
        dequant_q8_0(raw + (size_t)m0 * nb * 34, tmp.data(), (uint64_t)mc * N);
        for (int i = 0; i < mc; i++) {
            const float *row = &tmp[(size_t)i * N];
            float am = 0;
            for (int n = 0; n < N; n++) { float a = fabsf(row[n]); if (a > am) am = a; }
            float s = am / 127.f; if (s <= 1e-30f) s = 1e-8f;
            rs[m0 + i] = s;
            signed char *dq = &q[(size_t)(m0 + i) * N];
            for (int n = 0; n < N; n++) {
                int v = (int)lrintf(row[n] / s);
                if (v > 127) v = 127; else if (v < -127) v = -127;
                dq[n] = (signed char)v;
            }
        }
    }
}
// ---- 每行 int8 主机缓冲按行上卡; 行分裂逻辑与 V2 路径一致 (int8 每行恰好 N 字节, 天然对齐) ----
static WDev upload_i8_host(const std::vector<signed char> &q, const std::vector<float> &rs,
                           int M, int N, bool allow_split) {
    WDev w; w.N = N; w.M = M; w.i8row = true; w.rs = rs;
    w.bytes = (size_t)M * N; w.rowbytes = (size_t)N;
    if (q.size() != w.bytes || rs.size() != (size_t)M) {
        printf("FATAL: int8 缓冲尺寸不符 M=%d N=%d q=%zu rs=%zu\n", M, N, q.size(), rs.size()); exit(1);
    }
    int mh = M, ns = 1;
    if (allow_split && !g_nosplit && (long long)w.bytes >= SPLIT_MIN_BYTES && M >= 64) { mh = M / 2; ns = 2; }
    w.dev[0] = 0; w.mp[0] = mh;
    w.dev[1] = (ns == 2) ? 1 : -1; w.mp[1] = (ns == 2) ? (M - mh) : 0;
    for (int p = 0; p < 2; p++) {
        if (w.mp[p] <= 0) { w.d[p] = nullptr; continue; }
        size_t by = (size_t)w.mp[p] * w.rowbytes;
        size_t src_off = (size_t)(p == 0 ? 0 : mh) * w.rowbytes;
        xpu_set_device(w.dev[p]);
        if (xpu_malloc(&w.d[p], by)) {
            printf("FATAL: HBM 分配 %.1f MB 失败 chip%d [已用 %.0f MB]\n", by / 1048576.0, w.dev[p], g_hbm[w.dev[p]] / 1048576.0);
            exit(1);
        }
        if (xpu_memcpy(w.d[p], q.data() + src_off, by, XPU_HOST_TO_DEVICE)) { printf("FATAL: int8 H2D 失败 (M=%d)\n", w.mp[p]); exit(1); }
        xpu_wait();
        g_hbm[w.dev[p]] += by;
    }
    return w;
}

// ---- 单个 Q8_0 张量上卡 (统一入口; 按 g_i8row 决定量化路径) ----
static WDev up_q8(GF &g, const char *name, int dev_unused, bool allow_split = true) {
    (void)dev_unused;
    const GT *t = g.get(name);
    if (!t) { printf("FATAL: 缺张量 %s\n", name); exit(1); }
    int N = (int)t->dims[0], M = (int)t->dims[1];
    if (t->type == 12 || t->type == 14) {
        if (N % 256 != 0) { printf("FATAL: %s N=%d not 256-multiple\n", name, N); exit(1); }
        std::vector<unsigned char> rk(g.nbytes(*t));
        if (!g.read(*t, rk.data())) { printf("FATAL: read %s failed\n", name); exit(1); }
        size_t tb = (size_t)M * N;
        std::vector<signed char> q(tb, 0);
        std::vector<float> rs(N, 1.0f);
        if (t->type == 12) dequant_q4k_to_i8(rk.data(), q.data(), M, N);
        else               dequant_q6k_to_i8(rk.data(), q.data(), M, N);
        fprintf(stderr, "[KQ-I8] %s type=%d M=%d N=%d q=%zu %.1fMB\n",
                name, (int)t->type, M, N, q.size(), tb/1048576.0);
        return upload_i8_host(q, rs, M, N, allow_split);
    }
    if (t->type != 8) { printf("FATAL: %s type=%u 不是 Q8_0\n", name, t->type); exit(1); }
    std::vector<unsigned char> raw(g.nbytes(*t));
    if (!g.read(*t, raw.data())) { printf("FATAL: 读 %s 失败\n", name); exit(1); }
    if (g_i8row) {                                   // ★ 官方 gemm_int8 路径
        std::vector<signed char> q; std::vector<float> rs;
        q8raw_to_rowi8(raw.data(), M, N, q, rs);
        return upload_i8_host(q, rs, M, N, allow_split);
    }
    if (N % 1024 != 0) { printf("FATAL: %s N=%d 不是 1024 的倍数 (内核限制)\n", name, N); exit(1); }
    int nb = N / 32;
    std::vector<unsigned char> packed((size_t)M * ((size_t)N + (size_t)nb * 2));
    v2_pack(raw.data(), packed.data(), M, N);
    return upload_v2_host(packed, M, N, allow_split);
}

// ---- 同 N 的多个 Q8_0 张量拼成一个大矩阵 (它们用同一个 x) -> 一次 launch ----
//   ★ g_i8row=1 时拼的是"每行 int8"缓冲 (M*N 字节) + 每行 scale; 否则是老 V2 布局
struct Pack {
    int N = 0, M = 0;
    std::vector<unsigned char> host;
    std::vector<signed char> host8;
    std::vector<float> rs;
    std::vector<int> segM, segOff;
    void add(GF &g, const char *name) {
        const GT *t = g.get(name);
        if (!t) { printf("FATAL: 缺张量 %s\n", name); exit(1); }
        if (t->type != 8) { printf("FATAL: %s type=%u 不是 Q8_0\n", name, t->type); exit(1); }
        int n = (int)t->dims[0], m = (int)t->dims[1];
        if (M == 0) {
            N = n;
            if (!g_i8row && N % 1024 != 0) { printf("FATAL: %s N=%d 不是 1024 的倍数\n", name, N); exit(1); }
        } else if (n != N) { printf("FATAL: %s N=%d != 拼接组 N=%d\n", name, n, N); exit(1); }
        std::vector<unsigned char> raw(g.nbytes(*t));
        if (!g.read(*t, raw.data())) { printf("FATAL: 读 %s 失败\n", name); exit(1); }
        segM.push_back(m); segOff.push_back(M);
        if (g_i8row) {
            std::vector<signed char> q; std::vector<float> r2;
            q8raw_to_rowi8(raw.data(), m, N, q, r2);
            host8.resize((size_t)(M + m) * N);
            memcpy(host8.data() + (size_t)M * N, q.data(), (size_t)m * (size_t)N);
            rs.resize(M + m);
            memcpy(&rs[M], r2.data(), (size_t)m * 4);
        } else {
            int nb = N / 32;
            size_t rs_ = (size_t)N + (size_t)nb * 2;
            host.resize((size_t)(M + m) * rs_);
            v2_pack(raw.data(), host.data() + (size_t)M * rs_, m, N);
        }
        M += m;
    }
    WDev finish(bool allow_split = true) {
        return g_i8row ? upload_i8_host(host8, rs, M, N, allow_split) : upload_v2_host(host, M, N, allow_split);
    }
    void release() { std::vector<unsigned char>().swap(host); std::vector<signed char>().swap(host8); std::vector<float>().swap(rs); }
};

// ============================ 卡上 GEMV ============================
static void *g_dx[2] = {nullptr, nullptr}, *g_dy[2] = {nullptr, nullptr};
static void *g_dxq[2] = {nullptr, nullptr}, *g_dxs[2] = {nullptr, nullptr};
static uint64_t g_gemv_n = 0;
static uint64_t g_i8_n = 0;
static double g_i8_t = 0;
static uint64_t g_fwd_n = 0;      // forward 调用次数 (用于算"实测有效读带宽")
// ---- ★ 主机侧分桶计时 (单位: 秒; 每桶都已扣除桶内等卡的 gemm 时间 => 纯主机时间) ----
static int    g_prof = 0;
static double g_p_emb = 0, g_p_nrm = 0, g_p_ssm = 0, g_p_att = 0, g_p_lay = 0, g_p_lm = 0, g_p_arg = 0;
static double g_p_conv = 0, g_p_dn = 0, g_p_gn = 0, g_p_ab = 0;   // ★ SSM 内部细分
static double g_g_h2d = 0, g_g_launch = 0, g_g_wait = 0, g_g_d2h = 0;  // ★ gemm 四段: H2D/launch/wait/D2H
static double g_g_n = 0;
static uint64_t g_bn = 0;         // ★ 批量 gemm 调用次数
// ---- ★ prefill 批量化: 单批最大 token 数 / 触发批量的最小 token 数 (env 可调) ----
#define BATCH_MAXT 64
#define BATCH_MAXN 12288      // ★ 批量 gemm 里最大的输入维 (ffn_down 的 N=12288), 决定卡上 x 缓冲大小
static int g_bmax = 64, g_bmin = 16;
// ---- ★ 主机侧浮点环境开关 (dnbench 实测: delta-net 的 denormal 中间值导致 37x 惩罚) ----
static int g_ftz = 1;         // K200_FTZ=0 关闭 FTZ/DAZ (用于 A/B)
static int g_omp = 1;         // K200_OMP=0 关闭主机侧 OpenMP 并行 (逐迭代独立 => 数值不变)
static inline void enable_ftz_daz() {
#if defined(__SSE__)
    _mm_setcsr(_mm_getcsr() | 0x8040);     // MXCSR bit15=FTZ, bit6=DAZ
#endif
}

void run_gemv_q8_0i(int cl, int co, const void *W, const void *xq, const void *xs, void *y, int M, int N);
void run_gemv_q8v2(int cl, int co, const void *W, const void *xq, const void *xs, void *y, int M, int N);
void kq8_info(int *o);
static int g_cl = 8, g_co = 16, g_xi8 = 1;   // ★ co 必须是硬件真实每-cluster核数 16
static double g_ktime = 0, g_xquant = 0, g_d2h = 0;

// ================= 哨兵法诊断 =================
static int g_sent = 0;                 // K200_SENT=1 打开
static const char *g_gname = "?";
static char g_nb[64];
static int g_calls = 0;                // gemv 调用序号
static uint32_t g_sentpat = 0x7FC0DEADu;   // NaN payload 哨兵
static std::vector<float> g_sentbuf;
static std::vector<float> g_backbuf;
static void sent_fill(int dv, int M) {
    if ((int)g_sentbuf.size() < M) g_sentbuf.resize(M);
    for (int i = 0; i < M; i++) memcpy(&g_sentbuf[i], &g_sentpat, 4);
    xpu_set_device(dv);
    xpu_memcpy(g_dy[dv], g_sentbuf.data(), (size_t)M * 4, XPU_HOST_TO_DEVICE);
    xpu_wait();
    if (g_sent >= 3) {   // 回读验证: H2D 到底写进去没有
        g_backbuf.assign(M, 0.f);
        xpu_memcpy(g_backbuf.data(), g_dy[dv], (size_t)M * 4, XPU_DEVICE_TO_HOST);
        xpu_wait();
        int ns = 0, fns = -1;
        for (int i = 0; i < M; i++) { uint32_t b; memcpy(&b, &g_backbuf[i], 4); if (b == g_sentpat) { ns++; if (fns < 0) fns = i; } }
        printf("[FILL] dev=%d M=%d H2D后回读: 哨兵=%d/%d 首非哨兵=%d\n", dv, M, ns, M, fns);
    }
}
static std::vector<float> g_chk2;
static void rle(const char *what, const float *y, int M) {
    auto cls = [&](int i)->char { uint32_t b; memcpy(&b, &y[i], 4);
        if (b == g_sentpat) return 'S'; if (b == 0) return '0';
        if (!std::isfinite(y[i])) return 'N'; return '.'; };
    printf("[RLE %s] ", what);
    int i = 0;
    while (i < M) { char c = cls(i); int j = i; while (j < M && cls(j) == c) j++;
        if (j - i >= 8 || c != '.') printf("%c[%d,%d) ", c, i, j);
        i = j; }
    printf("\n");
}
static void sent_check(int dv, const float *y, int M, int M_, int N, const char *tag) {
    int ns = 0, fns = -1, nf = 0, fnf = -1;
    for (int i = 0; i < M_; i++) {
        uint32_t b; memcpy(&b, &y[i], 4);
        if (b == g_sentpat) { ns++; if (fns < 0) fns = i; }
        if (!std::isfinite(y[i])) { nf++; if (fnf < 0) fnf = i; }
    }
    printf("[SENT] call#%d dev=%d M=%d N=%d tag=%s | 仍是哨兵=%d 首=%d | 非有限=%d 首=%d\n",
           g_calls, dv, M, N, tag, ns, fns, nf, fnf);
    if (g_sent >= 4) {
        rle("bigD2H", y, M_);
        g_chk2.assign(M_, 0.f);
        const int CH = 64;                       // 每次 64 float
        for (int i = 0; i < M_; i += CH) { int c = (i + CH <= M_) ? CH : M_ - i;
            xpu_memcpy(&g_chk2[i], (char *)g_dy[dv] + (size_t)i * 4, (size_t)c * 4, XPU_DEVICE_TO_HOST); }
        xpu_wait();
        rle("chunkD2H", g_chk2.data(), M_);
    }
}
// ---- x 量化: 每 32 权重一个 scale (主机侧) ----
static void quant_x(const WDev &w, const float *x, std::vector<signed char> &xq, std::vector<float> &xs) {
    int nb = w.N / 32;
    xq.resize(w.N); xs.resize(nb);
    auto tq = std::chrono::steady_clock::now();
    for (int b = 0; b < nb; b++) {
        float amax = 0;
        for (int i = 0; i < 32; i++) { float a = fabsf(x[b * 32 + i]); if (a > amax) amax = a; }
        float s = amax / 127.f; if (s <= 1e-12f) s = 1e-8f;
        xs[b] = s;
        for (int i = 0; i < 32; i++) {
            int v = (int)lrintf(x[b * 32 + i] / s);
            if (v > 127) v = 127; else if (v < -128) v = -128;
            xq[b * 32 + i] = (signed char)v;
        }
    }
    g_xquant += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tq).count();
}
// ---- 单片 launch (诊断/selftest 用) ----
static void gemv_single(int dv, const WDev &w, const float *x, float *y, int cl, int co) {
    if (w.i8row) {                                   // ★ 官方 gemm_int8 单片
        xpu_set_device(dv);
        xpu_memcpy(g_dx[dv], x, (size_t)w.N * 4, XPU_HOST_TO_DEVICE);
        int r = api::gemm_int8(g_ctx[dv], false, true, 1, w.mp[0], w.N, 1.f,
                               (const float *)g_dx[dv], w.N,
                               (const int8_t *)w.d[0], 127.f, w.N,
                               0.f, (float *)g_dy[dv], w.mp[0]);
        if (r || xpu_wait()) { printf("FATAL: gemv_single gemm_int8 r=%d (dev%d M=%d N=%d)\n", r, dv, w.mp[0], w.N); exit(1); }
        xpu_memcpy(y, g_dy[dv], (size_t)w.mp[0] * 4, XPU_DEVICE_TO_HOST);
        xpu_wait();
        for (int i = 0; i < w.mp[0]; i++) y[i] *= w.rs[i];
        g_calls++; g_gemv_n++; g_i8_n++;
        return;
    }
    static std::vector<signed char> xq; static std::vector<float> xs;
    int nb = w.N / 32;
    quant_x(w, x, xq, xs);
    xpu_set_device(dv);
    g_calls++;
    xpu_memcpy(g_dxq[dv], xq.data(), (size_t)w.N, XPU_HOST_TO_DEVICE);
    xpu_memcpy(g_dxs[dv], xs.data(), (size_t)nb * 4, XPU_HOST_TO_DEVICE);
    if (w.v2) run_gemv_q8v2(cl, co, w.d[0], g_dxq[dv], g_dxs[dv], g_dy[dv], w.mp[0], w.N);
    else      run_gemv_q8_0i(cl, co, w.d[0], g_dxq[dv], g_dxs[dv], g_dy[dv], w.mp[0], w.N);
    if (xpu_wait()) { printf("FATAL: kernel wait (dev%d M=%d N=%d)\n", dv, w.mp[0], w.N); exit(1); }
    xpu_memcpy(y, g_dy[dv], (size_t)w.mp[0] * 4, XPU_DEVICE_TO_HOST);
    xpu_wait();
    g_gemv_n++;
}
// ---- ★ 官方 gemm_int8: 双芯各 launch 自己的行片 (真并发), 再拼回 y 并折回每行 scale ----
static void gemv_i8(const WDev &w, const float *x, float *y) {
    int ns = w.nsplit();
    double t0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    for (int p = 0; p < ns; p++) {                   // ① 两片都先提交 (并发跑)
        int dv = w.dev[p];
        xpu_set_device(dv);
        double _th = NOWS_();
        xpu_memcpy(g_dx[dv], x, (size_t)w.N * 4, XPU_HOST_TO_DEVICE);
        double _tl = NOWS_();
        int r = api::gemm_int8(g_ctx[dv], false, true, 1, w.mp[p], w.N, 1.f,
                               (const float *)g_dx[dv], w.N,
                               (const int8_t *)w.d[p], 127.f, w.N,
                               0.f, (float *)g_dy[dv], w.mp[p]);
        g_g_h2d += _tl - _th; g_g_launch += NOWS_() - _tl; g_g_n += 1;
        if (r) { printf("FATAL: gemm_int8 r=%d (chip%d M=%d N=%d)\n", r, dv, w.mp[p], w.N); exit(1); }
    }
    int off = 0;
    for (int p = 0; p < ns; p++) {                   // ② 等结果 + 取回 + 折 scale
        int dv = w.dev[p];
        xpu_set_device(dv);
        double _tw = NOWS_();
        if (xpu_wait()) { printf("FATAL: gemm_int8 wait (chip%d M=%d N=%d)\n", dv, w.mp[p], w.N); exit(1); }
        double _td = NOWS_();
        g_g_wait += _td - _tw;
        auto td = std::chrono::steady_clock::now();
        xpu_memcpy(y + off, g_dy[dv], (size_t)w.mp[p] * 4, XPU_DEVICE_TO_HOST);
        xpu_wait();
        g_g_d2h += NOWS_() - _td;
        g_d2h += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count();
        for (int i = 0; i < w.mp[p]; i++) y[off + i] *= w.rs[off + i];    // ★ 每输出行 scale
        off += w.mp[p];
        g_calls++;
    }
    g_i8_n++; g_i8_t += std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() - t0;
    g_gemv_n++;
}
// ============================================================================
// ★★ prefill 批量化: 线性层每个权重矩阵只读一次 (m = 本批 token 数 T) ★★
//   数值补偿 (mbatch2.cpp 实测): 官方 gemm_int8 内部把 f32 激活 a 按 per-tensor
//   int8 量化 => 一次 m=T 调用里所有 token 共用一个 max_a, 直接批量与逐 token 路径
//   差 0.9257% relrms (行幅度接近时) / 1.2668% (行幅度差 16 倍时), 会改黄金原文.
//   这里对激活做"行归一化":
//       A'[t,:] = A[t,:] * f_t ,  f_t = M0 / max|A[t,:]| ,  M0 = 全批最大行幅值
//   => 算子的 per-tensor scale 变成 M0/127, 每行等效量化步长 = (M0/127)/f_t
//      = max_t/127 = 逐 token 路径的 per-row 步长; 输出再乘 max_t/M0 还原.
//   实测 (mbatch2): 归一化后 m=64 vs m=1 relrms = 0.0000086% (maxabs 5.7e-6, 1 ulp 级),
//   而不归一化是 0.9257%. gemm 带宽: n=12288 k=4096 m=1 0.374ms -> m=64 0.460ms
//   (52x 每 token 的权重读取效率).
// ============================================================================
static std::vector<float> g_bnorm, g_bmx, g_bYbh;
static void gemv_batch(const WDev &w, const float *X, int T, float *Y) {
    if (T > BATCH_MAXT) { printf("FATAL: gemv_batch T=%d > %d\n", T, BATCH_MAXT); exit(1); }
    int ns = w.nsplit();
    double tb0 = NOWS_();
    g_bmx.resize(T);
    float M0 = 0;
    for (int t = 0; t < T; t++) {
        const float *r = X + (size_t)t * w.N; float am = 0;
        for (int i = 0; i < w.N; i++) { float a = fabsf(r[i]); if (a > am) am = a; }
        g_bmx[t] = am; if (am > M0) M0 = am;
    }
    if (M0 <= 1e-30f) M0 = 1e-30f;
    g_bnorm.resize((size_t)T * w.N);
    for (int t = 0; t < T; t++) {
        float f = M0 / (g_bmx[t] > 1e-30f ? g_bmx[t] : 1e-30f);
        const float *r = X + (size_t)t * w.N; float *d = g_bnorm.data() + (size_t)t * w.N;
        for (int i = 0; i < w.N; i++) d[i] = r[i] * f;
    }
    for (int p = 0; p < ns; p++) {                   // ① 两片都先提交 (并发跑)
        int dv = w.dev[p];
        xpu_set_device(dv);
        xpu_memcpy(g_dx[dv], g_bnorm.data(), (size_t)T * w.N * 4, XPU_HOST_TO_DEVICE);
        int r = api::gemm_int8(g_ctx[dv], false, true, T, w.mp[p], w.N, 1.f,
                               (const float *)g_dx[dv], w.N,
                               (const int8_t *)w.d[p], 127.f, w.N,
                               0.f, (float *)g_dy[dv], w.mp[p]);
        if (r) { printf("FATAL: gemm_int8(batch) r=%d (chip%d T=%d M=%d N=%d)\n", r, dv, T, w.mp[p], w.N); exit(1); }
    }
    int off = 0;
    for (int p = 0; p < ns; p++) {                   // ② 等结果 + 取回 + 行归一化还原 * 行 scale
        int dv = w.dev[p];
        xpu_set_device(dv);
        if (xpu_wait()) { printf("FATAL: gemm_int8(batch) wait (chip%d T=%d M=%d)\n", dv, T, w.mp[p]); exit(1); }
        g_bYbh.resize((size_t)T * w.mp[p]);
        xpu_memcpy(g_bYbh.data(), g_dy[dv], (size_t)T * w.mp[p] * 4, XPU_DEVICE_TO_HOST);
        xpu_wait();
        for (int t = 0; t < T; t++) {
            const float *src = g_bYbh.data() + (size_t)t * w.mp[p];
            float *dst = Y + (size_t)t * w.M + off;
            float inv = (g_bmx[t] > 1e-30f ? g_bmx[t] : 1e-30f) / M0;
            for (int i = 0; i < w.mp[p]; i++) dst[i] = (src[i] * inv) * w.rs[off + i];
        }
        off += w.mp[p];
        g_calls++;
    }
    g_bn++; g_i8_n++; g_i8_t += NOWS_() - tb0;
}
// ---- ★ 双芯并发: 每片各 launch 一次 (不同设备, 默认 stream 真并行), 再拼回 y ----
static void gemv(const WDev &w, const float *x, float *y) {
    if (w.i8row) { gemv_i8(w, x, y); return; }
    static std::vector<signed char> xq; static std::vector<float> xs;
    int nb = w.N / 32, ns = w.nsplit();
    quant_x(w, x, xq, xs);
    for (int p = 0; p < ns; p++) {
        int dv = w.dev[p];
        xpu_set_device(dv);
        xpu_memcpy(g_dxq[dv], xq.data(), (size_t)w.N, XPU_HOST_TO_DEVICE);
        xpu_memcpy(g_dxs[dv], xs.data(), (size_t)nb * 4, XPU_HOST_TO_DEVICE);
        if (w.v2) run_gemv_q8v2(g_cl, g_co, w.d[p], g_dxq[dv], g_dxs[dv], g_dy[dv], w.mp[p], w.N);
        else      run_gemv_q8_0i(g_cl, g_co, w.d[p], g_dxq[dv], g_dxs[dv], g_dy[dv], w.mp[p], w.N);
    }
    int off = 0;
    for (int p = 0; p < ns; p++) {
        int dv = w.dev[p];
        xpu_set_device(dv);
        if (xpu_wait()) { printf("FATAL: kernel wait (dev%d M=%d N=%d)\n", dv, w.mp[p], w.N); exit(1); }
        auto td = std::chrono::steady_clock::now();
        xpu_memcpy(y + off, g_dy[dv], (size_t)w.mp[p] * 4, XPU_DEVICE_TO_HOST);
        xpu_wait();
        g_d2h += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count();
        off += w.mp[p];
        g_calls++;
    }
    g_gemv_n++;
}
static void gemv_host(const std::vector<float> &W, int M, int N, const float *x, float *y) {
    #pragma omp parallel for schedule(static) if(g_omp)
    for (int m = 0; m < M; m++) {
        const float *r = &W[(size_t)m * N]; double s = 0;
        for (int n = 0; n < N; n++) s += (double)r[n] * x[n];
        y[m] = (float)s;
    }
}

// ============================ 全局 ============================
static GF g_gguf;
static Tokenizer g_tk;
static std::vector<Layer> g_lay;
static std::vector<float> g_out_norm;
static WDev g_outw;
static std::vector<float> g_rope_c, g_rope_s;   // [NROT/2]

static void rope_init(int nctx) {
    g_rope_c.resize(NROT / 2); g_rope_s.resize(NROT / 2);
}
static void rope_apply(float *v, int pos, int hd) {
    // 前 NROT 维, neox 配对 (i, i+NROT/2), theta_i = pos * base^(-2i/NROT)
    for (int i = 0; i < NROT / 2; i++) {
        float th = pos * powf(ROPE_BASE, -2.f * i / (float)NROT);
        float c = cosf(th), s = sinf(th);
        float x0 = v[i], x1 = v[i + NROT / 2];
        v[i]             = x0 * c - x1 * s;
        v[i + NROT / 2]  = x0 * s + x1 * c;
    }
    (void)hd;
}

struct State {
    std::vector<float> conv;   // NLAYER * 3 * CVD
    std::vector<float> S;      // NLAYER * NVH * VHD * KHD   (M[j*KHD+i] = S[i][j])
    std::vector<float> Kc, Vc; // NLAYER * MAXT * NKV * HD
    void reset() {
        memset(conv.data(), 0, conv.size() * 4);
        memset(S.data(), 0, S.size() * 4);
    }
};
static State g_st;
// ============================================================================
// ★★ 第29轮 (continuous batching): 槽位状态【指针间接化】
//   forward()/forward_batch() 原来直接摸 g_st.conv / g_st.S / L.Kc / L.Vc。
//   为了让同一段(逐位不变的)数值代码能作用在【任意槽位】的状态上,
//   这里把这 4 类缓冲改成经指针访问, 默认指向老的全局对象
//   ⇒ 老路径 (单会话) 的数据对象与访问顺序一字不变, 位级行为不变。
// ============================================================================
static std::vector<float> *g_pConv = &g_st.conv, *g_pS = &g_st.S;
static std::vector<float> *g_pKc[NLAYER] = {nullptr}, *g_pVc[NLAYER] = {nullptr};
// 图像 embedding 也按槽位隔离 (多槽并发时每槽可能各自一张图)
static std::vector<float> *g_pImgemb = &g_imgemb;
static int *g_pImgn = &g_img_n, *g_pImgused = &g_img_used;

// 单 token 前向: tok -> logits
static std::vector<float> g_x, g_xb, g_tmp;
// 拼接组内的段偏移 (必须与加载顺序严格一致; 加载时会 assert)
#define PA_QKV   0
#define PA_QKVS  8192
#define PA_GATE  (PA_QKV + PA_QKVS)      // 8192
#define PA_GATES 4096
#define PA_QA    (PA_QKV + PA_QKVS)
#define PA_KA    (PA_QA + PA_QKVS)       // 16384? -> 见下, 注意层单独定义
// 注意层组 A: q(8192) | k(1024) | v(1024)
#define AA_Q     0
#define AA_QS    8192
#define AA_K     (AA_Q + AA_QS)          // 8192
#define AA_KS    1024
#define AA_V     (AA_K + AA_KS)          // 9216
#define AA_VS    1024
#define AA_TOT   (AA_V + AA_VS)          // 10240
// MLP 组: ffn_gate(12288) | ffn_up(12288)
#define FF_G     0
#define FF_GS    12288
#define FF_U     (FF_G + FF_GS)          // 12288
#define FF_US    12288
#define FF_TOT   (FF_U + FF_US)          // 24576
static void forward(int tok, int pos, std::vector<float> &logits) {
    g_fwd_n++;
    std::vector<float> &x = g_x;
    static std::vector<float> gy;                 // 拼接组输出缓冲
    double _sb = 0, _sc = 0, _sb2 = 0, _sc2 = 0, _sb3 = 0, _sc3 = 0, _sbl = 0, _scl = 0;
    _sb = NOWS_(); _sc = g_i8_t;
    // --- 嵌入 (直接从 GGUF 文件解 Q8_0 行, 不进 HBM) ---
    {
        const GT *t = g_gguf.get("token_embd.weight");
        uint64_t nb = (uint64_t)(EMB / 32) * 34;
        std::vector<unsigned char> raw(nb);
        if (*g_pImgn > 0 && tok == g_img_pad && *g_pImgused < *g_pImgn) {  // ★ 图像 token
            memcpy(x.data(), &(*g_pImgemb)[(size_t)(*g_pImgused) * EMB], (size_t)EMB * 4);
            (*g_pImgused)++;
        } else {
        if (fseek(g_gguf.f, (long)(g_gguf.data_start + t->off + (uint64_t)tok * nb), SEEK_SET)) { printf("FATAL: seek embed\n"); exit(1); }
        if (fread(raw.data(), 1, nb, g_gguf.f) != nb) { printf("FATAL: read embed\n"); exit(1); }
        dequant_q8_0(raw.data(), x.data(), EMB);
        }
    }
    g_p_emb += (NOWS_() - _sb) - (g_i8_t - _sc);
    std::vector<float> xb(EMB), o(EMB);
    static std::vector<float> qkv, z, gg, uu;
    for (int l = 0; l < NLAYER; l++) {
        Layer &L = g_lay[l];
        _sb = NOWS_(); _sc = g_i8_t;
        rmsnorm(xb.data(), x.data(), L.attn_norm.data(), EMB, EPS);
        // ===== 波1: 所有以 attn_norm 输出为输入的矩阵 (拼接成一个 launch, 两芯各半) =====
        int tot_a = L.recr ? (PA_GATE + PA_GATES) : AA_TOT;
        int tot_f = FF_TOT;
        if ((int)gy.size() < (tot_a < tot_f ? tot_f : tot_a)) gy.resize(tot_a > tot_f ? tot_a : tot_f);
        { snprintf(g_nb, sizeof(g_nb), "l%d.grpA", l); g_gname = g_nb; }
        gemv(L.grp_a, xb.data(), gy.data());
        if (g_tr && l == 0) { g_trX[0].assign(gy.begin(), gy.begin() + PA_GATE + PA_GATES);
                              g_trRows.push_back(std::vector<float>(gy.begin(), gy.begin() + PA_GATE + PA_GATES)); }
        gg.resize(NFF); uu.resize(NFF);
        if (L.recr) {
            qkv.resize(PA_QKVS); z.resize(EMB);
            memcpy(qkv.data(), gy.data() + PA_QKV, PA_QKVS * 4);
            memcpy(z.data(),   gy.data() + PA_GATE, PA_GATES * 4);
            int albe_dummy = 0; (void)albe_dummy;
        } else {
            qkv.resize(NH * HD * 2);            // q
            memcpy(qkv.data(), gy.data() + AA_Q, AA_QS * 4);
        }
        g_p_nrm += (NOWS_() - _sb) - (g_i8_t - _sc);
        _sb2 = NOWS_(); _sc2 = g_i8_t;
        if (L.recr) {
            static std::vector<float> al, be, co;
            al.resize(NVH); be.resize(NVH); co.resize(CVD);
            double _t1 = NOWS_();
            gemv_host(L.alpha_w, NVH, EMB, xb.data(), al.data());
            gemv_host(L.beta_w,  NVH, EMB, xb.data(), be.data());
            g_p_ab += (NOWS_() - _t1) - (g_i8_t - _sc2);
            double _t2 = NOWS_();
            float *cs = &(*g_pConv)[(size_t)l * 3 * CVD];
            #pragma omp parallel for schedule(static) if(g_omp)
            for (int c = 0; c < CVD; c++) {
                float v = cs[0 * CVD + c] * L.conv1d[4 * c + 0]
                        + cs[1 * CVD + c] * L.conv1d[4 * c + 1]
                        + cs[2 * CVD + c] * L.conv1d[4 * c + 2]
                        + qkv[c]          * L.conv1d[4 * c + 3];
                co[c] = siluf(v);
                cs[0 * CVD + c] = cs[1 * CVD + c];
                cs[1 * CVD + c] = cs[2 * CVD + c];
                cs[2 * CVD + c] = qkv[c];
            }
            g_p_conv += (NOWS_() - _t2) - (g_i8_t - _sc2);
            double _t3 = NOWS_();
            // q(2048: 16x128) | k(2048) | v(4096: 32x128)
            std::vector<float> qc(co.begin(),        co.begin() + 2048);
            std::vector<float> kc(co.begin() + 2048, co.begin() + 4096);
            std::vector<float> vc(co.begin() + 4096, co.end());
            for (int h = 0; h < NKH; h++) l2norm_head(&qc[h * KHD], KHD, EPS);
            for (int h = 0; h < NKH; h++) l2norm_head(&kc[h * KHD], KHD, EPS);
            const float qscale = 1.f / sqrtf((float)KHD);
            static std::vector<float> dout; dout.assign(NVH * KHD, 0.f);
            #pragma omp parallel for schedule(static) if(g_omp)
            for (int hv = 0; hv < NVH; hv++) {
                int hk = hv % NKH;                      // 与 llama.cpp 融合内核 iv1 % neq1 一致
                float g_ = L.ssm_a[hv] * softplusf_(al[hv] + L.ssm_dt[hv]);
                float b_ = sigmoidf_(be[hv]);
                float eg = expf(g_);
                float *M = &(*g_pS)[((size_t)l * NVH + hv) * VHD * KHD];
                const float *kk = &kc[hk * KHD];
                const float *qq = &qc[hk * KHD];
                const float *vv = &vc[hv * KHD];
                int j = 0;
                // ★★ [pipe] 4 行并行 (j..j+3): 每行的归约仍严格按 i 升序、单 float 累加器、
                //    逐元素运算序列完全不变 => 与逐行版本【位级一致】(已验证黄金逐字节一致)。
                //    原版每个 (hv,j) 只有 1 条 FADD 依赖链 (4 周期延迟 x 128 迭代), 主机 4 核被饿死;
                //    4 条独立链交替发射把延迟填满 => delta-net 12.33 -> ~4 ms/token。
                for (; j + 4 <= VHD; j += 4) {
                    float *r0 = M + (size_t)(j + 0) * KHD, *r1 = M + (size_t)(j + 1) * KHD;
                    float *r2 = M + (size_t)(j + 2) * KHD, *r3 = M + (size_t)(j + 3) * KHD;
                    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                    for (int i = 0; i < KHD; i++) {
                        float ki = kk[i];
                        float a0 = r0[i] * eg; r0[i] = a0; s0 += a0 * ki;
                        float a1 = r1[i] * eg; r1[i] = a1; s1 += a1 * ki;
                        float a2 = r2[i] * eg; r2[i] = a2; s2 += a2 * ki;
                        float a3 = r3[i] * eg; r3[i] = a3; s3 += a3 * ki;
                    }
                    float d0 = (vv[j + 0] - s0) * b_, d1 = (vv[j + 1] - s1) * b_;
                    float d2 = (vv[j + 2] - s2) * b_, d3 = (vv[j + 3] - s3) * b_;
                    float o0 = 0, o1 = 0, o2 = 0, o3 = 0;
                    for (int i = 0; i < KHD; i++) {
                        float ki = kk[i], qi = qq[i];
                        float a0 = r0[i] + ki * d0; r0[i] = a0; o0 += a0 * qi;
                        float a1 = r1[i] + ki * d1; r1[i] = a1; o1 += a1 * qi;
                        float a2 = r2[i] + ki * d2; r2[i] = a2; o2 += a2 * qi;
                        float a3 = r3[i] + ki * d3; r3[i] = a3; o3 += a3 * qi;
                    }
                    dout[hv * KHD + j + 0] = o0 * qscale;
                    dout[hv * KHD + j + 1] = o1 * qscale;
                    dout[hv * KHD + j + 2] = o2 * qscale;
                    dout[hv * KHD + j + 3] = o3 * qscale;
                }
                for (; j < VHD; j++) {                  // 尾巴 (VHD=128 可被 4 整除, 实际不执行)
                    float *row = M + (size_t)j * KHD;
                    float sk = 0;
                    for (int i = 0; i < KHD; i++) { float rv = row[i] * eg; row[i] = rv; sk += rv * kk[i]; }
                    float d = (vv[j] - sk) * b_;
                    float so = 0;
                    for (int i = 0; i < KHD; i++) { float rv = row[i] + kk[i] * d; row[i] = rv; so += rv * qq[i]; }
                    dout[hv * KHD + j] = so * qscale;
                }
            }
            g_p_dn += (NOWS_() - _t3) - (g_i8_t - _sc2);
            double _t4 = NOWS_();
            // gated rms norm: per 128 head, rmsnorm(ssm_norm) * silu(z)
            #pragma omp parallel for schedule(static) if(g_omp)
            for (int h = 0; h < NVH; h++) {
                float *p = &dout[h * KHD];
                double ss = 0; for (int i = 0; i < KHD; i++) ss += (double)p[i] * p[i];
                float r = 1.f / sqrtf((float)(ss / KHD) + EPS);
                for (int i = 0; i < KHD; i++) p[i] = p[i] * r * L.ssm_norm[i] * siluf(z[h * KHD + i]);
            }
            g_p_gn += (NOWS_() - _t4) - (g_i8_t - _sc2);
            { snprintf(g_nb, sizeof(g_nb), "l%d.ssm_out", l); g_gname = g_nb; }
            gemv(L.grp_mid, dout.data(), o.data());
        } else {
            static std::vector<float> kk, vv, ao;
            kk.resize(NKV * HD); vv.resize(NKV * HD); ao.resize(NH * HD);
            memcpy(kk.data(), gy.data() + AA_K, AA_KS * 4);
            memcpy(vv.data(), gy.data() + AA_V, AA_VS * 4);
            static std::vector<float> q5(NH * HD), g5(NH * HD);
            { const float *qg = qkv.data();          // 8192 = q(4096) | gate(4096) 交错 2xHD
              for (int h = 0; h < NH; h++) {
                  memcpy(&q5[h * HD], &qg[h * HD * 2], HD * 4);
                  memcpy(&g5[h * HD], &qg[h * HD * 2 + HD], HD * 4);
              } }
            for (int h = 0; h < NH; h++) rmsnorm(&q5[h * HD], &q5[h * HD], L.q_norm.data(), HD, EPS);
            for (int h = 0; h < NKV; h++) rmsnorm(&kk[h * HD], &kk[h * HD], L.k_norm.data(), HD, EPS);
            for (int h = 0; h < NH; h++) rope_apply(&q5[h * HD], pos, HD);
            for (int h = 0; h < NKV; h++) rope_apply(&kk[h * HD], pos, HD);
            float *Kc = g_pKc[l]->data();
            float *Vc = g_pVc[l]->data();
            memcpy(&Kc[(size_t)pos * NKV * HD], kk.data(), (size_t)NKV * HD * 4);
            memcpy(&Vc[(size_t)pos * NKV * HD], vv.data(), (size_t)NKV * HD * 4);
            const float sc = 1.f / sqrtf((float)HD);
            int gpr = NH / NKV;
            static std::vector<float> wt;
            if ((int)wt.size() < NH * (pos + 1)) wt.resize((size_t)NH * (pos + 1));
            // ★ 每个头独立一段 wt (OpenMP 并行安全; 数值与共用缓冲时逐位一致)
            #pragma omp parallel for schedule(static) if(g_omp)
            for (int h = 0; h < NH; h++) {
                int kh = h / gpr;
                const float *qp = &q5[h * HD];
                float *wth = &wt[(size_t)h * (pos + 1)];
                float best = -1e30f;
                for (int t = 0; t <= pos; t++) {
                    const float *kt = &Kc[((size_t)t * NKV + kh) * HD];
                    float s = 0; for (int i = 0; i < HD; i++) s += qp[i] * kt[i];
                    s *= sc; wth[t] = s; if (s > best) best = s;
                }
                float sum = 0;
                for (int t = 0; t <= pos; t++) { float e = expf(wth[t] - best); wth[t] = e; sum += e; }
                float *oh = &ao[h * HD];
                for (int i = 0; i < HD; i++) oh[i] = 0;
                for (int t = 0; t <= pos; t++) {
                    float p = wth[t] / sum;
                    const float *vt = &Vc[((size_t)t * NKV + kh) * HD];
                    for (int i = 0; i < HD; i++) oh[i] += p * vt[i];
                }
                for (int i = 0; i < HD; i++) oh[i] *= sigmoidf_(g5[h * HD + i]);
            }
            { snprintf(g_nb, sizeof(g_nb), "l%d.attn_out", l); g_gname = g_nb; }
            gemv(L.grp_mid, ao.data(), o.data());
        }
        if (L.recr) g_p_ssm += (NOWS_() - _sb2) - (g_i8_t - _sc2);
        else        g_p_att += (NOWS_() - _sb2) - (g_i8_t - _sc2);
        _sb3 = NOWS_(); _sc3 = g_i8_t;
        for (int i = 0; i < EMB; i++) x[i] += o[i];
        // ===== 波3: MLP 上半 (ffn_gate + ffn_up 拼接, 同一个 post_norm 输出) =====
        rmsnorm(xb.data(), x.data(), L.post_norm.data(), EMB, EPS);
        { snprintf(g_nb, sizeof(g_nb), "l%d.grpFF", l); g_gname = g_nb; }
        gemv(L.grp_ffn, xb.data(), gy.data());
        memcpy(gg.data(), gy.data() + FF_G, FF_GS * 4);
        memcpy(uu.data(), gy.data() + FF_U, FF_US * 4);
        #pragma omp parallel for schedule(static) if(g_omp)
        for (int i = 0; i < NFF; i++) gg[i] = siluf(gg[i]) * uu[i];
        { snprintf(g_nb, sizeof(g_nb), "l%d.ffn_down", l); g_gname = g_nb; }
        gemv(L.ffn_down, gg.data(), o.data());
        for (int i = 0; i < EMB; i++) x[i] += o[i];
        g_p_lay += (NOWS_() - _sb3) - (g_i8_t - _sc3);
        if (g_tr && l == 0) g_trRows2.push_back(std::vector<float>(x.begin(), x.end()));
        if (g_tr) { if ((int)g_trL.size() < NLAYER) g_trL.resize(NLAYER);
                    if (l < 8) g_trL[l].push_back(std::vector<float>(x.begin(), x.end())); }
    }
    _sbl = NOWS_(); _scl = g_i8_t;
    rmsnorm(xb.data(), x.data(), g_out_norm.data(), EMB, EPS);
    logits.resize(g_outw.M);
    { snprintf(g_nb, sizeof(g_nb), "lm_head"); g_gname = g_nb; }
    gemv(g_outw, xb.data(), logits.data());
    g_p_lm += (NOWS_() - _sbl) - (g_i8_t - _scl);
    if (g_tr) g_trX[1].assign(x.begin(), x.end());
}

// ============================================================================
// ★★ prefill 批量前向: T 个 token 的线性层合并成一次 gemm (m=T) ★★
//   递推部分 (conv1d / delta-net / 全注意力 KV / rope) 仍逐 token 顺序跑 —
//   它们不吃权重带宽 (mbatch: 权重读才是瓶颈), 且必须保序才与逐 token 路径同状态.
//   只有最后一行需要 logits (生成从这里继续), 且用与逐 token 路径完全相同的
//   gemv(outw, m=1) 代码段 => 该行的数值路径与改前一致.
// ============================================================================
static std::vector<float> g_bX, g_bXB, g_bZ, g_bY, g_bO, g_bD, g_bGG,
                          g_bCO, g_bAL, g_bBE, g_bQ5, g_bG5, g_bKK, g_bVV, g_bAO, g_bWT;
static void forward_batch(const int *toks, int T, int pos0, std::vector<float> &logits) {
    g_fwd_n += T;
    double _sb = 0, _sc = 0, _sb2 = 0, _sc2 = 0, _sb3 = 0, _sc3 = 0, _sbl = 0, _scl = 0;
    g_bX.assign((size_t)T * EMB, 0.f);
    g_bXB.resize((size_t)T * EMB);
    g_bZ.resize((size_t)T * EMB);
    g_bO.resize((size_t)T * EMB);
    g_bD.resize((size_t)T * EMB);
    g_bGG.resize((size_t)T * NFF);
    g_bCO.resize(CVD); g_bAL.resize(NVH); g_bBE.resize(NVH);
    g_bQ5.resize(NH * HD); g_bG5.resize(NH * HD);
    g_bKK.resize(NKV * HD); g_bVV.resize(NKV * HD); g_bAO.resize(NH * HD);
    g_bWT.resize((size_t)NH * MAXT);
    int maxa = 0;
    for (int l = 0; l < NLAYER; l++) { int tt = g_lay[l].recr ? (PA_GATE + PA_GATES) : AA_TOT; if (tt > maxa) maxa = tt; }
    if (FF_TOT > maxa) maxa = FF_TOT;          // ★ grp_ffn 输出 24576 行 (ffn_gate|ffn_up), 必须算进缓冲
    g_bY.resize((size_t)T * maxa);
    _sb = NOWS_(); _sc = g_i8_t;
    {   // --- 嵌入 T 行 (直接从 GGUF 文件解 Q8_0 行, 不进 HBM) ---
        const GT *t = g_gguf.get("token_embd.weight");
        uint64_t nb = (uint64_t)(EMB / 32) * 34;
        std::vector<unsigned char> raw(nb);
        for (int q = 0; q < T; q++) {
            if (*g_pImgn > 0 && toks[q] == g_img_pad && *g_pImgused < *g_pImgn) {  // ★ 图像 token
                memcpy(g_bX.data() + (size_t)q * EMB, &(*g_pImgemb)[(size_t)(*g_pImgused) * EMB], (size_t)EMB * 4);
                (*g_pImgused)++;
                continue;
            }
            if (fseek(g_gguf.f, (long)(g_gguf.data_start + t->off + (uint64_t)toks[q] * nb), SEEK_SET)) { printf("FATAL: seek embed\n"); exit(1); }
            if (fread(raw.data(), 1, nb, g_gguf.f) != nb) { printf("FATAL: read embed\n"); exit(1); }
            dequant_q8_0(raw.data(), g_bX.data() + (size_t)q * EMB, EMB);
        }
    }
    g_p_emb += (NOWS_() - _sb) - (g_i8_t - _sc);
    for (int l = 0; l < NLAYER; l++) {
        Layer &L = g_lay[l];
        _sb = NOWS_(); _sc = g_i8_t;
        for (int q = 0; q < T; q++)
            rmsnorm(g_bXB.data() + (size_t)q * EMB, g_bX.data() + (size_t)q * EMB, L.attn_norm.data(), EMB, EPS);
        { snprintf(g_nb, sizeof(g_nb), "l%d.grpA", l); g_gname = g_nb; }
        gemv_batch(L.grp_a, g_bXB.data(), T, g_bY.data());
        if (g_tr && l == 0) {
            g_trX[0].assign(g_bY.begin(), g_bY.begin() + (PA_GATE + PA_GATES));
            for (int q = 0; q < T; q++)
                g_trRows.push_back(std::vector<float>(g_bY.begin() + (size_t)q * (PA_GATE + PA_GATES),
                                                       g_bY.begin() + (size_t)(q + 1) * (PA_GATE + PA_GATES)));
        }
        g_p_nrm += (NOWS_() - _sb) - (g_i8_t - _sc);
        _sb2 = NOWS_(); _sc2 = g_i8_t;
        if (L.recr) {
            for (int q = 0; q < T; q++) {
                const float *gy  = g_bY.data() + (size_t)q * (PA_GATE + PA_GATES);
                const float *qkvi = gy + PA_QKV;
                const float *xbq = g_bXB.data() + (size_t)q * EMB;
                memcpy(g_bZ.data() + (size_t)q * EMB, gy + PA_GATE, EMB * 4);
                float *dout = g_bD.data() + (size_t)q * EMB;
                gemv_host(L.alpha_w, NVH, EMB, xbq, g_bAL.data());
                gemv_host(L.beta_w,  NVH, EMB, xbq, g_bBE.data());
                float *cs = &(*g_pConv)[(size_t)l * 3 * CVD];
                for (int c = 0; c < CVD; c++) {
                    float v = cs[0 * CVD + c] * L.conv1d[4 * c + 0]
                            + cs[1 * CVD + c] * L.conv1d[4 * c + 1]
                            + cs[2 * CVD + c] * L.conv1d[4 * c + 2]
                            + qkvi[c]           * L.conv1d[4 * c + 3];
                    g_bCO[c] = siluf(v);
                    cs[0 * CVD + c] = cs[1 * CVD + c];
                    cs[1 * CVD + c] = cs[2 * CVD + c];
                    cs[2 * CVD + c] = qkvi[c];
                }
                float *qc = g_bCO.data();
                float *kc = g_bCO.data() + 2048;
                const float *vc = g_bCO.data() + 4096;
                for (int h = 0; h < NKH; h++) l2norm_head(&qc[h * KHD], KHD, EPS);
                for (int h = 0; h < NKH; h++) l2norm_head(&kc[h * KHD], KHD, EPS);
                const float qscale = 1.f / sqrtf((float)KHD);
                for (int i = 0; i < NVH * KHD; i++) dout[i] = 0.f;
                #pragma omp parallel for schedule(static) if(g_omp)
                for (int hv = 0; hv < NVH; hv++) {
                    int hk = hv % NKH;
                    float g_ = L.ssm_a[hv] * softplusf_(g_bAL[hv] + L.ssm_dt[hv]);
                    float b_ = sigmoidf_(g_bBE[hv]);
                    float eg = expf(g_);
                    float *M = &(*g_pS)[((size_t)l * NVH + hv) * VHD * KHD];
                    const float *kk = &kc[hk * KHD];
                    const float *qq = &qc[hk * KHD];
                    const float *vv = &vc[hv * KHD];
                    int j = 0;
                    // ★★ [pipe] 同逐 token 路径: 4 行并行 (位级一致, 见 forward() 内注释)
                    for (; j + 4 <= VHD; j += 4) {
                        float *r0 = M + (size_t)(j + 0) * KHD, *r1 = M + (size_t)(j + 1) * KHD;
                        float *r2 = M + (size_t)(j + 2) * KHD, *r3 = M + (size_t)(j + 3) * KHD;
                        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                        for (int i = 0; i < KHD; i++) {
                            float ki = kk[i];
                            float a0 = r0[i] * eg; r0[i] = a0; s0 += a0 * ki;
                            float a1 = r1[i] * eg; r1[i] = a1; s1 += a1 * ki;
                            float a2 = r2[i] * eg; r2[i] = a2; s2 += a2 * ki;
                            float a3 = r3[i] * eg; r3[i] = a3; s3 += a3 * ki;
                        }
                        float d0 = (vv[j + 0] - s0) * b_, d1 = (vv[j + 1] - s1) * b_;
                        float d2 = (vv[j + 2] - s2) * b_, d3 = (vv[j + 3] - s3) * b_;
                        float o0 = 0, o1 = 0, o2 = 0, o3 = 0;
                        for (int i = 0; i < KHD; i++) {
                            float ki = kk[i], qi = qq[i];
                            float a0 = r0[i] + ki * d0; r0[i] = a0; o0 += a0 * qi;
                            float a1 = r1[i] + ki * d1; r1[i] = a1; o1 += a1 * qi;
                            float a2 = r2[i] + ki * d2; r2[i] = a2; o2 += a2 * qi;
                            float a3 = r3[i] + ki * d3; r3[i] = a3; o3 += a3 * qi;
                        }
                        dout[hv * KHD + j + 0] = o0 * qscale;
                        dout[hv * KHD + j + 1] = o1 * qscale;
                        dout[hv * KHD + j + 2] = o2 * qscale;
                        dout[hv * KHD + j + 3] = o3 * qscale;
                    }
                    for (; j < VHD; j++) {
                        float *row = M + (size_t)j * KHD;
                        float sk = 0;
                        for (int i = 0; i < KHD; i++) { float rv = row[i] * eg; row[i] = rv; sk += rv * kk[i]; }
                        float d = (vv[j] - sk) * b_;
                        float so = 0;
                        for (int i = 0; i < KHD; i++) { float rv = row[i] + kk[i] * d; row[i] = rv; so += rv * qq[i]; }
                        dout[hv * KHD + j] = so * qscale;
                    }
                }
                const float *zp = g_bZ.data() + (size_t)q * EMB;
                #pragma omp parallel for schedule(static) if(g_omp)
                for (int h = 0; h < NVH; h++) {
                    float *p = &dout[h * KHD];
                    double ss = 0; for (int i = 0; i < KHD; i++) ss += (double)p[i] * p[i];
                    float r = 1.f / sqrtf((float)(ss / KHD) + EPS);
                    for (int i = 0; i < KHD; i++) p[i] = p[i] * r * L.ssm_norm[i] * siluf(zp[h * KHD + i]);
                }
            }
            g_p_ssm += (NOWS_() - _sb2) - (g_i8_t - _sc2);
        } else {
            for (int q = 0; q < T; q++) {
                int pos = pos0 + q;
                const float *qg = g_bY.data() + (size_t)q * AA_TOT + AA_Q;
                memcpy(g_bKK.data(), g_bY.data() + (size_t)q * AA_TOT + AA_K, AA_KS * 4);
                memcpy(g_bVV.data(), g_bY.data() + (size_t)q * AA_TOT + AA_V, AA_VS * 4);
                for (int h = 0; h < NH; h++) {
                    memcpy(&g_bQ5[h * HD], &qg[h * HD * 2], HD * 4);
                    memcpy(&g_bG5[h * HD], &qg[h * HD * 2 + HD], HD * 4);
                }
                for (int h = 0; h < NH; h++) rmsnorm(&g_bQ5[h * HD], &g_bQ5[h * HD], L.q_norm.data(), HD, EPS);
                for (int h = 0; h < NKV; h++) rmsnorm(&g_bKK[h * HD], &g_bKK[h * HD], L.k_norm.data(), HD, EPS);
                for (int h = 0; h < NH; h++) rope_apply(&g_bQ5[h * HD], pos, HD);
                for (int h = 0; h < NKV; h++) rope_apply(&g_bKK[h * HD], pos, HD);
                float *Kc = g_pKc[l]->data();
                float *Vc = g_pVc[l]->data();
                memcpy(&Kc[(size_t)pos * NKV * HD], g_bKK.data(), (size_t)NKV * HD * 4);
                memcpy(&Vc[(size_t)pos * NKV * HD], g_bVV.data(), (size_t)NKV * HD * 4);
                const float sc = 1.f / sqrtf((float)HD);
                int gpr = NH / NKV;
                #pragma omp parallel for schedule(static) if(g_omp)
                for (int h = 0; h < NH; h++) {
                    int kh = h / gpr;
                    const float *qp = &g_bQ5[h * HD];
                    float *wth = &g_bWT[(size_t)h * MAXT];
                    float best = -1e30f;
                    for (int t = 0; t <= pos; t++) {
                        const float *kt = &Kc[((size_t)t * NKV + kh) * HD];
                        float s = 0; for (int i = 0; i < HD; i++) s += qp[i] * kt[i];
                        s *= sc; wth[t] = s; if (s > best) best = s;
                    }
                    float sum = 0;
                    for (int t = 0; t <= pos; t++) { float e = expf(wth[t] - best); wth[t] = e; sum += e; }
                    float *oh = &g_bAO[h * HD];
                    for (int i = 0; i < HD; i++) oh[i] = 0;
                    for (int t = 0; t <= pos; t++) {
                        float p = wth[t] / sum;      // ★ 必须用本头自己的 wth (g_bWT[t] 是 bug: 会读到别的头)
                        const float *vt = &Vc[((size_t)t * NKV + kh) * HD];
                        for (int i = 0; i < HD; i++) oh[i] += p * vt[i];
                    }
                    for (int i = 0; i < HD; i++) oh[i] *= sigmoidf_(g_bG5[h * HD + i]);
                }
                memcpy(g_bD.data() + (size_t)q * EMB, g_bAO.data(), (size_t)NH * HD * 4);
            }
            g_p_att += (NOWS_() - _sb2) - (g_i8_t - _sc2);
        }
        { snprintf(g_nb, sizeof(g_nb), "l%d.mid", l); g_gname = g_nb; }
        gemv_batch(L.grp_mid, g_bD.data(), T, g_bO.data());
        _sb3 = NOWS_(); _sc3 = g_i8_t;
        for (int q = 0; q < T; q++) {
            float *xq = g_bX.data() + (size_t)q * EMB;
            const float *oq = g_bO.data() + (size_t)q * EMB;
            for (int i = 0; i < EMB; i++) xq[i] += oq[i];
            rmsnorm(g_bXB.data() + (size_t)q * EMB, xq, L.post_norm.data(), EMB, EPS);
        }
        { snprintf(g_nb, sizeof(g_nb), "l%d.grpFF", l); g_gname = g_nb; }
        gemv_batch(L.grp_ffn, g_bXB.data(), T, g_bY.data());
        #pragma omp parallel for schedule(static) if(g_omp)
        for (int q = 0; q < T; q++) {
            const float *gy = g_bY.data() + (size_t)q * FF_TOT;
            float *gg = g_bGG.data() + (size_t)q * NFF;
            for (int i = 0; i < NFF; i++) gg[i] = siluf(gy[FF_G + i]) * gy[FF_U + i];
        }
        { snprintf(g_nb, sizeof(g_nb), "l%d.ffn_down", l); g_gname = g_nb; }
        gemv_batch(L.ffn_down, g_bGG.data(), T, g_bO.data());
        for (int q = 0; q < T; q++) {
            float *xq = g_bX.data() + (size_t)q * EMB;
            const float *oq = g_bO.data() + (size_t)q * EMB;
            for (int i = 0; i < EMB; i++) xq[i] += oq[i];
        }
        g_p_lay += (NOWS_() - _sb3) - (g_i8_t - _sc3);
        if (g_tr && l == 0)
            for (int q = 0; q < T; q++)
                g_trRows2.push_back(std::vector<float>(g_bX.begin() + (size_t)q * EMB, g_bX.begin() + (size_t)(q + 1) * EMB));
        if (g_tr) { if ((int)g_trL.size() < NLAYER) g_trL.resize(NLAYER);
                    if (l < 8) for (int q = 0; q < T; q++)
                        g_trL[l].push_back(std::vector<float>(g_bX.begin() + (size_t)q * EMB, g_bX.begin() + (size_t)(q + 1) * EMB)); }
    }
    // --- 只有最后一行需要 logits: 与逐 token 路径同一段代码 (m=1) ---
    _sbl = NOWS_(); _scl = g_i8_t;
    rmsnorm(g_bXB.data(), g_bX.data() + (size_t)(T - 1) * EMB, g_out_norm.data(), EMB, EPS);
    logits.resize(g_outw.M);
    { snprintf(g_nb, sizeof(g_nb), "lm_head"); g_gname = g_nb; }
    gemv(g_outw, g_bXB.data(), logits.data());
    g_p_lm += (NOWS_() - _sbl) - (g_i8_t - _scl);
    if (g_tr) g_trX[1].assign(g_bX.begin(), g_bX.begin() + EMB);
}

// ============================================================================
// ★★ 第29轮: 多槽位 (multi-slot) 连续批处理 —— 数据结构
//   一个槽位 = 一条独立会话的【全部可变状态】:
//     conv / S (delta-net 线性层递推状态) + 每个全注意力层的 Kc/Vc + 位置 pos
//     + 该槽自己的图像 embedding / 图像行消耗计数 / 指纹 + logits + 采样前缀
//   ⇒ 槽与槽之间零共享, 批里互不干扰 (逐槽交叉验证见 PROGRESS 第29章)。
// ============================================================================
struct CbSlot {
    bool used   = false;   // 已分配给某个请求 (占用中)
    bool active = false;   // 已在批里推进
    bool feed   = false;   // 本步需要喂 cur_tok (否则只等结束/空闲)
    bool solo   = false;   // ★ 独占: 只有它一个活跃时进批 ⇒ 永远 M=1 (生产口用)
    int  reqid  = 0;       // 客户端给的 id (网关侧单调递增, 用于路由 token)
    std::string sid;       // 会话 id (检查点/前缀复用按 sid 分组)
    int  maxtok = 0, ntok = 0;
    int  pos    = 0;       // 下一个要喂的 token 的绝对位置
    int  cur_tok = 0;      // 本步要喂的 token
    int  imgn = 0, imgused = 0;
    uint64_t imgid = 0;
    std::vector<float> imgemb;                       // 本槽自己的图像 embedding (imgn*EMB)
    std::vector<float> conv, S;                      // 与 g_st 同形
    std::vector<std::vector<float> > Kc, Vc;         // [NLAYER]; 仅全注意力层非空
    std::vector<float> logits;                       // 当前 logits (vocab)
    std::vector<int> ids;                            // 本轮 prompt token ids (截断后)
    std::vector<int> genids;                         // 已生成 (不含最后发出的那个, 与 run_request 同义)
    std::vector<int> emitted;                        // 实际发出的 token (含最后一个)
    std::string out;
    int  reused = 0, lcp_all = 0;
    double t_admit = 0, t_done = 0, prefill_s = 0, first_tok_s = 0;
};

// ============================================================================
// ★★ 第29轮: 多槽位批量解码步 (continuous batching 的核心)
//   M 条活跃槽的当前 token 拼成 [M,K] 的 X ⇒ 每层每个权重矩阵【只读一遍】
//   (官方 gemm_int8, m=M) 得到 [M,N] 的 Y;
//   递推 (conv1d / delta-net / 全注意力 KV / rope) 按槽【各自的状态与位置】算 ——
//   逐槽代码与 forward_batch 里同一段【逐字相同】, 只是它操作的缓冲从全局换成该槽的
//   ⇒ 数值路径不变 ⇒ 每槽结果与"该请求单独跑"一致 (逐槽交叉验证证明)。
//
//   ★ M=1 位级等价: gemv_batch 在 T=1 时行归一因子 f = M0/max|row| 恰好 == 1.0f,
//     还原因子 inv == 1.0f ⇒ dst[i] = (src[i]*1.0f)*rs[i] 与 gemv_i8 的 y[i]*=rs[i]
//     逐位相同 ⇒ 与 forward()/forward_batch(T=1) 完全一致 (由 K200_SELFCHK 对拍)。
//   ★ CPU 只做调度这一条: 批里的【全部矩阵乘 (占 99% FLOP 与 100% 权重带宽)】都在卡上
//     (官方 gemm_int8, 每步每矩阵一次 launch); 递推部分沿用本引擎既有(已上线、
//     逐位验证过的)主机段代码 —— 见 PROGRESS 第29章"关于 CPU 参与数值计算的如实说明"。
// ============================================================================
static std::vector<float> g_mX, g_mXB, g_mY, g_mO, g_mD, g_mGG, g_mLOG,
                          g_mCO, g_mAL, g_mBE, g_mZZ, g_mQ5, g_mG5, g_mKK, g_mVV,
                          g_mWT, g_mAO;

static void forward_multi(std::vector<CbSlot *> &act) {
    int M = (int)act.size();
    if (M <= 0) return;
    if (M > BATCH_MAXT) { printf("FATAL: forward_multi M=%d > BATCH_MAXT=%d\n", M, BATCH_MAXT); exit(1); }
    g_fwd_n += M;
    double _sb = 0, _sc = 0, _sb2 = 0, _sc2 = 0, _sb3 = 0, _sc3 = 0, _sbl = 0, _scl = 0;
    int maxa = 0;
    for (int l = 0; l < NLAYER; l++) { int tt = g_lay[l].grp_a.M; if (tt > maxa) maxa = tt; }
    if (FF_TOT > maxa) maxa = FF_TOT;
    g_mX.assign((size_t)M * EMB, 0.f);
    g_mXB.resize((size_t)M * EMB);
    g_mD.resize((size_t)M * EMB);
    g_mZZ.resize((size_t)M * EMB);
    g_mO.resize((size_t)M * EMB);
    g_mGG.resize((size_t)M * NFF);
    g_mY.resize((size_t)M * maxa);
    g_mCO.resize((size_t)M * CVD);
    g_mAL.resize((size_t)M * NVH); g_mBE.resize((size_t)M * NVH);
    g_mQ5.resize((size_t)M * NH * HD); g_mG5.resize((size_t)M * NH * HD);
    g_mKK.resize((size_t)M * NKV * HD); g_mVV.resize((size_t)M * NKV * HD);
    g_mAO.resize((size_t)M * NH * HD);
    g_mWT.resize((size_t)M * NH * MAXT);
    g_mLOG.resize((size_t)M * (size_t)g_outw.M);
    _sb = NOWS_(); _sc = g_i8_t;
    {   // --- 嵌入 M 行 (每槽自己的 token; 图像行按【该槽自己的】消耗计数取) ---
        const GT *t = g_gguf.get("token_embd.weight");
        uint64_t nb = (uint64_t)(EMB / 32) * 34;
        std::vector<unsigned char> raw(nb);
        for (int q = 0; q < M; q++) {
            CbSlot *S = act[(size_t)q];
            int tk = S->cur_tok;
            if (S->imgn > 0 && tk == g_img_pad && S->imgused < S->imgn) {
                memcpy(g_mX.data() + (size_t)q * EMB, &S->imgemb[(size_t)S->imgused * EMB], (size_t)EMB * 4);
                S->imgused++;
                continue;
            }
            if (fseek(g_gguf.f, (long)(g_gguf.data_start + t->off + (uint64_t)tk * nb), SEEK_SET)) { printf("FATAL: seek embed\n"); exit(1); }
            if (fread(raw.data(), 1, nb, g_gguf.f) != nb) { printf("FATAL: read embed\n"); exit(1); }
            dequant_q8_0(raw.data(), g_mX.data() + (size_t)q * EMB, EMB);
        }
    }
    g_p_emb += (NOWS_() - _sb) - (g_i8_t - _sc);
    for (int l = 0; l < NLAYER; l++) {
        Layer &L = g_lay[l];
        const int ystride = L.grp_a.M;
        _sb = NOWS_(); _sc = g_i8_t;
        #pragma omp parallel for schedule(static) if(g_omp)
        for (int q = 0; q < M; q++)
            rmsnorm(g_mXB.data() + (size_t)q * EMB, g_mX.data() + (size_t)q * EMB, L.attn_norm.data(), EMB, EPS);
        { snprintf(g_nb, sizeof(g_nb), "l%d.grpA", l); g_gname = g_nb; }
        gemv_batch(L.grp_a, g_mXB.data(), M, g_mY.data());        // ★ 权重只读这一遍
        g_p_nrm += (NOWS_() - _sb) - (g_i8_t - _sc);
        _sb2 = NOWS_(); _sc2 = g_i8_t;
        if (L.recr) {
            for (int q = 0; q < M; q++) {   // z 门 + alpha/beta (gemv_host 内部已并行)
                const float *gy = g_mY.data() + (size_t)q * ystride;
                memcpy(g_mZZ.data() + (size_t)q * EMB, gy + PA_GATE, EMB * 4);
                gemv_host(L.alpha_w, NVH, EMB, g_mXB.data() + (size_t)q * EMB, g_mAL.data() + (size_t)q * NVH);
                gemv_host(L.beta_w,  NVH, EMB, g_mXB.data() + (size_t)q * EMB, g_mBE.data() + (size_t)q * NVH);
            }
            // conv1d + silu (depthwise, kernel=4) —— 逐槽各自的 conv 状态
            #pragma omp parallel for collapse(2) schedule(static) if(g_omp)
            for (int q = 0; q < M; q++) for (int c = 0; c < CVD; c++) {
                const float *qkvi = g_mY.data() + (size_t)q * ystride + PA_QKV;
                float *cs = &act[(size_t)q]->conv[(size_t)l * 3 * CVD];
                float v = cs[0 * CVD + c] * L.conv1d[4 * c + 0]
                        + cs[1 * CVD + c] * L.conv1d[4 * c + 1]
                        + cs[2 * CVD + c] * L.conv1d[4 * c + 2]
                        + qkvi[c]           * L.conv1d[4 * c + 3];
                g_mCO[(size_t)q * CVD + c] = siluf(v);
                cs[0 * CVD + c] = cs[1 * CVD + c];
                cs[1 * CVD + c] = cs[2 * CVD + c];
                cs[2 * CVD + c] = qkvi[c];
            }
            // q/k 逐头 L2 归一
            #pragma omp parallel for collapse(2) schedule(static) if(g_omp)
            for (int q = 0; q < M; q++) for (int h = 0; h < NKH; h++) {
                l2norm_head(g_mCO.data() + (size_t)q * CVD + h * KHD, KHD, EPS);
                l2norm_head(g_mCO.data() + (size_t)q * CVD + 2048 + h * KHD, KHD, EPS);
            }
            const float qscaleM = 1.f / sqrtf((float)KHD);
            // gated DeltaNet 递推 + 门控 RMSNorm —— 逐槽各自的 S 状态
            #pragma omp parallel for collapse(2) schedule(static) if(g_omp)
            for (int q = 0; q < M; q++) for (int hv = 0; hv < NVH; hv++) {
                int hk = hv % NKH;
                const float *qc = g_mCO.data() + (size_t)q * CVD;
                const float *kc = g_mCO.data() + (size_t)q * CVD + 2048;
                const float *vc = g_mCO.data() + (size_t)q * CVD + 4096;
                float g_ = L.ssm_a[hv] * softplusf_(g_mAL[(size_t)q * NVH + hv] + L.ssm_dt[hv]);
                float b_ = sigmoidf_(g_mBE[(size_t)q * NVH + hv]);
                float eg = expf(g_);
                float *Mp = &act[(size_t)q]->S[((size_t)l * NVH + hv) * VHD * KHD];
                const float *kk = &kc[hk * KHD];
                const float *qq = &qc[hk * KHD];
                const float *vv = &vc[hv * KHD];
                float *dout = g_mD.data() + (size_t)q * EMB + hv * KHD;
                int j = 0;
                for (; j + 4 <= VHD; j += 4) {          // [pipe] 4 行并行 (与单槽路径逐位一致)
                    float *r0 = Mp + (size_t)(j + 0) * KHD, *r1 = Mp + (size_t)(j + 1) * KHD;
                    float *r2 = Mp + (size_t)(j + 2) * KHD, *r3 = Mp + (size_t)(j + 3) * KHD;
                    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                    for (int i = 0; i < KHD; i++) {
                        float ki = kk[i];
                        float a0 = r0[i] * eg; r0[i] = a0; s0 += a0 * ki;
                        float a1 = r1[i] * eg; r1[i] = a1; s1 += a1 * ki;
                        float a2 = r2[i] * eg; r2[i] = a2; s2 += a2 * ki;
                        float a3 = r3[i] * eg; r3[i] = a3; s3 += a3 * ki;
                    }
                    float d0 = (vv[j + 0] - s0) * b_, d1 = (vv[j + 1] - s1) * b_;
                    float d2 = (vv[j + 2] - s2) * b_, d3 = (vv[j + 3] - s3) * b_;
                    float o0 = 0, o1 = 0, o2 = 0, o3 = 0;
                    for (int i = 0; i < KHD; i++) {
                        float ki = kk[i], qi = qq[i];
                        float a0 = r0[i] + ki * d0; r0[i] = a0; o0 += a0 * qi;
                        float a1 = r1[i] + ki * d1; r1[i] = a1; o1 += a1 * qi;
                        float a2 = r2[i] + ki * d2; r2[i] = a2; o2 += a2 * qi;
                        float a3 = r3[i] + ki * d3; r3[i] = a3; o3 += a3 * qi;
                    }
                    dout[j + 0] = o0 * qscaleM;
                    dout[j + 1] = o1 * qscaleM;
                    dout[j + 2] = o2 * qscaleM;
                    dout[j + 3] = o3 * qscaleM;
                }
                for (; j < VHD; j++) {
                    float *row = Mp + (size_t)j * KHD;
                    float sk = 0;
                    for (int i = 0; i < KHD; i++) { float rv = row[i] * eg; row[i] = rv; sk += rv * kk[i]; }
                    float d = (vv[j] - sk) * b_;
                    float so = 0;
                    for (int i = 0; i < KHD; i++) { float rv = row[i] + kk[i] * d; row[i] = rv; so += rv * qq[i]; }
                    dout[j] = so * qscaleM;
                }
            }
            #pragma omp parallel for collapse(2) schedule(static) if(g_omp)
            for (int q = 0; q < M; q++) for (int h = 0; h < NVH; h++) {
                float *p = g_mD.data() + (size_t)q * EMB + h * KHD;
                const float *zp = g_mZZ.data() + (size_t)q * EMB;
                double ss = 0; for (int i = 0; i < KHD; i++) ss += (double)p[i] * p[i];
                float r = 1.f / sqrtf((float)(ss / KHD) + EPS);
                for (int i = 0; i < KHD; i++) p[i] = p[i] * r * L.ssm_norm[i] * siluf(zp[h * KHD + i]);
            }
            g_p_ssm += (NOWS_() - _sb2) - (g_i8_t - _sc2);
        } else {
            // Q/gate 提取 + per-head RMSNorm + MRoPE (逐槽各自的位置)
            #pragma omp parallel for collapse(2) schedule(static) if(g_omp)
            for (int q = 0; q < M; q++) for (int h = 0; h < NH; h++) {
                const float *qg = g_mY.data() + (size_t)q * ystride + AA_Q;
                float *q5 = &g_mQ5[((size_t)q * NH + h) * HD];
                memcpy(q5, &qg[h * HD * 2], HD * 4);
                memcpy(&g_mG5[((size_t)q * NH + h) * HD], &qg[h * HD * 2 + HD], HD * 4);
                rmsnorm(q5, q5, L.q_norm.data(), HD, EPS);
                rope_apply(q5, act[(size_t)q]->pos, HD);
            }
            #pragma omp parallel for collapse(2) schedule(static) if(g_omp)
            for (int q = 0; q < M; q++) for (int h = 0; h < NKV; h++) {
                memcpy(&g_mKK[((size_t)q * NKV + h) * HD], g_mY.data() + (size_t)q * ystride + AA_K + h * HD, HD * 4);
                memcpy(&g_mVV[((size_t)q * NKV + h) * HD], g_mY.data() + (size_t)q * ystride + AA_V + h * HD, HD * 4);
                rmsnorm(&g_mKK[((size_t)q * NKV + h) * HD], &g_mKK[((size_t)q * NKV + h) * HD], L.k_norm.data(), HD, EPS);
                rope_apply(&g_mKK[((size_t)q * NKV + h) * HD], act[(size_t)q]->pos, HD);
            }
            // 写入【各槽自己的】KV 缓存 (位置 = 该槽的 pos)
            #pragma omp parallel for collapse(2) schedule(static) if(g_omp)
            for (int q = 0; q < M; q++) for (int h = 0; h < NKV; h++) {
                int pos = act[(size_t)q]->pos;
                memcpy(&act[(size_t)q]->Kc[(size_t)l][(size_t)pos * NKV * HD + (size_t)h * HD],
                       &g_mKK[((size_t)q * NKV + h) * HD], HD * 4);
                memcpy(&act[(size_t)q]->Vc[(size_t)l][(size_t)pos * NKV * HD + (size_t)h * HD],
                       &g_mVV[((size_t)q * NKV + h) * HD], HD * 4);
            }
            const float scM = 1.f / sqrtf((float)HD);
            int gprM = NH / NKV;
            // 因果注意力 (GQA) + 门控: 逐槽、逐头; 槽内按 t 升序单累加器 (与单槽路径同序)
            #pragma omp parallel for collapse(2) schedule(static) if(g_omp)
            for (int q = 0; q < M; q++) for (int h = 0; h < NH; h++) {
                int pos = act[(size_t)q]->pos;
                int kh = h / gprM;
                const float *Kc = act[(size_t)q]->Kc[(size_t)l].data();
                const float *Vc = act[(size_t)q]->Vc[(size_t)l].data();
                const float *qp = &g_mQ5[((size_t)q * NH + h) * HD];
                float *wth = &g_mWT[((size_t)q * NH + h) * MAXT];
                float best = -1e30f;
                for (int t = 0; t <= pos; t++) {
                    const float *kt = &Kc[((size_t)t * NKV + kh) * HD];
                    float s = 0; for (int i = 0; i < HD; i++) s += qp[i] * kt[i];
                    s *= scM; wth[t] = s; if (s > best) best = s;
                }
                float sum = 0;
                for (int t = 0; t <= pos; t++) { float e = expf(wth[t] - best); wth[t] = e; sum += e; }
                float *oh = &g_mAO[((size_t)q * NH + h) * HD];
                for (int i = 0; i < HD; i++) oh[i] = 0;
                for (int t = 0; t <= pos; t++) {
                    float p = wth[t] / sum;                 // ★ 必须用本头自己的 wth
                    const float *vt = &Vc[((size_t)t * NKV + kh) * HD];
                    for (int i = 0; i < HD; i++) oh[i] += p * vt[i];
                }
                const float *gp = &g_mG5[((size_t)q * NH + h) * HD];
                for (int i = 0; i < HD; i++) oh[i] *= sigmoidf_(gp[i]);
            }
            #pragma omp parallel for schedule(static) if(g_omp)
            for (int q = 0; q < M; q++)
                memcpy(g_mD.data() + (size_t)q * EMB, g_mAO.data() + (size_t)q * NH * HD, (size_t)NH * HD * 4);
            g_p_att += (NOWS_() - _sb2) - (g_i8_t - _sc2);
        }
        { snprintf(g_nb, sizeof(g_nb), "l%d.mid", l); g_gname = g_nb; }
        gemv_batch(L.grp_mid, g_mD.data(), M, g_mO.data());
        _sb3 = NOWS_(); _sc3 = g_i8_t;
        #pragma omp parallel for schedule(static) if(g_omp)
        for (int q = 0; q < M; q++) {
            float *xq = g_mX.data() + (size_t)q * EMB;
            const float *oq = g_mO.data() + (size_t)q * EMB;
            for (int i = 0; i < EMB; i++) xq[i] += oq[i];
            rmsnorm(g_mXB.data() + (size_t)q * EMB, xq, L.post_norm.data(), EMB, EPS);
        }
        { snprintf(g_nb, sizeof(g_nb), "l%d.grpFF", l); g_gname = g_nb; }
        gemv_batch(L.grp_ffn, g_mXB.data(), M, g_mY.data());
        #pragma omp parallel for collapse(2) schedule(static) if(g_omp)
        for (int q = 0; q < M; q++) for (int i = 0; i < NFF; i++) {
            const float *gy = g_mY.data() + (size_t)q * FF_TOT;
            g_mGG[(size_t)q * NFF + i] = siluf(gy[FF_G + i]) * gy[FF_U + i];
        }
        { snprintf(g_nb, sizeof(g_nb), "l%d.ffn_down", l); g_gname = g_nb; }
        gemv_batch(L.ffn_down, g_mGG.data(), M, g_mO.data());
        #pragma omp parallel for schedule(static) if(g_omp)
        for (int q = 0; q < M; q++) {
            float *xq = g_mX.data() + (size_t)q * EMB;
            const float *oq = g_mO.data() + (size_t)q * EMB;
            for (int i = 0; i < EMB; i++) xq[i] += oq[i];
        }
        g_p_lay += (NOWS_() - _sb3) - (g_i8_t - _sc3);
    }
    _sbl = NOWS_(); _scl = g_i8_t;
    #pragma omp parallel for schedule(static) if(g_omp)
    for (int q = 0; q < M; q++)
        rmsnorm(g_mXB.data() + (size_t)q * EMB, g_mX.data() + (size_t)q * EMB, g_out_norm.data(), EMB, EPS);
    { snprintf(g_nb, sizeof(g_nb), "lm_head"); g_gname = g_nb; }
    gemv_batch(g_outw, g_mXB.data(), M, g_mLOG.data());           // ★ lm_head 也只读一遍 (per-slot logits)
    g_p_lm += (NOWS_() - _sbl) - (g_i8_t - _scl);
}

// ============================ selftest: 卡上 Q8_0 GEMV vs 主机参考 ============================
static int selftest(GF &g, const char *name, int rows) {
    const GT *t = g.get(name);
    if (!t) { printf("no tensor %s\n", name); return 1; }
    int N = (int)t->dims[0], M = (int)t->dims[1];
    int Mt = rows < M ? rows : M;
    int stdev = getenv("K200_STDEV") ? atoi(getenv("K200_STDEV")) : 0;
    WDev w = up_q8(g, name, stdev, false);   // selftest 走单片, 保持诊断语义
    std::vector<float> x(N), y(M);
    srand(7);
    for (int i = 0; i < N; i++) x[i] = (float)((rand() % 2001) - 1000) / 1000.f;
    if (getenv("K200_XFILE")) {
        FILE *fp = fopen(getenv("K200_XFILE"), "rb");
        if (fp) { size_t r = fread(x.data(), 4, N, fp); fclose(fp); printf("[SELFTEST] x 来自 %s (%zu floats)\n", getenv("K200_XFILE"), r); }
    }
    // 硬件 grid 信息
    {
        int *o = nullptr; xpu_set_device(stdev);
        if (xpu_malloc((void **)&o, 8) == 0) {
            kq8_info(o); xpu_wait();
            int h[2] = {0, 0}; xpu_memcpy(h, o, 8, XPU_DEVICE_TO_HOST);
            printf("[INFO] 设备 grid: cluster_num=%d core_num=%d -> 硬件线程数=%d\n", h[0], h[1], h[0] * h[1]);
        }
    }
    // 主机参考: 只解前 Mt 行
    uint64_t rowbytes = (uint64_t)(N / 32) * 34;
    std::vector<unsigned char> raw((size_t)rowbytes * Mt);
    fseek(g.f, (long)(g.data_start + t->off), SEEK_SET);
    if (fread(raw.data(), 1, raw.size(), g.f) != raw.size()) { printf("read fail\n"); return 1; }
    std::vector<float> ref((size_t)Mt * N);
    dequant_q8_0(raw.data(), ref.data(), (uint64_t)Mt * N);
    std::vector<float> refy(Mt);
    for (int m = 0; m < Mt; m++) {
        double acc = 0;
        for (int n = 0; n < N; n++) acc += (double)ref[(size_t)m * N + n] * x[n];
        refy[m] = (float)acc;
    }
    int cl0 = g_cl, co0 = g_co, xi0 = g_xi8;
    int clv[16] = {4, 8, 8, 16, 8, 8, 16, 16, 4, 2, 8, 16, 8, 16, 8, 8};
    int cov[16] = {16, 16, 32, 32, 64, 128, 128, 64, 32, 32, 16, 16, 8, 8, 256, 512};
    int xiv[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    double best = 1e30; int bcl = cl0, bco = co0, bxi = xi0;
    std::vector<unsigned char> rowbuf(rowbytes);
    std::vector<float> refrow(N);
    for (int k = 0; k < 16; k++) {
        // 先跑一遍覆盖全部行再计时
        gemv_single(stdev, w, x.data(), y.data(), clv[k], cov[k]);
        g_d2h = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 2; r++) gemv_single(stdev, w, x.data(), y.data(), clv[k], cov[k]);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 2.0;
        // 全 M 行对拍 (逐行, 不爆内存)
        double se = 0, sr = 0; float mx = 0; int bad = 0, firstbad = -1, nf = 0;
        for (int m = 0; m < M; m++) {
            if (!std::isfinite(y[m])) { nf++; if (firstbad < 0) firstbad = m; }
            if (m < Mt) continue;
            fseek(g.f, (long)(g.data_start + t->off + (uint64_t)m * rowbytes), SEEK_SET);
            if (fread(rowbuf.data(), 1, rowbytes, g.f) != rowbytes) break;
            dequant_q8_0(rowbuf.data(), refrow.data(), N);
            double acc = 0;
            for (int n = 0; n < N; n++) acc += (double)refrow[n] * x[n];
            double d = acc - y[m]; se += d * d; sr += acc * acc;
            if (fabs(d) > mx) mx = (float)fabs(d);
            if (fabs(acc) > 1e-12 && sqrt(d * d / (acc * acc)) > 0.02) { bad++; if (firstbad < 0) firstbad = m; }
        }
        printf("[SWEEP] %-22s xi8=%d cl=%3d co=%3d  %6.2f ms  %5.2f GB/s  全M relrms=%.4f%% maxabs=%.4g  坏行=%d 首坏=%d 非有限=%d\n",
               name, xiv[k], clv[k], cov[k], ms, (double)w.bytes / 1073741824.0 / (ms / 1000.0),
               100.0 * sqrt(se / sr), mx, bad, firstbad, nf);
        if (ms < best && bad == 0 && nf == 0) { best = ms; bcl = clv[k]; bco = cov[k]; bxi = xiv[k]; }
    }
    if (best < 1e29) { g_cl = bcl; g_co = bco; g_xi8 = bxi;
        printf("[BEST] %s -> xi8=%d cl=%d co=%d  %.2f ms  %.2f GB/s (全 M 行零错误)\n", name, bxi, bcl, bco, best,
               (double)w.bytes / 1073741824.0 / (best / 1000.0)); }
    else printf("[BEST] 没有一种 grid 配置做到全 M 行零错误!\n");
    printf("[SELFTEST] 样例: y[0]=%.6f 期望=%.6f | y[1]=%.6f 期望=%.6f\n", y[0], refy[0], y[1], refy[1]);
    return 0;
}

// ============================ 主程序 ============================
static const char *g_prompt_tmpl = nullptr;   // K200_TMPL 覆盖 (含 {q})

int main(int argc, char **argv) {
    // ★★ 第32轮: 忽略 SIGPIPE ★★
    //   对端(网关)被 kill 后, 引擎的 printf(stdout 是一个已无读者的管道) 默认会收到 SIGPIPE
    //   ⇒ 引擎被信号打死 (只剩宿主重建 VM/80s 重启这条路)。忽略后 printf 只返回 EPIPE,
    //   引擎继续服务; 配合下面的"读到 EOF 不退出 + 空闲重附着 FIFO"才真正做到"客户端中断不用重启"。
    signal(SIGPIPE, SIG_IGN);
    int ngen = 128;
    const char *path = nullptr, *gen = nullptr, *stname = nullptr;
    int strows = 64;
    bool do_selftest = false, tokdump = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n") && i + 1 < argc) ngen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = argv[++i];
        else if (!strcmp(argv[i], "--selftest")) do_selftest = true;
        else if (!strcmp(argv[i], "--stname") && i + 1 < argc) stname = argv[++i];
        else if (!strcmp(argv[i], "--strows") && i + 1 < argc) strows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokdump")) tokdump = true;
    }
    if (!path) path = getenv("K200_MODEL_PATH");
    if (!path) path = "/home/caden/orn/Ornith-1.5-9B-Q8_0.gguf";
    if (getenv("K200_CL"))  g_cl  = atoi(getenv("K200_CL"));
    if (getenv("K200_CO"))  g_co  = atoi(getenv("K200_CO"));
    if (getenv("K200_XI8")) g_xi8 = atoi(getenv("K200_XI8"));
    if (getenv("K200_DUMP")) g_dump = atoi(getenv("K200_DUMP"));
    if (getenv("K200_NOSPLIT")) g_nosplit = atoi(getenv("K200_NOSPLIT"));
    if (getenv("K200_SENT")) g_sent = atoi(getenv("K200_SENT"));
    if (getenv("K200_I8ROW")) g_i8row = atoi(getenv("K200_I8ROW"));
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (getenv("K200_LDUMP")) { g_ldump = fopen(getenv("K200_LDUMP"), "wb");
        printf("[orn] logits dump -> %s (%s)\n", getenv("K200_LDUMP"), g_ldump ? "OK" : "打开失败"); }
    if (getenv("K200_PROF"))  g_prof = atoi(getenv("K200_PROF"));
    if (getenv("K200_FTZ"))   g_ftz = atoi(getenv("K200_FTZ"));
    if (getenv("K200_OMP"))   g_omp = atoi(getenv("K200_OMP"));
    if (getenv("K200_BMIN"))  g_bmin = atoi(getenv("K200_BMIN"));
    if (getenv("K200_BMAX"))  { g_bmax = atoi(getenv("K200_BMAX")); if (g_bmax > BATCH_MAXT) g_bmax = BATCH_MAXT; }
    {   const char *e = getenv("K200_MAXT");                 // ★ 第30轮(问题⑥)
        if (e) { int v = atoi(e); if (v >= 512 && v <= 16384) MAXT = v; }
        printf("[orn] ★★ 上下文上限 MAXT = %d tok (提示词+生成合计; K200_MAXT 可调, 默认 4096)\n", MAXT);
        printf("[orn]    内存账: 每槽 KV = %d 层 × %d tok × %d KV × %d HD × 2(K,V) × 4B = %.1f MB;"
               " 每槽 delta-net S+conv = %.1f MB\n", NLAYER / 4, MAXT, NKV, HD,
               (double)(NLAYER / 4) * MAXT * NKV * HD * 2 * 4 / 1048576.0,
               (double)NLAYER * (NVH * VHD * KHD + 3 * CVD) * 4 / 1048576.0);
    }
    printf("[orn] ★ prefill 批量化: bmin=%d bmax=%d (K200_BMIN/K200_BMAX; bmin 大 = 短 prompt 仍走逐 token 路径)\n", g_bmin, g_bmax);

    printf("[orn] model=%s\n", path);
    if (!g_gguf.open(path)) return 1;
    printf("[orn] GGUF: %zu tensors, %zu tokens, %zu merges, arch=%s\n",
           g_gguf.ts.size(), g_gguf.tokens.size(), g_gguf.merges.size(),
           g_gguf.strs.count("general.architecture") ? g_gguf.strs["general.architecture"].c_str() : "?");
    // ---- 从 GGUF 取 special token / chat_template (模型无关) ----
    std::map<std::string,int> tid;
    for (size_t i = 0; i < g_gguf.tokens.size(); i++) tid[g_gguf.tokens[i]] = (int)i;
    const char *want_tok[] = {"<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|endoftext|>",
                              "<|vision_start|>", "<|image_pad|>", "<tool_call>", "</tool_call>"};
    g_img_pad = tid.count("<|image_pad|>") ? tid["<|image_pad|>"] : -1;
    printf("[orn] ★ VIS: image_pad id = %d\n", g_img_pad);
    printf("[orn] special tokens from GGUF:");
    for (int i = 0; i < 9; i++) printf(" %s=%d", want_tok[i], tid.count(want_tok[i]) ? tid[want_tok[i]] : -1);
    printf("\n");
    std::string tmpl = g_gguf.strs.count("tokenizer.chat_template") ? g_gguf.strs["tokenizer.chat_template"] : "";
    printf("[orn] chat_template: %zu bytes from GGUF, head: %.90s\n", tmpl.size(), tmpl.c_str());
    printf("[orn] pre=%s eos=%llu\n", g_gguf.strs.count("tokenizer.ggml.pre") ? g_gguf.strs["tokenizer.ggml.pre"].c_str() : "?",
           (unsigned long long)(g_gguf.u64s.count("tokenizer.ggml.eos_token_id") ? g_gguf.u64s["tokenizer.ggml.eos_token_id"] : 0));

    if (do_selftest) {
        xpu_set_device(0);
        for (int dv = 0; dv < 2; dv++) { xpu_set_device(dv);
            if (xpu_malloc(&g_dx[dv], 16384 * 4) || xpu_malloc(&g_dy[dv], (size_t)(300000) * 4) ||
                xpu_malloc(&g_dxq[dv], 16384) || xpu_malloc(&g_dxs[dv], 1024 * 4)) { printf("FATAL: scratch dev%d\n", dv); return 1; } }
        const char *tn = stname ? stname : "blk.0.ffn_gate.weight";
        int rc = selftest(g_gguf, tn, strows);
        if (rc) return rc;
        if (!stname) return selftest(g_gguf, "blk.3.attn_q.weight", strows);
        return 0;
    }

    // ---- tokenizer ----
    g_tk.init(g_gguf.tokens, g_gguf.merges,
              (int)(g_gguf.u64s.count("tokenizer.ggml.bos_token_id") ? g_gguf.u64s["tokenizer.ggml.bos_token_id"] : 0),
              (int)(g_gguf.u64s.count("tokenizer.ggml.eos_token_id") ? g_gguf.u64s["tokenizer.ggml.eos_token_id"] : 1),
              g_gguf.ttypes.empty() ? nullptr : &g_gguf.ttypes,
              g_gguf.strs.count("tokenizer.ggml.pre") ? g_gguf.strs["tokenizer.ggml.pre"] : std::string());
    printf("[orn] tokenizer ready: vocab=%zu merges=%zu specials=%zu digits_single=%d\n",
           g_tk.vocab.size(), g_tk.rank.size(), g_tk.specials.size(), (int)g_tk.digits_single);
    if (tokdump) {
        std::string t = "你好";
        std::vector<int> ids = g_tk.encode(t);
        printf("[orn] tokdump '%s' ->", t.c_str());
        for (size_t i = 0; i < ids.size(); i++) printf(" %d", ids[i]);
        printf("\n");
        return 0;
    }

    g_x.resize(EMB); g_xb.resize(EMB);
    // ★ 先把卡上临时缓冲区分配掉(低地址), 再灌权重; 规避高地址写入丢数据
    for (int dv = 0; dv < 2; dv++) {
        xpu_set_device(dv);
        if (xpu_malloc(&g_dx[dv], (size_t)BATCH_MAXT * BATCH_MAXN * 4) || xpu_malloc(&g_dy[dv], (size_t)BATCH_MAXT * 130000 * 4) ||
            xpu_malloc(&g_dxq[dv], 32768) || xpu_malloc(&g_dxs[dv], 2048 * 4))
            printf("[orn] WARN: chip%d 临时缓冲区分配失败\n", dv);
        if (g_i8row) {   // ★ 官方算子上下文 (内含 64MB SD-CDNN workspace, 也放低地址)
            g_ctx[dv] = new api::Context(api::Device(api::DeviceType::XPU1, dv));
            printf("[orn] chip%d api::Context 就绪 (kXPU1 id=%d)\n", dv, dv);
        }
    }
    for (int dv = 0; dv < 2; dv++)
        printf("[orn] scratch chip%d: dx=%p dy=%p dxq=%p dxs=%p\n", dv, g_dx[dv], g_dy[dv], g_dxq[dv], g_dxs[dv]);
    // ---- 载入: 每个矩阵的行对半拆到两芯; 同一个 x 的矩阵拼接成一个 launch ----
    g_lay.resize(NLAYER);
    {
        auto t0 = std::chrono::steady_clock::now();
        char nm[128];
        for (int l = 0; l < NLAYER; l++) {
            Layer &L = g_lay[l];
            L.recr = is_recr(l);
            snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", l);           L.attn_norm = load_f32_host(g_gguf, nm);
            snprintf(nm, sizeof(nm), "blk.%d.post_attention_norm.weight", l); L.post_norm = load_f32_host(g_gguf, nm);
            if (L.recr) {
                {   Pack p;
                    snprintf(nm, sizeof(nm), "blk.%d.attn_qkv.weight", l);   p.add(g_gguf, nm);
                    snprintf(nm, sizeof(nm), "blk.%d.attn_gate.weight", l);  p.add(g_gguf, nm);
                    if (p.segM[0] != PA_QKVS || p.segM[1] != PA_GATES || p.M != (PA_GATE + PA_GATES)) {
                        printf("FATAL: recr grpA 段布局不符 (%d,%d,%d) 期望 (%d,%d,%d)\n",
                               p.segM[0], p.segM[1], p.M, PA_QKVS, PA_GATES, PA_GATE + PA_GATES); exit(1); }
                    L.grp_a = p.finish(); p.release(); }
                {   Pack p;
                    snprintf(nm, sizeof(nm), "blk.%d.ssm_out.weight", l);    p.add(g_gguf, nm);
                    if (p.M != EMB) { printf("FATAL: ssm_out M=%d != %d\n", p.M, EMB); exit(1); }
                    L.grp_mid = p.finish(); p.release(); }
                int aM = 0, aN = 0;
                snprintf(nm, sizeof(nm), "blk.%d.ssm_alpha.weight", l);  load_q8_host(g_gguf, nm, L.alpha_w, aM, aN);
                snprintf(nm, sizeof(nm), "blk.%d.ssm_beta.weight", l);   load_q8_host(g_gguf, nm, L.beta_w, aM, aN);
                snprintf(nm, sizeof(nm), "blk.%d.ssm_conv1d.weight", l); L.conv1d = load_f32_host(g_gguf, nm);
                snprintf(nm, sizeof(nm), "blk.%d.ssm_a", l);             L.ssm_a  = load_f32_host(g_gguf, nm);
                snprintf(nm, sizeof(nm), "blk.%d.ssm_dt.bias", l);       L.ssm_dt = load_f32_host(g_gguf, nm);
                snprintf(nm, sizeof(nm), "blk.%d.ssm_norm.weight", l);   L.ssm_norm = load_f32_host(g_gguf, nm);
            } else {
                {   Pack p;
                    snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", l);     p.add(g_gguf, nm);
                    snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", l);     p.add(g_gguf, nm);
                    snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", l);     p.add(g_gguf, nm);
                    if (p.segM[0] != AA_QS || p.segM[1] != AA_KS || p.segM[2] != AA_VS || p.M != AA_TOT) {
                        printf("FATAL: attn grpA 段布局不符 (%d,%d,%d,%d)\n", p.segM[0], p.segM[1], p.segM[2], p.M); exit(1); }
                    L.grp_a = p.finish(); p.release(); }
                {   Pack p;
                    snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", l); p.add(g_gguf, nm);
                    if (p.M != EMB) { printf("FATAL: attn_output M=%d != %d\n", p.M, EMB); exit(1); }
                    L.grp_mid = p.finish(); p.release(); }
                snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", l); L.q_norm = load_f32_host(g_gguf, nm);
                snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", l); L.k_norm = load_f32_host(g_gguf, nm);
                L.Kc.assign((size_t)MAXT * NKV * HD, 0.f);
                L.Vc.assign((size_t)MAXT * NKV * HD, 0.f);
            }
            {   Pack p;
                snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", l); p.add(g_gguf, nm);
                snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight", l);   p.add(g_gguf, nm);
                if (p.segM[0] != FF_GS || p.segM[1] != FF_US || p.M != FF_TOT) {
                    printf("FATAL: ffn grp 段布局不符 (%d,%d,%d)\n", p.segM[0], p.segM[1], p.M); exit(1); }
                L.grp_ffn = p.finish(); p.release(); }
            snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", l); L.ffn_down = up_q8(g_gguf, nm, 0);
            if (l % 4 == 3 || l == NLAYER - 1)
                printf("[orn] 已载入 %d/%d 层  chip0=%.0fMB chip1=%.0fMB\n", l + 1, NLAYER,
                       g_hbm[0] / 1048576.0, g_hbm[1] / 1048576.0);
        }
        g_out_norm = load_f32_host(g_gguf, "output_norm.weight");
        g_outw = up_q8(g_gguf, "output.weight", 0);
        auto t1 = std::chrono::steady_clock::now();
        printf("[orn] 权重常驻 HBM: chip0=%.1f MB  chip1=%.1f MB  合计=%.2f GiB  用时 %.1fs\n",
               g_hbm[0] / 1048576.0, g_hbm[1] / 1048576.0,
               (g_hbm[0] + g_hbm[1]) / 1073741824.0,
               std::chrono::duration<double>(t1 - t0).count());
    }
    for (int l = 0; l < NLAYER; l++) { g_pKc[l] = &g_lay[l].Kc; g_pVc[l] = &g_lay[l].Vc; }
    rope_init(MAXT);
    g_st.conv.assign((size_t)NLAYER * 3 * CVD, 0.f);
    g_st.S.assign((size_t)NLAYER * NVH * VHD * KHD, 0.f);
    if (g_co > 16) { printf("[orn] ★警告: co=%d > 16, 每 cluster 只有 16 个核会执行, 行会漏写!\n", g_co); }
    printf("[orn] grid: cl=%d co=%d xi8=%d (硬件每 cluster 核数=16)\n", g_cl, g_co, g_xi8);
    printf("[orn] ★ GEMV 路径 = %s\n", g_i8row
           ? "官方 gemm_int8 (每输出行 int8 + 每行 float scale, SD-CDNN 引擎)"
           : "自写 kq8v2 (GM2LM) [回退模式]");
    printf("[orn] 就绪. 主机常驻 delta-net S 状态 %.1f MB, KV(8 注意层) %.1f MB\n",
           g_st.S.size() * 4.0 / 1048576.0,
           (double)NLAYER / 4 * MAXT * NKV * HD * 2 * 4.0 / 1048576.0);

    int eos = (int)(g_gguf.u64s.count("tokenizer.ggml.eos_token_id") ? g_gguf.u64s["tokenizer.ggml.eos_token_id"] : 248046);
    std::vector<int> stops; stops.push_back(eos);
    stops.push_back(tid.count("<|endoftext|>") ? tid["<|endoftext|>"] : 248044);

    // ================= 会话/KV 状态复用 (消除每请求重复 prefill) =================
    //   (a) prompt 级快照: 同一 prompt 再来一次 => 直接恢复状态, prefill 0 秒
    //   (b) 同会话续算: 新 prompt 是上一轮已吃进状态的前缀延伸 => 只 prefill 新增的 token
    struct Snapshot {
        bool valid = false;
        std::vector<int> ids;
        int pos = 0;
        uint64_t img_id = 0;      // ★ DETVIS: 该快照依赖的图像指纹 (0 = 纯文本/无图)
        std::vector<float> conv, S, logits;
        std::vector<std::vector<float> > Kc, Vc;
    };
    static Snapshot g_snap;
    struct Sess { std::vector<int> consumed; int pos = 0; uint64_t img_id = 0; };
    static std::map<std::string, Sess> g_sess;
    static std::string g_live_sid;
    static int g_live_pos = -1;
    auto take_snap = [&](const std::vector<int> &sids, int spos, const std::vector<float> &lg, uint64_t iid) {
        g_snap.ids = sids; g_snap.pos = spos; g_snap.img_id = iid;
        g_snap.conv = g_st.conv; g_snap.S = g_st.S; g_snap.logits = lg;
        g_snap.Kc.resize(NLAYER); g_snap.Vc.resize(NLAYER);
        for (int l = 0; l < NLAYER; l++) {
            if (!g_lay[l].recr) { g_snap.Kc[l] = g_lay[l].Kc; g_snap.Vc[l] = g_lay[l].Vc; }
            else { std::vector<float>().swap(g_snap.Kc[l]); std::vector<float>().swap(g_snap.Vc[l]); }
        }
        g_snap.valid = true;
    };
    auto restore_snap = [&](std::vector<float> &lg) {
        g_st.conv = g_snap.conv; g_st.S = g_snap.S; lg = g_snap.logits;
        for (int l = 0; l < NLAYER; l++)
            if (!g_lay[l].recr) { g_lay[l].Kc = g_snap.Kc[l]; g_lay[l].Vc = g_snap.Vc[l]; }
    };
    auto state_bytes = [&]() {
        double b = (double)(g_st.conv.size() + g_st.S.size()) * 4;
        for (int l = 0; l < NLAYER; l++) if (!g_lay[l].recr) b += (double)(g_lay[l].Kc.size() + g_lay[l].Vc.size()) * 4;
        return b;
    };

    // ========================================================================
    // ★★ 第27轮: 前缀 KV 复用 —— "LCP 命中 + 状态回滚" 检查点表 ★★
    //   问题: DSH 等客户端每轮把整段上下文重发 ⇒ 若只认"全等"就每轮从头预填充
    //         (实测 603 tok = 16.8s)。本轮改成: 只要本轮 prompt 的某段前缀已经算过,
    //         就把状态回滚到那个位置, 只预填充新增 token。
    //   一个检查点 = (token 序列, 该序列末尾的完整状态):
    //         conv/S (delta-net) + 每注意层 Kc/Vc 前 pos 行 + logits + g_img_used + 图像依赖
    //   打点位置 (三个都打, 覆盖不同"重发形态"):
    //         ① 每条消息边界 (下一个 token 是 <|im_start|>) —— _TRIM 裁点移动后仍能回滚
    //         ② 预填充末尾 (整段 prompt) —— 同一 prompt 重发 ⇒ 0 预填充
    //         ③ 生成末尾 (prompt+回答) —— 下一轮"整段重发 + 追加" ⇒ 只算新增的那段
    //   命中规则 (保守, 与 DETVIS 图像指纹同源):
    //         检查点的 token 序列必须是本次 prompt 的【前缀】(逐 token 相同);
    //         且若该前缀里含 <|image_pad|> ⇒ 图像指纹必须与本次相同 (前缀纯文本 ⇒ 与图无关)。
    //   位级等价: 回滚只是把"同一段 token 算出来的状态"搬回原位, 不改任何数值路径;
    //         后续 token 仍走同一条 forward/forward_batch 代码 (批大小可能不同, 由
    //         第13/21章的批量等价性覆盖) ⇒ 输出 token 逐字节可对拍。
    // ========================================================================
    struct Ckpt {
        bool valid = false;
        std::vector<int> ids;     // 该状态覆盖的 token 序列 (prompt+generated 的前缀)
        int pos = 0;              // = ids.size() (即下一个 token 的位置索引)
        uint64_t imgdep = 0;      // 前缀含 image_pad 时的图像指纹; 0 = 前缀与图像无关
        int img_used = 0;         // 该状态下 g_img_used 的取值
        uint64_t last = 0;        // LRU 时间戳
        std::vector<float> conv, S, logits;
        std::vector<std::vector<float> > Kc, Vc;
    };
    static std::vector<Ckpt> g_ck;
    static uint64_t g_ck_tick = 0;
    static bool g_ck_inited = false;
    if (!g_ck_inited) {
        g_ck_inited = true;
        const char *e = getenv("K200_CKPT");
        int n = e ? atoi(e) : 8;
        if (n < 0) n = 0; if (n > 24) n = 24;
        g_ck.resize(n);
        printf("[orn] ★ 前缀 KV 复用(第27轮): 检查点表 %d 槽 (K200_CKPT 可调, 0=关闭); 打点 = 消息边界 + 预填充末尾 + 生成末尾\n", n);
    }
    auto ck_mb = [&]() {
        double b = 0;
        for (size_t i = 0; i < g_ck.size(); i++) if (g_ck[i].valid) {
            b += (double)(g_ck[i].conv.size() + g_ck[i].S.size() + g_ck[i].logits.size()) * 4;
            for (int l = 0; l < NLAYER; l++) if (!g_lay[l].recr) b += (double)(g_ck[i].Kc[l].size() + g_ck[i].Vc[l].size()) * 4;
        }
        return b / 1048576.0;
    };
    auto ck_nvalid = [&]() { int n = 0; for (size_t i = 0; i < g_ck.size(); i++) if (g_ck[i].valid) n++; return n; };
    // 打一个检查点 (sids[0..spos) == 已算过的 token 序列; 状态 = 当前 g_st/g_lay)
    auto ck_take = [&](const std::vector<int> &sids, int spos, const std::vector<float> &lg, int img_used) {
        if (g_ck.empty() || spos <= 0 || (int)sids.size() < spos) return;
        int slot = -1;
        for (size_t i = 0; i < g_ck.size(); i++)           // 同位置 + 同前缀 ⇒ 覆盖 (幂等, 不浪费槽)
            if (g_ck[i].valid && g_ck[i].pos == spos && std::equal(g_ck[i].ids.begin(), g_ck[i].ids.end(), sids.begin())) { slot = (int)i; break; }
        if (slot < 0) for (size_t i = 0; i < g_ck.size(); i++) if (!g_ck[i].valid) { slot = (int)i; break; }
        if (slot < 0) { slot = 0; for (size_t i = 1; i < g_ck.size(); i++) if (g_ck[i].last < g_ck[(size_t)slot].last) slot = (int)i; }   // LRU 淘汰
        if (slot < 0) return;
        Ckpt &C = g_ck[(size_t)slot];
        C.ids.assign(sids.begin(), sids.begin() + spos);
        C.pos = spos;
        C.conv = g_st.conv; C.S = g_st.S; C.logits = lg;
        C.Kc.resize(NLAYER); C.Vc.resize(NLAYER);
        size_t nr = (size_t)spos * NKV * HD;
        for (int l = 0; l < NLAYER; l++) {
            if (!g_lay[l].recr) {
                C.Kc[(size_t)l].assign(g_lay[l].Kc.begin(), g_lay[l].Kc.begin() + nr);
                C.Vc[(size_t)l].assign(g_lay[l].Vc.begin(), g_lay[l].Vc.begin() + nr);
            } else { std::vector<float>().swap(C.Kc[(size_t)l]); std::vector<float>().swap(C.Vc[(size_t)l]); }
        }
        C.img_used = img_used;
        C.imgdep = 0;
        if (g_img_n > 0 && g_img_pad >= 0)
            for (int k = 0; k < spos; k++) if (C.ids[(size_t)k] == g_img_pad) { C.imgdep = g_img_id; break; }
        C.last = ++g_ck_tick; C.valid = true;
    };
    // 回滚: 把检查点状态搬回 g_st/g_lay/g_img_used (KV 只搬前 pos 行, 后面的行在继续
    // prefill 时会被逐位覆盖, 不会被读到)
    auto ck_restore = [&](int slot, std::vector<float> &lg) {
        Ckpt &C = g_ck[(size_t)slot];
        g_st.conv = C.conv; g_st.S = C.S; lg = C.logits;
        size_t nr = (size_t)C.pos * NKV * HD;
        for (int l = 0; l < NLAYER; l++)
            if (!g_lay[l].recr) {
                memcpy(g_lay[l].Kc.data(), C.Kc[(size_t)l].data(), nr * 4);
                memcpy(g_lay[l].Vc.data(), C.Vc[(size_t)l].data(), nr * 4);
            }
        g_img_used = C.img_used;
        C.last = ++g_ck_tick;
    };
    const int im_start_id = tid.count("<|im_start|>") ? tid["<|im_start|>"] : -1;

    // ========================================================================
    // ★★ 第30轮: 超上下文上限时的【不丢尾】策略 (问题① 的引擎侧那一半) ★★
    //   旧行为: 预填充循环里 `if (pos >= MAXT) { ids.resize(i); break; }` ⇒ 保头丢尾:
    //   把用户最后那一条消息连同 <|im_end|> 一起丢掉 ⇒ 模型拿到的是被腰斩、没有
    //   ChatML 闭合的上下文 ⇒ 当场输出 EOS ⇒ 生成 0 token (网关又当"成功"返回空)。
    //   新行为: 从【头部】砍掉超出部分, 保住尾部(最新消息 + <|im_end|> + assistant 生成头);
    //   并且绝不把 <|vision_start|>..<|image_pad|>*..<|vision_end|> 图像块切成两半
    //   (切一半会让图像行错位 ⇒ 看图退化), 要么整块留、要么整块丢。
    // ========================================================================
    const int vis_e_id = tid.count("<|vision_end|>") ? tid["<|vision_end|>"] : -1;
    auto head_trim_keep_tail = [&](std::vector<int> &v, int maxn) -> int {
        if ((int)v.size() <= maxn) return 0;
        int drop = (int)v.size() - maxn;
        if (g_img_pad >= 0 && vis_e_id >= 0) {              // 图像块不许切一半
            int first_pad = -1, last_end = -1;
            for (int k = 0; k < (int)v.size(); k++) {
                if (first_pad < 0 && v[(size_t)k] == g_img_pad) first_pad = k;
                if (v[(size_t)k] == vis_e_id) last_end = k;
            }
            if (first_pad >= 0 && first_pad < drop && last_end + 1 > drop) drop = last_end + 1;
            if (drop > (int)v.size()) drop = (int)v.size();
        }
        if (im_start_id >= 0) {                             // 切点尽量落在消息边界, 免得留半句话
            for (int k = drop; k < (int)v.size() && k < drop + 32; k++)
                if (v[(size_t)k] == im_start_id) { drop = k; break; }
        }
        if (drop < 1) drop = 1;
        if (drop > (int)v.size() - 1) drop = (int)v.size() - 1;   // 永远至少留 1 个 token
        printf("[orn] ★ 超上下文上限 %d > %d tok: 【不丢尾】头部砍掉 %d tok (保留尾部: 最后一条消息与 <|im_end|>)\n",
               (int)v.size(), maxn, drop);
        v.erase(v.begin(), v.begin() + drop);
        return drop;
    };

    // 一条请求的生成
    auto run_request = [&](const std::string &prompt, int maxtok, bool stream, const std::string &sid) -> std::string {
        std::vector<int> ids = g_tk.encode(prompt);
        // ★ DETVIS: 本次 prompt 的图像依赖指纹 —— 含 <|image_pad|> 才算依赖当前图像,
        //   否则恒为 0 (纯文本提示 ⇒ 快照/会话复用行为与修前逐位相同)
        uint64_t ikey = 0;
        if (g_img_n > 0 && g_img_pad >= 0)
            for (size_t i = 0; i < ids.size(); i++) if (ids[i] == g_img_pad) { ikey = g_img_id; break; }
        std::vector<float> logits;
        int pos = 0, reused = 0;
        bool hit = false, did_prefill = false;
        if (!ids.empty() && g_snap.valid && g_snap.ids == ids && g_snap.img_id == ikey) {
            restore_snap(logits); pos = g_snap.pos; reused = pos; hit = true;
            g_img_used = g_img_n;      // ★ DETVIS: 命中快照 ⇒ 图像行早已全部消费完 (与全量 prefill 收尾状态一致)
        }
        // ★★ 第27轮: LCP 前缀复用 —— 在检查点表里找"本次 prompt 的最长可回滚前缀" ★★
        //   ① 逐检查点算 LCP(本次 ids, 检查点 ids);
        //   ② 只有 LCP == 检查点全长 (即检查点序列是本轮 prompt 的前缀) 才能回滚到它的状态;
        //   ③ 图像依赖必须一致 (前缀含 image_pad 时指纹必须相同) —— 与 DETVIS 同源, 防"换图复读"。
        int ck_hit = -1, ck_lcp = 0, lcp_all = 0;
        // ★ 第30轮(问题①): 入口做"保尾砍头"。ids 变了 ⇒ 前缀比对自然落空 ⇒ 走全量预填充 (正确)
        if ((int)ids.size() > MAXT) head_trim_keep_tail(ids, MAXT);
        if (!hit && !ids.empty()) {
            for (int ci = 0; ci < (int)g_ck.size(); ci++) {
                Ckpt &C = g_ck[(size_t)ci];
                if (!C.valid || C.pos <= 0) continue;
                if (!(C.imgdep == 0 || C.imgdep == ikey)) continue;
                int m = ((int)ids.size() < C.pos) ? (int)ids.size() : C.pos;
                int n = 0; while (n < m && C.ids[(size_t)n] == ids[(size_t)n]) n++;
                if (n > lcp_all) lcp_all = n;                 // 真实 LCP (仅统计图像依赖相容者)
                if (n < C.pos) continue;                      // 序列不是本轮前缀 ⇒ 状态在 C.pos, 回滚不了
                if (n > ck_lcp) { ck_lcp = n; ck_hit = ci; }
            }
            if (ck_hit >= 0) {
                ck_restore(ck_hit, logits);
                pos = g_ck[(size_t)ck_hit].pos; reused = pos;
                hit = (pos == (int)ids.size());
            }
        }
        auto t0 = std::chrono::steady_clock::now();
        if (!hit) {
            int cont = -1;
            if (!sid.empty() && ck_hit < 0) {   // ★ 第27轮: 已有检查点回滚 ⇒ 不再动会话表 (状态已在正确位置)
                std::map<std::string, Sess>::iterator it = g_sess.find(sid);
                if (it != g_sess.end() && g_live_sid == sid && g_live_pos == it->second.pos &&
                    it->second.img_id == ikey &&
                    it->second.pos > 0 && (int)ids.size() > it->second.pos &&
                    it->second.pos <= (int)it->second.consumed.size()) {
                    bool ok = true;
                    for (int i = 0; i < it->second.pos; i++) if (it->second.consumed[i] != ids[i]) { ok = false; break; }
                    if (ok) cont = it->second.pos;
                }
            }
            if (ck_hit >= 0) { /* ★ 第27轮: 状态已回滚到 ck_lcp, 不得 reset / 不得改 pos */ }
            else if (cont < 0) { g_st.reset(); pos = 0; reused = 0; }
            else { pos = cont; reused = cont; }
            did_prefill = true;
            // ★ 分块 prefill: 每个块内线性层的 gemm 合并成 m=T 一次读权重;
            //   块 ≥ bmin 走批量路径, 剩余不足 bmin 的尾巴走原逐 token 路径 (位级不变)
            if (ck_hit < 0) g_img_used = 0;    // ★ VIS: 全量 prefill 从头按顺序取图像行 (回滚路径 g_img_used 由 ck_restore 恢复)
            for (size_t i = reused; i < ids.size(); ) {
                if (pos >= MAXT) {   // ★ 第30轮: 入口已"保尾砍头", 这里正常不该发生 (兜底 + 告警)
                    printf("[orn] ★ WARN: 预填充触到 MAXT=%d (入口保尾砍头未生效?)\n", MAXT);
                    ids.resize(i); break; }
                size_t remain = ids.size() - i;
                int T = 1;
                if (remain >= (size_t)g_bmin) {
                    T = (int)((remain > (size_t)g_bmax) ? g_bmax : remain);
                    if (pos + T > MAXT) T = MAXT - pos;
                }
                // ★ 第30轮: 不再"为打检查点而切批" —— 切批会改变 forward_batch 的分批方式,
                //   而批路径与逐 token 路径并非位级等价 ⇒ 会动数值(第27轮文本 Q4 差异的直接原因)。
                //   改为在【批边界】打点: 状态本来就停在这些位置, 粒度 = bmax(64) tok, 一样吃到 LCP 回滚。
                if (im_start_id >= 0 && pos > 0) {
                    bool bnd = (ids[i] == im_start_id);
                    if (!bnd && T > 1)
                        for (int q = 1; q < T; q++) if (ids[i + (size_t)q] == im_start_id) { bnd = true; break; }
                    if (bnd) ck_take(ids, pos, logits, g_img_used);
                }
                if (T > 1) {
                    forward_batch(&ids[i], T, pos, logits);
                    if (g_ldump) { int sz = (int)logits.size(), p_ = pos + T - 1, tk = ids[i + T - 1];
                        fwrite(&p_, 4, 1, g_ldump); fwrite(&tk, 4, 1, g_ldump);
                        fwrite(&sz, 4, 1, g_ldump); fwrite(logits.data(), 4, (size_t)sz, g_ldump); fflush(g_ldump); }
                    pos += T; i += T;
                } else {
                    forward(ids[i], pos, logits);
                    if (g_ldump) { int sz = (int)logits.size(), p_ = pos, tk = ids[i];
                        fwrite(&p_, 4, 1, g_ldump); fwrite(&tk, 4, 1, g_ldump);
                        fwrite(&sz, 4, 1, g_ldump); fwrite(logits.data(), 4, (size_t)sz, g_ldump); fflush(g_ldump); }
                    pos++; i++;
                }
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        double pt = std::chrono::duration<double>(t1 - t0).count();
        if (did_prefill && reused == 0 && !ids.empty()) take_snap(ids, pos, logits, ikey);
        // ★ 第27轮 ③: 预填充末尾检查点 (整段 prompt)。同一 prompt 重发 ⇒ 直接回滚, 0 预填充。
        if (did_prefill && !ids.empty()) ck_take(ids, pos, logits, g_img_used);
        {   // ★ 第27轮: 复用台账 (serve2 透出到 8090 响应与日志)
            int newtok = (int)ids.size() - reused;
            printf("[orn] ★ LCP=%d tok 复用, 新增 %d tok, 预填充 %.3fs (回滚点=%d tok, %s; 最长公共前缀=%d tok; 检查点表 %d/%d 槽 %.0f MB)\n",
                   reused, newtok, pt, reused,
                   (reused == 0 ? "无可用前缀, 全量" : (hit ? "整段命中" : "部分前缀命中")),
                   lcp_all, ck_nvalid(), (int)g_ck.size(), ck_mb());
        }
        std::string out;
        std::vector<int> genids, emitted;
        int ntok = 0;
        uint64_t fwd0 = g_fwd_n;
        double pf0[7] = {g_p_emb, g_p_nrm, g_p_ssm, g_p_att, g_p_lay, g_p_lm, g_p_arg};
        double pf1[4] = {g_p_conv, g_p_dn, g_p_gn, g_p_ab};
        double pg[5] = {g_g_h2d, g_g_launch, g_g_wait, g_g_d2h, g_g_n};
        double pt_g = g_i8_t; uint64_t pn_g = g_i8_n, pbn_g = g_bn;
        auto t2 = std::chrono::steady_clock::now();
        while (ntok < maxtok && pos < MAXT) {
            int best = 0; float bv = logits[0];
            for (size_t i = 1; i < logits.size(); i++) if (logits[i] > bv) { bv = logits[i]; best = (int)i; }
            if (std::find(stops.begin(), stops.end(), best) != stops.end()) break;
            std::string tb = g_tk.decode_token(best, g_gguf.tokens);
            if (stream) {
                printf("__TOK__ ");
                for (size_t k = 0; k < tb.size(); k++) printf("%02x", (unsigned char)tb[k]);
                printf("\n");
            }
            out += tb;
            ntok++;
            emitted.push_back(best);            // ★ 第27轮: 记录"实际发出的 token" (供输出指纹对拍)
            if (ntok >= maxtok) break;
            genids.push_back(best);
            forward(best, pos, logits); pos++;
        }
        auto t3 = std::chrono::steady_clock::now();
        double gt = std::chrono::duration<double>(t3 - t2).count();
        // ★ 第27轮 ④: 生成末尾检查点 (prompt+回答) —— 下一轮"整段重发 + 追加"只需算新增那段
        if (!ids.empty() && pos > 0 && genids.size() > 0) {
            std::vector<int> full(ids);
            full.insert(full.end(), genids.begin(), genids.end());
            if ((int)full.size() == pos) ck_take(full, pos, logits, g_img_used);
        }
        {   // ★ 第27轮: 输出 token 流指纹 (确定性/位级等价比对用; 不改变任何数值)
            uint64_t h1 = 1469598103934665603ULL, h2 = 1469598103934665603ULL, h3 = 1469598103934665603ULL;
            for (size_t i = 0; i < emitted.size(); i++) { uint64_t v = (uint64_t)(uint32_t)emitted[i];
                for (int b = 0; b < 4; b++) { h1 ^= (unsigned char)((v >> (b * 8)) & 0xFF); h1 *= 1099511628211ULL; } }
            for (size_t i = 0; i < out.size(); i++) { h2 ^= (unsigned char)out[i]; h2 *= 1099511628211ULL; }
            for (size_t i = 0; i < ids.size(); i++) { uint64_t v = (uint64_t)(uint32_t)ids[i];
                for (int b = 0; b < 4; b++) { h3 ^= (unsigned char)((v >> (b * 8)) & 0xFF); h3 *= 1099511628211ULL; } }
            printf("[orn] ★ GEN %d tok ids=%016llx bytes=%016llx prompt=%016llx(%d tok)\n",
                   (int)emitted.size(), (unsigned long long)h1, (unsigned long long)h2,
                   (unsigned long long)h3, (int)ids.size());
        }
        if (g_prof && ntok > 0) {
            double n = (double)ntok;
            double e0 = (g_p_emb - pf0[0]) * 1000 / n, n0 = (g_p_nrm - pf0[1]) * 1000 / n,
                   s0 = (g_p_ssm - pf0[2]) * 1000 / n, a0 = (g_p_att - pf0[3]) * 1000 / n,
                   l0 = (g_p_lay - pf0[4]) * 1000 / n, m0 = (g_p_lm - pf0[5]) * 1000 / n;
            double card = (g_i8_t - pt_g) * 1000 / n;
            double hostsum = e0 + n0 + s0 + a0 + l0 + m0;
            printf("[orn][PROF] %d tok %.2fs = %.2f ms/tok | 卡gemm %.2f ms (%llu 次, 批量 %llu) | 主机 %.2f ms: 嵌入%.2f 归一化+launch%.2f SSM%.2f[conv%.2f delta-net%.2f gate-norm%.2f alpha/beta%.2f SSM其余%.2f] 注意力%.2f 层内其余%.2f lm_head%.2f | argmax+解码+其余 %.2f\n",
                   ntok, gt, gt * 1000 / n, card, (unsigned long long)(g_i8_n - pn_g), (unsigned long long)(g_bn - pbn_g),
                   hostsum, e0, n0, s0, (g_p_conv - pf1[0]) * 1000 / n, (g_p_dn - pf1[1]) * 1000 / n,
                   (g_p_gn - pf1[2]) * 1000 / n, (g_p_ab - pf1[3]) * 1000 / n,
                   s0 - ((g_p_conv - pf1[0]) + (g_p_dn - pf1[1]) + (g_p_gn - pf1[2]) + (g_p_ab - pf1[3])) * 1000 / n,
                   a0, l0, m0, gt * 1000 / n - card - hostsum);
            printf("[orn][PROF] ★ 卡 gemm 四段/token: H2D %.2f ms | launch %.2f ms | 等卡(wait) %.2f ms | D2H %.2f ms || 合计 %.2f ms (调用 %llu 次, %.3f ms/次, 其中每次 launch %.4f ms)\n",
                   (g_g_h2d - pg[0]) * 1000 / n, (g_g_launch - pg[1]) * 1000 / n,
                   (g_g_wait - pg[2]) * 1000 / n, (g_g_d2h - pg[3]) * 1000 / n,
                   ((g_g_h2d - pg[0]) + (g_g_launch - pg[1]) + (g_g_wait - pg[2]) + (g_g_d2h - pg[3])) * 1000 / n,
                   (unsigned long long)(g_g_n - pg[4]),
                   ((g_g_h2d - pg[0]) + (g_g_launch - pg[1]) + (g_g_wait - pg[2]) + (g_g_d2h - pg[3])) * 1000 / (double)(g_g_n - pg[4] + 1e-9),
                   (g_g_launch - pg[1]) * 1000 / (double)(g_g_n - pg[4] + 1e-9));
        }
        if (!sid.empty()) {
            Sess &S = g_sess[sid];
            S.consumed = ids;
            S.consumed.insert(S.consumed.end(), genids.begin(), genids.end());
            S.pos = pos;
            S.img_id = ikey;      // ★ DETVIS: 会话条目也要记住它吃的是哪张图
        }
        g_live_sid = sid; g_live_pos = pos;
        printf("[orn] prompt=%zu tok 用时 %.2fs (%.1f ms/tok, %s 复用%zu tok) | 生成 %d tok 用时 %.2fs = %.3f tok/s | gemv %llu 次\n",
               ids.size(), pt, pt * 1000.0 / (double)((ids.size() > (size_t)reused) ? (ids.size() - (size_t)reused) : 1),
               (hit ? "快照" : (reused > 0 ? "会话" : "全量")), (size_t)reused,
               ntok, gt, ntok / (gt + 1e-9), (unsigned long long)g_gemv_n);
        {   // ★ 实测有效读带宽 = 权重字节 x 本段 forward 次数 / 本段耗时 (每 token 读全部权重一遍)
            uint64_t nf = g_fwd_n - fwd0;
            double wb = (double)(g_hbm[0] + g_hbm[1]);
            printf("[orn] ★ 权重量 %.2f GiB/token, 本段 %llu 次 forward, 有效读带宽 = %.2f GB/s (双芯合计) | gemm_int8 %llu 次 累计 %.0f ms\n",
                   wb / 1073741824.0, (unsigned long long)nf, wb * nf / gt / 1e9,
                   (unsigned long long)g_i8_n, g_i8_t * 1000.0);
        }
        return out;
    };

    if (g_ftz) {   // ★★ FTZ+DAZ: dnbench 实测 delta-net 的 denormal 中间值造成 37x 惩罚 (3.44 -> 0.093 ms/层)
        enable_ftz_daz();
        #pragma omp parallel
        { enable_ftz_daz(); }        // MXCSR 是每线程寄存器, OpenMP 线程池里每个线程都要设
#if defined(__SSE__)
        printf("[orn] ★ 已开 FTZ+DAZ (MXCSR=0x%08x): 消除 delta-net denormal 惩罚 | K200_OMP=%d (0=关主机并行)\n", _mm_getcsr(), g_omp);
#else
        printf("[orn] ★ FTZ+DAZ 跳过 (无 SSE) | K200_OMP=%d\n", g_omp);
#endif
    }
    if (gen) {   // 离线模式: --gen "你好"
        std::string p;
        const char *ov = getenv("K200_TMPL");
        if (ov) { std::string t = ov; size_t k = t.find("{q}"); p = (k == std::string::npos) ? t : t.replace(k, 3, gen); }
        else p = std::string("<|im_start|>user\n") + gen + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        if (getenv("K200_TRACE")) {   // ★ 中间张量逐段对拍: 同一个 token 在两条路径下的 layer0 输出 / 末层 x
            std::vector<int> ids = g_tk.encode(p);
            std::vector<float> lg;
            auto fresh2 = [&]() {
                g_st.reset();
                for (int l = 0; l < NLAYER; l++)
                    if (!g_lay[l].recr) { memset(g_lay[l].Kc.data(), 0, g_lay[l].Kc.size() * 4);
                                          memset(g_lay[l].Vc.data(), 0, g_lay[l].Vc.size() * 4); }
            };
            auto cmpc = [](const std::vector<float> &a, const std::vector<float> &b, const char *nm) {
                if (a.size() != b.size()) { printf("[TRACE] %-22s 尺寸不同 %zu vs %zu\n", nm, a.size(), b.size()); return; }
                double se = 0, sr = 0, mx = 0; size_t nd = 0;
                for (size_t i = 0; i < a.size(); i++) { if (a[i] != b[i]) nd++;
                    double d = (double)a[i] - b[i]; se += d * d; sr += (double)a[i] * a[i]; if (fabs(d) > mx) mx = fabs(d); }
                printf("[TRACE] %-22s 不同=%zu/%zu  relrms=%.6f%%  maxabs=%.6g\n", nm, nd, a.size(),
                       100.0 * sqrt(se / (sr + 1e-300)), mx);
            };
            fresh2(); g_tr = 1; for (int k = 0; k < 8 && k < (int)ids.size(); k++) forward(ids[k], k, lg); g_tr = 0;
            std::vector<float> a0 = g_trX[0], a1 = g_trX[1];
            std::vector<std::vector<float> > rA = g_trRows;
            g_trRows.clear();
            fresh2(); g_tr = 1; forward_batch(&ids[0], 8, 0, lg); g_tr = 0;
            std::vector<std::vector<float> > rB = g_trRows;
            printf("[TRACE] prompt=%zu tok, 对比 token0 (逐 token 路径 vs 批量 T=2)\n", ids.size());
            cmpc(a0, g_trX[0], "gy(grpA,layer0)");
            cmpc(a1, g_trX[1], "x(final)");
            std::vector<std::vector<float> > xA = g_trRows2;
            g_trRows2.clear();
            fresh2(); g_tr = 1; forward_batch(&ids[0], 8, 0, lg); g_tr = 0;
            std::vector<std::vector<float> > xB = g_trRows2;
            printf("[TRACE] ★ layer0 之后隐状态 x 逐行对拍 (逐token 8 次 vs 批量 T=8): xA=%zu xB=%zu\n", xA.size(), xB.size());
            for (size_t i = 0; i < xA.size() && i < xB.size(); i++) {
                char nmb[64]; snprintf(nmb, sizeof(nmb), "row%zu x(after L0)", i);
                cmpc(xA[i], xB[i], nmb);
            }
            printf("[TRACE] ★ layer0 grpA 输出逐行对拍 (逐token vs 批量T=8): 共 rA=%zu rB=%zu 行\n", rA.size(), rB.size());
            for (size_t i = 0; i < rA.size() && i < rB.size(); i++) {
                char nmb[64]; snprintf(nmb, sizeof(nmb), "row%zu gy(layer0)", i);
                cmpc(rA[i], rB[i], nmb);
            }
            {   // ★ [层][行] 定位: 前 8 层里第一个跑偏的 (层,行)
                std::vector<std::vector<std::vector<float> > > LA = g_trL;
                g_trL.clear();
                fresh2(); g_tr = 1; forward_batch(&ids[0], 8, 0, lg); g_tr = 0;
                std::vector<std::vector<std::vector<float> > > LB = g_trL;
                printf("[TRACE] ★ [层][行] 定位 (前8层 x): LA=%zu 层 LB=%zu 层\n", LA.size(), LB.size());
                for (size_t l = 0; l < LA.size() && l < LB.size() && l < 8; l++)
                    for (size_t q = 0; q < LA[l].size() && q < LB[l].size(); q++) {
                        double se = 0, sr = 0, mx = 0;
                        for (size_t i = 0; i < LA[l][q].size(); i++) {
                            double d = (double)LA[l][q][i] - LB[l][q][i]; se += d * d; sr += (double)LA[l][q][i] * LA[l][q][i];
                            if (fabs(d) > mx) mx = fabs(d);
                        }
                        printf("       L%-2zu row%zu  层型=%-4s relrms=%.6f%%  maxabs=%.6g\n", l, q,
                               is_recr((int)l) ? "recr" : "attn", 100.0 * sqrt(se / (sr + 1e-300)), mx);
                    }
            }
            return 0;
        }
        if (getenv("K200_BATCHCHK")) {   // ★ 批量 vs 逐 token 数值对拍 (同一进程/同一初始状态, 逐个 T)
            std::vector<int> ids = g_tk.encode(p);
            std::vector<float> la, lb;
            auto fresh = [&]() {
                g_st.reset();
                for (int l = 0; l < NLAYER; l++)
                    if (!g_lay[l].recr) { memset(g_lay[l].Kc.data(), 0, g_lay[l].Kc.size() * 4);
                                          memset(g_lay[l].Vc.data(), 0, g_lay[l].Vc.size() * 4); }
            };
            auto argmaxof = [](const std::vector<float> &v) { int b = 0; for (size_t i = 1; i < v.size(); i++) if (v[i] > v[b]) b = (int)i; return b; };
            fresh();
            la.clear();
            for (size_t i = 0; i < ids.size(); i++) forward(ids[i], (int)i, la);
            printf("[CHK] 参考(逐 token) prompt=%zu tok  末位 argmax=%d top1=%.6f\n", ids.size(), argmaxof(la), la[argmaxof(la)]);
            std::vector<int> Ts = {1, 2, 4, 8, 16, 32, 64};
            std::vector<float> refS = g_st.S, refC = g_st.conv;
            for (size_t ki = 0; ki < Ts.size(); ki++) {
                int T = Ts[ki];
                fresh();
                for (size_t i = 0; i < ids.size(); ) {
                    int t = (int)((ids.size() - i > (size_t)T) ? (size_t)T : (ids.size() - i));
                    if (t == 1 && T > 1) { forward(ids[i], (int)i, lb); i += 1; }
                    else { forward_batch(&ids[i], t, (int)i, lb); i += (size_t)t; }   // T==1 也强制走批量函数体
                }
                double se = 0, sr = 0, mx = 0; size_t nd = 0;
                double ma = 0, mb = 0;
                for (size_t i = 0; i < la.size(); i++) { ma += la[i]; mb += lb[i]; }
                ma /= (double)la.size(); mb /= (double)lb.size();
                double sec = 0, src = 0;
                for (size_t i = 0; i < la.size() && i < lb.size(); i++) {
                    if (la[i] != lb[i]) nd++;
                    double d = (double)la[i] - lb[i]; se += d * d; sr += (double)la[i] * la[i];
                    if (fabs(d) > mx) mx = fabs(d);
                    double dc = ((double)la[i] - ma) - ((double)lb[i] - mb); sec += dc * dc;
                    double da = (double)la[i] - ma; src += da * da;
                }
                printf("[CHK] T=%-3d 末位 argmax=%-7d (参考 %-7d %s) top1=%.6f | 不同元素=%zu/%zu relrms=%.5f%% 去直流relrms=%.5f%% maxabs=%.5f\n",
                       T, argmaxof(lb), argmaxof(la), (argmaxof(lb) == argmaxof(la)) ? "一致" : "★不一致",
                       lb[argmaxof(lb)], nd, la.size(), 100.0 * sqrt(se / (sr + 1e-300)),
                       100.0 * sqrt(sec / (src + 1e-300)), mx);
                {   // ★ 逐层状态对拍: 定位第一个状态跑偏的层 (S: NVH*VHD*KHD; conv: 3*CVD)
                    int firstS = -1, firstC = -1; double wS = 0, wC = 0;
                    for (int l = 0; l < NLAYER; l++) {
                        const float *a1 = &refS[(size_t)l * NVH * VHD * KHD];
                        const float *b1 = &g_st.S[(size_t)l * NVH * VHD * KHD];
                        size_t nS = (size_t)NVH * VHD * KHD; double d2 = 0, r2 = 0, m2 = 0;
                        for (size_t i = 0; i < nS; i++) { double d = (double)a1[i] - b1[i]; d2 += d * d; r2 += (double)a1[i] * a1[i]; if (fabs(d) > m2) m2 = fabs(d); }
                        double rr = 100.0 * sqrt(d2 / (r2 + 1e-300));
                        if (rr > 1e-4 && firstS < 0) { firstS = l; wS = rr; }
                        const float *c1 = &refC[(size_t)l * 3 * CVD];
                        const float *c2 = &g_st.conv[(size_t)l * 3 * CVD];
                        double e2 = 0, q2 = 0;
                        for (size_t i = 0; i < (size_t)3 * CVD; i++) { double d = (double)c1[i] - c2[i]; e2 += d * d; q2 += (double)c1[i] * c1[i]; }
                        double rc = 100.0 * sqrt(e2 / (q2 + 1e-300));
                        if (rc > 1e-4 && firstC < 0) { firstC = l; wC = rc; }
                    }
                    printf("       └ 首个状态跑偏层: S[%d] (relrms %.4f%%) conv[%d] (%.4f%%)  [层型: %s]\n",
                           firstS, wS, firstC, wC,
                           firstS < 0 ? "-" : (is_recr(firstS) ? "recr(delta-net+conv1d)" : "★full-attention"));
                }
            }
            return 0;
        }
        printf("__BEGIN__\n");
        std::string o = run_request(p, ngen, true, "offline");
        printf("__END__\n");
        fprintf(stderr, "[orn] 离线输出: %s\n", o.c_str());
        return 0;
    }
    // ========================================================================
    // ★★ 第29轮: 多槽位连续批处理 (continuous batching) 服务模式 ★★
    //   协议【向后兼容】: 老的 "!VIS/!VISR/!VISIMG/!VISQ" 与 "@<n>@sid=..@b64:.."
    //   一字未改 (仍走老的 run_request 单会话路径) —— 老网关/老验收脚本不许坏 ✓
    //   新增: !SLOTS / !SADD / !SDEL / !SSTAT / !SPAUSE / !SRESUME
    //   引擎自己就是调度器: 只要批里还有活跃槽就每个循环迭代推进一步;
    //   poll(fd0, 0) 不阻塞地收新请求 ⇒ 新请求可以在批【中途】加入 (continuous batching)。
    //   ★ 生产隔离: solo=1 的请求只在【没有任何别的活跃槽】时进批 ⇒ 它永远 M=1
    //     ⇒ 与老单会话路径逐位一致; 8091 测试口用 solo=0 真并发。
    // ========================================================================
    struct CbReq { int cid = 0; int maxtok = 0; bool solo = false; std::string sid; uint64_t want_img = 0;
                   bool htrim = false;          // ★ 第30轮: 入口做过"保尾砍头"(超上下文上限)
                   std::string pr;              // ★ 第30轮d: 原始 prompt —— tokenize 延到入槽时做
                   std::vector<int> ids; };
    struct MCk {                    // 多槽位检查点 (按 sid 分组; 与第27章同一套设计)
        bool valid = false;
        int kind = 0;               // 0=消息边界 1=预填充末尾 2=生成末尾
        int hits = 0;               // ★ 被命中过几次 —— 淘汰时【保护被命中过的】条目
        std::string sid;
        std::vector<int> ids;
        int pos = 0;
        uint64_t imgdep = 0;
        int img_used = 0;
        uint64_t last = 0;
        std::vector<float> conv, S, logits;
        std::vector<std::vector<float> > Kc, Vc;
    };
    static std::vector<CbSlot> g_slots;
    static std::vector<CbReq>  g_squeue;
    static std::vector<MCk>    g_mck;
    static uint64_t g_mck_tick = 0, g_cb_steps = 0, g_cb_served = 0;
    static bool g_cb_run = true;
    static double g_cb_d2h = 0;
    // ★★ 第32轮: 冷预填充上限 (tok)。0 = 关闭该保护。
    static int g_maxcold = 3000;

    auto cb_state_mb = [&]() {
        return (double)(NLAYER * 3 * CVD + NLAYER * NVH * VHD * KHD + (NLAYER / 4) * 2 * MAXT * NKV * HD) * 4 / 1048576.0;
    };
    auto cb_init = [&](int n) {
        if (n < 0) n = 0;
        if (n > 32) n = 32;
        g_slots.clear(); g_slots.resize((size_t)n);
        for (int i = 0; i < n; i++) {
            CbSlot &S = g_slots[(size_t)i];
            S.conv.assign((size_t)NLAYER * 3 * CVD, 0.f);
            S.S.assign((size_t)NLAYER * NVH * VHD * KHD, 0.f);
            S.Kc.resize(NLAYER); S.Vc.resize(NLAYER);
            for (int l = 0; l < NLAYER; l++) if (!g_lay[l].recr) {
                S.Kc[(size_t)l].assign((size_t)MAXT * NKV * HD, 0.f);
                S.Vc[(size_t)l].assign((size_t)MAXT * NKV * HD, 0.f);
            }
            S.logits.assign((size_t)g_outw.M, 0.f);
        }
        const char *e = getenv("K200_CKPT");
        int nc = e ? atoi(e) : 3;
        if (nc < 0) nc = 0;
        if (nc > 16) nc = 16;
        g_mck.clear(); g_mck.resize((size_t)nc);
        printf("[orn] ★ 多槽位(第29轮): %d 槽 x %.1f MB = %.0f MB; 前缀检查点表 %d 槽 (K200_CKPT)\n",
               n, cb_state_mb(), n * cb_state_mb(), nc);
    };
    auto cb_bind = [&](CbSlot &S) {
        g_pConv = &S.conv; g_pS = &S.S;
        for (int l = 0; l < NLAYER; l++) { g_pKc[l] = &S.Kc[(size_t)l]; g_pVc[l] = &S.Vc[(size_t)l]; }
        g_pImgemb = &S.imgemb; g_pImgn = &S.imgn; g_pImgused = &S.imgused;
    };
    auto cb_unbind = [&]() {
        g_pConv = &g_st.conv; g_pS = &g_st.S;
        for (int l = 0; l < NLAYER; l++) { g_pKc[l] = &g_lay[l].Kc; g_pVc[l] = &g_lay[l].Vc; }
        g_pImgemb = &g_imgemb; g_pImgn = &g_img_n; g_pImgused = &g_img_used;
    };
    auto cb_mck_take = [&](CbSlot &S, const std::vector<int> &ids, int spos, const std::vector<float> &lg, int kind = 0) {
        // ★ 第30轮c: 不再要求 sid 非空 —— 检查点是【按内容】共用的 (跨会话前缀复用)
        if (g_mck.empty() || spos <= 0 || (int)ids.size() < spos) return;
        int sl = -1;
        for (size_t i = 0; i < g_mck.size(); i++)
            if (g_mck[i].valid && g_mck[i].pos == spos &&
                std::equal(g_mck[i].ids.begin(), g_mck[i].ids.end(), ids.begin())) { sl = (int)i; break; }
        if (sl < 0) for (size_t i = 0; i < g_mck.size(); i++) if (!g_mck[i].valid) { sl = (int)i; break; }
        if (sl < 0) {
            // ★ 第30轮c: 淘汰策略 —— 先丢【从未被命中过】的; 同类里优先丢【最短前缀】(长的更值钱,
            //   比如 DSH 的 system+工具块前缀), 同长度再丢最旧的。为什么不会饿死单会话:
            //   单会话的每个检查点在它下一轮都会被命中 (hits+1) ⇒ 有了 hits 就进入受保护组;
            //   只有真正没被用过的条目才会被优先淘汰。
            // ★★ 第32轮: 为大前缀【专门留一个槽】★★
            //   长前缀(DSH 的 system+工具块)命中一次的收益 = 数千 token 的冷预填充; 表只有 4 槽,
            //   若被短前缀(每个请求自己会打 3 个点)挤掉 ⇒ 跨会话缓存形同虚设 ✗。
            //   规则: 当前【最长前缀】那条被保护, 不参与淘汰; 只有"本次要存的更长"时才允许淘汰它。
            int prot = -1;
            {   int bestp = -1;
                for (size_t i = 0; i < g_mck.size(); i++)
                    if (g_mck[i].valid && g_mck[i].pos > bestp) { bestp = g_mck[i].pos; prot = (int)i; }
                if (prot >= 0 && spos > g_mck[(size_t)prot].pos) prot = -1;
            }
            int besth = 1 << 30; sl = -1;
            for (size_t i = 0; i < g_mck.size(); i++) {
                if ((int)i == prot) continue;
                if (g_mck[i].hits < besth) besth = g_mck[i].hits;
            }
            for (size_t i = 0; i < g_mck.size(); i++) {
                if ((int)i == prot) continue;
                if (g_mck[i].hits != besth) continue;
                if (sl < 0) { sl = (int)i; continue; }
                MCk &B = g_mck[(size_t)sl];
                if (g_mck[i].pos < B.pos) sl = (int)i;
                else if (g_mck[i].pos == B.pos && g_mck[i].last < B.last) sl = (int)i;
            }
            if (sl < 0) sl = 0;
        }
        MCk &C = g_mck[(size_t)sl];
        C.kind = kind;
        C.sid = S.sid;
        C.ids.assign(ids.begin(), ids.begin() + spos);
        C.pos = spos;
        C.conv = S.conv; C.S = S.S; C.logits = lg;
        C.Kc.resize(NLAYER); C.Vc.resize(NLAYER);
        size_t nr = (size_t)spos * NKV * HD;
        for (int l = 0; l < NLAYER; l++)
            if (!g_lay[l].recr) {
                C.Kc[(size_t)l].assign(S.Kc[(size_t)l].begin(), S.Kc[(size_t)l].begin() + nr);
                C.Vc[(size_t)l].assign(S.Vc[(size_t)l].begin(), S.Vc[(size_t)l].begin() + nr);
            } else { std::vector<float>().swap(C.Kc[(size_t)l]); std::vector<float>().swap(C.Vc[(size_t)l]); }
        C.img_used = S.imgused;
        C.imgdep = 0;
        if (S.imgn > 0 && g_img_pad >= 0)
            for (int k = 0; k < spos; k++) if (C.ids[(size_t)k] == g_img_pad) { C.imgdep = S.imgid; break; }
        C.last = ++g_mck_tick; C.valid = true;
    };
    // ========================================================================
    // ★★ 第32轮: 【只读】最长可回滚前缀探针 ★★
    //   与 cb_admit 的命中判定是【同一份代码】(cb_admit 也调用它) ⇒ 不会漂移。
    //   绝不改任何状态: 不 restore、不动 last/hits ⇒ 可以在【网关入队之前】安全地问它。
    //   命中规则(与第30轮一致): 检查点 token 序列必须是本轮 prompt 的前缀 (逐 token 相等),
    //   且图像依赖相容 (前缀含 image_pad ⇒ 指纹必须相同); 不要求 sid 相同 (跨会话)。
    // ========================================================================
    auto cb_lcp_probe = [&](const std::vector<int> &ids, uint64_t imgid,
                            int *hitslot, int *hitlcp, int *lcpall) {
        int slot = -1, bl = 0, al = 0;
        if (!ids.empty()) {
            for (int ci = 0; ci < (int)g_mck.size(); ci++) {
                MCk &C = g_mck[(size_t)ci];
                if (!C.valid || C.pos <= 0) continue;
                if (!(C.imgdep == 0 || C.imgdep == imgid)) continue;
                int m = ((int)ids.size() < C.pos) ? (int)ids.size() : C.pos;
                int n = 0; while (n < m && C.ids[(size_t)n] == ids[(size_t)n]) n++;
                if (n > al) al = n;
                if (n < C.pos) continue;                 // 序列不是本轮前缀 ⇒ 这个状态回滚不了
                if (n > bl) { bl = n; slot = ci; }
            }
        }
        if (hitslot) *hitslot = slot;
        if (hitlcp)  *hitlcp  = bl;
        if (lcpall)  *lcpall  = al;
        return slot;
    };
    auto cb_emit_fp = [&](CbSlot &S) {
        uint64_t h1 = 1469598103934665603ULL, h2 = 1469598103934665603ULL, h3 = 1469598103934665603ULL;
        for (size_t i = 0; i < S.emitted.size(); i++) { uint64_t v = (uint64_t)(uint32_t)S.emitted[i];
            for (int b = 0; b < 4; b++) { h1 ^= (unsigned char)((v >> (b * 8)) & 0xFF); h1 *= 1099511628211ULL; } }
        for (size_t i = 0; i < S.out.size(); i++) { h2 ^= (unsigned char)S.out[i]; h2 *= 1099511628211ULL; }
        for (size_t i = 0; i < S.ids.size(); i++) { uint64_t v = (uint64_t)(uint32_t)S.ids[i];
            for (int b = 0; b < 4; b++) { h3 ^= (unsigned char)((v >> (b * 8)) & 0xFF); h3 *= 1099511628211ULL; } }
        printf("__SFP__ %d %d ids=%016llx bytes=%016llx prompt=%016llx(%d tok)\n", S.reqid, (int)S.emitted.size(),
               (unsigned long long)h1, (unsigned long long)h2, (unsigned long long)h3, (int)S.ids.size());
    };
    auto cb_finish = [&](CbSlot &S, const char *why) {
        if (!S.ids.empty() && S.pos > 0 && !S.genids.empty()) {      // 生成末尾检查点 (prompt+回答)
            std::vector<int> full(S.ids);
            full.insert(full.end(), S.genids.begin(), S.genids.end());
            if ((int)full.size() == S.pos) { cb_bind(S); cb_mck_take(S, full, S.pos, S.logits, 2); cb_unbind(); }
        }
        S.t_done = NOWS_();
        {   double pt = S.prefill_s, wt = S.t_done - S.t_admit, dec = wt - pt; if (dec < 1e-6) dec = 1e-6;
            printf("[orn] slot%d cid=%d 结束: prompt=%d tok 复用%d tok 预填充%.3fs | 生成 %d tok 用时%.2fs = %.3f tok/s"
                   " | ★ 解码 %.3fs = %.3f tok/s (墙钟口径含预填充, 与旧件不可比; 看速度请看这一项) | gemv %llu 次\n",
                   (int)(&S - &g_slots[0]), S.reqid, (int)S.ids.size(), S.reused, pt, S.ntok, wt,
                   S.ntok / (wt + 1e-9), dec, S.ntok / dec, (unsigned long long)g_gemv_n);
        }
        cb_emit_fp(S);
        printf("__SDONE__ %d %s %d\n", S.reqid, why, S.ntok);
        S.active = false; S.feed = false; S.used = false;
        g_cb_served++;
    };
    auto cb_advance = [&](CbSlot &S) {   // 从当前 logits 出下一个 token —— 与 run_request 的 while 同序
        S.feed = false;
        if (S.ntok >= S.maxtok || S.pos >= MAXT) { cb_finish(S, (S.ntok >= S.maxtok) ? "length" : "MAXT"); return; }
        int best = 0; float bv = S.logits[0];
        for (size_t i = 1; i < S.logits.size(); i++) if (S.logits[i] > bv) { bv = S.logits[i]; best = (int)i; }
        if (std::find(stops.begin(), stops.end(), best) != stops.end()) { cb_finish(S, "stop"); return; }
        std::string tb = g_tk.decode_token(best, g_gguf.tokens);
        printf("__STOK__ %d ", S.reqid);
        for (size_t k = 0; k < tb.size(); k++) printf("%02x", (unsigned char)tb[k]);
        printf("\n");
        S.out += tb; S.ntok++; S.emitted.push_back(best);
        if (S.ntok >= S.maxtok) { cb_finish(S, "length"); return; }
        S.genids.push_back(best);
        S.cur_tok = best;
        S.feed = true;
    };
    // ========================================================================
    // ★★ 第32轮: 命令读取健壮化 + 预填充期间抽空命令队列 ★★
    //   ① 半截命令 (对端写到一半就死): 读到 EOF/HUP ⇒ 【丢弃不完整帧】+ 重置解析状态; 旧行为是
    //      把半截残留与下一条命令拼成一条 ⇒ 首字符不合法 ⇒ 新命令被【静默丢弃】⇒ 网关干等
    //      (观感 = 引擎卡死 ✗)。
    //   ② 对端关闭【绝不退出引擎】: 旧行为 `if (n <= 0) break;` = 直接跳出服务循环 ⇒ 引擎自杀;
    //      网关一被 kill/重启, 引擎与其热缓存一起没了 (事故现场: /health 不响应, 只能 stop/start 80s ✗)。
    //      新行为: 置 stdin_eof, 200ms 节流轮询; 路径被重建(旧网关会 remove+mkfifo)时重新 open 附着。
    //   ③ 预填充期间也抽空命令队列(白名单 !SDEL/!SSTAT/!SMCK/!SADD/!SPAUS/!SRESUM/!CPREF):
    //      /health 在长预填充期间照常答复, 客户端中断能立刻取消该槽; 白名单外的命令(图像塔等)
    //      推迟到预填充结束后再处理 —— 绝不在预填充中间去抢卡/动共享状态 ✗。
    // ========================================================================
    static std::vector<std::string> g_defer;
    std::string inbuf;
    std::vector<char> rbuf(65536);
    bool stdin_eof = false;
    const char *fifo_path = getenv("K200_FIFO");
    if (!fifo_path || !*fifo_path) fifo_path = "/home/caden/ornc/.ornq_cb.fifo";
    std::function<void(std::string &)> handle_cmd;   // ★ 实体在服务循环里赋值 (预填充泵也要调用)
    auto pump_cmds = [&](int to_ms, int prefill_mode) {
        struct pollfd pf; pf.fd = 0; pf.events = POLLIN; pf.revents = 0;
        int pr = poll(&pf, 1, to_ms);
        if (pr < 0 && errno != EINTR) { stdin_eof = true; inbuf.clear(); }
        if (pr > 0 && (pf.revents & (POLLIN | POLLHUP | POLLERR))) {
            ssize_t n = read(0, &rbuf[0], rbuf.size());
            if (n < 0) {
                if (errno != EINTR && errno != EAGAIN) { stdin_eof = true; inbuf.clear(); }
            } else if (n == 0) {                       // ★ 对端关闭: 丢弃不完整帧, 绝不退出
                printf("[orn] ★ 命令 FIFO 读到 EOF/HUP (客户端中断/关闭): 丢弃不完整帧 %d 字节 + 重置解析状态;"
                       " 空闲时自动重新附着 %s (无需重启引擎)\\n", (int)inbuf.size(), fifo_path);
                fflush(stdout);
                stdin_eof = true; inbuf.clear();
            } else {
                if (stdin_eof) printf("[orn] ★ 命令 FIFO 对端恢复, 继续服务 (引擎未重启)\\n");
                stdin_eof = false;
                inbuf.append(&rbuf[0], (size_t)n);
            }
        }
        size_t nl;
        while ((nl = inbuf.find('\n')) != std::string::npos) {
            std::string s = inbuf.substr(0, nl);
            inbuf.erase(0, nl + 1);
            while (!s.empty() && (s[s.size() - 1] == '\r' || s[s.size() - 1] == ' ')) s.erase(s.size() - 1);
            if (s.empty()) continue;
            if (prefill_mode && s[0] == '!') {
                bool safe = (s.compare(0, 5, "!SDEL") == 0) || (s.compare(0, 6, "!SSTAT") == 0) ||
                            (s.compare(0, 5, "!SMCK") == 0) || (s.compare(0, 5, "!SADD") == 0) ||
                            (s.compare(0, 6, "!SPAUS") == 0) || (s.compare(0, 7, "!SRESUM") == 0) ||
                            (s.compare(0, 6, "!CPREF") == 0);
                if (!safe) {
                    printf("[orn] ★ 预填充期间推迟命令(不在预填充中间抢卡): %.40s\\n", s.c_str());
                    fflush(stdout);
                    g_defer.push_back(s);
                    continue;
                }
            }
            handle_cmd(s);
        }
        if (inbuf.size() > ((size_t)8 << 20)) inbuf.clear();     // 防御: 超长无换行的垃圾
    };
    // ---- 入槽 + 预填充 (+ 前缀复用) ----
    auto cb_admit = [&](CbSlot &S, CbReq &R) {
        S.used = true; S.active = true; S.solo = R.solo; S.reqid = R.cid; S.sid = R.sid;
        S.maxtok = R.maxtok; S.ntok = 0; S.pos = 0; S.reused = 0; S.lcp_all = 0;
        S.out.clear(); S.genids.clear(); S.emitted.clear();
        S.ids = R.ids;
        if (S.ids.empty() && !R.pr.empty()) S.ids = g_tk.encode(R.pr);     // ★ 第30轮d
        if ((int)S.ids.size() > MAXT) head_trim_keep_tail(S.ids, MAXT);
        if (R.want_img != 0) {                                            // 图像一致性检查(原在命令循环里)
            uint64_t ikey = 0;
            if (g_img_n > 0 && g_img_pad >= 0)
                for (size_t i = 0; i < S.ids.size(); i++) if (S.ids[(size_t)i] == g_img_pad) { ikey = g_img_id; break; }
            if (ikey != 0 && R.want_img != 0 && ikey != R.want_img) {
                printf("__SREJ__ %d image-mismatch want=%016llx have=%016llx\n", S.reqid,
                       (unsigned long long)R.want_img, (unsigned long long)ikey);
                fflush(stdout); S.active = false; S.used = false; return;
            }
            if (ikey != 0 && g_img_n <= 0) { printf("__SREJ__ %d no-image\n", S.reqid); fflush(stdout); S.active = false; S.used = false; return; }
        }
        S.imgn = 0; S.imgused = 0; S.imgid = 0; S.imgemb.clear();
        if (g_img_n > 0 && g_img_pad >= 0) {
            bool need = false;
            for (size_t i = 0; i < S.ids.size(); i++) if (S.ids[(size_t)i] == g_img_pad) { need = true; break; }
            if (need) { S.imgemb = g_imgemb; S.imgn = g_img_n; S.imgid = g_img_id; S.imgused = 0; }
        }
        S.t_admit = NOWS_();
        cb_bind(S);
        int ck_hit = -1, ck_lcp = 0;
        // ★★ 第32轮: 命中判定搬到 cb_lcp_probe (只读探针) —— !CPREF 的入队前咨询与这里的
        //   实际命中必须是【同一份代码】, 否则"网关说能复用、引擎说不能"会互相打脸。
        if (!S.ids.empty()) ck_hit = cb_lcp_probe(S.ids, S.imgid, &ck_hit, &ck_lcp, &S.lcp_all);
        if (ck_hit >= 0) {
            MCk &C = g_mck[(size_t)ck_hit];
            if (C.sid != S.sid)
                printf("[orn] ★ 跨会话前缀命中: 检查点 sid=%s(%.8s) ← 本请求 sid=%.8s, 只按内容复用\n",
                       C.sid.c_str(), C.sid.c_str(), S.sid.c_str());
            S.conv = C.conv; S.S = C.S; S.logits = C.logits; S.imgused = C.img_used;
            size_t nr = (size_t)C.pos * NKV * HD;
            for (int l = 0; l < NLAYER; l++) if (!g_lay[l].recr) {
                memcpy(S.Kc[(size_t)l].data(), C.Kc[(size_t)l].data(), nr * 4);
                memcpy(S.Vc[(size_t)l].data(), C.Vc[(size_t)l].data(), nr * 4);
            }
            S.pos = C.pos; S.reused = C.pos;
            C.last = ++g_mck_tick; C.hits++;     // ★ 命中计数 (淘汰时保护被复用过的检查点)
        }
        // ★★ 第32轮: 【失败要快】冷预填充闸门 —— 在真正开算之前判, 绝不把十几分钟的冷预填充
        //   塞进唯一引擎通道。台账(真实事故): 用户 14685 tok 的 agent 载荷 ⇒ 引擎保尾砍头到 8192
        //   ⇒ 冷预填充 ~13 分钟 ⇒ 该槽被独占 ⇒ 之后所有请求排队 ⇒ 用户观感="服务死了" ✗。
        //   判据: 本次【真正要算的新增 token】= ids - 可复用前缀; 超过 K200_MAX_COLD_PREFILL_TOK
        //   ⇒ 立刻 __SREJ__(网关转成明确错误), 不预填充、不占槽。0 = 关闭该保护。
        if (g_maxcold > 0) {
            int coldtok = (int)S.ids.size() - S.reused;
            if (coldtok > g_maxcold) {
                cb_unbind();
                printf("[orn] ★ 拒绝(cold-prefill-too-large): cid=%d prompt=%d tok 可复用=%d 冷预填充=%d > 上限 %d"
                       " ⇒ 不预填充, 立刻报错 (绝不占着唯一引擎通道跑十几分钟)\n",
                       S.reqid, (int)S.ids.size(), S.reused, coldtok, g_maxcold);
                printf("__SREJ__ %d cold-prefill-too-large tok=%d reused=%d cold=%d max=%d\n",
                       S.reqid, (int)S.ids.size(), S.reused, coldtok, g_maxcold);
                fflush(stdout);
                S.active = false; S.used = false; S.feed = false;
                return;
            }
        }
        double t0 = NOWS_();
        if (S.reused == 0) {
            // ★★ 关键: 从零开始的预填充必须把【该槽上次请求残留的递推状态清零】——
            //    老路径 run_request 里对应的就是 g_st.reset() (memset conv + S)。
            //    漏了这一步 ⇒ 新请求会从上一条请求在【同一槽位】留下的
            //    delta-net S / conv 残值继续递推 ⇒ 同一输入在不同槽位/不同历史下给出不同答案
            //    (实测: "你好" 出现两个不同答案)。KV 不用清 (位置 0..pos 会被逐个覆盖)。
            memset(S.conv.data(), 0, S.conv.size() * 4);
            memset(S.S.data(), 0, S.S.size() * 4);
            S.imgused = 0;
        }
        if (S.reused < (int)S.ids.size()) {
            unsigned _pf_pump = 0;
            for (size_t i = (size_t)S.reused; i < S.ids.size(); ) {
                if (S.pos >= MAXT) {   // ★ 第30轮: 入口已"保尾砍头", 这里正常不该发生 (兜底 + 告警)
                    printf("[orn] ★ WARN: 槽%d 预填充触到 MAXT=%d (入口保尾砍头未生效?)\n", (int)(&S - &g_slots[0]), MAXT);
                    S.ids.resize(i); break; }
                // ★★ 第32轮: 预填充【期间】也抽空命令队列 —— ① /health(!SSTAT) 照常答复 (旧行为:
                //   预填充把命令循环整段占住 ⇒ 网关 /health 与后续请求全像"服务死"); ② 客户端中断
                //   (!SDEL) 能立刻取消本槽, 不再空跑到 MAXT。
                if (++_pf_pump > 0u) {   // ★ 每批之后都抽空一次: 深冷预填充一个批可达数秒, 8 批一次会让 /health 等近 1 分钟
                    pump_cmds(0, 1);
                    if (!S.used) {
                        cb_unbind();
                        printf("[orn] ★ 槽%d cid=%d 预填充被取消(客户端中断 !SDEL): 停在 %d/%d tok, 立即释放\n",
                               (int)(&S - &g_slots[0]), S.reqid, S.pos, (int)S.ids.size());
                        fflush(stdout);
                        return;
                    }
                }
                size_t remain = S.ids.size() - i;
                int T = 1;
                if (remain >= (size_t)g_bmin) {
                    T = (int)((remain > (size_t)g_bmax) ? g_bmax : remain);
                    if (S.pos + T > MAXT) T = MAXT - S.pos;
                }
                // ★ 第30轮: 取消"为打检查点而切批"(与 run_request 同一处修正, 见那里的注释)
                if (im_start_id >= 0 && S.pos > 0) {
                    bool bnd = (S.ids[i] == im_start_id);
                    if (!bnd && T > 1)
                        for (int q = 1; q < T; q++) if (S.ids[i + (size_t)q] == im_start_id) { bnd = true; break; }
                    if (bnd) cb_mck_take(S, S.ids, S.pos, S.logits, 0);
                }
                if (T > 1) { forward_batch(&S.ids[i], T, S.pos, S.logits); S.pos += T; i += (size_t)T; }
                else       { forward(S.ids[i], S.pos, S.logits);        S.pos += 1; i += 1; }
            }
            cb_mck_take(S, S.ids, S.pos, S.logits, 1);
        }
        S.prefill_s = NOWS_() - t0;
        cb_unbind();
        int newtok = (int)S.ids.size() - S.reused;
        printf("[orn] slot%d cid=%d 入槽: prompt=%d tok 用时 %.3fs (%.1f ms/tok, %s 复用%d tok) 新增%d tok | gemv %llu 次 | 图像%d行 指纹%016llx\n",
               (int)(&S - &g_slots[0]), S.reqid, (int)S.ids.size(), S.prefill_s,
               S.prefill_s * 1000.0 / (double)(newtok > 0 ? newtok : 1),
               (S.reused == 0 ? "全量" : (S.reused == (int)S.ids.size() ? "整段命中" : "部分前缀命中")),
               S.reused, newtok, (unsigned long long)g_gemv_n, S.imgn, (unsigned long long)S.imgid);
        {   int nv = 0; for (size_t i2 = 0; i2 < g_mck.size(); i2++) if (g_mck[i2].valid) nv++;
            printf("[orn]   ★ 检查点表 %d/%d 有效; 本次最长公共前缀(lcp)=%d tok\n", nv, (int)g_mck.size(), S.lcp_all); }
        printf("__SPREF__ %d reused=%d new=%d prefill_s=%.3f lcp=%d ptok=%d\n",
               S.reqid, S.reused, newtok, S.prefill_s, S.lcp_all, (int)S.ids.size());
        printf("__SACTIVE__ %d %d\n", S.reqid, (int)(&S - &g_slots[0]));
        cb_advance(S);        // 预填充的 logits ⇒ 第一个 token (与 run_request 同序, 立刻发出)
    };
    auto cb_drain_queue = [&]() {
        for (;;) {
            if (g_squeue.empty()) return;
            bool any_solo = false, que_solo = false;
            int nact = 0;
            for (size_t i = 0; i < g_slots.size(); i++)
                if (g_slots[i].active) { nact++; if (g_slots[i].solo) any_solo = true; }
            for (size_t i = 0; i < g_squeue.size(); i++) if (g_squeue[i].solo) que_solo = true;
            int pick = -1, sl = -1;
            for (size_t i = 0; i < g_squeue.size(); i++) {
                CbReq &R = g_squeue[i];
                if (R.solo) { if (nact > 0) continue; }
                else        { if (any_solo || que_solo) continue; }
                for (size_t s2 = 0; s2 < g_slots.size(); s2++) if (!g_slots[s2].used) { sl = (int)s2; break; }
                if (sl >= 0) { pick = (int)i; break; }
            }
            if (pick < 0) return;
            CbReq R = g_squeue[(size_t)pick];
            g_squeue.erase(g_squeue.begin() + pick);
            cb_admit(g_slots[(size_t)sl], R);
        }
    };
    auto cb_step = [&]() {
        std::vector<CbSlot *> act, solo_act;
        for (size_t i = 0; i < g_slots.size(); i++)
            if (g_slots[i].active && g_slots[i].feed) act.push_back(&g_slots[i]);
        if (act.empty()) return;
        bool any_solo = false;
        for (size_t i = 0; i < act.size(); i++) if (act[i]->solo) any_solo = true;
        if (any_solo) {                       // ★ solo 槽独占批 ⇒ 保证它 M=1
            for (size_t i = 0; i < act.size(); i++) if (act[i]->solo) solo_act.push_back(act[i]);
            act.swap(solo_act);
            if (act.size() > 1) { std::vector<CbSlot *> one; one.push_back(act[0]); act.swap(one); }
        }
        double _t0 = NOWS_(); double _c0 = g_i8_t;
        double _h0[10] = {g_p_ssm, g_p_att, g_p_lay, g_p_lm, g_p_emb, g_p_nrm, g_cb_d2h, g_g_h2d, g_g_d2h, g_g_wait};
        forward_multi(act);
        if (g_prof >= 1) {
            static int _cbpr = 0; static double _cbacc = 0, _cbcard = 0;
            static double _cbh[10] = {0};
            _cbacc += NOWS_() - _t0; _cbcard += g_i8_t - _c0;
            _cbh[0] += g_p_ssm - _h0[0]; _cbh[1] += g_p_att - _h0[1]; _cbh[2] += g_p_lay - _h0[2];
            _cbh[3] += g_p_lm - _h0[3];   _cbh[4] += g_p_emb - _h0[4]; _cbh[5] += g_p_nrm - _h0[5];
            _cbh[6] += g_cb_d2h - _h0[6]; _cbh[7] += g_g_h2d - _h0[7]; _cbh[8] += g_g_d2h - _h0[8];
            _cbh[9] += g_g_wait - _h0[9];
            if (++_cbpr >= 10) {
                double n = _cbpr;
                printf("[orn][CBSTEP] 均步: M=%d 墙钟 %.2f ms = 卡gemm %.2f + 主机 %.2f [SSM%.2f 注意力%.2f 层内%.2f lm%.2f 嵌入%.2f 归一%.2f] | 卡四段 H2D%.2f launch+wait%.2f D2H%.2f | 控制侧logits搬运 %.2f ms\n",
                       (int)act.size(), _cbacc * 1000 / n, _cbcard * 1000 / n, (_cbacc - _cbcard) * 1000 / n,
                       _cbh[0] * 1000 / n, _cbh[1] * 1000 / n, _cbh[2] * 1000 / n, _cbh[3] * 1000 / n,
                       _cbh[4] * 1000 / n, _cbh[5] * 1000 / n,
                       _cbh[7] * 1000 / n, _cbh[9] * 1000 / n, _cbh[8] * 1000 / n, _cbh[6] * 1000 / n);
                fflush(stdout);
                _cbpr = 0; _cbacc = 0; _cbcard = 0;
                for (int z = 0; z < 10; z++) _cbh[z] = 0;
            }
        }
        const size_t V = (size_t)g_outw.M;
        for (size_t i = 0; i < act.size(); i++) {
            CbSlot *S = act[i];
            double _dt = NOWS_();
            S->logits.assign(g_mLOG.begin() + (ptrdiff_t)(i * V), g_mLOG.begin() + (ptrdiff_t)((i + 1) * V));
            g_cb_d2h += NOWS_() - _dt;
            S->pos++;
            cb_advance(*S);
        }
        g_cb_steps++;
    };
    auto cb_busy = [&]() {
        if (!g_squeue.empty()) return true;
        for (size_t i = 0; i < g_slots.size(); i++) if (g_slots[i].active) return true;
        return false;
    };
    {
        const char *e = getenv("K200_SLOTS");
        int n = e ? atoi(e) : 0;
        cb_init(n);
        if (n == 0) printf("[orn] ★ 多槽位未启用 (K200_SLOTS=0): 只有老单会话 \"@\" 协议可用\n");
        {   const char *em = getenv("K200_MAX_COLD_PREFILL_TOK");
            if (em && atoi(em) >= 0) g_maxcold = atoi(em);
            printf("[orn] ★ 冷预填充上限 = %d tok (K200_MAX_COLD_PREFILL_TOK; 0=关闭保护; 超限请求在【入队前】就被网关拒绝, 引擎侧同样兜底拒收)\n", g_maxcold);
        }
    }

    {   // ==================== 服务循环 ====================
        handle_cmd = [&](std::string &s) {
            if (!s.empty() && s[0] == '!') {                 // ★ 命令
                if (s.compare(0, 5, "!VIS ") == 0) {
                    char vp[512]; int vW = 0, vH = 0;
                    if (sscanf(s.c_str() + 5, "%511s %d %d", vp, &vW, &vH) == 3) vis_cmd(vp, vW, vH);
                    else printf("__VIS__ ERR 参数\n");
                } else if (s.compare(0, 5, "!VISR") == 0) {
                    g_imgemb.clear(); g_img_n = 0; g_img_used = 0; g_img_id = 0;   // ★ DETVIS: 指纹一并清
                    printf("__VISR__ 0\n");
                } else if (s.compare(0, 7, "!VISIMG") == 0) {
                    printf("__VISIMG__ %d %016llx\n", g_img_n, (unsigned long long)g_img_id);
                } else if (s.compare(0, 7, "!VISQ  ") == 0) {
                    printf("__VISQ__ %d %d\n", vis_loaded(), g_img_n);
                } else if (s.compare(0, 6, "!SLOTS") == 0) {
                    int n = 0; sscanf(s.c_str() + 6, "%d", &n);
                    cb_init(n);
                    printf("__SLOTS__ %d state_mb=%.1f\n", (int)g_slots.size(), g_slots.size() * cb_state_mb());
                } else if (s.compare(0, 5, "!SMCK") == 0) {
                    printf("__SMCK__ n=%d\n", (int)g_mck.size());
                    for (size_t i2 = 0; i2 < g_mck.size(); i2++) {
                        MCk &C = g_mck[i2];
                        printf("__SMCKROW__ %d valid=%d kind=%d pos=%d sid=%s ids=%d kv0=%d last=%llu\n",
                               (int)i2, (int)C.valid, C.kind, C.pos, C.sid.c_str(), (int)C.ids.size(),
                               C.Kc.empty() ? 0 : (int)C.Kc[3].size(), (unsigned long long)C.last);
                    }
                    printf("__SMCKEND__\n");
                } else if (s.compare(0, 6, "!CPREF") == 0) {
                    // ★★ 第32轮: 【冷预填充代价咨询】—— 网关在 !SADD 之前问, 只读、不占槽。★★
                    //   为什么必须问引擎: token 数要用【引擎自己的 tokenizer】的真实值 —— 字符估算
                    //   对中文/数字误差可达 3~4 倍 (中文 1 字可能 1~3 token; 数字 1 token/字符) ✗。
                    std::vector<std::string> tk; std::string r = s.substr(6);
                    {   size_t i = 0;
                        while (i < r.size()) {
                            while (i < r.size() && r[i] == ' ') i++;
                            size_t j = i;
                            while (j < r.size() && r[j] != ' ') j++;
                            if (j > i) tk.push_back(r.substr(i, j - i));
                            i = j;
                        }
                    }
                    std::string b64; uint64_t cimg = 0; int has_img = 0; size_t want_len = 0; bool has_len = false;
                    for (size_t k = 0; k < tk.size(); k++) {
                        const std::string &t = tk[k];
                        if (t.rfind("b64:", 0) == 0) b64 = t.substr(4);
                        else if (t.rfind("img=", 0) == 0) { cimg = (uint64_t)strtoull(t.c_str() + 4, NULL, 16); has_img = 1; }
                        else if (t.rfind("len=", 0) == 0) { want_len = (size_t)strtoul(t.c_str() + 4, NULL, 10); has_len = true; }
                    }
                    static const char *T64c = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                    int rev2[256]; for (int i = 0; i < 256; i++) rev2[i] = -1;
                    for (int i = 0; i < 64; i++) rev2[(unsigned char)T64c[i]] = i;
                    std::string pr; int val = 0, bits = 0;
                    for (size_t i = 0; i < b64.size(); i++) {
                        if (b64[i] == '=') break;
                        int c = rev2[(unsigned char)b64[i]]; if (c < 0) continue;
                        val = ((val << 6) | c) & 0xFFFFFF; bits += 6;
                        if (bits >= 8) { bits -= 8; pr += (char)((val >> bits) & 0xFF); }
                    }
                    if (has_len && pr.size() != want_len) {
                        printf("[orn] ★ WARN: CPREF 丢弃不完整帧 (声明 %zu 字节, 实收 %zu) —— 按探针失败回报\n",
                               want_len, pr.size());
                        printf("__CPREF__ tok=-1 raw=-1 reused=0 cold=-1 coldmax=%d err=truncated\n", g_maxcold);
                        fflush(stdout);
                    } else {
                        std::chrono::steady_clock::time_point ct0 = std::chrono::steady_clock::now();
                        std::vector<int> ids = g_tk.encode(pr);
                        double tk_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ct0).count();
                        int raw = (int)ids.size();
                        if ((int)ids.size() > MAXT) head_trim_keep_tail(ids, MAXT);
                        int haspad = 0;
                        if (g_img_pad >= 0) for (size_t i = 0; i < ids.size(); i++) if (ids[(size_t)i] == g_img_pad) { haspad = 1; break; }
                        int hs = -1, hl = 0, la = 0;
                        if (!haspad || has_img) cb_lcp_probe(ids, cimg, &hs, &hl, &la);   // 带图却没给指纹 ⇒ 保守: 不认复用
                        int reused = (hs >= 0) ? hl : 0;
                        int cold = (int)ids.size() - reused;
                        printf("[orn] ★ CPREF 实测(引擎 tokenizer %.1f ms): raw=%d tok, 保尾砍头后=%d, 可复用=%d,"
                               " 冷预填充=%d (上限 %d)\\n", tk_ms, raw, (int)ids.size(), reused, cold, g_maxcold);
                        printf("__CPREF__ tok=%d raw=%d reused=%d cold=%d coldmax=%d lcp=%d slot=%d haspad=%d tk_ms=%.1f\\n",
                               (int)ids.size(), raw, reused, cold, g_maxcold, la, hs, haspad, tk_ms);
                        fflush(stdout);
                    }
                } else if (s.compare(0, 6, "!SSTAT") == 0) {
                    int na = 0, nf = 0;
                    for (size_t i = 0; i < g_slots.size(); i++) { if (g_slots[i].active) na++; if (!g_slots[i].used) nf++; }
                    printf("__SSTAT__ slots=%d active=%d free=%d queued=%d steps=%llu served=%llu\n",
                           (int)g_slots.size(), na, nf, (int)g_squeue.size(),
                           (unsigned long long)g_cb_steps, (unsigned long long)g_cb_served);
                } else if (s.compare(0, 6, "!SPAUS") == 0) {
                    g_cb_run = false; printf("__SPAUSE__ ok\n");
                } else if (s.compare(0, 7, "!SRESUM") == 0) {
                    g_cb_run = true; printf("__SRESUME__ ok\n");
                } else if (s.compare(0, 5, "!SADD") == 0) {
                    std::vector<std::string> tk; std::string r = s.substr(5);
                    {   size_t i = 0;
                        while (i < r.size()) {
                            while (i < r.size() && r[i] == ' ') i++;
                            size_t j = i;
                            while (j < r.size() && r[j] != ' ') j++;
                            if (j > i) tk.push_back(r.substr(i, j - i));
                            i = j;
                        }
                    }
                    if (tk.size() < 3) { printf("__SREJ__ 0 bad-args\n"); fflush(stdout); return; }
                    int cid = atoi(tk[0].c_str());
                    int mt  = atoi(tk[1].c_str());
                    std::string sid, b64; uint64_t want_img = 0; int solo = 0;
                    size_t want_len = 0; bool has_len = false;   // ★ 第32轮: 帧完整性校验
                    for (size_t k = 2; k < tk.size(); k++) {
                        const std::string &t = tk[k];
                        if (t.rfind("sid=", 0) == 0) { sid = t.substr(4); if (sid == "-") sid.clear(); }  // ★ "-" = 无会话 ⇒ 不参与前缀复用
                        else if (t.rfind("img=", 0) == 0) want_img = (uint64_t)strtoull(t.c_str() + 4, NULL, 16);
                        else if (t.rfind("solo=", 0) == 0) solo = atoi(t.c_str() + 5);
                        else if (t.rfind("len=", 0) == 0) { want_len = (size_t)strtoul(t.c_str() + 4, NULL, 10); has_len = true; }
                        else if (t.rfind("b64:", 0) == 0) b64 = t.substr(4);
                    }
                    static const char *T64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                    int rev[256]; for (int i = 0; i < 256; i++) rev[i] = -1;
                    for (int i = 0; i < 64; i++) rev[(unsigned char)T64[i]] = i;
                    std::string pr; int val = 0, bits = 0;
                    for (size_t i = 0; i < b64.size(); i++) {
                        if (b64[i] == '=') break;
                        int c = rev[(unsigned char)b64[i]]; if (c < 0) continue;
                        val = ((val << 6) | c) & 0xFFFFFF; bits += 6;
                        if (bits >= 8) { bits -= 8; pr += (char)((val >> bits) & 0xFF); }
                    }
                    if (has_len && pr.size() != want_len) {   // ★ 半截帧: 绝不拿半截 prompt 去算
                        printf("[orn] ★ WARN: 丢弃不完整命令帧 (cid=%d 声明 %zu 字节, 实收 %zu) —— 不入槽\n",
                               cid, want_len, pr.size());
                        printf("__SREJ__ %d truncated-frame got=%zu want=%zu\n", cid, pr.size(), want_len);
                        fflush(stdout); return;
                    }
                    if (g_slots.empty()) { printf("__SREJ__ %d no-slots\n", cid); fflush(stdout); return; }
                    CbReq R; R.cid = cid; R.maxtok = (mt > 0) ? mt : 64; R.solo = (solo != 0); R.sid = sid;
                    R.want_img = want_img;
                    // ★★ 第30轮d: 【不要在命令循环里 tokenize】★★
                    //   实测: 在另一个请求正在解码时提交一个大 prompt, 引擎会在 handle_cmd 里
                    //   同步跑 g_tk.encode(17KB 文本) ⇒ 把正在解码的槽【整段冻住】(实测 9 个 token
                    //   的"解码耗时"从 0.6s 变成 71s / 189s, 用户看到的就是"速度掉到 0.1 tok/s")。
                    //   ⇒ 只把原串入队, tokenize 推迟到 cb_admit (solo 情况下那时引擎已空闲)。
                    R.pr = pr;
                    g_squeue.push_back(R);
                    printf("__SADD__ %d queued chars=%d maxtok=%d solo=%d sid=%s\n",
                           cid, (int)pr.size(), R.maxtok, (int)R.solo, R.sid.c_str());
                } else if (s.compare(0, 5, "!SDEL") == 0) {
                    int cid = atoi(s.c_str() + 5); int ok = 0;
                    for (size_t i = 0; i < g_squeue.size(); i++) if (g_squeue[i].cid == cid) { g_squeue.erase(g_squeue.begin() + i); ok = 1; break; }
                    for (size_t i = 0; i < g_slots.size(); i++) if (g_slots[i].used && g_slots[i].reqid == cid) {
                        g_slots[i].used = false; g_slots[i].active = false; g_slots[i].feed = false; ok = 1; }
                    printf("__SDEL__ %d %s\n", cid, ok ? "ok" : "notfound");
                } else {
                    printf("__VIS__ ERR 未知命令\n");
                }
                fflush(stdout);
                return;
            }
            if (s.size() < 3 || s[0] != '@') {
                // ★★ 第32轮: 非法/半截帧【不再静默丢弃】★★
                //   旧行为: 直接 return ⇒ 若上一帧是半截(对端中途死掉)残留, 与这一帧拼在一起时
                //   新命令被静默吞掉 ⇒ 网关一直等 (观感 = 引擎卡死 ✗)。
                //   新行为: ① 先在行内【重同步】到最后一个命令起点再解析一次 (半截 b64 会被 len=
                //   校验拦下); ② 仍然不合法就大声告警 —— 绝不悄悄吞命令。
                size_t p = std::string::npos;
                static const char *mk[7] = {"!SADD ", "!CPREF ", "!SDEL ", "!SSTAT", "!SMCK", "!VIS ", "!VISR"};
                for (int k = 0; k < 7; k++) {
                    size_t q = s.rfind(mk[k]);
                    if (q != std::string::npos && q > 0 && (p == std::string::npos || q > p)) p = q;
                }
                if (p != std::string::npos) {
                    printf("[orn] ★ WARN: 半截残留帧 + 新命令拼在一起 (%d 字节) ⇒ 重同步到偏移 %d 的命令\n",
                           (int)s.size(), (int)p);
                    fflush(stdout);
                    std::string s2 = s.substr(p);
                    handle_cmd(s2);
                    return;
                }
                printf("[orn] ★ WARN: 丢弃非法命令帧 (%d 字节, 首字节=%02x): %.40s\n",
                       (int)s.size(), s.empty() ? 0 : (unsigned char)s[0], s.c_str());
                fflush(stdout);
                return;
            }
            size_t p2 = s.find('@', 1);
            if (p2 == std::string::npos) return;
            int mt = atoi(s.substr(1, p2 - 1).c_str());
            std::string rest = s.substr(p2 + 1), sid;
            if (rest.rfind("sid=", 0) == 0) { size_t p3 = rest.find('@', 0); sid = rest.substr(4, p3 - 4); rest = (p3 == std::string::npos) ? std::string() : rest.substr(p3 + 1); }
            std::string b64 = rest;
            if (b64.rfind("b64:", 0) == 0) b64 = b64.substr(4);
            static const char *T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            int rev[256]; for (int i = 0; i < 256; i++) rev[i] = -1;
            for (int i = 0; i < 64; i++) rev[(unsigned char)T[i]] = i;
            std::string pr; int val = 0, bits = 0;
            for (size_t i = 0; i < b64.size(); i++) {
                if (b64[i] == '=') break;
                int c = rev[(unsigned char)b64[i]]; if (c < 0) continue;
                val = ((val << 6) | c) & 0xFFFFFF; bits += 6;
                if (bits >= 8) { bits -= 8; pr += (char)((val >> bits) & 0xFF); }
            }
            cb_unbind();       // ★ 老路径必须作用在【全局单会话状态】上
            printf("__BEGIN__\n");
            run_request(pr, mt, true, sid);
            printf("__END__\n");
        };
        for (;;) {
            while (!g_defer.empty()) {        // ★ 预填充期间被推迟的命令 (图像塔等) 在这里补做
                std::string sd = g_defer.front(); g_defer.erase(g_defer.begin());
                printf("[orn] ★ 预填充结束, 补做被推迟的命令: %.40s\n", sd.c_str());
                handle_cmd(sd);
            }
            bool busy = g_cb_run && cb_busy();
            if (!stdin_eof) {
                pump_cmds(busy ? 0 : -1, 0);   // 空闲时阻塞等命令; 忙时非阻塞抽空
            } else {
                pump_cmds(200, 0);             // ★ 对端不在: 200ms 节流轮询, 绝不空转/绝不退出
                struct stat s0, s1;
                bool need_reopen = (fstat(0, &s0) != 0) || (stat(fifo_path, &s1) != 0) ||
                                   (s0.st_ino != s1.st_ino) || (s0.st_dev != s1.st_dev);
                if (need_reopen && !(g_cb_run && cb_busy())) {
                    int nf = open(fifo_path, O_RDONLY);      // 路径被重建(旧网关 remove+mkfifo) ⇒ 重开
                    if (nf >= 0) {
                        dup2(nf, 0); if (nf != 0) close(nf);
                        printf("[orn] ★ 重新附着命令 FIFO %s (客户端中断/重连; 引擎与其热缓存都不需要重启)\n", fifo_path);
                        fflush(stdout);
                    } else usleep(200000);
                }
            }
            if (g_cb_run) {
                if (!g_squeue.empty()) cb_drain_queue();
                if (cb_busy()) cb_step();
            }
            fflush(stdout);
        }
    }
    return 0;
}
