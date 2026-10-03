// orn.cpp — Ornith-1.5-9B (arch=qwen35: 24 层 gated-delta-net 线性注意力 + 8 层全注意力, 32 层主干, blk.32 为 MTP 跳过)
// 昆仑 K200 双芯:
//   * 权重常驻两芯 HBM: Q8_0 → 纯 int8 [行=out][列=in] 行优先, 沿 K 分 CH 块各一个 max
//   * 全部 GEMV 走官方 gemm_int8 (SD-CDNN 引擎, trans_b=true, 逐块 beta=1 累加), 双芯行分裂并发
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
#include <xpu/runtime.h>
// ---- ★ 官方 gemm_int8 (SD-CDNN 引擎, 实测 110~130 GB/s): 替代自写 GM2LM GEMV ----
//   y[M,N] = x[M,K] · W[N,K]^T ; trans_b=true, B 就是 GGUF 原始行序 [out,in], max_b = max|W|(不是步长)
#include <xpu/refactor/nn.h>
#include "xpu/refactor/context/xpu_act_type.h"
namespace baidu { namespace xpu { namespace api {
int gemm_int8(Context* ctx, const bool trans_a, const bool trans_b, int m, int n, int k,
              float alpha, const float* a, int lda, const int8_t* b, float max_b, int ldb,
              float beta, float* c, int ldc);
}}}
namespace api = baidu::xpu::api;
static api::Context *g_ctx[2] = {nullptr, nullptr};   // 双芯各一个 Context

void run_gemv_q8_0(int cl, int co, const void *W, const void *x, void *y, int M, int N);

// ---- ★ 数值回归钩子: 逐位置 logits dump (K200_LDUMP=<文件>) — 纯落盘, 不参与计算 ----
static FILE *g_ldump = nullptr;

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
static const int MAXT = 1024;

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
    size_t bytes = 0;                        // int8 权重总字节 = M*N
    bool  i8 = true;                         // 走官方 gemm_int8
    int   CH = 1;                            // K 分块数 (每块一个 max)
    std::vector<float> maxv;                 // [CH] 每块 max|w|
    int   nsplit() const { return d[1] ? 2 : 1; }
};
static int g_nosplit = 0;                              // K200_NOSPLIT=1 关闭分裂 (诊断/A-B 用)
static uint64_t g_hbm[2] = {0, 0};
// ---- K 分块数 CH: 官方 op 每张量只有一个 max_b ⇒ 沿 K 切 CH 块, 每块各自 max 并 beta=1 累加 ----
static int g_ch       = 16;   // K200_CH 覆盖 (默认值, 上级实测: CH=16 误差 ~2~10%)
static int g_ch_down  = 32;   // K200_CH_DOWN: ffn_down (12288 列) 最敏感, 单独提高
static double g_launch = 0, g_wait = 0, g_pack = 0;
static uint64_t g_launch_cnt = 0;
static std::map<int, int> g_ch_hist;
static uint64_t g_calls_per_tok = 0;   // 每 token 的 gemm_int8 调用数
static api::Context *ctx_of(int dv) {
    if (!g_ctx[dv]) {
        xpu_set_device(dv);
        g_ctx[dv] = new api::Context(api::kXPU1);
        printf("[orn] 官方 api::Context chip%d 就绪 (%p)\n", dv, (void *)g_ctx[dv]);
    }
    return g_ctx[dv];
}

// ---- 一组张量 (同一个 x, 但官方 op 下每个张量必须各自调用: 不同张量 max 不同) ----
struct Grp {
    std::vector<WDev> ws;
    std::vector<int>  off;
    int M = 0, N = 0;
    void add(GF &g, const char *name);          // 定义见下 (需要 up_i8)
};

struct Layer {
    bool recr = false;
    std::vector<float> attn_norm, post_norm;
    // 线性注意力 (gated delta net)
    Grp grp_a, grp_mid;                      // grp_a = 同读 attn_norm 输出的张量组; grp_mid = ssm_out/o
    std::vector<float> alpha_w, beta_w;      // [32 x 4096] host (小, 不值得上卡)
    std::vector<float> conv1d;               // [c*4 + tap] = 4 per channel
    std::vector<float> ssm_a, ssm_dt, ssm_norm;
    // 全注意力
    std::vector<float> q_norm, k_norm;
    std::vector<float> Kc, Vc;   // KV cache (仅注意层, MAXT*NKV*HD)
    // MLP
    Grp grp_ffn;                             // {ffn_gate, ffn_up} (同读 post_norm 输出)
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

// ============================================================================
// ★ 官方 gemm_int8 权重加载: Q8_0 → 按 K 分 CH 块对称 int8 (每块一个 max)
//   HBM 布局: 每片 = CH 个 plane, plane c = [该片 mp 行][KC] 行优先 (KC = N/CH)
//   官方 op 语义: y = x·W^T (trans_b=true), B = 行优先 [行=输出, 列=输入] = GGUF 原始行序,
//                 max_b = max|W| (不是步长!), beta=1 逐块累加
// ============================================================================
// 扫 raw 的 rows 行, 更新每 K 块 max|w| (= 精确的 max|Q8_0 反量化值|, |d|*maxabs(q))
static void i8_scan_max(const unsigned char *raw, int rows, int N, int CH, std::vector<float> &mx) {
    int nb = N / 32, nbc = (N / CH) / 32;
    for (int m = 0; m < rows; m++) {
        const unsigned char *rp = raw + (size_t)m * nb * 34;
        for (int c = 0; c < CH; c++) {
            const unsigned char *cp = rp + (size_t)c * nbc * 34;
            float best = mx[c];
            for (int b = 0; b < nbc; b++) {
                float d = h2f(*(const unsigned short *)(cp + (size_t)b * 34));
                const signed char *q = (const signed char *)(cp + (size_t)b * 34 + 2);
                int qm = 0;
                for (int i = 0; i < 32; i++) { int v = q[i]; if (v < 0) v = -v; if (v > qm) qm = v; }
                float a = fabsf(d) * (float)qm;
                if (a > best) best = a;
            }
            mx[c] = best;
        }
    }
}
// 量化 raw 的 rows 行 (全局行号从 row0 起) 写入 host (已按 片/块 plane 布局)
static void i8_pack_rows(const unsigned char *raw, int rows, int row0, int M, int N, int CH,
                         const std::vector<float> &mx, int mh, unsigned char *host) {
    int nb = N / 32, KC = N / CH, nbc = KC / 32;
    for (int r = 0; r < rows; r++) {
        int m  = row0 + r;
        int p  = (m < mh) ? 0 : 1;
        int lr = (p == 0) ? m : m - mh;
        int mp = (p == 0) ? mh : M - mh;
        const unsigned char *rp = raw + (size_t)r * nb * 34;
        for (int c = 0; c < CH; c++) {
            unsigned char *dp = host + ((size_t)p * CH * mp + (size_t)c * mp + (size_t)lr) * KC;
            float f = 127.f / mx[c];
            for (int b = 0; b < nbc; b++) {
                float d = h2f(*(const unsigned short *)(rp + (size_t)(c * nbc + b) * 34));
                const signed char *q = (const signed char *)(rp + (size_t)(c * nbc + b) * 34 + 2);
                signed char *dq = (signed char *)dp + b * 32;
                float gg = d * f;
                for (int i = 0; i < 32; i++) {
                    int v = (int)lrintf((float)q[i] * gg);
                    if (v > 127) v = 127; else if (v < -127) v = -127;
                    dq[i] = (signed char)v;
                }
            }
        }
    }
}
// ---- 一个 Q8_0 张量 → 纯 int8 (K 分 CH 块) 上卡; 自动决定是否两芯行分裂 ----
static WDev up_i8(GF &g, const char *name, int CH, bool allow_split = true) {
    const GT *t = g.get(name);
    if (!t) { printf("FATAL: 缺张量 %s\n", name); exit(1); }
    if (t->type != 8) { printf("FATAL: %s type=%u 不是 Q8_0\n", name, t->type); exit(1); }
    int N = (int)t->dims[0], M = (int)t->dims[1];    // N = 输入维(K), M = 输出行
    if (N > 16384) { printf("FATAL: %s N=%d > 激活缓冲 16384\n", name, N); exit(1); }
    if (M > 600000) { printf("FATAL: %s M=%d > 输出缓冲 600000\n", name, M); exit(1); }
    if (CH < 1) CH = 1;
    if (N % (CH * 32) != 0) { printf("FATAL: %s N=%d 不能被 CH=%d (×32 权重块) 整除\n", name, N, CH); exit(1); }
    uint64_t rowbytes = (uint64_t)(N / 32) * 34;
    const int RB = 256;                              // 每次 256 行 (峰值 ~ 256*N*4 字节)
    std::vector<unsigned char> rb((size_t)rowbytes * RB);
    std::vector<float> mx(CH, 0.f);
    auto t0 = std::chrono::steady_clock::now();
    for (int m0 = 0; m0 < M; m0 += RB) {             // pass A: 每块 max
        int c = (m0 + RB <= M) ? RB : M - m0;
        if (fseek(g.f, (long)(g.data_start + t->off + (uint64_t)m0 * rowbytes), SEEK_SET)) { printf("FATAL: seek %s\n", name); exit(1); }
        if (fread(rb.data(), 1, (size_t)rowbytes * c, g.f) != ((size_t)rowbytes * c)) { printf("FATAL: read %s\n", name); exit(1); }
        i8_scan_max(rb.data(), c, N, CH, mx);
    }
    for (int c = 0; c < CH; c++) if (!(mx[c] > 0.f)) mx[c] = 1e-30f;
    int mh = M, ns = 1;
    if (allow_split && !g_nosplit && M >= 64) { mh = M / 2; ns = 2; }
    WDev w; w.N = N; w.M = M; w.i8 = true; w.CH = CH; w.maxv = mx;
    w.bytes = (size_t)M * (size_t)N;
    w.dev[0] = 0; w.mp[0] = mh;
    w.dev[1] = (ns == 2) ? 1 : -1; w.mp[1] = (ns == 2) ? (M - mh) : 0;
    std::vector<unsigned char> host(w.bytes);
    for (int m0 = 0; m0 < M; m0 += RB) {             // pass B: 量化
        int c = (m0 + RB <= M) ? RB : M - m0;
        if (fseek(g.f, (long)(g.data_start + t->off + (uint64_t)m0 * rowbytes), SEEK_SET)) { printf("FATAL: seek %s\n", name); exit(1); }
        if (fread(rb.data(), 1, (size_t)rowbytes * c, g.f) != ((size_t)rowbytes * c)) { printf("FATAL: read %s\n", name); exit(1); }
        i8_pack_rows(rb.data(), c, m0, M, N, CH, mx, mh, host.data());
    }
    std::vector<unsigned char>().swap(rb);
    g_pack += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (int p = 0; p < 2; p++) {
        if (w.mp[p] <= 0) { w.d[p] = nullptr; continue; }
        size_t by  = (size_t)w.mp[p] * (size_t)N;
        size_t src = (size_t)(p == 0 ? 0 : mh) * (size_t)N;
        xpu_set_device(w.dev[p]);
        if (xpu_malloc(&w.d[p], by)) {
            printf("FATAL: HBM 分配 %.1f MB 失败 chip%d [已用 %.0f MB]\n", by / 1048576.0, w.dev[p], g_hbm[w.dev[p]] / 1048576.0);
            exit(1);
        }
        if (xpu_memcpy(w.d[p], host.data() + src, by, XPU_HOST_TO_DEVICE)) { printf("FATAL: H2D 失败 (M=%d)\n", w.mp[p]); exit(1); }
        xpu_wait();
        g_hbm[w.dev[p]] += by;
    }
    g_ch_hist[CH]++;                                        // 统计 (必须在 d[1] 填好后)
    g_calls_per_tok += (uint64_t)w.nsplit() * (uint64_t)CH;
    return w;
}
// CH 选择: ffn_down (输入 12288 列) 是上级实测最敏感张量 ⇒ 单独提高; 其余用 g_ch
static int ch_for(const char *name) {
    int c = strstr(name, "ffn_down") ? g_ch_down : g_ch;
    return c < 1 ? 1 : c;
}
void Grp::add(GF &g, const char *name) {
    const GT *t = g.get(name);
    if (!t) { printf("FATAL: 缺张量 %s\n", name); exit(1); }
    int n = (int)t->dims[0], m = (int)t->dims[1];
    if (N == 0) N = n; else if (n != N) { printf("FATAL: %s N=%d != 组内 N=%d\n", name, n, N); exit(1); }
    off.push_back(M);
    ws.push_back(up_i8(g, name, ch_for(name)));
    M += m;
}

// ============================ 卡上 GEMV ============================
static void *g_dx[2] = {nullptr, nullptr}, *g_dy[2] = {nullptr, nullptr};
static void *g_dxq[2] = {nullptr, nullptr}, *g_dxs[2] = {nullptr, nullptr};
static uint64_t g_gemv_n = 0;
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
// ---- 一个 (张量, 片): CH 次 gemm_int8, 首块 beta=0 之后 beta=1 累加 ----
static void gemm_tensor_chip(int dv, const WDev &w, int p, const void *xd, void *cd) {
    int CH = w.CH, KC = w.N / CH, mp = w.mp[p];
    api::Context *cx = ctx_of(dv);
    for (int c = 0; c < CH; c++) {
        const float  *A = (const float *)xd + (size_t)c * KC;                      // 激活按 K 偏移
        const int8_t *B = (const int8_t *)w.d[p] + (size_t)c * (size_t)mp * KC;    // 块 plane c
        auto t0 = std::chrono::steady_clock::now();
        int r = api::gemm_int8(cx, false, true, 1, mp, KC, 1.f, A, w.N,
                               B, w.maxv[c], KC, c == 0 ? 0.f : 1.f, (float *)cd, mp);
        g_launch += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        g_launch_cnt++;
        if (r) { printf("FATAL: gemm_int8 r=%d (dev%d rows=%d K=%d KC=%d chunk=%d/%d)\n", r, dv, mp, w.N, KC, c, CH); exit(1); }
    }
}
// ---- 单片 launch (诊断/selftest 用) ----
static void gemv_single(int dv, const WDev &w, const float *x, float *y, int cl, int co) {
    (void)cl; (void)co;
    xpu_set_device(dv);
    xpu_memcpy(g_dx[dv], x, (size_t)w.N * 4, XPU_HOST_TO_DEVICE);
    gemm_tensor_chip(dv, w, 0, g_dx[dv], g_dy[dv]);
    auto t0 = std::chrono::steady_clock::now();
    if (xpu_wait()) { printf("FATAL: kernel wait (dev%d M=%d N=%d)\n", dv, w.M, w.N); exit(1); }
    g_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    xpu_memcpy(y, g_dy[dv], (size_t)w.M * 4, XPU_DEVICE_TO_HOST);
    xpu_wait();
    g_gemv_n++;
}
// ---- ★ 双芯并发: 每片 CH 次 launch 全部排完, 再每片 wait 一次 + D2H 拼回 y ----
static void gemv(const WDev &w, const float *x, float *y) {
    int ns = w.nsplit();
    for (int p = 0; p < ns; p++) {
        int dv = w.dev[p];
        xpu_set_device(dv);
        xpu_memcpy(g_dx[dv], x, (size_t)w.N * 4, XPU_HOST_TO_DEVICE);
        gemm_tensor_chip(dv, w, p, g_dx[dv], g_dy[dv]);
    }
    int off = 0;
    for (int p = 0; p < ns; p++) {
        int dv = w.dev[p];
        xpu_set_device(dv);
        auto t0 = std::chrono::steady_clock::now();
        if (xpu_wait()) { printf("FATAL: kernel wait (dev%d rows=%d K=%d)\n", dv, w.mp[p], w.N); exit(1); }
        g_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        auto td = std::chrono::steady_clock::now();
        xpu_memcpy(y + off, g_dy[dv], (size_t)w.mp[p] * 4, XPU_DEVICE_TO_HOST);
        xpu_wait();
        g_d2h += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count();
        off += w.mp[p];
    }
    g_gemv_n++;
}
// ---- 张量组: 每张量各自调用 (官方 op 每张量只有一个 max_b, 不能拼) ----
static void gemv_grp(const Grp &G, const float *x, float *y) {
    for (size_t i = 0; i < G.ws.size(); i++) gemv(G.ws[i], x, y + G.off[i]);
}
static void gemv_host(const std::vector<float> &W, int M, int N, const float *x, float *y) {
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
    std::vector<float> &x = g_x;
    static std::vector<float> gy;                 // 拼接组输出缓冲
    // --- 嵌入 (直接从 GGUF 文件解 Q8_0 行, 不进 HBM) ---
    {
        const GT *t = g_gguf.get("token_embd.weight");
        uint64_t nb = (uint64_t)(EMB / 32) * 34;
        std::vector<unsigned char> raw(nb);
        if (fseek(g_gguf.f, (long)(g_gguf.data_start + t->off + (uint64_t)tok * nb), SEEK_SET)) { printf("FATAL: seek embed\n"); exit(1); }
        if (fread(raw.data(), 1, nb, g_gguf.f) != nb) { printf("FATAL: read embed\n"); exit(1); }
        dequant_q8_0(raw.data(), x.data(), EMB);
    }
    std::vector<float> xb(EMB), o(EMB);
    static std::vector<float> qkv, z, gg, uu;
    for (int l = 0; l < NLAYER; l++) {
        Layer &L = g_lay[l];
        rmsnorm(xb.data(), x.data(), L.attn_norm.data(), EMB, EPS);
        // ===== 波1: 所有以 attn_norm 输出为输入的矩阵 (拼接成一个 launch, 两芯各半) =====
        int tot_a = L.recr ? (PA_GATE + PA_GATES) : AA_TOT;
        int tot_f = FF_TOT;
        if ((int)gy.size() < (tot_a < tot_f ? tot_f : tot_a)) gy.resize(tot_a > tot_f ? tot_a : tot_f);
        { snprintf(g_nb, sizeof(g_nb), "l%d.grpA", l); g_gname = g_nb; }
        gemv_grp(L.grp_a, xb.data(), gy.data());
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
        if (L.recr) {
            static std::vector<float> al, be, co;
            al.resize(NVH); be.resize(NVH); co.resize(CVD);
            gemv_host(L.alpha_w, NVH, EMB, xb.data(), al.data());
            gemv_host(L.beta_w,  NVH, EMB, xb.data(), be.data());
            float *cs = &g_st.conv[(size_t)l * 3 * CVD];
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
            // q(2048: 16x128) | k(2048) | v(4096: 32x128)
            std::vector<float> qc(co.begin(),        co.begin() + 2048);
            std::vector<float> kc(co.begin() + 2048, co.begin() + 4096);
            std::vector<float> vc(co.begin() + 4096, co.end());
            for (int h = 0; h < NKH; h++) l2norm_head(&qc[h * KHD], KHD, EPS);
            for (int h = 0; h < NKH; h++) l2norm_head(&kc[h * KHD], KHD, EPS);
            const float qscale = 1.f / sqrtf((float)KHD);
            static std::vector<float> dout; dout.assign(NVH * KHD, 0.f);
            for (int hv = 0; hv < NVH; hv++) {
                int hk = hv % NKH;                      // 与 llama.cpp 融合内核 iv1 % neq1 一致
                float g_ = L.ssm_a[hv] * softplusf_(al[hv] + L.ssm_dt[hv]);
                float b_ = sigmoidf_(be[hv]);
                float eg = expf(g_);
                float *M = &g_st.S[((size_t)l * NVH + hv) * VHD * KHD];
                const float *kk = &kc[hk * KHD];
                const float *qq = &qc[hk * KHD];
                const float *vv = &vc[hv * KHD];
                for (int j = 0; j < VHD; j++) {
                    float *row = M + (size_t)j * KHD;
                    for (int i = 0; i < KHD; i++) row[i] *= eg;      // S <- S * exp(g)
                    float sk = 0; for (int i = 0; i < KHD; i++) sk += row[i] * kk[i];
                    float d = (vv[j] - sk) * b_;
                    for (int i = 0; i < KHD; i++) row[i] += kk[i] * d;
                    float so = 0; for (int i = 0; i < KHD; i++) so += row[i] * qq[i];
                    dout[hv * KHD + j] = so * qscale;
                }
            }
            // gated rms norm: per 128 head, rmsnorm(ssm_norm) * silu(z)
            for (int h = 0; h < NVH; h++) {
                float *p = &dout[h * KHD];
                double ss = 0; for (int i = 0; i < KHD; i++) ss += (double)p[i] * p[i];
                float r = 1.f / sqrtf((float)(ss / KHD) + EPS);
                for (int i = 0; i < KHD; i++) p[i] = p[i] * r * L.ssm_norm[i] * siluf(z[h * KHD + i]);
            }
            { snprintf(g_nb, sizeof(g_nb), "l%d.ssm_out", l); g_gname = g_nb; }
            gemv_grp(L.grp_mid, dout.data(), o.data());
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
            float *Kc = L.Kc.data();
            float *Vc = L.Vc.data();
            memcpy(&Kc[(size_t)pos * NKV * HD], kk.data(), (size_t)NKV * HD * 4);
            memcpy(&Vc[(size_t)pos * NKV * HD], vv.data(), (size_t)NKV * HD * 4);
            const float sc = 1.f / sqrtf((float)HD);
            int gpr = NH / NKV;
            static std::vector<float> wt;
            if ((int)wt.size() < pos + 1) wt.resize(pos + 1);
            for (int h = 0; h < NH; h++) {
                int kh = h / gpr;
                const float *qp = &q5[h * HD];
                float best = -1e30f;
                for (int t = 0; t <= pos; t++) {
                    const float *kt = &Kc[((size_t)t * NKV + kh) * HD];
                    float s = 0; for (int i = 0; i < HD; i++) s += qp[i] * kt[i];
                    s *= sc; wt[t] = s; if (s > best) best = s;
                }
                float sum = 0;
                for (int t = 0; t <= pos; t++) { float e = expf(wt[t] - best); wt[t] = e; sum += e; }
                float *oh = &ao[h * HD];
                for (int i = 0; i < HD; i++) oh[i] = 0;
                for (int t = 0; t <= pos; t++) {
                    float p = wt[t] / sum;
                    const float *vt = &Vc[((size_t)t * NKV + kh) * HD];
                    for (int i = 0; i < HD; i++) oh[i] += p * vt[i];
                }
                for (int i = 0; i < HD; i++) oh[i] *= sigmoidf_(g5[h * HD + i]);   // 注意输出按维门控
            }
            { snprintf(g_nb, sizeof(g_nb), "l%d.attn_out", l); g_gname = g_nb; }
            gemv_grp(L.grp_mid, ao.data(), o.data());
        }
        for (int i = 0; i < EMB; i++) x[i] += o[i];
        // ===== 波3: MLP 上半 (ffn_gate + ffn_up 拼接, 同一个 post_norm 输出) =====
        rmsnorm(xb.data(), x.data(), L.post_norm.data(), EMB, EPS);
        { snprintf(g_nb, sizeof(g_nb), "l%d.grpFF", l); g_gname = g_nb; }
        gemv_grp(L.grp_ffn, xb.data(), gy.data());
        memcpy(gg.data(), gy.data() + FF_G, FF_GS * 4);
        memcpy(uu.data(), gy.data() + FF_U, FF_US * 4);
        for (int i = 0; i < NFF; i++) gg[i] = siluf(gg[i]) * uu[i];
        { snprintf(g_nb, sizeof(g_nb), "l%d.ffn_down", l); g_gname = g_nb; }
        gemv(L.ffn_down, gg.data(), o.data());
        for (int i = 0; i < EMB; i++) x[i] += o[i];
    }
    rmsnorm(xb.data(), x.data(), g_out_norm.data(), EMB, EPS);
    logits.resize(g_outw.M);
    { snprintf(g_nb, sizeof(g_nb), "lm_head"); g_gname = g_nb; }
    gemv(g_outw, xb.data(), logits.data());
}

// ============================ selftest: 官方 gemm_int8 三方对拍 ============================
//   ① 卡上官方 op (双芯行分裂 + K 分 CH 块 beta=1 累加)
//   ② 主机 double 参考, 用**同一份** int8 量化值 (⇒ ①vs② 只检验 op 语义/布局, 不掺量化误差)
//   ③ 主机 double 参考, 用 Q8_0 原始反量化值 (⇒ ②vs③ 就是这次权重量化的保真度)
static double relrms_pct(const std::vector<float> &a, const std::vector<float> &b, int n) {
    double se = 0, s2 = 0;
    for (int i = 0; i < n; i++) { double d = (double)a[i] - (double)b[i]; se += d * d; s2 += (double)b[i] * (double)b[i]; }
    return 100.0 * sqrt(se / (s2 > 0 ? s2 : 1e-30));
}
static int selftest(GF &g, const char *name, int rows) {
    const GT *t = g.get(name);
    if (!t) { printf("no tensor %s\n", name); return 1; }
    if (t->type != 8) { printf("selftest 只支持 Q8_0 (type=%u)\n", t->type); return 1; }
    int N = (int)t->dims[0], M = (int)t->dims[1];
    int Mt = (rows > 0 && rows < M) ? rows : M;
    int CH = ch_for(name);
    printf("[ST] %s  M=%d N=%d  CH=%d  (对拍 %d 行)\n", name, M, N, CH, Mt);
    std::vector<float> x(N);
    srand(7);
    for (int i = 0; i < N; i++) x[i] = (float)((rand() % 2001) - 1000) / 1000.f;
    if (getenv("K200_XFILE")) {
        FILE *fp = fopen(getenv("K200_XFILE"), "rb");
        if (fp) { size_t r = fread(x.data(), 4, N, fp); fclose(fp); printf("[ST] x 来自 %s (%zu floats)\n", getenv("K200_XFILE"), r); }
    }
    auto t0 = std::chrono::steady_clock::now();
    WDev w = up_i8(g, name, CH, true);          // ★ 双芯行分裂
    double ld = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("[ST] 上卡: chip0 %d 行 / chip1 %d 行  int8 %.1f MB  加载 %.1fs\n",
           w.mp[0], w.nsplit() == 2 ? w.mp[1] : 0, w.bytes / 1048576.0, ld);
    printf("[ST] 每块 max|w|:");
    for (int c = 0; c < w.CH; c++) printf(" %.5g", w.maxv[c]);
    printf("\n");
    std::vector<float> y((size_t)w.M, 0.f);
    t0 = std::chrono::steady_clock::now();
    gemv(w, x.data(), y.data());
    double gms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("[ST] 内核 %d 片 × %d 块 = %d 次调用, %.2f ms => 权重带宽 %.2f GB/s\n",
           w.nsplit(), w.CH, w.nsplit() * w.CH, gms, w.bytes / 1073741824.0 / (gms / 1000.0));
    // 主机参考
    uint64_t rowbytes = (uint64_t)(N / 32) * 34;
    std::vector<unsigned char> raw((size_t)rowbytes * (size_t)Mt);
    if (fseek(g.f, (long)(g.data_start + t->off), SEEK_SET)) { printf("seek fail\n"); return 1; }
    if (fread(raw.data(), 1, raw.size(), g.f) != raw.size()) { printf("read fail\n"); return 1; }
    int KC = N / CH;
    std::vector<float> refq(Mt, 0.f), ref8(Mt, 0.f);
    std::vector<double> hmax(CH, 0.0);
    for (int m = 0; m < Mt; m++) {
        const unsigned char *rp = raw.data() + (size_t)m * rowbytes;
        double aq = 0, a8 = 0;
        for (int n = 0; n < N; n++) {
            int blk = n / 32;
            float d = h2f(*(const unsigned short *)(rp + (size_t)blk * 34));
            double wv = (double)d * (double)((const signed char *)(rp + (size_t)blk * 34 + 2))[n % 32];
            int c = n / KC;
            if (fabs(wv) > hmax[c]) hmax[c] = fabs(wv);
            int v = (int)lrint(wv * 127.0 / (double)w.maxv[c]);
            if (v > 127) v = 127; else if (v < -127) v = -127;
            aq += (double)v * (double)w.maxv[c] / 127.0 * (double)x[n];
            a8 += wv * (double)x[n];
        }
        refq[m] = (float)aq; ref8[m] = (float)a8;
    }
    printf("[ST] max 校验 (上卡 / 主机前 %d 行):", Mt);
    for (int c = 0; c < CH; c++) printf(" %.6g/%.6g", (double)w.maxv[c], hmax[c]);
    printf("\n");
    printf("[ST] (1) 内核 vs 主机double(int8 同 max) relrms = %.4f%%   <-- 必须 <1%%\n", relrms_pct(y, refq, Mt));
    printf("[ST] (2) int8 量化 vs Q8_0 原始           relrms = %.4f%%\n", relrms_pct(refq, ref8, Mt));
    printf("[ST] 样例 y[0]=%.6f ref_i8=%.6f ref_q8=%.6f | y[%d]=%.6f ref_i8=%.6f\n",
           y[0], refq[0], ref8[0], Mt / 2, y[Mt / 2], refq[Mt / 2]);
    return 0;
}

// ============================ 主程序 ============================
static const char *g_prompt_tmpl = nullptr;   // K200_TMPL 覆盖 (含 {q})

int main(int argc, char **argv) {
    int ngen = 128;
    const char *path = nullptr, *gen = nullptr, *stname = nullptr;
    int strows = 0;   // 0 = selftest 对拍全部行 (两个芯都要被覆盖)
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
    if (getenv("K200_CH"))      g_ch      = atoi(getenv("K200_CH"));
    if (getenv("K200_CH_DOWN")) g_ch_down = atoi(getenv("K200_CH_DOWN"));
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (getenv("K200_LDUMP")) { g_ldump = fopen(getenv("K200_LDUMP"), "wb");
        printf("[orn] logits dump -> %s (%s)\n", getenv("K200_LDUMP"), g_ldump ? "OK" : "打开失败"); }

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
        printf("[orn] CH=%d (ffn_down=%d)  K200_CH / K200_CH_DOWN 可覆盖\n", g_ch, g_ch_down);
        const char *tn = stname ? stname : "blk.3.attn_k.weight";     // M=1024 N=4096 (双芯行分裂)
        int rc = selftest(g_gguf, tn, strows);
        if (rc) return rc;
        if (!stname) {
            rc = selftest(g_gguf, "blk.0.ffn_down.weight", strows);   // M=4096 N=12288 (最敏感张量)
            if (rc) return rc;
            rc = selftest(g_gguf, "blk.3.attn_q.weight", strows);     // M=8192 N=4096
            if (rc) return rc;
        }
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
        if (xpu_malloc(&g_dx[dv], 16384 * 4) || xpu_malloc(&g_dy[dv], (size_t)300000 * 4) ||
            xpu_malloc(&g_dxq[dv], 32768) || xpu_malloc(&g_dxs[dv], 2048 * 4))
            printf("[orn] WARN: chip%d 临时缓冲区分配失败\n", dv);
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
                {   Grp p;
                    snprintf(nm, sizeof(nm), "blk.%d.attn_qkv.weight", l);   p.add(g_gguf, nm);
                    snprintf(nm, sizeof(nm), "blk.%d.attn_gate.weight", l);  p.add(g_gguf, nm);
                    if (p.ws[0].M != PA_QKVS || p.ws[1].M != PA_GATES || p.M != (PA_GATE + PA_GATES)) {
                        printf("FATAL: recr grpA 段布局不符 (%d,%d,%d) 期望 (%d,%d,%d)\n",
                               p.ws[0].M, p.ws[1].M, p.M, PA_QKVS, PA_GATES, PA_GATE + PA_GATES); exit(1); }
                    L.grp_a = p; }
                {   Grp p;
                    snprintf(nm, sizeof(nm), "blk.%d.ssm_out.weight", l);    p.add(g_gguf, nm);
                    if (p.M != EMB) { printf("FATAL: ssm_out M=%d != %d\n", p.M, EMB); exit(1); }
                    L.grp_mid = p; }
                int aM = 0, aN = 0;
                snprintf(nm, sizeof(nm), "blk.%d.ssm_alpha.weight", l);  load_q8_host(g_gguf, nm, L.alpha_w, aM, aN);
                snprintf(nm, sizeof(nm), "blk.%d.ssm_beta.weight", l);   load_q8_host(g_gguf, nm, L.beta_w, aM, aN);
                snprintf(nm, sizeof(nm), "blk.%d.ssm_conv1d.weight", l); L.conv1d = load_f32_host(g_gguf, nm);
                snprintf(nm, sizeof(nm), "blk.%d.ssm_a", l);             L.ssm_a  = load_f32_host(g_gguf, nm);
                snprintf(nm, sizeof(nm), "blk.%d.ssm_dt.bias", l);       L.ssm_dt = load_f32_host(g_gguf, nm);
                snprintf(nm, sizeof(nm), "blk.%d.ssm_norm.weight", l);   L.ssm_norm = load_f32_host(g_gguf, nm);
            } else {
                {   Grp p;
                    snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", l);     p.add(g_gguf, nm);
                    snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", l);     p.add(g_gguf, nm);
                    snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", l);     p.add(g_gguf, nm);
                    if (p.ws[0].M != AA_QS || p.ws[1].M != AA_KS || p.ws[2].M != AA_VS || p.M != AA_TOT) {
                        printf("FATAL: attn grpA 段布局不符 (%d,%d,%d,%d)\n", p.ws[0].M, p.ws[1].M, p.ws[2].M, p.M); exit(1); }
                    L.grp_a = p; }
                {   Grp p;
                    snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", l); p.add(g_gguf, nm);
                    if (p.M != EMB) { printf("FATAL: attn_output M=%d != %d\n", p.M, EMB); exit(1); }
                    L.grp_mid = p; }
                snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", l); L.q_norm = load_f32_host(g_gguf, nm);
                snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", l); L.k_norm = load_f32_host(g_gguf, nm);
                L.Kc.assign((size_t)MAXT * NKV * HD, 0.f);
                L.Vc.assign((size_t)MAXT * NKV * HD, 0.f);
            }
            {   Grp p;
                snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", l); p.add(g_gguf, nm);
                snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight", l);   p.add(g_gguf, nm);
                if (p.ws[0].M != FF_GS || p.ws[1].M != FF_US || p.M != FF_TOT) {
                    printf("FATAL: ffn grp 段布局不符 (%d,%d,%d)\n", p.ws[0].M, p.ws[1].M, p.M); exit(1); }
                L.grp_ffn = p; }
            snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", l); L.ffn_down = up_i8(g_gguf, nm, ch_for(nm));
            if (l % 4 == 3 || l == NLAYER - 1)
                printf("[orn] 已载入 %d/%d 层  chip0=%.0fMB chip1=%.0fMB\n", l + 1, NLAYER,
                       g_hbm[0] / 1048576.0, g_hbm[1] / 1048576.0);
        }
        g_out_norm = load_f32_host(g_gguf, "output_norm.weight");
        g_outw = up_i8(g_gguf, "output.weight", g_ch);
        auto t1 = std::chrono::steady_clock::now();
        printf("[orn] 权重常驻 HBM: chip0=%.1f MB  chip1=%.1f MB  合计=%.2f GiB  用时 %.1fs (量化打包 %.1fs)\n",
               g_hbm[0] / 1048576.0, g_hbm[1] / 1048576.0,
               (g_hbm[0] + g_hbm[1]) / 1073741824.0,
               std::chrono::duration<double>(t1 - t0).count(), g_pack / 1000.0);
        printf("[orn] 权重量化: K 分块 CH 分布");
        for (std::map<int, int>::iterator it = g_ch_hist.begin(); it != g_ch_hist.end(); ++it)
            printf("  CH=%d:%d 张量", it->first, it->second);
        printf("  | 每 token 官方调用 = %llu 次\n", (unsigned long long)g_calls_per_tok);
    }
    rope_init(MAXT);
    g_st.conv.assign((size_t)NLAYER * 3 * CVD, 0.f);
    g_st.S.assign((size_t)NLAYER * NVH * VHD * KHD, 0.f);
    if (g_co > 16) { printf("[orn] ★警告: co=%d > 16, 每 cluster 只有 16 个核会执行, 行会漏写!\n", g_co); }
    printf("[orn] grid: cl=%d co=%d xi8=%d (硬件每 cluster 核数=16)\n", g_cl, g_co, g_xi8);
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
        std::vector<float> conv, S, logits;
        std::vector<std::vector<float> > Kc, Vc;
    };
    static Snapshot g_snap;
    struct Sess { std::vector<int> consumed; int pos = 0; };
    static std::map<std::string, Sess> g_sess;
    static std::string g_live_sid;
    static int g_live_pos = -1;
    auto take_snap = [&](const std::vector<int> &sids, int spos, const std::vector<float> &lg) {
        g_snap.ids = sids; g_snap.pos = spos;
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

    // 一条请求的生成
    auto run_request = [&](const std::string &prompt, int maxtok, bool stream, const std::string &sid) -> std::string {
        std::vector<int> ids = g_tk.encode(prompt);
        std::vector<float> logits;
        int pos = 0, reused = 0;
        bool hit = false, did_prefill = false;
        if (!ids.empty() && g_snap.valid && g_snap.ids == ids) {
            restore_snap(logits); pos = g_snap.pos; reused = pos; hit = true;
        }
        auto t0 = std::chrono::steady_clock::now();
        if (!hit) {
            int cont = -1;
            if (!sid.empty()) {
                std::map<std::string, Sess>::iterator it = g_sess.find(sid);
                if (it != g_sess.end() && g_live_sid == sid && g_live_pos == it->second.pos &&
                    it->second.pos > 0 && (int)ids.size() > it->second.pos &&
                    it->second.pos <= (int)it->second.consumed.size()) {
                    bool ok = true;
                    for (int i = 0; i < it->second.pos; i++) if (it->second.consumed[i] != ids[i]) { ok = false; break; }
                    if (ok) cont = it->second.pos;
                }
            }
            if (cont < 0) { g_st.reset(); pos = 0; reused = 0; }
            else { pos = cont; reused = cont; }
            did_prefill = true;
            for (size_t i = reused; i < ids.size(); i++) {
                if (pos >= MAXT) { ids.resize(i); break; }
                forward(ids[i], pos, logits);
                if (g_ldump) { int sz = (int)logits.size(), p_ = pos, tk = ids[i];
                    fwrite(&p_, 4, 1, g_ldump); fwrite(&tk, 4, 1, g_ldump);
                    fwrite(&sz, 4, 1, g_ldump); fwrite(logits.data(), 4, (size_t)sz, g_ldump); fflush(g_ldump); }
                pos++;
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        double pt = std::chrono::duration<double>(t1 - t0).count();
        if (did_prefill && reused == 0 && !ids.empty()) take_snap(ids, pos, logits);
        std::string out;
        std::vector<int> genids;
        int ntok = 0;
        auto t2 = std::chrono::steady_clock::now();
        double L0 = g_launch, W0 = g_wait, D0 = g_d2h; uint64_t C0 = g_launch_cnt; uint64_t G0 = g_gemv_n;
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
            if (ntok >= maxtok) break;
            genids.push_back(best);
            forward(best, pos, logits); pos++;
        }
        auto t3 = std::chrono::steady_clock::now();
        double gt = std::chrono::duration<double>(t3 - t2).count();
        if (!sid.empty()) {
            Sess &S = g_sess[sid];
            S.consumed = ids;
            S.consumed.insert(S.consumed.end(), genids.begin(), genids.end());
            S.pos = pos;
        }
        g_live_sid = sid; g_live_pos = pos;
        printf("[orn] prompt=%zu tok 用时 %.2fs (%s 复用%zu tok) | 生成 %d tok 用时 %.2fs = %.3f tok/s | gemv %llu 次\n",
               ids.size(), pt, (hit ? "快照" : (reused > 0 ? "会话" : "全量")), (size_t)reused,
               ntok, gt, ntok / (gt + 1e-9), (unsigned long long)g_gemv_n);
        if (ntok > 0)
            printf("[orn] 计时/生成: 官方 gemm_int8 调用 %llu 次 (%.1f 次/tok) | launch(主机) %.1f ms | wait %.1f ms | D2H %.1f ms || 每 tok: launch %.2f ms wait %.2f ms\n",
                   (unsigned long long)(g_launch_cnt - C0), (double)(g_launch_cnt - C0) / ntok,
                   g_launch - L0, g_wait - W0, g_d2h - D0,
                   (g_launch - L0) / ntok, (g_wait - W0) / ntok);
        (void)G0;
        return out;
    };

    if (gen) {   // 离线模式: --gen "你好"
        std::string p;
        const char *ov = getenv("K200_TMPL");
        if (ov) { std::string t = ov; size_t k = t.find("{q}"); p = (k == std::string::npos) ? t : t.replace(k, 3, gen); }
        else p = std::string("<|im_start|>user\n") + gen + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        printf("__BEGIN__\n");
        std::string o = run_request(p, ngen, true, "offline");
        printf("__END__\n");
        fprintf(stderr, "[orn] 离线输出: %s\n", o.c_str());
        return 0;
    }
    // 服务模式: stdin "@<n>@b64:<prompt>" 或 "@<n>@sid=<id>@b64:<prompt>"
    {
        char *line = NULL; size_t cap = 0;
        while (getline(&line, &cap, stdin) > 0) {
            std::string s(line);
            while (!s.empty() && (s[s.size()-1] == '\n' || s[s.size()-1] == '\r')) s.erase(s.size()-1);
            if (s.size() < 3 || s[0] != '@') continue;
            size_t p2 = s.find('@', 1);
            if (p2 == std::string::npos) continue;
            int mt = atoi(s.substr(1, p2 - 1).c_str());
            std::string rest = s.substr(p2 + 1), sid;
            if (rest.rfind("sid=", 0) == 0) { size_t p3 = rest.find('@', 0); sid = rest.substr(4, p3 - 4); rest = (p3 == std::string::npos) ? std::string() : rest.substr(p3 + 1); }
            std::string b64 = rest;
            if (b64.rfind("b64:", 0) == 0) b64 = b64.substr(4);
        // base64 解码
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
        printf("__BEGIN__\n");
            run_request(pr, mt, true, sid);
            printf("__END__\n");
        }
    }
    return 0;
}
