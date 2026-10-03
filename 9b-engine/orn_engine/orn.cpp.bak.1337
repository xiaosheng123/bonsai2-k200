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
#include <xpu/runtime.h>

void run_gemv_q8_0(int cl, int co, const void *W, const void *x, void *y, int M, int N);

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
struct WDev { void *d = nullptr; int dev = -1; int M = 0, N = 0; size_t bytes = 0; size_t rowbytes = 0; bool v2 = false; };
static uint64_t g_hbm[2] = {0, 0};

struct Layer {
    bool recr = false;
    std::vector<float> attn_norm, post_norm;
    // 线性注意力 (gated delta net)
    WDev qkv, gate, ssm_out;
    std::vector<float> alpha_w, beta_w;      // [32 x 4096] host (小, 不值得上卡)
    std::vector<float> conv1d;               // [c*4 + tap] = 4 per channel
    std::vector<float> ssm_a, ssm_dt, ssm_norm;
    // 全注意力
    WDev q, k, v, o;
    std::vector<float> q_norm, k_norm;
    std::vector<float> Kc, Vc;   // KV cache (仅注意层, MAXT*NKV*HD)
    // MLP
    WDev ffn_gate, ffn_up, ffn_down;
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

static WDev up_q8(GF &g, const char *name, int dev) {
    const GT *t = g.get(name);
    if (!t) { printf("FATAL: 缺张量 %s\n", name); exit(1); }
    if (t->type != 8) { printf("FATAL: %s type=%u 不是 Q8_0\n", name, t->type); exit(1); }
    WDev w; w.N = (int)t->dims[0]; w.M = (int)t->dims[1]; w.bytes = g.nbytes(*t); w.dev = dev;
    if (w.N % 1024 != 0) { printf("FATAL: %s N=%d 不是 1024 的倍数 (内核限制)\n", name, w.N); exit(1); }
    std::vector<unsigned char> raw(w.bytes);
    if (!g.read(*t, raw.data())) { printf("FATAL: 读 %s 失败\n", name); exit(1); }
    // ★ V2 行重排: 每 4096 权重 -> [4096 int8][128 f16 scale] (对齐, 便于 int32 打包 MAC); 总字节不变
    if (!getenv("K200_KV2") || atoi(getenv("K200_KV2")) != 0) {
        int nb = w.N / 32;
        size_t rowbytes = (size_t)w.N + (size_t)nb * 2;
        std::vector<unsigned char> packed((size_t)w.M * rowbytes);
        for (int m = 0; m < w.M; m++) {
            const unsigned char *src = raw.data() + (size_t)m * nb * 34;
            unsigned char *dst = packed.data() + (size_t)m * rowbytes;
            for (int c = 0; c < w.N / 4096; c++) {
                unsigned char *qc = dst + (size_t)c * 4352;
                unsigned char *scc = qc + 4096;
                for (int b = 0; b < 128; b++) {
                    const unsigned char *sp = src + (size_t)(c * 128 + b) * 34;
                    memcpy(scc + b * 2, sp, 2);
                    memcpy(qc + b * 32, sp + 2, 32);
                }
            }
        }
        raw.swap(packed);
        w.bytes = raw.size(); w.rowbytes = rowbytes; w.v2 = true;
    }
    xpu_set_device(dev);
    if (xpu_malloc(&w.d, w.bytes)) { printf("FATAL: HBM 分配 %s (%.1f MB) 失败 chip%d [已用 %.0f MB]\n",
                                          name, w.bytes / 1048576.0, dev, g_hbm[dev] / 1048576.0); exit(1); }
    if (xpu_memcpy(w.d, raw.data(), w.bytes, XPU_HOST_TO_DEVICE)) { printf("FATAL: H2D %s 失败\n", name); exit(1); }
    xpu_wait();
    if (getenv("K200_ALLOCLOG")) printf("[ALLOC] %-32s dev=%d M=%6d N=%6d bytes=%9zu ptr=%p next=%p\n", name, dev, w.M, w.N, w.bytes, w.d, (char*)w.d + w.bytes);
    g_hbm[dev] += w.bytes;
    return w;
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
static void gemv_sel(int dv, const WDev &w, const float *x, float *y, int xi8, int cl, int co) {
    xpu_set_device(dv);
    g_calls++;
    if (g_sent >= 1) sent_fill(dv, w.M);
    if (g_sent == 1) printf("[GEMV] call#%d %s dev=%d M=%d N=%d cl=%d co=%d xi8=%d dy=%p\n", g_calls, g_gname, dv, w.M, w.N, cl, co, xi8, g_dy[dv]);
    if (xi8) {
        static std::vector<signed char> xq; static std::vector<float> xs;
        int nb = w.N / 32;
        xq.resize(w.N); xs.resize(nb);
        auto tq = std::chrono::steady_clock::now();
        for (int b = 0; b < nb; b++) {
            float amax = 0;
            for (int i = 0; i < 32; i++) { float a = fabsf(x[b*32+i]); if (a > amax) amax = a; }
            float s = amax / 127.f; if (s <= 1e-12f) s = 1e-8f;
            xs[b] = s;
            for (int i = 0; i < 32; i++) { int v = (int)lrintf(x[b*32+i] / s);
                if (v > 127) v = 127; else if (v < -128) v = -128; xq[b*32+i] = (signed char)v; }
        }
        auto tq2 = std::chrono::steady_clock::now();
        g_xquant += std::chrono::duration<double, std::milli>(tq2 - tq).count();
        xpu_memcpy(g_dxq[dv], xq.data(), (size_t)w.N, XPU_HOST_TO_DEVICE);
        xpu_memcpy(g_dxs[dv], xs.data(), (size_t)nb * 4, XPU_HOST_TO_DEVICE);
        if (w.v2) run_gemv_q8v2(cl, co, w.d, g_dxq[dv], g_dxs[dv], g_dy[dv], w.M, w.N);
        else      run_gemv_q8_0i(cl, co, w.d, g_dxq[dv], g_dxs[dv], g_dy[dv], w.M, w.N);
    } else {
        xpu_memcpy(g_dx[dv], x, (size_t)w.N * 4, XPU_HOST_TO_DEVICE);
        run_gemv_q8_0(cl, co, w.d, g_dx[dv], g_dy[dv], w.M, w.N);
    }
    if (xpu_wait()) { printf("FATAL: kernel wait (dev%d M=%d N=%d)\n", dv, w.M, w.N); exit(1); }
    auto td = std::chrono::steady_clock::now();
    if (g_dump && dv == 1) {
        memset(y, 0, (size_t)w.M * 4);                       // 清 0, 看设备到底返回什么
        xpu_memcpy(y, g_dy[dv], (size_t)w.M * 4, XPU_DEVICE_TO_HOST);
        xpu_wait();
        printf("[DBG] D2H#1 y[190]=%.6g y[195]=%.6g y[210]=%.6g (半路重读)\n", y[190], y[195], y[210]);
        float probe[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        xpu_memcpy(&probe[0], (char *)g_dy[dv] + 190 * 4, 8 * 4, XPU_DEVICE_TO_HOST);
        xpu_wait();
        printf("[DBG] 单独 8 float D2H @190: %.6g %.6g %.6g %.6g\n", probe[0], probe[1], probe[2], probe[3]);
    } else xpu_memcpy(y, g_dy[dv], (size_t)w.M * 4, XPU_DEVICE_TO_HOST);
    g_gemv_n++;
    if (g_sent) sent_check(dv, y, w.M, w.M, w.N, g_gname);
    g_d2h += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count();
    g_gemv_n++;
}
static void gemv(const WDev &w, const float *x, float *y) { gemv_sel(w.dev, w, x, y, g_xi8, g_cl, g_co); }
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
static void forward(int tok, int pos, std::vector<float> &logits) {
    std::vector<float> &x = g_x;
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
    for (int l = 0; l < NLAYER; l++) {
        Layer &L = g_lay[l];
        rmsnorm(xb.data(), x.data(), L.attn_norm.data(), EMB, EPS);
        if (g_dump && pos == 12 && l < 4) { char b[64]; snprintf(b, sizeof(b), "attn_norm-%d", l); dumpv(b, xb.data(), EMB); }
        if (L.recr) {
            static std::vector<float> qkv, z, al, be, co;
            qkv.resize(CVD); z.resize(EMB); al.resize(NVH); be.resize(NVH); co.resize(CVD);
            { snprintf(g_nb, sizeof(g_nb), "l%d.qkv", l); g_gname = g_nb; }
            gemv(L.qkv, xb.data(), qkv.data());
            { snprintf(g_nb, sizeof(g_nb), "l%d.gate", l); g_gname = g_nb; }
            gemv(L.gate, xb.data(), z.data());
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
            if (g_dump && l == 0 && pos == 0) {
                FILE *fp = fopen("/tmp/xb0.bin", "wb"); fwrite(xb.data(), 4, EMB, fp); fclose(fp);
                printf("[DBG] l0 pos0 xb0..5: %.6g %.6g %.6g %.6g %.6g %.6g  xb_absmax=%.4g\n",
                       xb[0], xb[1], xb[2], xb[3], xb[4], xb[5], absmax(xb.data(), EMB));
                printf("[DBG] l0 pos0 gemv_qkv_y0..5: %.6g %.6g %.6g %.6g %.6g %.6g  yabsmax=%.4g (dev=%d M=%d N=%d)\n",
                       qkv[0], qkv[1], qkv[2], qkv[3], qkv[4], qkv[5], absmax(qkv.data(), CVD), L.qkv.dev, L.qkv.M, L.qkv.N);
                printf("[DBG] l0 pos0 gemv_gate_z0..5: %.6g %.6g %.6g %.6g %.6g %.6g\n", z[0], z[1], z[2], z[3], z[4], z[5]);
                { int nf = 0, first = -1; for (int i = 0; i < CVD; i++) if (!std::isfinite(qkv[i])) { if (first < 0) first = i; nf++; }
                  for (int i = 190; i < 200; i++) { unsigned b; memcpy(&b, &qkv[i], 4); printf("[DBG] qkv[%d]=%.6g bits=0x%08x\n", i, qkv[i], b); }
                  printf("[DBG] qkv 非有限元素: %d 个, 第一个 index=%d\n", nf, first);
                  int nf2 = 0, f2 = -1; for (int i = 0; i < EMB; i++) if (!std::isfinite(z[i])) { if (f2 < 0) f2 = i; nf2++; }
                  for (int i = 190; i < 200; i++) { unsigned b; memcpy(&b, &z[i], 4); printf("[DBG] z[%d]=%.6g bits=0x%08x\n", i, z[i], b); }
                  printf("[DBG] z 非有限元素: %d 个, 第一个 index=%d\n", nf2, f2); }
            }
            if (g_dump && l == 0) {
                printf("[DBG] l0 qkv_absmax=%.4g co_absmax=%.4g al=%g be=%g ssm_a[0]=%g dt[0]=%g xb_absmax=%.4g\n",
                       absmax(qkv.data(), CVD), absmax(co.data(), CVD), al[0], be[0], L.ssm_a[0], L.ssm_dt[0],
                       absmax(xb.data(), EMB));
            }
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
            if (g_dump && l == 0) printf("[DBG] l0 dout_absmax=%.4g z_absmax=%.4g\n", absmax(dout.data(), NVH * KHD), absmax(z.data(), EMB));
            // gated rms norm: per 128 head, rmsnorm(ssm_norm) * silu(z)
            for (int h = 0; h < NVH; h++) {
                float *p = &dout[h * KHD];
                double ss = 0; for (int i = 0; i < KHD; i++) ss += (double)p[i] * p[i];
                float r = 1.f / sqrtf((float)(ss / KHD) + EPS);
                for (int i = 0; i < KHD; i++) p[i] = p[i] * r * L.ssm_norm[i] * siluf(z[h * KHD + i]);
            }
            { snprintf(g_nb, sizeof(g_nb), "l%d.ssm_out", l); g_gname = g_nb; }
            gemv(L.ssm_out, dout.data(), o.data());
            if (g_dump && l == 0) printf("[DBG] l0 ssm_out_absmax=%.4g\n", absmax(o.data(), EMB));
        } else {
            static std::vector<float> qg, kk, vv, ao;
            qg.resize(NH * HD * 2); kk.resize(NKV * HD); vv.resize(NKV * HD); ao.resize(NH * HD);
            gemv(L.q, xb.data(), qg.data());
            gemv(L.k, xb.data(), kk.data());
            gemv(L.v, xb.data(), vv.data());
            static std::vector<float> q5(NH * HD), g5(NH * HD);
            for (int h = 0; h < NH; h++) {
                memcpy(&q5[h * HD], &qg[h * HD * 2], HD * 4);
                memcpy(&g5[h * HD], &qg[h * HD * 2 + HD], HD * 4);
            }
            for (int h = 0; h < NH; h++) rmsnorm(&q5[h * HD], &q5[h * HD], L.q_norm.data(), HD, EPS);
            for (int h = 0; h < NKV; h++) rmsnorm(&kk[h * HD], &kk[h * HD], L.k_norm.data(), HD, EPS);
            for (int h = 0; h < NH; h++) rope_apply(&q5[h * HD], pos, HD);
            for (int h = 0; h < NKV; h++) rope_apply(&kk[h * HD], pos, HD);
            // KV cache (每层一份, 只有注意层分配)
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
                for (int i = 0; i < HD; i++) oh[i] *= sigmoidf_(g5[h * HD + i]);   // 注意输出按维门控(gate 与 q 同头打包)
            }
            gemv(L.o, ao.data(), o.data());
        }
        for (int i = 0; i < EMB; i++) x[i] += o[i];
        // MLP
        rmsnorm(xb.data(), x.data(), L.post_norm.data(), EMB, EPS);
        static std::vector<float> gg, uu;
        gg.resize(NFF); uu.resize(NFF);
        { snprintf(g_nb, sizeof(g_nb), "l%d.ffn_gate", l); g_gname = g_nb; }
        gemv(L.ffn_gate, xb.data(), gg.data());
        { snprintf(g_nb, sizeof(g_nb), "l%d.ffn_up", l); g_gname = g_nb; }
        gemv(L.ffn_up,   xb.data(), uu.data());
        if (g_dump && l == 0) printf("[DBG] l0 ffn_in_absmax=%.4g gg_absmax=%.4g uu_absmax=%.4g\n", absmax(xb.data(), EMB), absmax(gg.data(), NFF), absmax(uu.data(), NFF));
        for (int i = 0; i < NFF; i++) gg[i] = siluf(gg[i]) * uu[i];
        { snprintf(g_nb, sizeof(g_nb), "l%d.ffn_down", l); g_gname = g_nb; }
        gemv(L.ffn_down, gg.data(), o.data());
        for (int i = 0; i < EMB; i++) x[i] += o[i];
        if (g_dump && pos == 12 && l < 4) { char b[64]; snprintf(b, sizeof(b), "l_out-%d", l); dumpv(b, x.data(), EMB); }
    }
    rmsnorm(xb.data(), x.data(), g_out_norm.data(), EMB, EPS);
    if (g_dump && pos == 12) dumpv("result_norm", xb.data(), EMB);
    logits.resize(g_outw.M);
    { snprintf(g_nb, sizeof(g_nb), "lm_head"); g_gname = g_nb; }
    gemv(g_outw, xb.data(), logits.data());
}

// ============================ selftest: 卡上 Q8_0 GEMV vs 主机参考 ============================
static int selftest(GF &g, const char *name, int rows) {
    const GT *t = g.get(name);
    if (!t) { printf("no tensor %s\n", name); return 1; }
    int N = (int)t->dims[0], M = (int)t->dims[1];
    int Mt = rows < M ? rows : M;
    int stdev = getenv("K200_STDEV") ? atoi(getenv("K200_STDEV")) : 0;
    WDev w = up_q8(g, name, stdev);
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
        gemv_sel(stdev, w, x.data(), y.data(), xiv[k], clv[k], cov[k]);
        g_d2h = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 2; r++) gemv_sel(stdev, w, x.data(), y.data(), xiv[k], clv[k], cov[k]);
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
    if (getenv("K200_SENT")) g_sent = atoi(getenv("K200_SENT"));
    setvbuf(stdout, NULL, _IOLBF, 0);

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
        if (xpu_malloc(&g_dx[dv], 16384 * 4) || xpu_malloc(&g_dy[dv], (size_t)300000 * 4) ||
            xpu_malloc(&g_dxq[dv], 32768) || xpu_malloc(&g_dxs[dv], 2048 * 4))
            printf("[orn] WARN: chip%d 临时缓冲区分配失败\n", dv);
    }
    for (int dv = 0; dv < 2; dv++)
        printf("[orn] scratch chip%d: dx=%p dy=%p dxq=%p dxs=%p\n", dv, g_dx[dv], g_dy[dv], g_dxq[dv], g_dxs[dv]);
    // ---- 层字节预算 -> 两芯分层 ----
    g_lay.resize(NLAYER);
    std::vector<size_t> lb(NLAYER, 0);
    for (int l = 0; l < NLAYER; l++) {
        bool rc_ = is_recr(l);
        char nm[128];
        const char *rec[] = {"attn_qkv.weight","attn_gate.weight","ssm_alpha.weight","ssm_beta.weight","ssm_out.weight"};
        const char *att[] = {"attn_q.weight","attn_k.weight","attn_v.weight","attn_output.weight"};
        for (int i = 0; i < (rc_ ? 5 : 4); i++) {
            snprintf(nm, sizeof(nm), rc_ ? "blk.%d.%s" : "blk.%d.%s", l, rc_ ? rec[i] : att[i]);
            const GT *t = g_gguf.get(nm);
            if (t && t->type == 8) lb[l] += g_gguf.nbytes(*t);
        }
        for (int i = 0; i < 3; i++) {
            snprintf(nm, sizeof(nm), "blk.%d.ffn_%s.weight", l, i == 0 ? "gate" : (i == 1 ? "up" : "down"));
            const GT *t = g_gguf.get(nm);
            if (t && t->type == 8) lb[l] += g_gguf.nbytes(*t);
        }
    }
    {
        const GT *t = g_gguf.get("output.weight");
        size_t ob = t ? g_gguf.nbytes(*t) : 0;
        uint64_t bud[2] = {ob, 0};            // 预算用独立计数器(真实占用另算)
        g_hbm[0] = ob;                         // lm_head 放 chip0
        printf("[orn] 双芯分层: chip0 起始 lm_head %.0f MB\n", ob / 1048576.0);
        std::vector<int> ldev(NLAYER, 0);
        for (int l = 0; l < NLAYER; l++) {
            int dv2 = (bud[0] <= bud[1]) ? 0 : 1;
            ldev[l] = dv2; bud[dv2] += lb[l];
        }
        printf("[orn] 预算: chip0=%.0fMB chip1=%.0fMB\n", bud[0] / 1048576.0, bud[1] / 1048576.0);
        printf("[orn] layer->chip: ");
        for (int l = 0; l < NLAYER; l++) {
            g_lay[l].recr = is_recr(l);
            printf("%d", ldev[l]);
            if (l != NLAYER - 1) printf(",");
        }
        printf("\n");
        // ---- 载入 ----
        auto t0 = std::chrono::steady_clock::now();
        for (int l = 0; l < NLAYER; l++) {
            Layer &L = g_lay[l];
            int dv = ldev[l];
            char nm[128];
            snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", l);           L.attn_norm = load_f32_host(g_gguf, nm);
            snprintf(nm, sizeof(nm), "blk.%d.post_attention_norm.weight", l); L.post_norm = load_f32_host(g_gguf, nm);
            if (L.recr) {
                snprintf(nm, sizeof(nm), "blk.%d.attn_qkv.weight", l);   L.qkv  = up_q8(g_gguf, nm, dv);
                snprintf(nm, sizeof(nm), "blk.%d.attn_gate.weight", l);  L.gate = up_q8(g_gguf, nm, dv);
                snprintf(nm, sizeof(nm), "blk.%d.ssm_out.weight", l);    L.ssm_out = up_q8(g_gguf, nm, dv);
                int aM = 0, aN = 0;
                snprintf(nm, sizeof(nm), "blk.%d.ssm_alpha.weight", l);  load_q8_host(g_gguf, nm, L.alpha_w, aM, aN);
                snprintf(nm, sizeof(nm), "blk.%d.ssm_beta.weight", l);   load_q8_host(g_gguf, nm, L.beta_w, aM, aN);
                snprintf(nm, sizeof(nm), "blk.%d.ssm_conv1d.weight", l); L.conv1d = load_f32_host(g_gguf, nm);
                snprintf(nm, sizeof(nm), "blk.%d.ssm_a", l);             L.ssm_a  = load_f32_host(g_gguf, nm);
                snprintf(nm, sizeof(nm), "blk.%d.ssm_dt.bias", l);       L.ssm_dt = load_f32_host(g_gguf, nm);
                snprintf(nm, sizeof(nm), "blk.%d.ssm_norm.weight", l);   L.ssm_norm = load_f32_host(g_gguf, nm);
            } else {
                snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", l);     L.q = up_q8(g_gguf, nm, dv);
                snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", l);     L.k = up_q8(g_gguf, nm, dv);
                snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", l);     L.v = up_q8(g_gguf, nm, dv);
                snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", l);L.o = up_q8(g_gguf, nm, dv);
                snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", l);L.q_norm = load_f32_host(g_gguf, nm);
                snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", l);L.k_norm = load_f32_host(g_gguf, nm);
                L.Kc.assign((size_t)MAXT * NKV * HD, 0.f);
                L.Vc.assign((size_t)MAXT * NKV * HD, 0.f);
            }
            snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", l); L.ffn_gate = up_q8(g_gguf, nm, dv);
            snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight", l);   L.ffn_up   = up_q8(g_gguf, nm, dv);
            snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", l); L.ffn_down = up_q8(g_gguf, nm, dv);
            if (l % 8 == 7 || l == NLAYER - 1)
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

    // 一条请求的生成
    auto run_request = [&](const std::string &prompt, int maxtok, bool stream) -> std::string {
        std::vector<int> ids = g_tk.encode(prompt);
        g_st.reset();
        std::vector<float> logits;
        auto t0 = std::chrono::steady_clock::now();
        int pos = 0;
        for (size_t i = 0; i < ids.size(); i++) { if (pos >= MAXT) { ids.resize(i); break; } forward(ids[i], pos, logits); pos++; }
        auto t1 = std::chrono::steady_clock::now();
        double pt = std::chrono::duration<double>(t1 - t0).count();
        std::string out;
        int ntok = 0;
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
            if (ntok >= maxtok) break;
            forward(best, pos, logits); pos++;
        }
        auto t3 = std::chrono::steady_clock::now();
        double gt = std::chrono::duration<double>(t3 - t2).count();
        printf("[orn] prompt=%zu tok 用时 %.2fs (%.2f tok/s) | 生成 %d tok 用时 %.2fs = %.3f tok/s | gemv %llu 次\n",
               ids.size(), pt, ids.size() / (pt + 1e-9), ntok, gt, ntok / (gt + 1e-9), (unsigned long long)g_gemv_n);
        return out;
    };

    if (gen) {   // 离线模式: --gen "你好"
        std::string p;
        const char *ov = getenv("K200_TMPL");
        if (ov) { std::string t = ov; size_t k = t.find("{q}"); p = (k == std::string::npos) ? t : t.replace(k, 3, gen); }
        else p = std::string("<|im_start|>user\n") + gen + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        printf("__BEGIN__\n");
        std::string o = run_request(p, ngen, true);
        printf("__END__\n");
        fprintf(stderr, "[orn] 离线输出: %s\n", o.c_str());
        return 0;
    }
    // 服务模式: stdin "@<n>@b64:<prompt>"
    char *line = NULL; size_t cap = 0;
    while (getline(&line, &cap, stdin) > 0) {
        std::string s(line);
        while (!s.empty() && (s[s.size()-1] == '\n' || s[s.size()-1] == '\r')) s.erase(s.size()-1);
        if (s.size() < 3 || s[0] != '@') continue;
        size_t p2 = s.find('@', 1);
        if (p2 == std::string::npos) continue;
        int mt = atoi(s.substr(1, p2 - 1).c_str());
        std::string b64 = s.substr(p2 + 1);
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
        run_request(pr, mt, true);
        printf("__END__\n");
    }
    return 0;
}
