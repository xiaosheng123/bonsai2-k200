// llref.cpp — 用官方 libllama 生成 token 级真值: 贪心解码, 打印 token id + 文本
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <ctime>
#include <chrono>
#include <vector>
#include <cmath>
#include "llama.h"


// ---------------- 参考中间激活 dump ----------------
static const char *g_targets[] = {"model.input_embed","attn_norm-0","l_out-0","l_out-1","l_out-2",
    "attn_post_norm-0","ffn_out-0","l_out-3","l_out-31","result_norm","result_output"};
static bool g_first_decode = true;
static bool g_dumped[32] = {false};

static bool dump_cb(struct ggml_tensor *t, bool ask, void *ud) {
    if (ask) {
        for (size_t i = 0; i < sizeof(g_targets)/sizeof(g_targets[0]); i++)
            if (t->name && !strcmp(t->name, g_targets[i])) return true;
        return false;
    }
    if (!g_first_decode) return true;
    int idx = -1;
    for (size_t i = 0; i < sizeof(g_targets)/sizeof(g_targets[0]); i++)
        if (t->name && !strcmp(t->name, g_targets[i])) idx = (int)i;
    if (idx < 0 || g_dumped[idx]) return true;
    g_dumped[idx] = true;
    int64_t n0 = t->ne[0], n1 = t->ne[1] > 0 ? t->ne[1] : 1;
    size_t bytes = ggml_nbytes(t);
    std::vector<char> buf(bytes);
    ggml_backend_tensor_get(t, buf.data(), 0, bytes);
    const float *f = (const float *)buf.data();
    if (t->type != GGML_TYPE_F32) { printf("[DUMP] %s type!=F32\n", t->name); return true; }
    int64_t tc = n1 - 1;                       // 最后一个 token 列
    double ss = 0;
    for (int64_t i = 0; i < n0; i++) { double v = f[i + tc * n0]; ss += v * v; }
    printf("[DUMP] %-20s ne=[%lld,%lld] L2=%.6f v:", t->name, (long long)n0, (long long)n1, sqrt(ss));
    for (int i = 0; i < 6; i++) printf(" %.6f", f[i + tc * n0]);
    printf("\n");
    return true;
}

int main(int argc, char **argv) {
    const char *model_path = "/home/caden/orn/Ornith-1.5-9B-Q8_0.gguf";
    int ngen = 16;
    const char *prompt = NULL;
    const char *pfile = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) ngen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-f") && i + 1 < argc) pfile = argv[++i];
    }
    std::string ptxt;
    if (pfile) {
        FILE *fp = fopen(pfile, "rb");
        if (!fp) { printf("open %s fail\n", pfile); return 1; }
        char buf[4096]; size_t r = fread(buf, 1, sizeof(buf), fp); fclose(fp);
        ptxt.assign(buf, r);
    } else {
        ptxt = prompt ? prompt : "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { printf("load fail\n"); return 1; }
    const llama_vocab *vocab = llama_model_get_vocab(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 512; cp.n_threads = 4;
    cp.n_ubatch = 512;
    cp.cb_eval = dump_cb; cp.cb_eval_user_data = NULL;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { printf("ctx fail\n"); return 1; }

    std::vector<llama_token> toks(ptxt.size() + 8);
    int nt = llama_tokenize(vocab, ptxt.c_str(), (int)ptxt.size(), toks.data(), (int)toks.size(), false, true);
    if (nt < 0) { printf("tokenize fail %d\n", nt); return 1; }
    toks.resize(nt);
    printf("[ref] prompt %zu bytes -> %d tokens:", ptxt.size(), nt);
    for (int i = 0; i < nt; i++) printf(" %d", toks[i]);
    printf("\n");

    auto t0 = std::chrono::steady_clock::now();
    llama_batch b = llama_batch_get_one(toks.data(), nt);
    if (llama_decode(ctx, b)) { printf("decode fail\n"); return 1; }
    g_first_decode = false;
    auto t1 = std::chrono::steady_clock::now();
    printf("[ref] prefill %d tok 用时 %.2fs\n", nt, std::chrono::duration<double>(t1 - t0).count());

    std::string out;
    auto t2 = std::chrono::steady_clock::now();
    int n = 0;
    for (int i = 0; i < ngen; i++) {
        const float *logits = llama_get_logits_ith(ctx, -1);
        int nv = llama_vocab_n_tokens(vocab);
        int best = 0; float bv = logits[0];
        for (int k = 1; k < nv; k++) if (logits[k] > bv) { bv = logits[k]; best = k; }
        char buf[256]; int nb = llama_token_to_piece(vocab, best, buf, sizeof(buf), 0, true);
        printf("[ref] tok %2d id=%-7d piece='%s'\n", i, best, std::string(buf, nb > 0 ? nb : 0).c_str());
        out += std::string(buf, nb > 0 ? nb : 0);
        n++;
        if (llama_vocab_is_eog(vocab, best)) { printf("[ref] EOG\n"); break; }
        llama_batch b2 = llama_batch_get_one(&best, 1);
        if (llama_decode(ctx, b2)) { printf("decode fail2\n"); break; }
    }
    auto t3 = std::chrono::steady_clock::now();
    printf("[ref] 生成 %d tok 用时 %.2fs = %.3f tok/s\n", n,
           std::chrono::duration<double>(t3 - t2).count(), n / std::chrono::duration<double>(t3 - t2).count());
    printf("[ref] 全文: %s\n", out.c_str());
    return 0;
}
