// mb3.cpp — 驱动
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <xpu/runtime.h>

#define MAXO 200000
void m_info(int *o);
void m_rd(const void *W, long long span, int chunk, int nch, void *out, int cl, int co);
void m_rda(const void *W, long long span, int chunk, int nch, int depth, void *out, int cl, int co);
void m_c_simd(float *o, int reps, int cl, int co);
void m_c_i8(float *o, int reps, int cl, int co);
void m_c_i8p(float *o, int reps, int cl, int co);
void m_c_cvt(float *o, int reps, int cl, int co);
void m_c_fmac(float *o, int reps, int cl, int co);
void m_c_old(float *o, int reps, int cl, int co);
void m_g_rd_mac(const void *W, long long span, int chunk, int nch, int mode, void *out, void *chk, int cl, int co);

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

typedef void (*cufn)(float *, int, int, int);

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    const char *stage = argc > 2 ? argv[2] : "all";
    xpu_set_device(dev);
    printf("[mb3] dev=%d stage=%s\n", dev, stage);

    // ---------- E1 grid 探测 ----------
    if (strstr(stage, "info") || !strcmp(stage, "all")) {
        int *o = nullptr;
        if (xpu_malloc((void **)&o, MAXO * 4)) { printf("info malloc fail\n"); return 1; }
        std::vector<int> h(MAXO, -1);
        xpu_memcpy(h.data(), o, MAXO * 4, XPU_HOST_TO_DEVICE);
        m_info(o);
        int w = xpu_wait();
        xpu_memcpy(h.data(), o, MAXO * 4, XPU_DEVICE_TO_HOST);
        int nrun = 0, maxcid = -1, maxcore = -1, coremap[64];
        for (int i = 0; i < 64; i++) coremap[i] = 0;
        for (int i = 0; i < 100000; i++) if (h[i] >= 0) {
            int cid = h[i] / 1000, core = h[i] % 1000;
            nrun++; if (cid > maxcid) maxcid = cid; if (core > maxcore) maxcore = core;
            if (core < 64) coremap[core] = 1;
        }
        int ncore = 0; for (int i = 0; i < 64; i++) ncore += coremap[i];
        printf("[E1] wait=%d 实际执行线程=%d 最大cluster_id=%d 最大core_id=%d 出现core_id数=%d\n",
               w, nrun, maxcid, maxcore, ncore);
        printf("[E1] load_param: cluster_num=%d core_num=%d HW_CORE=%d\n", h[100000], h[100001], h[100002]);
        int dc = 0; xpu_device_count(&dc);
        printf("[E1] xpu_device_count=%d\n", dc);
        xpu_free(o);
    }

    // ---------- 大缓冲 ----------
    size_t WBYTES = 512ull << 20;
    void *W = nullptr;
    if (xpu_malloc(&W, WBYTES)) { printf("malloc W fail\n"); return 1; }
    {
        std::vector<unsigned char> t(1 << 20);
        for (size_t i = 0; i < t.size(); i++) t[i] = (unsigned char)(i * 7 + 3);
        for (size_t off = 0; off < WBYTES; off += t.size())
            xpu_memcpy((char *)W + off, t.data(), t.size(), XPU_HOST_TO_DEVICE);
        xpu_wait();
    }
    void *O = nullptr; xpu_malloc(&O, 65536 * 4); xpu_wait();

    const long long SPAN = 2ll << 20;      // 每线程 2MB

    // ---------- A1 阻塞纯读 ----------
    if (strstr(stage, "read")) {
        int chs[] = {16, 32, 64, 128, 256, 1024, 4096};
        int cls[] = {1, 4, 8};
        for (int ci = 0; ci < 3; ci++) {
            int cl = cls[ci], co = 16, nthr = cl * 16;
            long long tot = (long long)nthr * SPAN;
            if ((size_t)tot > WBYTES) { printf("[A1] cl=%d 超缓冲, 跳过\n", cl); continue; }
            for (int k = 0; k < 7; k++) {
                int chunk = chs[k];
                int nch = (int)(SPAN / chunk);
                if (nch > 200000) nch = 200000;
                m_rd(W, SPAN, chunk, nch, O, cl, co); int w = xpu_wait();
                if (w) { printf("[A1] cl=%d chunk=%d wait=%d FAIL\n", cl, chunk, w); continue; }
                double t0 = now_ms();
                for (int r = 0; r < 3; r++) { m_rd(W, SPAN, chunk, nch, O, cl, co); xpu_wait(); }
                double ms = (now_ms() - t0) / 3.0;
                double bytes = (double)nthr * chunk * (double)nch;
                printf("[A1] 阻塞读 cl=%d co=16 chunk=%5dB 线程=%3d  %8.2f ms  %7.2f GB/s\n",
                       cl, chunk, nthr, ms, bytes / ms / 1e6);
            }
        }
    }

    // ---------- A2 异步纯读 ----------
    if (strstr(stage, "aread")) {
        int chs[] = {64, 256, 1024};
        int dps[] = {2, 4, 8};
        for (int ci = 0; ci < 3; ci++) {
            int cl = 4, co = 16, nthr = cl * 16;
            for (int k = 0; k < 3; k++) for (int d = 0; d < 3; d++) {
                int chunk = chs[k], depth = dps[d];
                if (chunk * depth > 8192) continue;
                int nch = (int)(SPAN / chunk); if (nch > 200000) nch = 200000;
                m_rda(W, SPAN, chunk, nch, depth, O, cl, co); int w = xpu_wait();
                if (w) { printf("[A2] wait=%d FAIL\n", w); continue; }
                double t0 = now_ms();
                for (int r = 0; r < 3; r++) { m_rda(W, SPAN, chunk, nch, depth, O, cl, co); xpu_wait(); }
                double ms = (now_ms() - t0) / 3.0;
                double bytes = (double)nthr * chunk * (double)nch;
                printf("[A2] 异步 read cl=%d chunk=%5dB depth=%d   %8.2f ms  %7.2f GB/s\n",
                       cl, chunk, depth, ms, bytes / ms / 1e6);
            }
        }
    }

    // ---------- B 计算吞吐 ----------
    if (strstr(stage, "comp")) {
        struct { const char *nm; cufn fn; int reps; double elems; } c[] = {
            {"B1 float SIMD vvmul+vvadd   ", m_c_simd, 6000, 512.0},
            {"B2 标量 int8 MAC 4链         ", m_c_i8,   6000, 512.0},
            {"B3 int32打包 int8 MAC        ", m_c_i8p,  6000, 512.0},
            {"B4 int8->float cvt + float SIMD", m_c_cvt, 6000, 512.0},
            {"B5 标量 float MAC 4链        ", m_c_fmac, 6000, 512.0},
            {"B6 老内核风格 (标量+scale)   ", m_c_old,  6000, 512.0},
        };
        for (int ci = 0; ci < 2; ci++) {
            int cl = ci ? 8 : 4, co = 16, nthr = cl * 16;
            for (int i = 0; i < 6; i++) {
                c[i].fn((float *)O, c[i].reps / 10, cl, co); int w = xpu_wait();
                if (w) { printf("[%s] wait=%d FAIL\n", c[i].nm, w); continue; }
                double t0 = now_ms();
                for (int r = 0; r < 3; r++) { c[i].fn((float *)O, c[i].reps, cl, co); xpu_wait(); }
                double ms = (now_ms() - t0) / 3.0;
                double el = (double)c[i].reps * c[i].elems * (double)nthr;
                printf("[%s] cl=%d 线程=%3d  %8.2f ms  %8.2f Melem/s  %6.2f 时钟/元素(0.9G)\n",
                       c[i].nm, cl, nthr, ms, el / ms / 1e3, ms * 1e-3 * 0.9e9 * 16 * (cl / 4.0) / el);
            }
        }
    }

    // ---------- C GEMV 仿真 ----------
    if (strstr(stage, "gemv")) {
        const char *mn[4] = {"C0 标量int8MAC", "C1 cvt+SIMD   ", "C2 int32打包  ", "C3 标量float  "};
        int chs[] = {16, 64, 256, 1024};
        int cl = 8, co = 16, nthr = 128;
        for (int mdl = 0; mdl < 4; mdl++) {
            for (int k = 0; k < 4; k++) {
                int chunk = chs[k];
                int nch = (int)(SPAN / chunk); if (nch > 200000) nch = 200000;
                m_g_rd_mac(W, SPAN, chunk, nch, mdl, O, nullptr, cl, co); int w = xpu_wait();
                if (w) { printf("[C] mode=%d chunk=%d wait=%d FAIL\n", mdl, chunk, w); continue; }
                double t0 = now_ms();
                for (int r = 0; r < 3; r++) { m_g_rd_mac(W, SPAN, chunk, nch, mdl, O, nullptr, cl, co); xpu_wait(); }
                double ms = (now_ms() - t0) / 3.0;
                double el = (double)nthr * (double)chunk * (double)nch;
                double gbs = el / ms / 1e6;
                printf("[%s] cl=8 chunk=%5dB  %8.2f ms  %7.2f Gelem/s  %7.2f GB/s(权重)  预测双芯 %5.2f tok/s\n",
                       mn[mdl], chunk, ms, el / ms / 1e9, gbs, gbs * 2.0 / 9.2);
            }
        }
    }

    xpu_free(W); xpu_free(O);
    printf("=== done ===\n");
    return 0;
}
