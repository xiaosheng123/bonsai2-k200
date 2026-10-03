// mbench.cpp — 驱动
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <cmath>
#include <xpu/runtime.h>

#define MAXO 200000
void m_info(int *o);
void m_rd(const void *W, int span, int chunk, int nch, void *out, int cl, int co);
void m_rda(const void *W, int span, int chunk, int nch, int depth, void *out, int cl, int co);
void m_rdg(const void *W, int span, int nchunk, int chunk, void *out, int cl, int co);
void m_fma(float *o, int reps, int cl, int co);
void m_i8(float *o, int reps, int cl, int co);
void m_simd(float *o, int reps, int cl, int co);
void m_cvt(float *o, int reps, int cl, int co);
void m_cvt2(float *o, int reps, int cl, int co);
void m_gmac(const void *W, int span, int n32, void *o, float xv, int cl, int co);

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    const char *stage = argc > 2 ? argv[2] : "all";
    bool do_info = strstr(stage, "info") || !strcmp(stage, "all");
    bool do_read = strstr(stage, "read") || !strcmp(stage, "all");
    bool do_comp = strstr(stage, "comp") || !strcmp(stage, "all");
    xpu_set_device(dev);
    printf("[mb] dev=%d\n", dev);

    // ---------- E1 grid 探测 ----------
    if (do_info) {
        int *o = nullptr; xpu_malloc((void **)&o, MAXO * 4);
        std::vector<int> h(MAXO, -1);
        xpu_memcpy(h.data(), o, MAXO * 4, XPU_HOST_TO_DEVICE);
        m_info(o); int w = xpu_wait();
        xpu_memcpy(h.data(), o, MAXO * 4, XPU_DEVICE_TO_HOST);
        int nrun = 0, maxcid = -1, maxcore = -1;
        int coremap[64]; for (int i = 0; i < 64; i++) coremap[i] = 0;
        for (int i = 0; i < MAXO; i++) if (h[i] >= 0) {
            int cid = h[i] / 1000, core = h[i] % 1000;
            nrun++; if (cid > maxcid) maxcid = cid; if (core > maxcore) maxcore = core;
            if (core < 64) coremap[core] = 1;
        }
        int ncore = 0; for (int i = 0; i < 64; i++) ncore += coremap[i];
        printf("[E1] wait=%d 实际执行线程=%d  最大 cluster_id=%d  最大 core_id=%d  出现的 core_id 数=%d\n", w, nrun, maxcid, maxcore, ncore);
        printf("[E1] 设备数=?"); int dc = 0; xpu_device_count(&dc); printf("%d\n", dc);
        xpu_free(o);
    }

    // ---------- 大权重缓冲 (纯读测试用) ----------
    size_t WBYTES = 512ull << 20;               // 512MB
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

    // ---------- E2 纯读 (GM2LM, 阻塞) ----------
    int cls[] = {4, 8};
    if (do_read) {
    for (int ci = 0; ci < 2; ci++) {
        int cl = cls[ci], co = 16;
        int nthr = cl * 16;
        for (int chunk = 1024; chunk <= 8192; chunk *= 2) {
            int nch = 4096;
            int span = (unsigned long long)nch * chunk;
            if ((unsigned long long)nthr * span > WBYTES) continue;
            m_rd(W, span, chunk, nch, O, cl, co); int w = xpu_wait();
            if (w) { printf("[E2] cl=%d chunk=%d wait=%d FAIL\n", cl, chunk, w); continue; }
            double t0 = now_ms();
            for (int r = 0; r < 3; r++) { m_rd(W, span, chunk, nch, O, cl, co); xpu_wait(); }
            double ms = (now_ms() - t0) / 3.0;
            double bytes = (double)nthr * span;
            printf("[E2] GM2LM 阻塞 cl=%2d co=%d chunk=%5dB nch=%d  线程=%4d  %8.2f ms  %7.2f GB/s\n",
                   cl, co, chunk, nch, nthr, ms, bytes / ms / 1e6);
        }
    }

    // ---------- E3 ----------
    if (0)
    for (int cl_ = 4; cl_ <= 8; cl_ += 4) {
        int co = 16, nthr = cl_ * 16;
        int chunk = 4096, nch = 2048;
        int span = (unsigned long long)nch * chunk;
        if ((unsigned long long)nthr * span > WBYTES) continue;
        m_rdg(W, span, nch, chunk, O, cl_, co); int w = xpu_wait();
        if (w) { printf("[E3] wait=%d FAIL\n", w); continue; }
        double t0 = now_ms();
        for (int r = 0; r < 3; r++) { m_rdg(W, span, nch, chunk, O, cl_, co); xpu_wait(); }
        double ms = (now_ms() - t0) / 3.0;
        double bytes = (double)nthr * span;
        double cyc_per_elem = ms * 1e-3 * 64 * 0.9e9 / bytes;
        printf("[E3] 直接全局读 cl=%2d co=%d  线程=%4d  %8.2f ms  %7.2f GB/s  (%.2f 时钟/字节)\n",
               cl_, co, nthr, ms, bytes / ms / 1e6, cyc_per_elem);
    }

    // ---------- 计算吞吐 ----------
    if (do_comp) {
        struct { const char *nm; void (*fn)(float *, int, int, int); double reps; double elems; } c[] = {
            {"E4 local float 2load+FMA", m_fma,  4000, 512.0},
            {"E5 local int8 dot       ", m_i8,   4000, 512.0},
            {"E6 256bit SIMD vvmul+add", m_simd, 8000, 512.0},
            {"E7 int8->f32 cvt + SIMD ", m_cvt,  8000, 512.0},
            {"E7b int32 解包 + SIMD   ", m_cvt2, 8000, 512.0},
        };
        for (int i = 0; i < 5; i++) {
            int cl = 4, co = 16;
            c[i].fn((float *)O, (int)c[i].reps, cl, co); int w = xpu_wait();
            if (w) { printf("[%s] wait=%d FAIL\n", c[i].nm, w); continue; }
            double t0 = now_ms();
            for (int r = 0; r < 3; r++) { c[i].fn((float *)O, (int)c[i].reps, cl, co); xpu_wait(); }
            double ms = (now_ms() - t0) / 3.0;
            double el = c[i].reps * c[i].elems * 64;
            double cyc = ms * 1e-3 * 64 * 0.9e9;
            printf("[%s] cl=%d co=%d  %8.2f ms  %8.2f Melem/s  %6.2f 时钟/元素  %6.2f Gop/s\n",
                   c[i].nm, cl, co, ms, el / ms / 1e3, cyc / el, el / ms / 1e6);
        }
    }

    xpu_free(W); xpu_free(O);
    printf("=== done ===\n");
    return 0;
}
