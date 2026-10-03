// ri.cpp — 驱动: 先 info, 再纯读带宽扫描。严格参数边界。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <xpu/runtime.h>

void rdinfo_info(int *out, int cl, int co);
void rdinfo_rd(const void *W, long long span, long long total, int chunk, int nch, int *out, int cl, int co);

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    const char *stage = argc > 2 ? argv[2] : "all";
    xpu_set_device(dev);
    printf("[ri] dev=%d stage=%s\n", dev, stage);

    // ---------- info: 真拓扑 ----------
    if (!strcmp(stage, "info") || !strcmp(stage, "all")) {
        int *o = nullptr;
        if (xpu_malloc((void **)&o, 4096 * 4)) { printf("info malloc fail\n"); return 1; }
        std::vector<int> h(4096, -7);
        xpu_memcpy(o, h.data(), 4096 * 4, XPU_HOST_TO_DEVICE);
        xpu_wait();
        int cls[] = {1, 4, 8, 16};
        int cos[] = {8, 16, 64};
        for (int ci = 0; ci < 4; ci++) for (int ki = 0; ki < 3; ki++) {
            int cl = cls[ci], co = cos[ki];
            for (int i = 0; i < 4096; i++) h[i] = -7;
            xpu_memcpy(o, h.data(), 4096 * 4, XPU_HOST_TO_DEVICE);
            rdinfo_info(o, cl, co);
            int w = xpu_wait();
            xpu_memcpy(h.data(), o, 4096 * 4, XPU_DEVICE_TO_HOST);
            xpu_wait();
            if (w) { printf("[info] cl=%d co=%d wait=%d FAIL\n", cl, co, w); continue; }
            int n = 0, maxcid = -1, maxcore = -1, cn = -1, cnum = -1;
            int cidseen[64] = {0}, coreseen[64] = {0};
            for (int i = 0; i < 1024; i++) {
                if (h[i * 4] < 0) continue;
                int a = h[i * 4], b = h[i * 4 + 1];
                if (n == 0) { cn = h[i * 4 + 2]; cnum = h[i * 4 + 3]; }
                n++; if (a > maxcid) maxcid = a; if (b > maxcore) maxcore = b;
                if (a < 64) cidseen[a] = 1;
                if (b < 64) coreseen[b] = 1;
            }
            int nc = 0, nd = 0;
            for (int i = 0; i < 64; i++) { nc += cidseen[i]; nd += coreseen[i]; }
            printf("[info] launch cl=%2d co=%3d -> 落地线程=%4d 最大cluster_id=%3d 最大core_id=%3d 出现cluster数=%d 出现core_id数=%d | cluster_num()=%d core_num()=%d\n",
                   cl, co, n, maxcid, maxcore, nc, nd, cn, cnum);
        }
        xpu_free(o);
    }

    // ---------- 纯读带宽扫描 ----------
    if (!strcmp(stage, "read") || !strcmp(stage, "all")) {
        const long long TOTAL = 512ll << 20;      // 512 MB
        void *W = nullptr;
        if (xpu_malloc(&W, (size_t)TOTAL)) { printf("malloc W fail\n"); return 1; }
        {
            std::vector<unsigned char> t(1 << 20);
            for (size_t i = 0; i < t.size(); i++) t[i] = (unsigned char)(i * 7 + 3);
            for (long long off = 0; off < TOTAL; off += (long long)t.size())
                xpu_memcpy((char *)W + off, t.data(), t.size(), XPU_HOST_TO_DEVICE);
            xpu_wait();
        }
        int *O = nullptr;
        if (xpu_malloc((void **)&O, 4096 * 4)) { printf("malloc O fail\n"); return 1; }
        std::vector<int> h(4096);

        int chunks[] = {16, 64, 256, 1024, 4096, 8192, 16384, 32768};
        int cls[] = {4, 8, 16};
        printf("[read] 纯 GM2LM 读带宽 (每线程 span 固定 1MB, 总读 = 线程数*1MB)\n");
        for (int ci = 0; ci < 3; ci++) {
            int cl = cls[ci], co = 16;
            int nthr = cl * 16;
            long long span = 1ll << 20;
            if ((long long)nthr * span > TOTAL) { printf("[read] cl=%d 超缓冲, 跳过\n", cl); continue; }
            for (int k = 0; k < 8; k++) {
                int chunk = chunks[k];
                int nch = (int)(span / chunk);
                rdinfo_rd(W, span, TOTAL, chunk, nch, O, cl, co);
                int w = xpu_wait();
                if (w) { printf("[read] cl=%d chunk=%d wait=%d FAIL\n", cl, chunk, w); continue; }
                // 校验没有线程越界 (-1/-2)
                xpu_memcpy(h.data(), O, 4096 * 4, XPU_DEVICE_TO_HOST); xpu_wait();
                int bad = 0; for (int i = 0; i < nthr; i++) if (h[i] == -1 || h[i] == -2) bad++;
                double ms = 0;
                const int R = 3;
                double t0 = now_ms();
                for (int r = 0; r < R; r++) { rdinfo_rd(W, span, TOTAL, chunk, nch, O, cl, co); xpu_wait(); }
                ms = (now_ms() - t0) / R;
                double bytes = (double)nthr * span;
                printf("[read] cl=%2d 线程=%3d chunk=%6dB  %8.3f ms  %7.2f GB/s  越界线程=%d\n",
                       cl, nthr, chunk, ms, bytes / ms / 1e6, bad);
            }
        }
        xpu_free(O); xpu_free(W);
    }
    printf("=== ri done ===\n");
    return 0;
}
