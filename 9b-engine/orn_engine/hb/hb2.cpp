// hb2.cpp — 第二阶段分解测量, 全部只用【已验证】的 kq8 内核 (kq8.proxy.o/kq8_host.o)。
//   T6 传输粒度敏感度: V2(每行 1 次 4352B DMA) vs 非V2(每行 4 次 1088B DMA), 同字节数
//   T7 每次 launch 的固定开销 vs cl
//   T8 双芯并发: 单芯全量 vs 双芯各半 (验证两芯默认 stream 是否真并行)
//   T9 拼接矩阵 (M=24576, 同 N) 单次调用 -> 验证"合并 launch"路线可用
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <xpu/runtime.h>

void run_gemv_q8v2(int cl, int co, const void *W, const void *xq, const void *xs, void *y, int M, int N);
void run_gemv_q8_0i(int cl, int co, const void *W, const void *xq, const void *xs, void *y, int M, int N);

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static const int N = 4096, nb = N / 32, ROWB = N + nb * 2;   // 4352 (V2 行字节)
static const int MAXM = 32768;

struct Dev {
    void *W = nullptr, *xq = nullptr, *xs = nullptr, *y = nullptr;
    int id = -1;
};

static Dev make_dev(int id, size_t wbytes) {
    Dev d; d.id = id;
    xpu_set_device(id);
    if (xpu_malloc(&d.W, wbytes)) { printf("dev%d malloc W fail\n", id); exit(1); }
    if (xpu_malloc(&d.xq, 65536) || xpu_malloc(&d.xs, 8192) || xpu_malloc(&d.y, (size_t)MAXM * 4 + 65536)) {
        printf("dev%d malloc scratch fail\n", id); exit(1);
    }
    std::vector<unsigned char> t(1 << 20);
    for (size_t i = 0; i < t.size(); i++) t[i] = (unsigned char)((i * 131 + 7) & 0xff);
    for (size_t off = 0; off < wbytes; off += t.size()) {
        size_t n = (wbytes - off < t.size()) ? (wbytes - off) : t.size();
        xpu_memcpy((char *)d.W + off, t.data(), n, XPU_HOST_TO_DEVICE);
    }
    std::vector<unsigned char> hx(N, 3), hs(nb * 4, 0);
    { float *f = (float *)hs.data(); for (int i = 0; i < nb; i++) f[i] = 0.01f; }
    xpu_memcpy(d.xq, hx.data(), N, XPU_HOST_TO_DEVICE);
    xpu_memcpy(d.xs, hs.data(), nb * 4, XPU_HOST_TO_DEVICE);
    if (xpu_wait()) { printf("dev%d warmup wait fail\n", id); exit(1); }
    return d;
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    const char *stage = argc > 1 ? argv[1] : "all";
    size_t wbytes = (size_t)MAXM * ROWB;      // 142 MB
    printf("[hb2] stage=%s W=%.0f MB\n", stage, wbytes / 1048576.0);
    Dev d0 = make_dev(0, wbytes);
    Dev d1;
    bool have1 = false;
    if (!strcmp(stage, "dual") || !strcmp(stage, "all")) { d1 = make_dev(1, wbytes); have1 = true; }

    // ---------------- T6 传输粒度: V2(4352B/行) vs 非V2(1088B/行 x4) ----------------
    if (!strcmp(stage, "gran") || !strcmp(stage, "all")) {
        int M = 12288;
        double bytes = (double)M * ROWB;
        xpu_set_device(0);
        const int R = 4;
        run_gemv_q8v2(16, 16, d0.W, d0.xq, d0.xs, d0.y, M, N); xpu_wait();
        double t0 = now_ms();
        for (int r = 0; r < R; r++) { run_gemv_q8v2(16, 16, d0.W, d0.xq, d0.xs, d0.y, M, N); xpu_wait(); }
        double ms1 = (now_ms() - t0) / R;
        run_gemv_q8_0i(16, 16, d0.W, d0.xq, d0.xs, d0.y, M, N); xpu_wait();
        t0 = now_ms();
        for (int r = 0; r < R; r++) { run_gemv_q8_0i(16, 16, d0.W, d0.xq, d0.xs, d0.y, M, N); xpu_wait(); }
        double ms2 = (now_ms() - t0) / R;
        printf("[T6] V2  (4352B/行, 每行1次DMA) M=%d %8.3f ms  %6.2f GB/s\n", M, ms1, bytes / ms1 / 1e6);
        printf("[T6] 非V2(1088B/行x4, 每行4次DMA) M=%d %8.3f ms  %6.2f GB/s  (比值 %.2fx)\n",
               M, ms2, bytes / ms2 / 1e6, ms2 / ms1);
    }

    // ---------------- T7 launch 固定开销 vs cl (M=32 极小) ----------------
    if (!strcmp(stage, "launch") || !strcmp(stage, "all")) {
        int clv[] = {1, 2, 4, 8, 16};
        xpu_set_device(0);
        for (int k = 0; k < 5; k++) {
            int cl = clv[k];
            run_gemv_q8v2(cl, 16, d0.W, d0.xq, d0.xs, d0.y, 32, N); xpu_wait();
            const int R = 50;
            double t0 = now_ms();
            for (int r = 0; r < R; r++) { run_gemv_q8v2(cl, 16, d0.W, d0.xq, d0.xs, d0.y, 32, N); xpu_wait(); }
            printf("[T7] cl=%2d co=16 极小内核 launch+wait = %8.3f ms/次\n", cl, (now_ms() - t0) / R);
        }
        for (int k = 0; k < 3; k++) {
            int co = (int[]){8, 32, 64}[k];
            run_gemv_q8v2(8, co, d0.W, d0.xq, d0.xs, d0.y, 32, N); xpu_wait();
            const int R = 50;
            double t0 = now_ms();
            for (int r = 0; r < R; r++) { run_gemv_q8v2(8, co, d0.W, d0.xq, d0.xs, d0.y, 32, N); xpu_wait(); }
            printf("[T7] cl= 8 co=%2d 极小内核 launch+wait = %8.3f ms/次\n", co, (now_ms() - t0) / R);
        }
    }

    // ---------------- T8 双芯并发 ----------------
    if (!strcmp(stage, "dual") || !strcmp(stage, "all")) {
        int MBIG = 24576, MHALF = MBIG / 2;
        double bytes = (double)MBIG * ROWB;
        const int R = 4;
        // 单芯全量
        xpu_set_device(0);
        run_gemv_q8v2(16, 16, d0.W, d0.xq, d0.xs, d0.y, MBIG, N); xpu_wait();
        double t0 = now_ms();
        for (int r = 0; r < R; r++) { run_gemv_q8v2(16, 16, d0.W, d0.xq, d0.xs, d0.y, MBIG, N); xpu_wait(); }
        double ms1 = (now_ms() - t0) / R;
        printf("[T8] 单芯 全量      M=%d %8.3f ms  %6.2f GB/s\n", MBIG, ms1, bytes / ms1 / 1e6);
        // 双芯各半 (不同设备, 各自默认 stream)
        xpu_set_device(0); run_gemv_q8v2(16, 16, d0.W, d0.xq, d0.xs, d0.y, MHALF, N);
        xpu_set_device(1); run_gemv_q8v2(16, 16, d1.W, d1.xq, d1.xs, d1.y, MHALF, N);
        xpu_set_device(0); xpu_wait(); xpu_set_device(1); xpu_wait();
        t0 = now_ms();
        for (int r = 0; r < R; r++) {
            xpu_set_device(0); run_gemv_q8v2(16, 16, d0.W, d0.xq, d0.xs, d0.y, MHALF, N);
            xpu_set_device(1); run_gemv_q8v2(16, 16, d1.W, d1.xq, d1.xs, d1.y, MHALF, N);
            xpu_set_device(0); xpu_wait(); xpu_set_device(1); xpu_wait();
        }
        double ms2 = (now_ms() - t0) / R;
        printf("[T8] 双芯 各半(cl=16) M=%d %8.3f ms  %6.2f GB/s(总)  加速比=%.2fx\n", MBIG, ms2, bytes / ms2 / 1e6, ms1 / ms2);
        // 双芯各半 cl=8
        xpu_set_device(0); run_gemv_q8v2(8, 16, d0.W, d0.xq, d0.xs, d0.y, MHALF, N);
        xpu_set_device(1); run_gemv_q8v2(8, 16, d1.W, d1.xq, d1.xs, d1.y, MHALF, N);
        xpu_set_device(0); xpu_wait(); xpu_set_device(1); xpu_wait();
        t0 = now_ms();
        for (int r = 0; r < R; r++) {
            xpu_set_device(0); run_gemv_q8v2(8, 16, d0.W, d0.xq, d0.xs, d0.y, MHALF, N);
            xpu_set_device(1); run_gemv_q8v2(8, 16, d1.W, d1.xq, d1.xs, d1.y, MHALF, N);
            xpu_set_device(0); xpu_wait(); xpu_set_device(1); xpu_wait();
        }
        ms2 = (now_ms() - t0) / R;
        printf("[T8] 双芯 各半(cl= 8) M=%d %8.3f ms  %6.2f GB/s(总)  加速比=%.2fx\n", MBIG, ms2, bytes / ms2 / 1e6, ms1 / ms2);
    }

    // ---------------- T9 拼接矩阵单次调用 ----------------
    if (!strcmp(stage, "cat") || !strcmp(stage, "all")) {
        int M = 36864;                       // qkv+gate+ffn_gate+ffn_up 之和
        double bytes = (double)M * ROWB;
        xpu_set_device(0);
        run_gemv_q8v2(16, 16, d0.W, d0.xq, d0.xs, d0.y, M, N); xpu_wait();
        const int R = 4;
        double t0 = now_ms();
        for (int r = 0; r < R; r++) { run_gemv_q8v2(16, 16, d0.W, d0.xq, d0.xs, d0.y, M, N); xpu_wait(); }
        double ms = (now_ms() - t0) / R;
        printf("[T9] 拼接 M=%d 单次调用 %8.3f ms  %6.2f GB/s  (若拆 4 次则多 3x%.2fms launch)\n",
               M, ms, bytes / ms / 1e6, 3.3);
    }
    printf("=== hb2 done ===\n");
    return 0;
}
