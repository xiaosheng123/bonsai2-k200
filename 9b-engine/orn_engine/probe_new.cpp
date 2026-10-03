#include <xpu/runtime.h>
#include <cstdio>
#include <vector>
#include <sys/time.h>

extern void pw(int *o, int n);
extern void pw2(int *o, int n, int stride, int cnt);

int main() {
    const int N = 1024;
    int *d = nullptr;
    xpu_set_device(1);
    if (xpu_malloc((void **)&d, N * 4)) { printf("malloc fail\n"); return 1; }

    {
        std::vector<int> h(N, -1);
        xpu_memcpy(d, h.data(), N * 4, XPU_HOST_TO_DEVICE); xpu_wait();
        pw(d, N);
        xpu_wait();
        xpu_memcpy(h.data(), d, N * 4, XPU_DEVICE_TO_HOST); xpu_wait();
        int miss = 0, firstmiss = -1;
        for (int i = 0; i < N; i++) if (h[i] != i) { miss++; if (firstmiss < 0) firstmiss = i; }
        printf("=== 1) o[tid]=tid, grid(8,128)  正确槽位=%d/%d 首错=%d ===\n", N - miss, N, firstmiss);
        for (int r = 0; r < 8; r++) {
            printf("    cl%d: ", r);
            for (int c = 0; c < 128; c++) printf("%c", h[r * 128 + c] == r * 128 + c ? '.' : 'X');
            printf("\n");
        }
    }

    {
        const int stride = 1024, cnt = 4;
        const int N2 = 4096;
        int *d2 = nullptr;
        if (xpu_malloc((void **)&d2, N2 * 4)) { printf("malloc2 fail\n"); return 1; }
        std::vector<int> h(N2, -1);
        xpu_memcpy(d2, h.data(), N2 * 4, XPU_HOST_TO_DEVICE); xpu_wait();
        pw2(d2, N2, stride, cnt);
        xpu_wait();
        xpu_memcpy(h.data(), d2, N2 * 4, XPU_DEVICE_TO_HOST); xpu_wait();
        int miss = 0, firstmiss = -1;
        for (int i = 0; i < N2; i++) if (h[i] < 0) { miss++; if (firstmiss < 0) firstmiss = i; }
        printf("=== 2) 跨步写 stride=%d cnt=%d (4096 槽) 未写入=%d 首未写=%d ===\n", stride, cnt, miss, firstmiss);
        for (int r = 0; r < 32; r++) {
            printf("    [%4d]: ", r * 128);
            for (int c = 0; c < 128; c++) printf("%c", h[r * 128 + c] >= 0 ? '.' : 'X');
            printf("\n");
        }
    }
    printf("DONE\n");
    return 0;
}
