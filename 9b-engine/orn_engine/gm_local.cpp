// gm_local.cpp — local 预算分档探测: 用法 ./gm_local <bytes>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <xpu/runtime.h>

extern void run_gk6144(float*, long*, int, int, int);
extern void run_gk6656(float*, long*, int, int, int);
extern void run_gk7168(float*, long*, int, int, int);
extern void run_gk7680(float*, long*, int, int, int);

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    int bytes = argc > 1 ? atoi(argv[1]) : 6144;
    if (xpu_set_device(0)) { printf("SETDEV_FAIL\n"); return 2; }
    void *dsrc = 0, *dsink = 0;
    if (xpu_malloc(&dsrc, 4096) || xpu_malloc(&dsink, 8)) { printf("MALLOC_FAIL\n"); return 2; }
    float h[1024] = {1.5f};
    xpu_memcpy(dsrc, h, 4096, XPU_HOST_TO_DEVICE);
    xpu_wait();
    const int N = 256;
    if (bytes == 6144)      run_gk6144((float*)dsrc, (long*)dsink, 8, 16, N);
    else if (bytes == 6656) run_gk6656((float*)dsrc, (long*)dsink, 8, 16, N);
    else if (bytes == 7168) run_gk7168((float*)dsrc, (long*)dsink, 8, 16, N);
    else if (bytes == 7680) run_gk7680((float*)dsrc, (long*)dsink, 8, 16, N);
    else { printf("BAD_BYTES\n"); return 2; }
    int rc = xpu_wait();
    printf("LOCAL_%d %s\n", bytes, rc ? "CRASH" : "OK");
    return rc;
}
