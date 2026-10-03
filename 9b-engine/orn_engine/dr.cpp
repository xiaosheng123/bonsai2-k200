#include <cstdio>
#include <xpu/runtime.h>
extern void run_dr(const int *in, int *out, int n);
int main() {
    xpu_set_device(0);
    int in[128], out[128];
    for (int i = 0; i < 128; i++) in[i] = i * 3 + 1;
    int *di, *do_;
    xpu_malloc((void**)&di, 512);
    xpu_malloc((void**)&do_, 512);
    xpu_memcpy(di, in, 512, XPU_HOST_TO_DEVICE);
    xpu_wait();
    run_dr(di, do_, 128);
    xpu_wait();
    xpu_memcpy(out, do_, 512, XPU_DEVICE_TO_HOST);
    xpu_wait();
    for (int i = 0; i < 128; i++) printf("%d ", out[i]);
    printf("\n");
    return 0;
}