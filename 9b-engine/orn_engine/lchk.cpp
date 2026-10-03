#include <cstdio>
#include <vector>
#include <xpu/runtime.h>
extern void run_lchk(float *out);
int main() {
    xpu_set_device(0);
    std::vector<float> out(16*256, -1.f);
    float *do_;
    xpu_malloc((void**)&do_, out.size()*4);
    run_lchk(do_); xpu_wait();
    xpu_memcpy(out.data(), do_, out.size()*4, XPU_DEVICE_TO_HOST); xpu_wait();
    int shared = 0, priv = 0;
    for (int c = 0; c < 16; c++) { if (out[c*256] == 42.f) shared++; else priv++; }
    printf("LCHK: cores seeing core0 value: shared=%d private=%d\n", shared, priv);
    return 0;
}
