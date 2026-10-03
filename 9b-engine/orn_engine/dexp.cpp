#include <cstdio>
#include <cmath>
#include <vector>
#include <xpu/runtime.h>
extern void run_dexp(const float *in, float *out);
int main() {
    xpu_set_device(0);
    std::vector<float> in(32), out(32, -999.f);
    for (int i = 0; i < 32; i++) in[i] = (i - 16) * 0.01f;
    float *di, *doo;
    xpu_malloc((void**)&di, 128); xpu_malloc((void**)&doo, 128);
    xpu_memcpy(di, in.data(), 128, XPU_HOST_TO_DEVICE); xpu_wait();
    run_dexp(di, doo); xpu_wait();
    xpu_memcpy(out.data(), doo, 128, XPU_DEVICE_TO_HOST); xpu_wait();
    printf("in0..4: %.3f %.3f %.3f %.3f %.3f\n", in[0], in[1], in[2], in[3], in[4]);
    printf("out0..4: %.4f %.4f %.4f %.4f %.4f\n", out[0], out[1], out[2], out[3], out[4]);
    printf("expect : %.4f %.4f %.4f %.4f %.4f\n", exp(in[0]), exp(in[1]), exp(in[2]), exp(in[3]), exp(in[4]));
    return 0;
}
