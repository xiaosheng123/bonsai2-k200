#include <cstdio>
#include <chrono>
#include <cmath>
#include <vector>
#include <xpu/runtime.h>
extern void run_dchk(const float *S, float *out);
int main() {
    xpu_set_device(0);
    std::vector<float> in(128*128), out(128*128, -999.f);
    for (size_t i = 0; i < in.size(); i++) in[i] = (float)(i % 977) * 0.25f - 3.f;
    float *di, *doo;
    xpu_malloc((void**)&di, in.size()*4);
    xpu_malloc((void**)&doo, out.size()*4);
    xpu_memcpy(di, in.data(), in.size()*4, XPU_HOST_TO_DEVICE); xpu_wait();

    xpu_memcpy(out.data(), doo, out.size()*4, XPU_DEVICE_TO_HOST); xpu_wait();
    int bad = 0, firstbad = -1; double maxe = 0;
    for (size_t i = 0; i < in.size(); i++) {
        double e = fabs(out[i] - in[i]);
        if (e > maxe) maxe = e;
        if (e > 1e-6) { bad++; if (firstbad < 0) firstbad = (int)i; }
    }
    printf("DCHK row-copy: bad=%d/%zu maxerr=%.2e firstbad=%d\n", bad, in.size(), maxe, firstbad);
    return 0;
}
