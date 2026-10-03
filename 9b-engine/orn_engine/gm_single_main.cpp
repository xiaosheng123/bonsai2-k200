// gm_single_main.cpp — host: 最简 GM2LM 验证, 确定性输入 src[i]=i
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <xpu/runtime.h>

extern void run_gm_single(int clusters, int cores, float *src, float *dst);

int main() {
    // src[i] = i (确定性), 每 cl 读 1024 个
    std::vector<float> src(8*1024);
    for (int i = 0; i < 8*1024; i++) src[i] = (float)(i % 1024);
    float *d_src, *d_dst;
    xpu_malloc((void**)&d_src, 8*1024*4);
    xpu_malloc((void**)&d_dst, 16*8*4);
    xpu_memcpy(d_src, src.data(), 8*1024*4, XPU_HOST_TO_DEVICE);
    xpu_wait();
    run_gm_single(8, 16, d_src, d_dst);
    xpu_wait();
    std::vector<float> r(16*8);
    xpu_memcpy(r.data(), d_dst, r.size()*4, XPU_DEVICE_TO_HOST);
    xpu_wait();
    // 期望: sum(i for i in 0..1023) * 1.0001 = (1023*1024/2) * 1.0001
    double expect = (1023.0*1024/2) * 1.0001;
    float maxerr = 0;
    for (int i = 0; i < 16*8; i++) {
        float e = fabsf(r[i] - (float)expect);
        if (e > maxerr) maxerr = e;
    }
    printf("GM2LM single deterministic: expect=%.1f got[0]=%.1f maxerr=%.4f %s\n",
           expect, r[0], maxerr, maxerr < 1.0f ? "PASS" : "FAIL");
    return 0;
}