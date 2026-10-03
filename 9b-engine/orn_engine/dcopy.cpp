#include <cstdio>
#include <chrono>
#include <vector>
#include <xpu/runtime.h>
extern void run_dcopy(float*,const float*,const float*,const float*,const float*,const float*,float*);
int main() {
    xpu_set_device(0);
    size_t ns=(size_t)32*128*128, nk=(size_t)32*16*128, nv=32*128;
    std::vector<float> z(nk,0.5f), zv(nv,0.5f), zg(32,0.5f);
    float *dS,*dk,*dv,*dq,*dg,*db,*do_;
    xpu_malloc((void**)&dS,ns*4); xpu_malloc((void**)&dk,nk*4); xpu_malloc((void**)&dv,nv*4);
    xpu_malloc((void**)&dq,nk*4); xpu_malloc((void**)&dg,128); xpu_malloc((void**)&db,128); xpu_malloc((void**)&do_,nv*4);
    xpu_memcpy(dS,z.data(),ns*4,XPU_HOST_TO_DEVICE); xpu_memcpy(dk,z.data(),nk*4,XPU_HOST_TO_DEVICE);
    xpu_memcpy(dq,z.data(),nk*4,XPU_HOST_TO_DEVICE);
    xpu_memcpy(dv,zv.data(),nv*4,XPU_HOST_TO_DEVICE); xpu_memcpy(dg,zg.data(),128,XPU_HOST_TO_DEVICE); xpu_wait();
    run_dcopy(dS,dk,dv,dq,dg,db,do_); xpu_wait();
    double best=1e9;
    for(int t=0;t<30;t++){auto a=std::chrono::high_resolution_clock::now();
        run_dcopy(dS,dk,dv,dq,dg,db,do_); xpu_wait();
        auto b=std::chrono::high_resolution_clock::now(); double ms=std::chrono::duration<double,std::milli>(b-a).count(); if(ms<best)best=ms;}
    double traffic=(ns*2+ (nv+3*nk)*32/1.0)*4;
    printf("DCOPY (same DMA shape, no math): best=%.3f ms\n",best);
    return 0;
}
