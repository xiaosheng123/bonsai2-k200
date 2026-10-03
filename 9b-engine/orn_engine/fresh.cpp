#include <cstdio>
#include <xpu/runtime.h>
extern void run_fresh(const int *in, int *out, int n);
int main(){
    xpu_set_device(0);
    int in[32], out[32];
    for (int i=0;i<32;i++) in[i]=i*7+1;
    int *di,*do_;
    xpu_malloc((void**)&di,128);
    xpu_malloc((void**)&do_,128);
    xpu_memcpy(di,in,128,XPU_HOST_TO_DEVICE);
    xpu_wait();
    run_fresh(di,do_,32);
    xpu_wait();
    xpu_memcpy(out,do_,128,XPU_DEVICE_TO_HOST);
    xpu_wait();
    for(int i=0;i<32;i++) printf("%d ", out[i]);
    printf("\n");
    return 0;
}