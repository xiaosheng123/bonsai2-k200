#include <cstdio>
#include <xpu/runtime.h>
extern void run_selfrw(int *buf, int *out, int n);
int main(){
    xpu_set_device(0);
    int buf[32], out[32];
    int *dbuf, *dout;
    xpu_malloc((void**)&dbuf,128);
    xpu_malloc((void**)&dout,128);
    run_selfrw(dbuf,dout,32);
    xpu_wait();
    xpu_memcpy(out,dout,128,XPU_DEVICE_TO_HOST);
    xpu_memcpy(buf,dbuf,128,XPU_DEVICE_TO_HOST);
    xpu_wait();
    printf("buf: "); for(int i=0;i<32;i++) printf("%d ",buf[i]); printf("\n");
    printf("out: "); for(int i=0;i<32;i++) printf("%d ",out[i]); printf("\n");
    return 0;
}