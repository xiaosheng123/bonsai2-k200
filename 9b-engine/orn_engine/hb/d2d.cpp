// d2d.cpp — 纯 xpu_memcpy 带宽测量 (零内核风险): 设备内 D2D + 跨设备 peer 拷贝
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <xpu/runtime.h>
static double ms_now(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
int main(int argc,char**argv){
    setbuf(stdout,NULL);
    xpu_set_device(0);
    size_t sizes[]={1ull<<20,16ull<<20,64ull<<20,256ull<<20,512ull<<20};
    printf("[d2d] 设备内 D2D 拷贝 (XPU_DEVICE_TO_DEVICE)\n");
    for(int k=0;k<5;k++){
        size_t sz=sizes[k];
        void *a=nullptr,*b=nullptr;
        if(xpu_malloc(&a,sz)){printf("  malloc a %zu fail\n",sz);continue;}
        if(xpu_malloc(&b,sz)){printf("  malloc b %zu fail\n",sz);continue;}
        std::vector<unsigned char> t(1<<20); for(size_t i=0;i<t.size();i++)t[i]=(unsigned char)i;
        for(size_t off=0;off<sz;off+=t.size()) xpu_memcpy((char*)a+off,t.data(),t.size(),XPU_HOST_TO_DEVICE);
        xpu_wait();
        int rc=xpu_memcpy(b,a,sz,XPU_DEVICE_TO_DEVICE); int w=xpu_wait();
        if(rc||w){printf("  D2D %zu MB rc=%d wait=%d FAIL (不支持?)\n",sz>>20,rc,w); xpu_free(a);xpu_free(b); continue;}
        const int R=3; double t0=ms_now();
        for(int r=0;r<R;r++){xpu_memcpy(b,a,sz,XPU_DEVICE_TO_DEVICE); xpu_wait();}
        double ms=(ms_now()-t0)/R;
        printf("  D2D %4zu MB  %9.3f ms  拷贝带宽=%7.2f GB/s  (单读口径=%7.2f GB/s)\n",
               sz>>20, ms, sz/ms/1e6, sz/ms/2e6);
        xpu_free(a); xpu_free(b);
    }
    // 跨设备 peer: dev0 -> dev1
    {
        size_t sz=256ull<<20;
        xpu_set_device(0); void*a=nullptr; if(xpu_malloc(&a,sz)){printf("peer malloc0 fail\n");return 0;}
        xpu_set_device(1); void*b=nullptr; if(xpu_malloc(&b,sz)){printf("peer malloc1 fail\n");return 0;}
        int rc=xpu_memcpy_peer(1,b,0,a,sz); int w=xpu_wait();
        if(rc||w){printf("[d2d] peer 256MB rc=%d wait=%d FAIL\n",rc,w);}
        else{ const int R=3; double t0=ms_now();
            for(int r=0;r<R;r++){xpu_memcpy_peer(1,b,0,a,sz); xpu_wait();}
            double ms=(ms_now()-t0)/R;
            printf("[d2d] 跨设备 peer dev0->dev1 256MB %9.3f ms  %7.2f GB/s\n",ms,sz/ms/1e6);}
    }
    printf("=== d2d done ===\n"); return 0;
}
