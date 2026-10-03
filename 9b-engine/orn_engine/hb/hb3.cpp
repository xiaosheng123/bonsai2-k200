// hb3.cpp — 只用已验证的 V2 内核: 测 cl 扩展性 (区分 "共享DMA瓶颈" vs "每核计算瓶颈")
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <xpu/runtime.h>
void run_gemv_q8v2(int cl,int co,const void*W,const void*xq,const void*xs,void*y,int M,int N);
static double now_ms(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
int main(int argc,char**argv){
    setbuf(stdout,NULL);
    int dev=argc>1?atoi(argv[1]):0; xpu_set_device(dev);
    const int N=4096, nb=N/32, ROWB=N+nb*2, M=16384;
    size_t wb=(size_t)M*ROWB;
    void*W=nullptr,*xq=nullptr,*xs=nullptr,*y=nullptr;
    if(xpu_malloc(&W,wb)||xpu_malloc(&xq,65536)||xpu_malloc(&xs,8192)||xpu_malloc(&y,(size_t)M*4+65536)){printf("malloc fail\n");return 1;}
    std::vector<unsigned char> t(1<<20); for(size_t i=0;i<t.size();i++)t[i]=(unsigned char)((i*131+7)&0xff);
    for(size_t off=0;off<wb;off+=t.size()) xpu_memcpy((char*)W+off,t.data(),t.size(),XPU_HOST_TO_DEVICE);
    std::vector<unsigned char> hx(N,3),hs(nb*4,0); {float*f=(float*)hs.data();for(int i=0;i<nb;i++)f[i]=0.01f;}
    xpu_memcpy(xq,hx.data(),N,XPU_HOST_TO_DEVICE); xpu_memcpy(xs,hs.data(),nb*4,XPU_HOST_TO_DEVICE);
    if(xpu_wait()){printf("warm fail\n");return 1;}
    double bytes=(double)M*ROWB;
    printf("[hb3] dev=%d M=%d 读 %.1f MB/次 (cl 扩展性)\n",dev,M,bytes/1048576.0);
    int cls[]={1,2,4,8,16};
    for(int k=0;k<5;k++){
        int cl=cls[k];
        run_gemv_q8v2(cl,16,W,xq,xs,y,M,N); if(xpu_wait()){printf("cl=%d wait FAIL\n",cl);continue;}
        const int R=4; double t0=now_ms();
        for(int r=0;r<R;r++){run_gemv_q8v2(cl,16,W,xq,xs,y,M,N); xpu_wait();}
        double ms=(now_ms()-t0)/R;
        printf("[hb3] cl=%2d co=16  %9.3f ms  %6.2f GB/s   每行 %.3f us  (=%.2f GB/s/簇)\n",
               cl,ms,bytes/ms/1e6,ms*1000.0/M, (bytes/ms/1e6)/cl);
    }
    printf("=== hb3 done ===\n"); return 0;
}
