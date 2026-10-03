// git8.cpp -- 手动最小验证: 官方 gemm_int8 语义 + 误差 + 分块累加 + 全尺寸吞吐
// 安全: 只调官方 API, 尺寸从小到大, 全程边界检查; 绝不用 *_maxptr (会把卡打进 ERROR)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <vector>
#include <chrono>
#include <xpu/runtime.h>
#include "xpu/refactor/nn.h"
#include "xpu/refactor/context/xpu_act_type.h"
namespace api = baidu::xpu::api;
namespace baidu { namespace xpu { namespace api {
int gemm_int8(Context* ctx, const bool trans_a, const bool trans_b, int m, int n, int k,
              float alpha, const float* a, int lda, const int8_t* b, float max_b, int ldb,
              float beta, float* c, int ldc);
}}}
static double NOW(){ using namespace std::chrono; return duration<double>(steady_clock::now().time_since_epoch()).count(); }
static double relrms(const std::vector<float>&a,const std::vector<float>&b){
    double se=0,s2=0; for(size_t i=0;i<a.size();i++){ double d=a[i]-b[i]; se+=d*d; s2+=(double)b[i]*b[i]; }
    return 100.0*sqrt(se/ (s2>0?s2:1e-30));
}
// 量化: 对称 int8, 返回 scale
static float quant(const std::vector<float>&w, std::vector<signed char>&q, float* out_max=0){
    float mx=0; for(size_t i=0;i<w.size();i++) mx=fmaxf(mx,fabsf(w[i]));
    if(out_max) *out_max=mx;
    float s = mx>0? mx/127.f : 1.f; q.resize(w.size());
    for(size_t i=0;i<w.size();i++){ int v=(int)lrintf(w[i]/s); if(v>127)v=127; if(v<-127)v=-127; q[i]=(signed char)v; }
    return s;
}
int main(int argc,char**argv){
    assert(xpu_set_device(0)==0);
    api::Context ctx(api::kXPU1);

    // ================= T1: 小尺寸语义对拍 (M=1, N=64, K=256) =================
    {
        int M=1,N=64,K=256;
        std::vector<float> x(K); for(int i=0;i<K;i++) x[i]=sinf(i*0.11f)*0.5f;
        std::vector<float> W((size_t)N*K); for(size_t i=0;i<W.size();i++) W[i]=cosf(i*0.017f);
        std::vector<signed char> Wq; float wmax=0; float s=quant(W,Wq,&wmax);
        std::vector<float> ref((size_t)M*N,0.f);
        for(int j=0;j<N;j++){ double acc=0; for(int i=0;i<K;i++) acc+=(double)x[i]*(double)Wq[(size_t)j*K+i]*s; ref[j]=(float)acc; }
        void *dA=0,*dB=0,*dC=0;
        assert(xpu_malloc(&dA,(size_t)M*K*4)==0); assert(xpu_malloc(&dB,(size_t)N*K)==0); assert(xpu_malloc(&dC,(size_t)M*N*4)==0);
        xpu_memcpy(dA,x.data(),(size_t)M*K*4,XPU_HOST_TO_DEVICE);
        xpu_memcpy(dB,Wq.data(),(size_t)N*K,XPU_HOST_TO_DEVICE);
        { std::vector<float> _z((size_t)M*N,0.f); xpu_memcpy(dC,_z.data(),(size_t)M*N*4,XPU_HOST_TO_DEVICE); }
        for(int tb=0;tb<2;tb++){
            int r=api::gemm_int8(&ctx,false,tb==1,M,N,K,1.f,(const float*)dA,K,(const int8_t*)dB,wmax,K,0.f,(float*)dC,N);
            int w=xpu_wait();
            std::vector<float> y((size_t)M*N); xpu_memcpy(y.data(),dC,(size_t)M*N*4,XPU_DEVICE_TO_HOST);
            printf("[T1 trans_b=%d] r=%d wait=%d relrms=%.3f%%  y0=%.5f ref0=%.5f\n", tb, r, w, relrms(y,ref), y[0], ref[0]);
        }
        xpu_free(dA);xpu_free(dB);xpu_free(dC);
    }

    // ========== T2: K 分 4 块, 每块各自 scale, beta=1 累加 (精度接近 per-block) ==========
    {
        int M=1,N=64,K=256, CH=4, KC=K/CH;
        std::vector<float> x(K); for(int i=0;i<K;i++) x[i]=sinf(i*0.11f)*0.5f;
        std::vector<float> W((size_t)N*K); for(size_t i=0;i<W.size();i++) W[i]=cosf(i*0.017f)*(1.0f+3.0f*((i%97)==0));
        void *dA=0,*dC=0; assert(xpu_malloc(&dA,(size_t)M*K*4)==0); assert(xpu_malloc(&dC,(size_t)M*N*4)==0);
        xpu_memcpy(dA,x.data(),(size_t)M*K*4,XPU_HOST_TO_DEVICE);
        { std::vector<float> _z((size_t)M*N,0.f); xpu_memcpy(dC,_z.data(),(size_t)M*N*4,XPU_HOST_TO_DEVICE); }
        std::vector<void*> dB(CH,0); std::vector<std::vector<signed char>> Q(CH); std::vector<float> S(CH);
        std::vector<float> ref((size_t)M*N,0.f);
        for(int c=0;c<CH;c++){
            std::vector<float> Wc((size_t)N*KC);
            for(int j=0;j<N;j++) for(int i=0;i<KC;i++) Wc[(size_t)j*KC+i]=W[(size_t)j*K+c*KC+i];
            float mc=0; float s=quant(Wc,Q[c],&mc); S[c]=mc;
            assert(xpu_malloc(&dB[c],(size_t)N*KC)==0);
            xpu_memcpy(dB[c],Q[c].data(),(size_t)N*KC,XPU_HOST_TO_DEVICE);
            for(int j=0;j<N;j++){ double acc=0; for(int i=0;i<KC;i++) acc+=(double)x[c*KC+i]*(double)Q[c][(size_t)j*KC+i]*s; ref[j]+=(float)acc; }
        }
        for(int c=0;c<CH;c++){
            int r=api::gemm_int8(&ctx,false,true,M,N,KC,1.f,(const float*)((char*)dA+ (size_t)c*KC*4),K,(const int8_t*)dB[c],S[c],KC,1.f,(float*)dC,N);
            if(c==0) printf("[T2 chunk0] r=%d\n", r);
        }
        int w=xpu_wait(); std::vector<float> y((size_t)M*N); xpu_memcpy(y.data(),dC,(size_t)M*N*4,XPU_DEVICE_TO_HOST);
        printf("[T2 K分4块 beta=1 累加] wait=%d relrms=%.3f%%  y0=%.5f ref0=%.5f\n", w, relrms(y,ref), y[0], ref[0]);
        xpu_free(dA);xpu_free(dC); for(int c=0;c<CH;c++) xpu_free(dB[c]);
    }

    // ================= T3: 全尺寸吞吐 (M=1, N=4096, K=8192 ⇒ int8 权重 32MB) =================
    {
        int M=1,N=4096,K=8192;
        std::vector<float> x(K,0.01f); std::vector<float> W((size_t)N*K); for(size_t i=0;i<W.size();i++) W[i]=cosf(i*0.017f);
        std::vector<signed char> Wq; float wmax=0; float s=quant(W,Wq,&wmax);
        void *dA=0,*dB=0,*dC=0;
        assert(xpu_malloc(&dA,(size_t)M*K*4)==0); assert(xpu_malloc(&dB,(size_t)N*K)==0); assert(xpu_malloc(&dC,(size_t)M*N*4)==0);
        xpu_memcpy(dA,x.data(),(size_t)M*K*4,XPU_HOST_TO_DEVICE); xpu_memcpy(dB,Wq.data(),(size_t)N*K,XPU_HOST_TO_DEVICE);
        double t0=NOW(); int r=api::gemm_int8(&ctx,false,true,M,N,K,1.f,(const float*)dA,K,(const int8_t*)dB,wmax,K,0.f,(float*)dC,N); double t1=NOW();
        int w=xpu_wait(); double t2=NOW();
        double bytes=(double)N*K; 
        printf("[T3 全尺寸] r=%d wait=%d  host=%.3fms  wait=%.3fms  权重=%.1fMB  throughput=%.2f GB/s (host口径) / %.2f GB/s (wait口径)\n",
               r,w,(t1-t0)*1e3,(t2-t1)*1e3, bytes/1048576.0, bytes/(t1-t0)/1e9, bytes/(t2-t1)/1e9);
        xpu_free(dA);xpu_free(dB);xpu_free(dC);
    }
    printf("ALL DONE\n");
    return 0;
}
