// tm3.cpp - reproduce prod crash: allocate lots of device mem (simulate weights) then run merged T=17
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <xpu/runtime.h>
#define VHD 128
#define KHD 128
#define NVH 32
#define PACKED_PER_HV 386
extern void run_delta_net_batch_merged(int c,int co,int T,float*S,const float*pi,float*ot);
int main(){
  int T=17; setbuf(stdout,NULL);
  xpu_set_device(0);
  size_t S_sz=(size_t)NVH*VHD*KHD, pack_sz=(size_t)T*NVH*PACKED_PER_HV, out_sz=(size_t)T*NVH*VHD;
  // Simulate prod: allocate ~6GB on device (like loaded weights), then the S + batch buffers.
  // K200 has ~8GB HBM per chip. Prod loads 3426MB chip0 + 130MB ATN + S 64MB + batch ~1MB.
  const size_t FAKE = 6500u*1024u*1024u; // ~6.3GB
  float *fake=nullptr;
  int mr = xpu_malloc((void**)&fake, FAKE);
  printf("malloc fake 6.3GB: %d ptr=%p\n", mr, (void*)fake);
  float *dS=nullptr,*dpi=nullptr,*dot=nullptr;
  int a1=xpu_malloc((void**)&dS,S_sz*4);
  int a2=xpu_malloc((void**)&dpi,pack_sz*4);
  int a3=xpu_malloc((void**)&dot,out_sz*4);
  printf("malloc dS=%d(%p) dpi=%d(%p) dot=%d(%p)\n",a1,(void*)dS,a2,(void*)dpi,a3,(void*)dot);
  std::vector<float> S0(S_sz,0.001f), pk(pack_sz,0.5f);
  int c1=xpu_memcpy(dS,S0.data(),S_sz*4,XPU_HOST_TO_DEVICE);
  int c2=xpu_memcpy(dpi,pk.data(),pack_sz*4,XPU_HOST_TO_DEVICE);
  int w0=xpu_wait(); printf("copy c1=%d c2=%d wait=%d\n",c1,c2,w0);
  for(int i=0;i<3;i++){
    run_delta_net_batch_merged(8,16,T,dS,dpi,dot);
    int w=xpu_wait(); printf("call %d: wait=%d\n",i,w);
  }
  return 0;
}