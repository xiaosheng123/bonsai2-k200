#include <cstdio>
#include <xpu/runtime.h>
int main(){
    setbuf(stdout,NULL);
    int dc=0; xpu_device_count(&dc);
    printf("xpu_device_count=%d\n", dc);
    const char* nm[]={"PCI_ADDRESS","DEVID","BOARDID","ONBOARDID","MODEL","MEM_MAIN_CAPACITY","MEM_L3_CAPACITY","NUM_CLUSTER","NUM_SDNN","NUM_DMACH","NUM_HWQ","NUM_ENC","NUM_DEC","NUM_IMGPROC"};
    uint32_t maj=0,min=0; xpu_get_driver_version(&maj,&min); printf("driver=%u.%u\n",maj,min);
    xpu_get_runtime_version(&maj,&min); printf("runtime=%u.%u\n",maj,min);
    for(int d=0; d<dc; d++){
        printf("--- dev%d ---\n", d);
        for(int a=0;a<=13;a++){
            uint64_t v=~0ull; int r=xpu_device_get_attr(&v,(XPUDeviceAttr)a,d);
            if (a==5||a==6) printf("  %-18s rc=%d v=%llu (%.1f GB)\n", nm[a], r, (unsigned long long)v, v/1073741824.0);
            else printf("  %-18s rc=%d v=%llu\n", nm[a], r, (unsigned long long)v);
        }
    }
    return 0;
}
