#include <xpu/runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

using namespace std::chrono;

void test_memcpy(const char* name, void* src, void* dst, size_t size, int iters) {
    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        xpu_memcpy(dst, src, size, XPU_HOST_TO_DEVICE);
        xpu_wait();
    }
    auto t1 = high_resolution_clock::now();
    double ms = duration<double, std::milli>(t1 - t0).count() / iters;
    double gb = (size / 1e9) / (ms / 1000.0);
    printf("%s: %.3f ms, %.2f GB/s\n", name, ms, gb);
}

int main() {
    xpu_set_device(0);
    
    // Test sizes
    size_t sizes[] = {16*1024*1024, 64*1024*1024, 256*1024*1024};
    const int iters = 50;
    
    // 1. Regular malloc (pageable)
    printf("=== PAGEABLE (malloc) ===\n");
    for (size_t sz : sizes) {
        void *dsrc, *ddst;
        xpu_malloc(&dsrc, sz);
        xpu_malloc(&ddst, sz);
        char *hsrc = (char*)malloc(sz);
        char *hdst = (char*)malloc(sz);
        test_memcpy("H2D", hsrc, dsrc, sz, iters);
        test_memcpy("D2H", dsrc, hdst, sz, iters);
        test_memcpy("D2D", dsrc, ddst, sz, iters);
        free(hsrc); free(hdst);
        xpu_free(dsrc); xpu_free(ddst);
    }
    
    // 2. mmap MAP_LOCKED (pinned)
    printf("\n=== PINNED (mmap MAP_LOCKED) ===\n");
    for (size_t sz : sizes) {
        void *dsrc, *ddst;
        xpu_malloc(&dsrc, sz);
        xpu_malloc(&ddst, sz);
        char *hsrc = (char*)mmap(NULL, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_LOCKED, -1, 0);
        char *hdst = (char*)mmap(NULL, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_LOCKED, -1, 0);
        if (hsrc == MAP_FAILED || hdst == MAP_FAILED) {
            printf("mmap failed for %zu bytes\n", sz);
            continue;
        }
        test_memcpy("H2D", hsrc, dsrc, sz, iters);
        test_memcpy("D2H", dsrc, hdst, sz, iters);
        test_memcpy("D2D", dsrc, ddst, sz, iters);
        munmap(hsrc, sz); munmap(hdst, sz);
        xpu_free(dsrc); xpu_free(ddst);
    }
    
    // 3. posix_memalign (aligned, may help)
    printf("\n=== ALIGNED (posix_memalign 4K) ===\n");
    for (size_t sz : sizes) {
        void *dsrc, *ddst;
        xpu_malloc(&dsrc, sz);
        xpu_malloc(&ddst, sz);
        char *hsrc = nullptr, *hdst = nullptr;
        posix_memalign((void**)&hsrc, 4096, sz);
        posix_memalign((void**)&hdst, 4096, sz);
        test_memcpy("H2D", hsrc, dsrc, sz, iters);
        test_memcpy("D2H", dsrc, hdst, sz, iters);
        test_memcpy("D2D", dsrc, ddst, sz, iters);
        free(hsrc); free(hdst);
        xpu_free(dsrc); xpu_free(ddst);
    }
    
    return 0;
}