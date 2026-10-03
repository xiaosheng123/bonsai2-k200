// memprobe.cpp — 探测昆仑 K200 每芯可用 HBM
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <xpu/runtime.h>
int main(int argc, char **argv) {
    int ndev = 0;
    xpu_device_count(&ndev);
    printf("xpu_device_count = %d\n", ndev);
    size_t step = argc > 1 ? (size_t)atol(argv[1]) << 20 : (size_t)64 << 20;
    for (int d = 0; d < ndev && d < 2; d++) {
        xpu_set_device(d);
        std::vector<void *> v;
        size_t tot = 0;
        while (tot < ((size_t)16 << 30)) {
            void *p = nullptr;
            int rc = xpu_malloc(&p, step);
            if (rc) { printf("dev%d: 分配失败 rc=%d, 已成功 %.1f MB\n", d, rc, tot / 1048576.0); break; }
            v.push_back(p); tot += step;
        }
        printf("dev%d 可分配 %.1f MB (%.2f GiB)\n", d, tot / 1048576.0, tot / 1073741824.0);
        for (size_t i = 0; i < v.size(); i++) xpu_free(v[i]);
    }
    return 0;
}
