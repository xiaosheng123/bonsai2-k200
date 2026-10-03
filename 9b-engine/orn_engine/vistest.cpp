#include "vis.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc >= 2 && !strcmp(argv[1], "gemmtest")) return vis_gemm_selftest();
    if (argc >= 2 && !strcmp(argv[1], "cardAtest")) return vis_cardA_selftest();
    if (argc >= 6 && !strcmp(argv[1], "run")) {
        const char* mm = argv[2];
        const char* ip = argv[3];
        int W = atoi(argv[4]), H = atoi(argv[5]);
        const char* dd = (argc > 6) ? argv[6] : "";
        if (W <= 0 || H <= 0 || W % 16 || H % 16) { printf("FATAL: 非法尺寸 %dx%d\n", W, H); return 2; }
        if (vis_init(mm)) return 1;
        std::vector<float> in((size_t)W * H * 3);
        FILE* f = fopen(ip, "rb");
        if (!f) { printf("FATAL: 打不开 %s\n", ip); return 1; }
        size_t got = fread(in.data(), 4, in.size(), f);
        fclose(f);
        if (got != in.size()) { printf("FATAL: 输入尺寸不符 (%zu != %zu)\n", got, in.size()); return 1; }
        std::vector<float> out;
        int ntok = 0;
        int rc = vis_forward(in.data(), W, H, out, ntok, dd);
        if (rc) return rc;
        printf("OK n_tok=%d\n", ntok);
        return 0;
    }
    printf("用法: vistest gemmtest | vistest run <mmproj> <in.f32> <W> <H> [dumpdir]\n");
    return 2;
}
