// 量化误差对比: Q8_0 (每32块 scale) 精确 vs 重新量化为每行 scale
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include "gf_extract.h"

int main(int argc, char **argv) {
    const char *path = "/home/caden/orn/Ornith-1.5-9B-Q8_0.gguf";
    const char *tn = argc > 1 ? argv[1] : "blk.0.ffn_gate.weight";
    GF g;
    if (!g.open(path)) { printf("open fail\n"); return 1; }
    const GT *t = g.get(tn);
    if (!t) { printf("no tensor %s\n", tn); return 1; }
    int N = (int)t->dims[0], M = (int)t->dims[1];
    int rows = M > 512 ? 512 : M;
    size_t rowbytes = (size_t)(N / 32) * 34;
    std::vector<unsigned char> raw(rowbytes * rows);
    if (fseek(g.f, (long)(g.data_start + t->off), SEEK_SET)) { printf("seek fail\n"); return 1; }
    if (fread(raw.data(), 1, raw.size(), g.f) != raw.size()) { printf("read fail\n"); return 1; }
    printf("tensor %s  dims [%d, %d]  检查前 %d 行\n", tn, N, M, rows);

    double se32 = 0, sr = 0, serow = 0, sepair = 0;
    for (int m = 0; m < rows; m++) {
        const unsigned char *r = raw.data() + (size_t)m * rowbytes;
        std::vector<float> w(N);
        float amax_row = 0;
        for (int b = 0; b < N / 32; b++) {
            unsigned short hh = *(const unsigned short *)(r + b * 34);
            unsigned s = (unsigned)(hh & 0x8000) << 16; int e = (hh >> 10) & 0x1f; unsigned mm = hh & 0x3ff, u;
            if (e == 0) { if (!mm) u = s; else { int sh = 0; while (!(mm & 0x400)) { mm <<= 1; sh++; } mm &= 0x3ff; u = s | ((unsigned)(127 - 15 - sh) << 23) | (mm << 13); } }
            else if (e == 31) u = s | 0x7f800000u | (mm << 13);
            else u = s | ((unsigned)(e - 15 + 127) << 23) | (mm << 13);
            float d; memcpy(&d, &u, 4);
            const signed char *q = (const signed char *)(r + b * 34 + 2);
            for (int i = 0; i < 32; i++) { w[b * 32 + i] = q[i] * d; float a = fabsf(w[b * 32 + i]); if (a > amax_row) amax_row = a; }
        }
        // 每 32 块 scale
        float amax32 = 0;
        for (int b = 0; b < N / 32; b++) {
            const signed char *q = (const signed char *)(r + b * 34 + 2);
            unsigned short hh = *(const unsigned short *)(r + b * 34);
            unsigned s = (unsigned)(hh & 0x8000) << 16; int e = (hh >> 10) & 0x1f; unsigned mm = hh & 0x3ff, u;
            if (e == 0) { if (!mm) u = s; else { int sh = 0; while (!(mm & 0x400)) { mm <<= 1; sh++; } mm &= 0x3ff; u = s | ((unsigned)(127 - 15 - sh) << 23) | (mm << 13); } }
            else if (e == 31) u = s | 0x7f800000u | (mm << 13);
            else u = s | ((unsigned)(e - 15 + 127) << 23) | (mm << 13);
            float d; memcpy(&d, &u, 4);
            for (int i = 0; i < 32; i++) { float a = fabsf(q[i] * d); if (a > amax32) amax32 = a; }
            (void)amax32;
        }
        float step_row = amax_row / 127.f;
        for (int b = 0; b < N / 32; b++) {
            unsigned short hh = *(const unsigned short *)(r + b * 34);
            unsigned s = (unsigned)(hh & 0x8000) << 16; int e = (hh >> 10) & 0x1f; unsigned mm = hh & 0x3ff, u;
            if (e == 0) { if (!mm) u = s; else { int sh = 0; while (!(mm & 0x400)) { mm <<= 1; sh++; } mm &= 0x3ff; u = s | ((unsigned)(127 - 15 - sh) << 23) | (mm << 13); } }
            else if (e == 31) u = s | 0x7f800000u | (mm << 13);
            else u = s | ((unsigned)(e - 15 + 127) << 23) | (mm << 13);
            float d; memcpy(&d, &u, 4);
            float ab = 0;
            const signed char *q = (const signed char *)(r + b * 34 + 2);
            for (int i = 0; i < 32; i++) { float a = fabsf(q[i]); if (a > ab) ab = a; }
            (void)ab;
            float step_b = d;                       // 每块误差 = d*sqrt(1/12) 量级
            for (int i = 0; i < 32; i++) {
                float wv = w[b * 32 + i];
                int qr = (int)lrintf(wv / step_row);
                if (qr > 127) qr = 127; else if (qr < -128) qr = -128;
                float ew = qr * step_row;
                se32 += (double)(step_b * step_b) / 12.0;        // Q8_0 理论误差
                serow += (double)(ew - wv) * (ew - wv);
                sr += (double)wv * wv;
            }
        }
    }
    printf("  relrms(Q8_0 每32块, 量化前理论) = %.4f%%\n", 100.0 * sqrt(se32 / (double)(rows * N) / (sr / (double)(rows * N))));
    printf("  relrms(每行 scale 重量化)      = %.4f%%   (n=%d)\n", 100.0 * sqrt(serow / sr), rows * N);
    return 0;
}
