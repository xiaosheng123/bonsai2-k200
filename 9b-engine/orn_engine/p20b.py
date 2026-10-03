#!/usr/bin/env python3
# p20b.py — 给 probe.cpp 追加第 20 轮新算子/新用法 的极小尺寸单发验证
#   T_REDMIN3 : reduce(MIN, 3D, rdims={2})
#   T_ADDROW  : elementwise_add_2d 行广播 [m,n] + [1,n]   (卡上 bias)
#   T_EWMAXL  : elementwise_max (flat)                    (max|A| 用)
#   T_NEGL    : neg (flat)                                (max|A| 用)
#   T_MAXABS3 : ★ 完整卡上 max|A| 归一链 vs 主机 f32 参考 (A 含负值)
#   T_GELUZ   : gelu 出/入不同缓冲 + padding 尾部 (0) 保持 0
import sys
p = "/home/caden/orn_engine/probe.cpp"
s = open(p, encoding="utf-8").read()
if "T_MAXABS3" in s:
    print("already patched"); raise SystemExit

fn = r'''
static int T_REDMIN3(int m, int nc, int cs) {
    printf("[probe] reduce MIN 3D dim2  [%d,%d,%d] -> [%d,%d]\n", m, nc, cs, m, nc);
    std::vector<float> x((size_t)m * nc * cs), y, ref((size_t)m * nc);
    fill(x, 2323u, 3.0f);
    for (int t = 0; t < m; t++) for (int c = 0; c < nc; c++) {
        float mn = 1e30f;
        for (int j = 0; j < cs; j++) { float v = x[((size_t)t * nc + c) * cs + j]; if (v < mn) mn = v; }
        ref[(size_t)t * nc + c] = mn;
    }
    DBuf dx, dy; dx.alloc((size_t)m * nc * cs * 4); dy.alloc((size_t)m * nc * 4); dx.up(x.data(), (size_t)m * nc * cs * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    int xd[3] = {m, nc, cs}, rd[1] = {2};
    int r = api::reduce(&ctx, (const float*)dx.p, (float*)dy.p, xd, 3, rd, 1, REDUCE_MIN);
    int ww = xpu_wait();
    y.resize((size_t)m * nc); dy.down(y.data(), (size_t)m * nc * 4);
    printf("[probe] reduce MIN 3D r=%d wait=%d relrms=%.5f%%  y0=%.6f ref0=%.6f\n", r, ww, relrms(y, ref), y[0], ref[0]);
    return (r == 0 && relrms(y, ref) < 0.5) ? 0 : 1;
}

static int T_ADDROW(int m, int n) {
    printf("[probe] elementwise_add_2d 行广播 a[m,n] + b[1,n]  m=%d n=%d\n", m, n);
    std::vector<float> a((size_t)m * n), b(n), z, ref((size_t)m * n);
    fill(a, 5151u, 2.0f); fill(b, 5252u, 1.0f);
    for (size_t i = 0; i < (size_t)m * n; i++) ref[i] = a[i] + b[i % n];
    DBuf da, db, dz; da.alloc((size_t)m * n * 4); db.alloc(n * 4); dz.alloc((size_t)m * n * 4);
    da.up(a.data(), (size_t)m * n * 4); db.up(b.data(), n * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    int r = api::elementwise_add_2d(&ctx, (const float*)da.p, (const float*)db.p, (float*)dz.p, m, n, 1, n);
    int ww = xpu_wait();
    z.resize((size_t)m * n); dz.down(z.data(), (size_t)m * n * 4);
    printf("[probe] add_2d[a(m,n)+b(1,n)] r=%d wait=%d relrms=%.5f%%  (应 =0, 加法精确)\n", r, ww, relrms(z, ref));
    return (r == 0 && relrms(z, ref) < 0.5) ? 0 : 1;
}

static int T_MAXABS3(int m, int nc, int cs) {
    printf("[probe] ★ 卡上 max|A| 归一链  m=%d nc=%d cs=%d (A 含负值)\n", m, nc, cs);
    std::vector<float> x((size_t)m * nc * cs);
    fill(x, 6363u, 3.0f);                       // 有正有负
    std::vector<float> ref((size_t)m * nc);
    for (int t = 0; t < m; t++) for (int c = 0; c < nc; c++) {
        float am = 0;
        for (int j = 0; j < cs; j++) { float a = fabsf(x[((size_t)t * nc + c) * cs + j]); if (a > am) am = a; }
        ref[(size_t)t * nc + c] = am;           // 主机 fabsf 参考
    }
    std::vector<float> refMc(nc, 0.f);
    for (int t = 0; t < m; t++) for (int c = 0; c < nc; c++) if (ref[(size_t)t * nc + c] > refMc[c]) refMc[c] = ref[(size_t)t * nc + c];
    // 卡上: reduce MAX / reduce MIN / neg / elementwise_max / reduce MAX(Mc) / div f
    DBuf dx, dmx, dmn, dng, dmc, df, dones;
    dx.alloc((size_t)m * nc * cs * 4); dmx.alloc((size_t)m * nc * 4); dmn.alloc((size_t)m * nc * 4);
    dng.alloc((size_t)m * nc * 4); dmc.alloc(nc * 4); df.alloc((size_t)m * nc * 4); dones.alloc(nc * 4);
    dx.up(x.data(), (size_t)m * nc * cs * 4);
    std::vector<float> ones(nc, 1.0f); dones.up(ones.data(), nc * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    int xd3[3] = {m, nc, cs}, rd2[1] = {2}, xd2[2] = {m, nc}, rd0[1] = {0};
    int r = api::reduce(&ctx, (const float*)dx.p, (float*)dmx.p, xd3, 3, rd2, 1, REDUCE_MAX);
    r |= api::reduce(&ctx, (const float*)dx.p, (float*)dmn.p, xd3, 3, rd2, 1, REDUCE_MIN);
    r |= api::neg(&ctx, (const float*)dmn.p, (float*)dng.p, m * nc);
    r |= api::elementwise_max_2d(&ctx, (const float*)dmx.p, (const float*)dng.p, (float*)dmx.p, m, nc, m, nc);
    r |= api::reduce(&ctx, (const float*)dmx.p, (float*)dmc.p, xd2, 2, rd0, 1, REDUCE_MAX);
    r |= api::elementwise_div_2d(&ctx, (const float*)dmc.p, (const float*)dmx.p, (float*)df.p, 1, nc, m, nc);
    int ww = xpu_wait();
    std::vector<float> absmax((size_t)m * nc), Mc(nc), f((size_t)m * nc);
    dmx.down(absmax.data(), (size_t)m * nc * 4); dmc.down(Mc.data(), nc * 4); df.down(f.data(), (size_t)m * nc * 4);
    double amr = relrms(absmax, ref);
    std::vector<float> reff((size_t)m * nc);
    for (size_t i = 0; i < reff.size(); i++) reff[i] = refMc[i % nc] / ref[i];
    double fr = relrms(f, reff);
    double mcr = 0; for (int c = 0; c < nc; c++) mcr = fmax(mcr, fabs((double)Mc[c] - (double)refMc[c]));
    printf("[probe] max|A| 链 r=%d wait=%d | absmax vs 主机 fabsf relrms=%.5f%% | Mc maxabs差=%.3e | f=Mc/absmax relrms=%.5f%%\n",
           r, ww, amr, mcr, fr);
    return (r == 0 && amr < 0.5 && mcr == 0.0) ? 0 : 1;
}

static int T_GELUZ(int m, int n, int nreal) {
    printf("[probe] gelu 出/入异缓冲 + padding 尾部保持 0  m=%d n=%d nreal=%d\n", m, n, nreal);
    std::vector<float> x((size_t)m * n, 0.f), ref((size_t)m * n, 0.f), y;
    fill(x, 7474u, 4.0f);
    for (int t = 0; t < m; t++) for (int j = nreal; j < n; j++) x[(size_t)t * n + j] = 0.f;   // padding 尾部
    for (size_t i = 0; i < x.size(); i++) { float v = x[i]; ref[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))); }
    DBuf dx, dy; dx.alloc((size_t)m * n * 4); dy.alloc((size_t)m * n * 4); dx.up(x.data(), (size_t)m * n * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    int r = api::gelu(&ctx, (const float*)dx.p, (float*)dy.p, m * n);
    int ww = xpu_wait();
    y.resize((size_t)m * n); dy.down(y.data(), (size_t)m * n * 4);
    int tailbad = 0;
    for (int t = 0; t < m; t++) for (int j = nreal; j < n; j++) if (y[(size_t)t * n + j] != 0.f) tailbad++;
    printf("[probe] gelu r=%d wait=%d relrms=%.5f%%  padding 尾部非 0 个数=%d (期望 0)\n", r, ww, relrms(y, ref), tailbad);
    return (r == 0 && relrms(y, ref) < 0.5 && tailbad == 0) ? 0 : 1;
}

int main(int argc, char** argv) {'''

s = s.replace("\nint main(int argc, char** argv) {", fn, 1)
s = s.replace('    if (!strcmp(t, "reddim0")) rc |= T_REDDIM0(64, 24);',
              '    if (!strcmp(t, "reddim0")) rc |= T_REDDIM0(64, 24);\n'
              '    if (!strcmp(t, "redmin3")) rc |= T_REDMIN3(8, 4, 16);\n'
              '    if (!strcmp(t, "addrow"))  rc |= T_ADDROW(8, 45);\n'
              '    if (!strcmp(t, "addrowM")) rc |= T_ADDROW(128, 4320);\n'
              '    if (!strcmp(t, "maxabs"))  rc |= T_MAXABS3(8, 4, 16);\n'
              '    if (!strcmp(t, "maxabsM")) rc |= T_MAXABS3(64, 45, 96);\n'
              '    if (!strcmp(t, "geluz"))   rc |= T_GELUZ(8, 96, 80);\n'
              '    if (!strcmp(t, "geluzM"))  rc |= T_GELUZ(64, 4320, 4304);')
open(p, "w", encoding="utf-8").write(s)
print("probe patched")
