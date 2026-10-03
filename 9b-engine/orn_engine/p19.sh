#!/bin/bash
# p19.sh — 给 probe.cpp 追加 elementwise_div_2d 极小尺寸单发验证 (新算子, 先探针后大尺寸)
set -eu
cd /home/caden/orn_engine
cp -a probe.cpp probe.cpp.pre19.bak
python3 - <<'PY'
p = "/home/caden/orn_engine/probe.cpp"
s = open(p).read()
if "T_DIV2D" in s:
    print("already patched"); raise SystemExit
fn = r'''
static int T_DIV2D(int m, int n) {
    printf("[probe] elementwise_div_2d  a[m,n] / b[m,1]  m=%d n=%d\n", m, n);
    std::vector<float> a((size_t)m * n), b(m), z, ref((size_t)m * n);
    fill(a, 1313u, 2.0f); fill(b, 1414u, 1.0f);
    for (int i = 0; i < m; i++) b[i] = 0.5f + 0.5f * fabsf(b[i]);   // 远离 0, 避免天然放大相对误差
    for (size_t i = 0; i < (size_t)m * n; i++) ref[i] = a[i] / b[i / n];
    DBuf da, db, dz; da.alloc((size_t)m * n * 4); db.alloc(m * 4); dz.alloc((size_t)m * n * 4);
    da.up(a.data(), (size_t)m * n * 4); db.up(b.data(), m * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    int r = api::elementwise_div_2d(&ctx, (const float*)da.p, (const float*)db.p, (float*)dz.p, m, n, m, 1);
    int ww = xpu_wait();
    z.resize((size_t)m * n); dz.down(z.data(), (size_t)m * n * 4);
    printf("[probe] div_2d[a(m,n)/b(m,1)] r=%d wait=%d relrms=%.5f%%  z0=%.6f ref0=%.6f\n",
           r, ww, relrms(z, ref), z[0], ref[0]);
    return (r == 0 && relrms(z, ref) < 0.5) ? 0 : 1;
}

int main(int argc, char** argv) {'''
s = s.replace("\nint main(int argc, char** argv) {", fn, 1)
s = s.replace('    if (!strcmp(t, "inpl"))   rc |= T_MUL2D_INPLACE(16, 64);',
              '    if (!strcmp(t, "inpl"))   rc |= T_MUL2D_INPLACE(16, 64);\n'
              '    if (!strcmp(t, "div2d"))  rc |= T_DIV2D(8, 64);\n'
              '    if (!strcmp(t, "div2dM")) rc |= T_DIV2D(256, 1152);')
open(p, "w").write(s)
print("patched")
PY
bash buildprobe.sh 2>&1 | tail -5
