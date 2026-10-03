#!/bin/bash
# p19b.sh — 追加 probe: reduce(MAX, 2D, dim0) 的 [1,nc] 输出布局 (本轮卡上归一链用到的新用法)
set -eu
cd /home/caden/orn_engine
python3 - <<'PY'
p = "/home/caden/orn_engine/probe.cpp"
s = open(p).read()
if "T_REDDIM0" in s:
    print("already patched"); raise SystemExit
fn = r'''
static int T_REDDIM0(int m, int nc) {
    printf("[probe] reduce MAX 2D dim0  [%d,%d] -> [1,%d]\n", m, nc, nc);
    std::vector<float> x((size_t)m * nc), y, ref(nc, -1e30f);
    fill(x, 1717u, 3.0f);
    for (int i = 0; i < m; i++) for (int c = 0; c < nc; c++) if (x[(size_t)i * nc + c] > ref[c]) ref[c] = x[(size_t)i * nc + c];
    DBuf dx, dy; dx.alloc((size_t)m * nc * 4); dy.alloc(nc * 4); dx.up(x.data(), (size_t)m * nc * 4);
    Context ctx(Device(DeviceType::XPU1, 0));
    int xd[2] = {m, nc}, rd[1] = {0};
    int r = api::reduce(&ctx, (const float*)dx.p, (float*)dy.p, xd, 2, rd, 1, REDUCE_MAX);
    int ww = xpu_wait();
    y.resize(nc); dy.down(y.data(), nc * 4);
    printf("[probe] reduce MAX 2D dim0 r=%d wait=%d relrms=%.5f%%  y0=%.6f ref0=%.6f\n", r, ww, relrms(y, ref), y[0], ref[0]);
    return (r == 0 && relrms(y, ref) < 0.5) ? 0 : 1;
}

int main(int argc, char** argv) {'''
s = s.replace("\nint main(int argc, char** argv) {", fn, 1)
s = s.replace('    if (!strcmp(t, "div2dM")) rc |= T_DIV2D(256, 1152);',
              '    if (!strcmp(t, "div2dM")) rc |= T_DIV2D(256, 1152);\n'
              '    if (!strcmp(t, "reddim0")) rc |= T_REDDIM0(64, 24);')
open(p, "w").write(s)
print("patched")
PY
bash buildprobe.sh 2>&1 | tail -3
bash buildcards.sh 2>&1 | tail -3
bash buildorn3v19.sh 2>&1 | tail -4
