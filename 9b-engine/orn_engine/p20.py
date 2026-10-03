#!/usr/bin/env python3
# p20.py — 第 20 轮: (A) FFN 搬卡 (VIS_CARDFFN) + (B) 16 头输出堆叠 (VIS_HEAP)
#   起点: 线上 vis.cpp (第 19 轮交付)  ->  产出 vis.cpp.cards (实验二进制用)
import sys

P = "/home/caden/orn_engine/vis.cpp"
OUT = "/home/caden/orn_engine/vis.cpp.cards"
s = open(P, encoding="utf-8").read()
orig = s
nrep = 0

def rep(old, new, cnt=1):
    global s, nrep
    c = s.count(old)
    if c != cnt:
        print("!! 匹配数 %d (期望 %d) 于:\n%s\n----" % (c, cnt, old[:500]))
        sys.exit(1)
    s = s.replace(old, new)
    nrep += 1

# ---------------------------------------------------------------- 1) 开关 + 缓冲
rep("""static int g_vcarda = 1;    // VIS_CARDA=0 退回老路径(主机归一 + A 上下卡), 仅作 A/B 对照
static double g_t_gemm = 0, g_t_host = 0;""",
"""static int g_vcarda = 1;    // VIS_CARDA=0 退回老路径(主机归一 + A 上下卡), 仅作 A/B 对照
// ★★ 第 20 轮 (全部为【实验开关】, 默认 0) ★★
//   VIS_CARDFFN: 0=关(默认)  1=(A)全卡: ffn_up 留卡 + 卡上 bias/gelu + 卡上 max|A| 归一 + 分块 gemm
//                           2=(A)半卡: 主机 bias/gelu(与老路径逐位相同) + 卡上 max|A| 归一 + 分块 gemm
//   VIS_HEAP   : 1=(B) 16 头 O gemm 输出堆叠在卡上, 每层每芯只做一次 D2H (32→2 次/层)
static int g_vcardffn = 0;
static int g_vheap = 0;
static int g_vkpad = 0;      // FFN padding 布局 (Kp = 45*96 = 4320): 与老的 96+80 非均匀分块逐位等价
static void* g_vUpP[2] = {nullptr, nullptr};   // (A) 卡上 FFN 激活 (padding 布局 [m][Kp])
static void* g_vBup[2] = {nullptr, nullptr};   // (A) ffn_up.bias 的 padding 副本 [VNL][Kp] (尾部 0)
static void* g_vSlO[2] = {nullptr, nullptr};   // (B) 16 头 O gemm 输出堆叠 [16][Mh][VDH]
static size_t g_vUpPsz = 0, g_vSlOsz = 0, g_vBupsz = 0;
static void* g_vMg[2] = {nullptr, nullptr};    // ★ 全 -1 的 [1][nc] 向量 (算 -minA 用; api::neg 不在 api.h 里)
static int g_vkpadn = VFFN;                    // padding 后的 FFN 维度 (4320)
static double g_t_gemm = 0, g_t_host = 0;""")

# ---------------------------------------------------------------- 2) vgemm_chunked: 输出落点可指定
rep("""static void vgemm_chunked(int N, int K, int nc, const std::vector<int>& cs, const std::vector<int>& koff,
                          const signed char* const* bdev, const void* const* rsdev,
                          const float* A, int M, float* C, int leave_on_card = 0) {""",
"""static void vgemm_chunked(int N, int K, int nc, const std::vector<int>& cs, const std::vector<int>& koff,
                          const signed char* const* bdev, const void* const* rsdev,
                          const float* A, int M, float* C, int leave_on_card = 0,
                          float* const* accdst = nullptr) {""")
rep("""        if (m <= 0) continue;
        xpu_set_device(dv);
        {   // H2D 1) An 本分区行""",
"""        if (m <= 0) continue;
        xpu_set_device(dv);
        float* accp = accdst ? accdst[dv] : (float*)g_vCa[dv];   // ★ 第20轮: 结果可落在别的卡上缓冲
        {   // H2D 1) An 本分区行""")
rep("""                  int r = api::reduce(g_vctx[dv], stk, (float*)g_vCa[dv], xd, 3, rd, 1, api::REDUCE_SUM);""",
"""                  int r = api::reduce(g_vctx[dv], stk, accp, xd, 3, rd, 1, api::REDUCE_SUM);""")
rep("""                  r = api::elementwise_add(g_vctx[dv], (const float*)g_vCa[dv], (const float*)g_vCr[dv], (float*)g_vCa[dv], m * N);""",
"""                  r = api::elementwise_add(g_vctx[dv], (const float*)accp, (const float*)g_vCr[dv], (float*)accp, m * N);""")
rep("""            if (xpu_memcpy(C + (size_t)r0 * N, g_vCa[dv], (size_t)m * N * 4, XPU_DEVICE_TO_HOST)) { printf("[vis] FATAL: D2H C 分区\\n"); exit(1); }""",
"""            if (xpu_memcpy(C + (size_t)r0 * N, accp, (size_t)m * N * 4, XPU_DEVICE_TO_HOST)) { printf("[vis] FATAL: D2H C 分区\\n"); exit(1); }""")

# ---------------------------------------------------------------- 3) vgemm: 透传
rep("""static void vgemm(const VW& w, const float* A, int M, float* C) {""",
"""static void vgemm(const VW& w, const float* A, int M, float* C, int leave_on_card = 0,
                  float* const* accdst = nullptr) {""")
rep("""    vgemm_chunked(N, K, w.nc, w.cs, w.koff, bd.data(), rd, A, M, C);
}""",
"""    vgemm_chunked(N, K, w.nc, w.cs, w.koff, bd.data(), rd, A, M, C, leave_on_card, accdst);
}""")

# ---------------------------------------------------------------- 4) cardA: 核心抽出 + absmax + 输出重定向
old_fn = s[s.index("static void vgemm_cardA(const std::vector<signed char>& qpack, const std::vector<float>& rs,"):]
old_fn = old_fn[:old_fn.index("\n}\n") + 3]
assert "cardA 折回" in old_fn and old_fn.count("api::gemm_int8") == 1, "cardA 定位失败"

new_fn = """// ★ 第 20 轮: 核心多了 3 个参数 (全部默认 = 第 19 轮行为)
//   bdev/rsdev : B 与折回列 scale 的【设备指针表】(索引 p*nc+c) —— 便于直接用常驻卡上的权重
//   accdst     : 结果落点 (默认 g_vCa); skip_d2h: 只算不下载 (供 (B) 堆叠后一次性 D2H)
//   absmax_mode: 1 = A 含负值, 用 max(reduce MAX, -reduce MIN) 求 max|A| (与主机 fabsf 逐位相同)
static void vdump(const char* dir, const char* name, const float* p, size_t n);
static void vgemm_cardA_core(int N, int K, int cs, const signed char* const* bdev,
                             const void* const* rsdev, const void* const* A_dev, int M, float* C,
                             int leave_on_card, float* const* accdst, int skip_d2h, int absmax_mode) {
    if (cs <= 0 || K % cs) { printf("[vis] FATAL: cardA 分块不整除 K=%d cs=%d\\n", K, cs); exit(1); }
    int nc = K / cs;
    if ((size_t)nc * N * 4 > g_vRdsz) { printf("[vis] FATAL: cardA Rd 越界\\n"); exit(1); }
    if (A_dev[0] == nullptr || A_dev[1] == nullptr) { printf("[vis] FATAL: cardA A_dev 为空\\n"); exit(1); }
    int Mh = (M + 1) / 2;
    for (int p = 0; p < 2; p++) {
        int dv = p, r0 = p * Mh, m = Mh; if (r0 + m > M) m = M - r0;
        if (m <= 0) continue;
        xpu_set_device(dv);
        if ((size_t)m * K * 4 > g_vAnsz) { printf("[vis] FATAL: cardA An 越界 m=%d K=%d (%zu>%zu)\\n", m, K, (size_t)m * K * 4, g_vAnsz); exit(1); }
        if ((size_t)m * nc * 4 > g_vFxsz) { printf("[vis] FATAL: cardA 行因子越界 m=%d nc=%d\\n", m, nc); exit(1); }
        if ((size_t)m * N * 4 > g_vSesz) { printf("[vis] FATAL: cardA 次 reduce 越界 m=%d N=%d\\n", m, N); exit(1); }
        const float* A = (const float*)A_dev[p];        // 本分区的 m 行已就位
        float* mxA = (float*)g_vFx[dv];     // [m][nc] 逐块逐行 max
        float* fAr = (float*)g_vFa[dv];     // [m][nc] 归一因子 f = Mc/mx   (与主机同式)
        float* tmp = (float*)g_vFb[dv];     // [m][nc] 临时 (mx*invMc)
        float* mFe = (float*)g_vFe[dv];     // [nc][m] 折回行因子 (转置后)
        float* mBc = (float*)g_vFi[dv];     // [1][nc] 每块全局 max Mc
        float* invM = mBc + nc;             // [1][nc] 1/Mc
        float* An  = (float*)g_vAn[dv];
        float* acc = accdst ? accdst[dv] : (float*)g_vCa[dv];
        float* sec = (float*)g_vSe[dv];
        {   // 1) 逐块逐行 MAX: [m, nc, cs] --dim2--> [m, nc]
            int xd[3]; xd[0] = m; xd[1] = nc; xd[2] = cs;
            int rd[1]; rd[0] = 2;
            int r = api::reduce(g_vctx[dv], A, mxA, xd, 3, rd, 1, api::REDUCE_MAX);
            if (r) { printf("[vis] FATAL: cardA reduce MAX r=%d (m=%d nc=%d cs=%d)\\n", r, m, nc, cs); exit(1); }
        }
        if (absmax_mode) {   // ★ 1b~1d) A 含负值: max|A| = max( maxA, -minA )  (与主机 fabsf 逐位相同)
            float* mnA = tmp;
            int xd[3]; xd[0] = m; xd[1] = nc; xd[2] = cs;
            int rd[1]; rd[0] = 2;
            int r = api::reduce(g_vctx[dv], A, mnA, xd, 3, rd, 1, api::REDUCE_MIN);
            if (r) { printf("[vis] FATAL: cardA reduce MIN r=%d\\n", r); exit(1); }
            r = api::elementwise_mul_2d(g_vctx[dv], (const float*)mnA, (const float*)g_vMg[dv], fAr, m, nc, 1, nc);
            if (r) { printf("[vis] FATAL: cardA neg(-1 乘) r=%d\\n", r); exit(1); }
            r = api::elementwise_max_2d(g_vctx[dv], (const float*)mxA, (const float*)fAr, mxA, m, nc, m, nc);
            if (r) { printf("[vis] FATAL: cardA elementwise_max r=%d\\n", r); exit(1); }
        }
        {   // 2) 每块全局 MAX: [m,nc] --dim0--> [1,nc]
            int xd[2]; xd[0] = m; xd[1] = nc;
            int rd[1]; rd[0] = 0;
            int r = api::reduce(g_vctx[dv], mxA, mBc, xd, 2, rd, 1, api::REDUCE_MAX);
            if (r) { printf("[vis] FATAL: cardA reduce Mc r=%d (m=%d nc=%d)\\n", r, m, nc); exit(1); }
        }
        {   // 3) f = Mc / mx  (与主机 float f = Mc / m 同式, 为的是【位级复现】主机路径)
            int r = api::elementwise_div_2d(g_vctx[dv], mBc, mxA, fAr, 1, nc, m, nc);
            if (r) { printf("[vis] FATAL: cardA div f r=%d\\n", r); exit(1); }
        }
        {   // 4) invMc = 1/Mc  (与主机 float invMc = 1.0f/Mc 同式)
            int r = api::elementwise_div_2d(g_vctx[dv], (const float*)g_vOn[dv], mBc, invM, 1, nc, 1, nc);
            if (r) { printf("[vis] FATAL: cardA div invMc r=%d\\n", r); exit(1); }
        }
        {   // 5) 折回行因子 = mx*invMc (与主机 mx[t] = m*invMc 同式)
            int r = api::elementwise_mul_2d(g_vctx[dv], mxA, invM, tmp, m, nc, 1, nc);
            if (r) { printf("[vis] FATAL: cardA mul 行因子 r=%d\\n", r); exit(1); }
        }
        {   // 6) 转置 [m,nc] -> [nc,m]
            int sh[2]; sh[0] = m; sh[1] = nc;
            int pm[2]; pm[0] = 1; pm[1] = 0;
            int r = api::transpose(g_vctx[dv], tmp, mFe, sh, pm, 2);
            if (r) { printf("[vis] FATAL: cardA transpose r=%d (m=%d nc=%d)\\n", r, m, nc); exit(1); }
        }
        {   // 7) An = A * f  (与主机 d[j] = r[j]*f 同式; A 视作[m*nc,cs], f 视作[m*nc,1])
            int r = api::elementwise_mul_2d(g_vctx[dv], A, fAr, An, m * nc, cs, m * nc, 1);
            if (r) { printf("[vis] FATAL: cardA mul An r=%d (m=%d n=%d)\\n", r, m * nc, cs); exit(1); }
        }
        size_t per = (size_t)m * N;
        int G = (int)(g_vCssz / (per * 4)); if (G < 1) G = 1; if (G > nc) G = nc;
        float* stk = (float*)g_vCs[dv];
        float* sct = (float*)g_vTc[dv];
        for (int c0 = 0; c0 < nc; c0 += G) {
            int c1 = c0 + G; if (c1 > nc) c1 = nc; int gg = c1 - c0;
            for (int c = c0; c < c1; c++) {
                const int8_t* Bc = bdev[(size_t)p * nc + c];
                int r = api::gemm_int8(g_vctx[dv], false, true, m, N, cs, 1.f,
                                       An + (size_t)c * cs, K, Bc, 127.f, cs, 0.f,
                                       stk + (size_t)(c - c0) * per, N);
                g_p_gemm += 0; g_p_n_gemm++;
                if (r) { printf("[vis] FATAL: cardA gemm r=%d (m=%d n=%d k=%d 块%d/%d)\\n", r, m, N, cs, c, nc); exit(1); }
            }
            if (xpu_wait()) { printf("[vis] FATAL: cardA wait\\n"); exit(1); }
            {   // 行因子 (块主序 [nc][m]) —— 同一算子, 因子来自卡上 reduce
                int r = api::elementwise_mul_2d(g_vctx[dv], stk, mFe + (size_t)c0 * m, sct, gg * m, N, gg * m, 1);
                if (r) { printf("[vis] FATAL: cardA mul 行 r=%d\\n", r); exit(1); }
            }
            for (int c = c0; c < c1; c++) {   // 列 scale (每块每行一个 scale)
                int r = api::elementwise_mul_2d(g_vctx[dv], sct + (size_t)(c - c0) * per,
                                                (const float*)rsdev[p] + (size_t)c * N,
                                                stk + (size_t)(c - c0) * per, m, N, 1, N);
                if (r) { printf("[vis] FATAL: cardA mul 列 r=%d\\n", r); exit(1); }
            }
            {   // 跨块 reduce SUM -> acc (多组时再 elementwise_add 累加)
                int xd[3]; xd[0] = gg; xd[1] = m; xd[2] = N;
                int rd[1]; rd[0] = 0;
                int r;
                if (c0 == 0) r = api::reduce(g_vctx[dv], stk, acc, xd, 3, rd, 1, api::REDUCE_SUM);
                else {
                    r = api::reduce(g_vctx[dv], stk, sec, xd, 3, rd, 1, api::REDUCE_SUM);
                    if (!r) r = api::elementwise_add(g_vctx[dv], acc, sec, acc, m * N);
                }
                if (r) { printf("[vis] FATAL: cardA 折回 r=%d\\n", r); exit(1); }
            }
        }
        if (!leave_on_card && !skip_d2h) {
            double t = vnow();
            if (xpu_memcpy(C + (size_t)r0 * N, acc, per * 4, XPU_DEVICE_TO_HOST)) { printf("[vis] FATAL: cardA D2H\\n"); exit(1); }
            xpu_wait();
            g_p_d2h += vnow() - t; g_p_n_d2h++;
        }
    }
}

// 第 19 轮原接口 (注意力路径唯一入口) —— 行为一字未改; 第 20 轮只多了 heap_head
//   heap_head >= 0: (B) 16 头输出堆叠, 写第 heap_head 段, 不 D2H
static void vgemm_cardA(const std::vector<signed char>& qpack, const std::vector<float>& rs,
                        int N, int K, int cs, const void* const* A_dev, int M,
                        float* C, int leave_on_card = 0, int heap_head = -1) {
    if (cs <= 0 || K % cs) { printf("[vis] FATAL: cardA 分块不整除 K=%d cs=%d\\n", K, cs); exit(1); }
    int nc = K / cs;
    if ((size_t)qpack.size() != (size_t)N * K) { printf("[vis] FATAL: cardA 打包尺寸 %zu != %d\\n", qpack.size(), N * K); exit(1); }
    if ((size_t)rs.size() != (size_t)nc * N) { printf("[vis] FATAL: cardA rs 尺寸 %zu != %d\\n", rs.size(), nc * N); exit(1); }
    if ((size_t)N * K > g_vBsz) { printf("[vis] FATAL: cardA Bd 越界 n=%d K=%d\\n", N, K); exit(1); }
    if ((size_t)nc * N * 4 > g_vRdsz) { printf("[vis] FATAL: cardA Rd 越界\\n"); exit(1); }
    if (A_dev[0] == nullptr || A_dev[1] == nullptr) { printf("[vis] FATAL: cardA A_dev 为空\\n"); exit(1); }
    for (int p = 0; p < 2; p++) {      // B / rs 每芯一份 (与 vgemm_dynB 同约定)
        xpu_set_device(p);
        double t = vnow();
        if (xpu_memcpy(g_vBd[p], qpack.data(), (size_t)N * K, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: cardA H2D B\\n"); exit(1); }
        if (xpu_memcpy(g_vRd[p], rs.data(), (size_t)nc * N * 4, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: cardA H2D rs\\n"); exit(1); }
        g_p_h2d += vnow() - t; g_p_n_h2d += 2;
    }
    std::vector<const int8_t*> bd((size_t)2 * nc, nullptr);
    for (int p = 0; p < 2; p++)
        for (int c = 0; c < nc; c++)
            bd[(size_t)p * nc + c] = (const int8_t*)g_vBd[p] + (size_t)c * (size_t)N * cs;
    const void* rd[2] = {g_vRd[0], g_vRd[1]};
    float* accd[2] = {nullptr, nullptr};
    int skip = 0;
    if (heap_head >= 0) {
        int Mh = (M + 1) / 2;
        if (g_vSlO[0] == nullptr || g_vSlO[1] == nullptr) { printf("[vis] FATAL: heap 堆叠缓冲未分配\\n"); exit(1); }
        if ((size_t)(heap_head + 1) * Mh * N * 4 > g_vSlOsz) { printf("[vis] FATAL: heap 越界\\n"); exit(1); }
        for (int p = 0; p < 2; p++) accd[p] = (float*)g_vSlO[p] + (size_t)heap_head * Mh * N;
        skip = 1;
    }
    vgemm_cardA_core(N, K, cs, bd.data(), rd, A_dev, M, C, leave_on_card, accd, skip, 0);
}

// ★ (A) 第 20 轮: FFN 的 Wdn 分块 gemm —— A = gelu 输出(含负值) 常驻卡上
//   K = Kp = 45*96 = 4320 (padding 布局), 与老路径 96+80 非均匀分块【逐位等价】(见 PROGRESS 第 20 章)
static void vgemm_ffn_dn(const VW& w, const void* const* A_dev, int M, float* C) {
    int Kp = g_vkpadn;
    if (Kp % 96) { printf("[vis] FATAL: FFN Kp=%d 不是 96 的倍数\\n", Kp); exit(1); }
    if (w.K != Kp || w.N != VNE) { printf("[vis] FATAL: Wdn 形状 N=%d K=%d != %d/%d\\n", w.N, w.K, VNE, Kp); exit(1); }
    int cs = 96, nc = Kp / cs;
    std::vector<const int8_t*> bd((size_t)2 * nc, nullptr);
    for (int p = 0; p < 2; p++)
        for (int c = 0; c < nc; c++)
            bd[(size_t)p * nc + c] = (const int8_t*)w.d[p] + w.coff[p][c];
    const void* rd[2] = {w.rsdev[0], w.rsdev[1]};
    vgemm_cardA_core(w.N, Kp, cs, bd.data(), rd, A_dev, M, C, 0, nullptr, 0, 1);
}

// ★ (A) 模式 1: 卡上 bias(广播加) + gelu ([m][Kp]; padding 尾部恒 0 ⇒ gelu(0)=0 仍为 0)
static void vffn_card_bias_gelu(int T, int il) {
    int Kp = g_vkpadn, Mh = (T + 1) / 2;
    for (int p = 0; p < 2; p++) {
        int dv = p, r0 = p * Mh, m = Mh; if (r0 + m > T) m = T - r0;
        if (m <= 0) continue;
        xpu_set_device(dv);
        float* up = (float*)g_vUpP[dv];
        float* gl = (float*)g_vAn[dv];       // 暂存 (g_vAn 在归一阶段才用)
        const float* bias = (const float*)g_vBup[dv] + (size_t)il * Kp;
        double t = vnow();
        int r = api::elementwise_add_2d(g_vctx[dv], (const float*)up, bias, gl, m, Kp, 1, Kp);
        if (r) { printf("[vis] FATAL: FFN 卡上 bias r=%d (m=%d Kp=%d)\\n", r, m, Kp); exit(1); }
        r = api::gelu(g_vctx[dv], (const float*)gl, (float*)up, m * Kp);
        if (r) { printf("[vis] FATAL: FFN 卡上 gelu r=%d (m=%d Kp=%d)\\n", r, m, Kp); exit(1); }
        if (xpu_wait()) { printf("[vis] FATAL: FFN bias/gelu wait\\n"); exit(1); }
        g_p_gelu += vnow() - t;
    }
}

// ★ (A) 模式 2: 主机 bias+gelu 后把 padding 布局的 Up 传到卡上
static void vffn_up_upload(int T, const float* Up) {
    int Kp = g_vkpadn, Mh = (T + 1) / 2;
    for (int p = 0; p < 2; p++) {
        int dv = p, r0 = p * Mh, m = Mh; if (r0 + m > T) m = T - r0;
        if (m <= 0) continue;
        xpu_set_device(dv);
        double t = vnow();
        if (xpu_memcpy(g_vUpP[dv], Up + (size_t)r0 * Kp, (size_t)m * Kp * 4, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: H2D FFN Up\\n"); exit(1); }
        xpu_wait();
        g_p_h2d += vnow() - t; g_p_n_h2d++;
    }
}

// ★ (A) 调试: 卡上 g_vUpP 的 [T][Kp] 落盘 (与 VIS_CARDFFN=0 的 l0_up 对照)
static void vffn_dump_up(int T, const char* dumpdir, const char* fn) {
    if (!dumpdir || !dumpdir[0]) return;
    int Kp = g_vkpadn, Mh = (T + 1) / 2;
    std::vector<float> h((size_t)T * Kp);
    for (int p = 0; p < 2; p++) {
        int dv = p, r0 = p * Mh, m = Mh; if (r0 + m > T) m = T - r0;
        if (m <= 0) continue;
        xpu_set_device(dv);
        if (xpu_memcpy(h.data() + (size_t)r0 * Kp, g_vUpP[dv], (size_t)m * Kp * 4, XPU_DEVICE_TO_HOST)) { printf("[vis] FATAL: D2H FFN Up(dump)\\n"); exit(1); }
        xpu_wait();
    }
    vdump(dumpdir, fn, h.data(), h.size());
}
"""
s = s.replace(old_fn, new_fn, 1)
nrep += 1

# ---------------------------------------------------------------- 5) init: 开关 + 缓冲
rep("""    g_vcarda = getenv("VIS_CARDA") ? atoi(getenv("VIS_CARDA")) : 1;   // ★ 第 19 轮: A 常驻卡上 (0=老路径, 仅对照)""",
"""    g_vcarda = getenv("VIS_CARDA") ? atoi(getenv("VIS_CARDA")) : 1;   // ★ 第 19 轮: A 常驻卡上 (0=老路径, 仅对照)
    // ★★ 第 20 轮开关 (默认全关) ★★
    g_vcardffn = getenv("VIS_CARDFFN") ? atoi(getenv("VIS_CARDFFN")) : 0;
    g_vheap    = getenv("VIS_HEAP")    ? atoi(getenv("VIS_HEAP"))    : 0;
    g_vkpad    = getenv("VIS_PAD")     ? atoi(getenv("VIS_PAD"))     : (g_vcardffn ? 1 : 0);
    {   // FFN padding 布局: 找 >= VFFN 的 (g_vch 整数倍) (4304 -> 4320 = 45*96)
        int kp = g_vkpad ? ((VFFN + g_vch - 1) / g_vch) * g_vch : VFFN;
        if (kp != VFFN && (g_vch <= 0 || kp % g_vch)) { printf("[vis] ★ FFN padding 不可用 (CH=%d), 退回老布局\\n", g_vch); kp = VFFN; g_vkpad = 0; }
        g_vkpadn = kp;
        if (kp != VFFN) printf("[vis] ★★ 第20轮 FFN padding 布局: VFFN %d -> Kp %d (=%d*%d, 尾部补 0) —— 与老 96+80 分块逐位等价\\n",
                                VFFN, kp, g_vch, kp / g_vch);
    }""")
rep("""    printf("[vis] ★ A 常驻卡上: %s —— 卡上 reduce(MAX)+elementwise_div_2d 逐块行归一, A 不再上下卡\\n",
           g_vcarda ? "ON (第19轮)" : "OFF (老路径: 主机归一 + A 上下卡)");""",
"""    printf("[vis] ★ A 常驻卡上: %s —— 卡上 reduce(MAX)+elementwise_div_2d 逐块行归一, A 不再上下卡\\n",
           g_vcarda ? "ON (第19轮)" : "OFF (老路径: 主机归一 + A 上下卡)");
    printf("[vis] ★★ 第 20 轮开关: VIS_CARDFFN=%d (%s)  VIS_HEAP=%d%s\\n", g_vcardffn,
           g_vcardffn == 1 ? "全卡 bias/gelu/归一/gemm" : (g_vcardffn == 2 ? "主机 bias/gelu + 卡上归一/gemm" : "关(第19轮路径)"),
           g_vheap, g_vheap ? " (16 头输出堆叠, D2H 32->2 次/层)" : "");""")

rep("""        g_vSesz = (size_t)(VMA_MAX / 2) * VKMAX * 4;
        if (xpu_malloc(&g_vSe[dv], g_vSesz)) { printf("[vis] FATAL: alloc 次 reduce chip%d\\n", dv); return 1; }""",
"""        g_vSesz = (size_t)(VMA_MAX / 2) * VKMAX * 4;
        if (xpu_malloc(&g_vSe[dv], g_vSesz)) { printf("[vis] FATAL: alloc 次 reduce chip%d\\n", dv); return 1; }
        // ★★ 第 20 轮: (A) FFN 卡上激活缓冲 + bias 副本;  (B) 16 头输出堆叠
        if (g_vcardffn) {
            g_vUpPsz = (size_t)(VMA_MAX / 2) * g_vkpadn * 4;
            if (xpu_malloc(&g_vUpP[dv], g_vUpPsz)) { printf("[vis] FATAL: alloc FFN UpP chip%d\\n", dv); return 1; }
            g_vBupsz = (size_t)VNL * g_vkpadn * 4;
            if (xpu_malloc(&g_vBup[dv], g_vBupsz)) { printf("[vis] FATAL: alloc FFN bias chip%d\\n", dv); return 1; }
            printf("[vis] (A) FFN 卡上激活缓冲 chip%d = %.1f MB (Kp=%d, m<=%d)\\n", dv, g_vUpPsz / 1048576.0, g_vkpadn, VMA_MAX / 2);
        }
        if (g_vheap) {
            g_vSlOsz = (size_t)VNH * (VMA_MAX / 2) * VDH * 4;
            if (xpu_malloc(&g_vSlO[dv], g_vSlOsz)) { printf("[vis] FATAL: alloc heap 堆叠 chip%d\\n", dv); return 1; }
            printf("[vis] (B) 16 头输出堆叠缓冲 chip%d = %.1f MB\\n", dv, g_vSlOsz / 1048576.0);
        }""")

rep("""        {   // 全 1 数组 (算 1/Mc 用)
            std::vector<float> ones((VKMAX / 8 + 8), 1.0f);
            if (xpu_memcpy(g_vOn[dv], ones.data(), g_vFisz, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: H2D ones chip%d\\n", dv); return 1; }
            xpu_wait();
        }""",
"""        {   // 全 1 数组 (算 1/Mc 用) + 全 -1 数组 (算 -minA 用, api::neg 不在 api.h 里)
            std::vector<float> ones((VKMAX / 8 + 8), 1.0f);
            if (xpu_memcpy(g_vOn[dv], ones.data(), g_vFisz, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: H2D ones chip%d\\n", dv); return 1; }
            if (xpu_malloc(&g_vMg[dv], g_vFisz)) { printf("[vis] FATAL: alloc -1 向量 chip%d\\n", dv); return 1; }
            std::vector<float> mgs((VKMAX / 8 + 8), -1.0f);
            if (xpu_memcpy(g_vMg[dv], mgs.data(), g_vFisz, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: H2D -1 向量 chip%d\\n", dv); return 1; }
            xpu_wait();
        }""")

# ---------------------------------------------------------------- 6) 权重装载: FFN padding
rep("""        VLOAD_WI(g_Wup,  "ffn_up",   VFFN, VNE);
        VLOAD_WI(g_Wdn,  "ffn_down", VNE, VFFN);""",
"""        #define VLOAD_WI_PAD(field, suffix, N_, K_, NP_, KP_) { \\
            snprintf(nm, sizeof(nm), "v.blk.%d.%s.weight", il, suffix); \\
            if (!g_vg.read_f32(nm, W)) return 1; \\
            if ((int)W.size() != (N_) * (K_)) { printf("[vis] FATAL: %s 形状 %zu != %d*%d\\n", nm, W.size(), (N_), (K_)); return 1; } \\
            if ((NP_) != (N_) || (KP_) != (K_)) { \\
                std::vector<float> Wp((size_t)(NP_) * (KP_), 0.f); \\
                for (int n = 0; n < (N_); n++) memcpy(&Wp[(size_t)n * (KP_)], &W[(size_t)n * (K_)], (size_t)(K_) * 4); \\
                vw_from_f32(Wp, (NP_), (KP_), field[il]); \\
            } else vw_from_f32(W, (N_), (K_), field[il]); }
        VLOAD_WI_PAD(g_Wup,  "ffn_up",   VFFN, VNE, g_vkpadn, VNE);
        VLOAD_WI_PAD(g_Wdn,  "ffn_down", VNE, VFFN, VNE, g_vkpadn);
        #undef VLOAD_WI_PAD""")
rep("""        VLOAD_V(g_bup, "ffn_up.bias"); VLOAD_V(g_bdn, "ffn_down.bias");""",
"""        VLOAD_V(g_bup, "ffn_up.bias"); VLOAD_V(g_bdn, "ffn_down.bias");
        if (g_vkpadn != VFFN && (int)g_bup[il].size() == VFFN) g_bup[il].resize(g_vkpadn, 0.f);   // ★ padding 尾部恒 0""")

# ---------------------------------------------------------------- 7) bias 副本上卡
rep("""    printf("[vis] 权重点卡完成: chip0 %.0f MB, chip1 %.0f MB\\n", g_vhbm[0] / 1048576.0, g_vhbm[1] / 1048576.0);
    g_vinit = 1;""",
"""    printf("[vis] 权重点卡完成: chip0 %.0f MB, chip1 %.0f MB\\n", g_vhbm[0] / 1048576.0, g_vhbm[1] / 1048576.0);
    if (g_vcardffn) {   // ★ (A): ffn_up.bias 的 padding 副本 (每层一份) 上卡
        std::vector<float> bp((size_t)VNL * g_vkpadn, 0.f);
        for (int il = 0; il < VNL; il++) memcpy(&bp[(size_t)il * g_vkpadn], g_bup[il].data(), (size_t)g_bup[il].size() * 4);
        for (int dv = 0; dv < 2; dv++) {
            xpu_set_device(dv);
            if (xpu_memcpy(g_vBup[dv], bp.data(), (size_t)VNL * g_vkpadn * 4, XPU_HOST_TO_DEVICE)) { printf("[vis] FATAL: H2D FFN bias\\n"); return 1; }
            xpu_wait();
        }
    }
    g_vinit = 1;""")

# ---------------------------------------------------------------- 8) 前向: FFN 分支
rep("""        vgemm(g_Wup[il], Y.data(), T, Up.data());
        if (il == 0) vdump(dumpdir, "l0_up.f32", Up.data(), Up.size());
        double tge = vnow();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < Up.size(); i++) Up[i] = vgelu(Up[i] + g_bup[il][i % VFFN]);
        g_p_gelu += vnow() - tge;
        vgemm(g_Wdn[il], Up.data(), T, Dn.data());""",
"""        const int Kp = g_vkpadn;   // ★ 第20轮: FFN 维度 (padding 布局, 尾部恒 0)
        if (g_vcardffn) {
            if (g_vcardffn == 1) {
                // ★ (A) 模式1: ffn_up 结果【留卡】→ 卡上 bias + gelu → 卡上 max|A| 归一 + 分块 gemm
                float* upd[2] = {(float*)g_vUpP[0], (float*)g_vUpP[1]};
                vgemm(g_Wup[il], Y.data(), T, Up.data(), 1, upd);
                if (il == 0 && dumpdir && dumpdir[0]) vffn_dump_up(T, dumpdir, "l0_up.f32");
                vffn_card_bias_gelu(T, il);
            } else {
                // ★ (A) 模式2: 主机 bias+gelu (与老路径逐位相同) → 上卡 → 卡上 max|A| 归一 + 分块 gemm
                vgemm(g_Wup[il], Y.data(), T, Up.data());
                if (il == 0 && dumpdir && dumpdir[0]) vdump(dumpdir, "l0_up.f32", Up.data(), Up.size());
                double th = vnow();
                #pragma omp parallel for schedule(static)
                for (int t = 0; t < T; t++) {
                    float* u = Up.data() + (size_t)t * Kp;
                    const float* b = g_bup[il].data();
                    for (int i = 0; i < Kp; i++) u[i] = vgelu(u[i] + b[i]);
                }
                g_p_gelu += vnow() - th;
                vffn_up_upload(T, Up.data());
            }
            const void* Ad2f[2] = {g_vUpP[0], g_vUpP[1]};
            vgemm_ffn_dn(g_Wdn[il], Ad2f, T, Dn.data());
        } else {
            // ---- 第 19 轮老路径 (默认); 只把 bias+gelu 的取模换成等价两重循环 (位级相同、更快) ----
            vgemm(g_Wup[il], Y.data(), T, Up.data());
            if (il == 0) vdump(dumpdir, "l0_up.f32", Up.data(), Up.size());
            double tge = vnow();
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < T; t++) {
                float* u = Up.data() + (size_t)t * Kp;
                const float* b = g_bup[il].data();
                for (int i = 0; i < Kp; i++) u[i] = vgelu(u[i] + b[i]);
            }
            g_p_gelu += vnow() - tge;
            vgemm(g_Wdn[il], Up.data(), T, Dn.data());
        }""")

# ---------------------------------------------------------------- 9) (B) 头循环
rep("""            if (useCA) {   // ★ 第 19 轮: A = softmax 输出常驻卡上, 卡上逐块行归一 ⇒ 零 A 往返
                vpack_T(vt.data(), T, VDH, csO, qv, rs3, csB, koB);      // B[d][t] = V[t][d]
                const void* Ad2[2] = {g_vCr[0], g_vCr[1]};
                vgemm_cardA(qv, rs3, VDH, T, csO, Ad2, T, C2.data());    // C2[qt][d]
            } else {""",
"""            if (useCA) {   // ★ 第 19 轮: A = softmax 输出常驻卡上, 卡上逐块行归一 ⇒ 零 A 往返
                vpack_T(vt.data(), T, VDH, csO, qv, rs3, csB, koB);      // B[d][t] = V[t][d]
                const void* Ad2[2] = {g_vCr[0], g_vCr[1]};
                // ★ (B) VIS_HEAP=1: 每头结果写卡上堆叠缓冲第 h 段, 不 D2H (头循环后一次取回)
                vgemm_cardA(qv, rs3, VDH, T, csO, Ad2, T, C2.data(), 0, g_vheap ? h : -1);
            } else {""")
rep("""            #pragma omp parallel for schedule(static)
            for (int t = 0; t < T; t++)
                for (int d = 0; d < VDH; d++)
                    Ao[(size_t)(h * VDH + d) + (size_t)VNE * t] = C2[(size_t)t * VDH + d];
        }
        g_p_attn += vnow() - tat;""",
"""            if (!g_vheap) {
                #pragma omp parallel for schedule(static)
                for (int t = 0; t < T; t++)
                    for (int d = 0; d < VDH; d++)
                        Ao[(size_t)(h * VDH + d) + (size_t)VNE * t] = C2[(size_t)t * VDH + d];
            }
        }
        if (g_vheap) {   // ★ (B) 头循环结束: 每芯【一次】D2H 取回 16 头输出 (32 -> 2 次/层)
            int Mh = (T + 1) / 2;
            std::vector<float> C2s((size_t)2 * VNH * Mh * VDH);
            for (int p = 0; p < 2; p++) {
                xpu_set_device(p);
                double t = vnow();
                if (xpu_memcpy(&C2s[(size_t)p * VNH * Mh * VDH], g_vSlO[p], (size_t)VNH * Mh * VDH * 4, XPU_DEVICE_TO_HOST)) { printf("[vis] FATAL: D2H heap\\n"); exit(1); }
                xpu_wait();
                g_p_d2h += vnow() - t; g_p_n_d2h++;
            }
            #pragma omp parallel for schedule(static)
            for (int h = 0; h < VNH; h++)
                for (int t = 0; t < T; t++) {
                    int p = (t < Mh) ? 0 : 1, i = t - p * Mh;
                    const float* sc = &C2s[((size_t)p * VNH * Mh + (size_t)h * Mh + i) * VDH];
                    for (int d = 0; d < VDH; d++) Ao[(size_t)(h * VDH + d) + (size_t)VNE * t] = sc[d];
                }
        }
        g_p_attn += vnow() - tat;""")

# ---------------------------------------------------------------- 10) Up 缓冲按 Kp
rep("""    std::vector<float> Sm((size_t)T * T), C2((size_t)T * VDH), Ao((size_t)VNE * T), Up((size_t)T * VFFN), Dn((size_t)T * VNE);""",
"""    std::vector<float> Sm((size_t)T * T), C2((size_t)T * VDH), Ao((size_t)VNE * T), Up((size_t)T * g_vkpadn), Dn((size_t)T * VNE);""")

open(OUT, "w", encoding="utf-8").write(s)
print("replacements:", nrep, "| bytes:", len(orig), "->", len(s))
