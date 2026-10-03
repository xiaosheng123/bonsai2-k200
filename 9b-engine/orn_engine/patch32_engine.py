#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# patch32_engine.py — 第32轮引擎改动 (orn3.cb30.cpp -> orn3.cb32.cpp)
#   ① 冷预填充闸门 (cb_admit, 入槽即判, 超限立刻 __SREJ__, 绝不预填充)
#   ② 只读探针 cb_lcp_probe + 新命令 !CPREF (网关入队前问"真实 tok / 可复用 tok")
#   ③ 检查点淘汰: 为大前缀专门留一个槽 (最长前缀优先保留)
#   ④ 命令读取健壮化: 半截帧丢弃 + 重同步 + 对端关闭不退出 + 空闲重附着 FIFO + len= 校验
#   ⑤ 预填充期间抽空命令队列(白名单): !SSTAT/!SDEL 等照常答复
#   每个替换都断言命中次数, 命中数不对就整体失败 (绝不"大概改上了")。
# =============================================================================
import sys, os, shutil

SRC = "/home/caden/orn_engine/orn3.cb30.cpp"
DST = "/home/caden/orn_engine/orn3.cb32.cpp"

orig = open(SRC, encoding="utf-8").read()
s = orig
applied = []


def rep(name, old, new, expect=1):
    global s
    n = s.count(old)
    if n != expect:
        print("FATAL[%s]: 命中 %d 次 (期望 %d) ⇒ 中止, 不改文件" % (name, n, expect))
        sys.exit(2)
    s = s.replace(old, new)
    applied.append(name)
    print("  ok  %s" % name)


def rep_span(name, start_anchor, end_anchor, new, expect=1):
    """从 start_anchor 到 end_anchor(含) 的整段替换"""
    global s
    i0 = s.find(start_anchor)
    if i0 < 0 or s.count(start_anchor) != expect:
        print("FATAL[%s]: 起始锚点命中 %d 次" % (name, s.count(start_anchor)))
        sys.exit(2)
    i1 = s.find(end_anchor, i0)
    if i1 < 0:
        print("FATAL[%s]: 结束锚点未找到" % name)
        sys.exit(2)
    i1 += len(end_anchor)
    s = s[:i0] + new + s[i1:]
    applied.append(name)
    print("  ok  %s (替换 %d 字节)" % (name, i1 - i0))


# ---------------------------------------------------------------- 0) include
rep("include-sys-stat",
    "#include <poll.h>\n",
    "#include <poll.h>\n#include <sys/stat.h>\n#include <fcntl.h>\n#include <csignal>\n")

rep("sigpipe-ignore",
    "int main(int argc, char **argv) {\n",
    "int main(int argc, char **argv) {\n"
    "    // ★★ 第32轮: 忽略 SIGPIPE ★★\n"
    "    //   对端(网关)被 kill 后, 引擎的 printf(stdout 是一个已无读者的管道) 默认会收到 SIGPIPE\n"
    "    //   ⇒ 引擎被信号打死 (只剩宿主重建 VM/80s 重启这条路)。忽略后 printf 只返回 EPIPE,\n"
    "    //   引擎继续服务; 配合下面的\"读到 EOF 不退出 + 空闲重附着 FIFO\"才真正做到\"客户端中断不用重启\"。\n"
    "    signal(SIGPIPE, SIG_IGN);\n")

# ------------------------------------------------- 1) 冷预填充上限 + !CPREF 用的全局
rep("g_maxcold-decl",
    "    static double g_cb_d2h = 0;\n",
    "    static double g_cb_d2h = 0;\n"
    "    // ★★ 第32轮: 冷预填充上限 (tok)。0 = 关闭该保护。\n"
    "    static int g_maxcold = 3000;\n")

rep("g_maxcold-init",
    '        if (n == 0) printf("[orn] ★ 多槽位未启用 (K200_SLOTS=0): 只有老单会话 \\"@\\" 协议可用\\n");\n',
    '        if (n == 0) printf("[orn] ★ 多槽位未启用 (K200_SLOTS=0): 只有老单会话 \\"@\\" 协议可用\\n");\n'
    '        {   const char *em = getenv("K200_MAX_COLD_PREFILL_TOK");\n'
    '            if (em && atoi(em) >= 0) g_maxcold = atoi(em);\n'
    '            printf("[orn] ★ 冷预填充上限 = %d tok (K200_MAX_COLD_PREFILL_TOK; 0=关闭保护; 超限请求在【入队前】就被网关拒绝, 引擎侧同样兜底拒收)\\n", g_maxcold);\n'
    '        }\n')

# ------------------------------------------------- 2) 只读探针 (插在 cb_emit_fp 之前)
PROBE = r'''    // ========================================================================
    // ★★ 第32轮: 【只读】最长可回滚前缀探针 ★★
    //   与 cb_admit 的命中判定是【同一份代码】(cb_admit 也调用它) ⇒ 不会漂移。
    //   绝不改任何状态: 不 restore、不动 last/hits ⇒ 可以在【网关入队之前】安全地问它。
    //   命中规则(与第30轮一致): 检查点 token 序列必须是本轮 prompt 的前缀 (逐 token 相等),
    //   且图像依赖相容 (前缀含 image_pad ⇒ 指纹必须相同); 不要求 sid 相同 (跨会话)。
    // ========================================================================
    auto cb_lcp_probe = [&](const std::vector<int> &ids, uint64_t imgid,
                            int *hitslot, int *hitlcp, int *lcpall) {
        int slot = -1, bl = 0, al = 0;
        if (!ids.empty()) {
            for (int ci = 0; ci < (int)g_mck.size(); ci++) {
                MCk &C = g_mck[(size_t)ci];
                if (!C.valid || C.pos <= 0) continue;
                if (!(C.imgdep == 0 || C.imgdep == imgid)) continue;
                int m = ((int)ids.size() < C.pos) ? (int)ids.size() : C.pos;
                int n = 0; while (n < m && C.ids[(size_t)n] == ids[(size_t)n]) n++;
                if (n > al) al = n;
                if (n < C.pos) continue;                 // 序列不是本轮前缀 ⇒ 这个状态回滚不了
                if (n > bl) { bl = n; slot = ci; }
            }
        }
        if (hitslot) *hitslot = slot;
        if (hitlcp)  *hitlcp  = bl;
        if (lcpall)  *lcpall  = al;
        return slot;
    };
'''
rep("probe-insert", "    auto cb_emit_fp = [&](CbSlot &S) {", PROBE + "    auto cb_emit_fp = [&](CbSlot &S) {")

# ------------------------------------------------- 3) cb_admit 改用探针 (同一份判定)
SCAN_NEW = '''        int ck_hit = -1, ck_lcp = 0;
        // ★★ 第32轮: 命中判定搬到 cb_lcp_probe (只读探针) —— !CPREF 的入队前咨询与这里的
        //   实际命中必须是【同一份代码】, 否则"网关说能复用、引擎说不能"会互相打脸。
        if (!S.ids.empty()) ck_hit = cb_lcp_probe(S.ids, S.imgid, &ck_hit, &ck_lcp, &S.lcp_all);
        if (ck_hit >= 0) {
            MCk &C = g_mck[(size_t)ck_hit];
            if (C.sid != S.sid)
                printf("[orn] ★ 跨会话前缀命中: 检查点 sid=%s(%.8s) ← 本请求 sid=%.8s, 只按内容复用\\n",
                       C.sid.c_str(), C.sid.c_str(), S.sid.c_str());
            S.conv = C.conv; S.S = C.S; S.logits = C.logits; S.imgused = C.img_used;
            size_t nr = (size_t)C.pos * NKV * HD;
            for (int l = 0; l < NLAYER; l++) if (!g_lay[l].recr) {
                memcpy(S.Kc[(size_t)l].data(), C.Kc[(size_t)l].data(), nr * 4);
                memcpy(S.Vc[(size_t)l].data(), C.Vc[(size_t)l].data(), nr * 4);
            }
            S.pos = C.pos; S.reused = C.pos;
            C.last = ++g_mck_tick; C.hits++;     // ★ 命中计数 (淘汰时保护被复用过的检查点)
        }'''
rep_span("admit-scan-to-probe",
         "        int ck_hit = -1, ck_lcp = 0;\n",
         "                C.last = ++g_mck_tick; C.hits++;     // ★ 命中计数 (淘汰时保护被复用过的检查点)\n            }\n        }",
         SCAN_NEW)

# ------------------------------------------------- 4) 冷预填充闸门 (预填充之前)
GUARD = '''        // ★★ 第32轮: 【失败要快】冷预填充闸门 —— 在真正开算之前判, 绝不把十几分钟的冷预填充
        //   塞进唯一引擎通道。台账(真实事故): 用户 14685 tok 的 agent 载荷 ⇒ 引擎保尾砍头到 8192
        //   ⇒ 冷预填充 ~13 分钟 ⇒ 该槽被独占 ⇒ 之后所有请求排队 ⇒ 用户观感="服务死了" ✗。
        //   判据: 本次【真正要算的新增 token】= ids - 可复用前缀; 超过 K200_MAX_COLD_PREFILL_TOK
        //   ⇒ 立刻 __SREJ__(网关转成明确错误), 不预填充、不占槽。0 = 关闭该保护。
        if (g_maxcold > 0) {
            int coldtok = (int)S.ids.size() - S.reused;
            if (coldtok > g_maxcold) {
                cb_unbind();
                printf("[orn] ★ 拒绝(cold-prefill-too-large): cid=%d prompt=%d tok 可复用=%d 冷预填充=%d > 上限 %d"
                       " ⇒ 不预填充, 立刻报错 (绝不占着唯一引擎通道跑十几分钟)\\n",
                       S.reqid, (int)S.ids.size(), S.reused, coldtok, g_maxcold);
                printf("__SREJ__ %d cold-prefill-too-large tok=%d reused=%d cold=%d max=%d\\n",
                       S.reqid, (int)S.ids.size(), S.reused, coldtok, g_maxcold);
                fflush(stdout);
                S.active = false; S.used = false; S.feed = false;
                return;
            }
        }
'''
rep("guard-admit",
    "        double t0 = NOWS_();\n        if (S.reused == 0) {",
    GUARD + "        double t0 = NOWS_();\n        if (S.reused == 0) {")

# ------------------------------------------------- 5) 检查点淘汰: 为大前缀留一个槽
EVICT_OLD = '''            int besth = 1 << 30; sl = -1;
            for (size_t i = 0; i < g_mck.size(); i++) if (g_mck[i].hits < besth) besth = g_mck[i].hits;
            for (size_t i = 0; i < g_mck.size(); i++) {
                if (g_mck[i].hits != besth) continue;
                if (sl < 0) { sl = (int)i; continue; }
                MCk &B = g_mck[(size_t)sl];
                if (g_mck[i].pos < B.pos) sl = (int)i;
                else if (g_mck[i].pos == B.pos && g_mck[i].last < B.last) sl = (int)i;
            }
            if (sl < 0) sl = 0;'''
EVICT_NEW = '''            // ★★ 第32轮: 为大前缀【专门留一个槽】★★
            //   长前缀(DSH 的 system+工具块)命中一次的收益 = 数千 token 的冷预填充; 表只有 4 槽,
            //   若被短前缀(每个请求自己会打 3 个点)挤掉 ⇒ 跨会话缓存形同虚设 ✗。
            //   规则: 当前【最长前缀】那条被保护, 不参与淘汰; 只有"本次要存的更长"时才允许淘汰它。
            int prot = -1;
            {   int bestp = -1;
                for (size_t i = 0; i < g_mck.size(); i++)
                    if (g_mck[i].valid && g_mck[i].pos > bestp) { bestp = g_mck[i].pos; prot = (int)i; }
                if (prot >= 0 && spos > g_mck[(size_t)prot].pos) prot = -1;
            }
            int besth = 1 << 30; sl = -1;
            for (size_t i = 0; i < g_mck.size(); i++) {
                if ((int)i == prot) continue;
                if (g_mck[i].hits < besth) besth = g_mck[i].hits;
            }
            for (size_t i = 0; i < g_mck.size(); i++) {
                if ((int)i == prot) continue;
                if (g_mck[i].hits != besth) continue;
                if (sl < 0) { sl = (int)i; continue; }
                MCk &B = g_mck[(size_t)sl];
                if (g_mck[i].pos < B.pos) sl = (int)i;
                else if (g_mck[i].pos == B.pos && g_mck[i].last < B.last) sl = (int)i;
            }
            if (sl < 0) sl = 0;'''
rep("evict-protect-long", EVICT_OLD, EVICT_NEW)

# ------------------------------------------------- 6) 命令读取健壮化 (插在 cb_admit 之前)
PUMP = r'''    // ========================================================================
    // ★★ 第32轮: 命令读取健壮化 + 预填充期间抽空命令队列 ★★
    //   ① 半截命令 (对端写到一半就死): 读到 EOF/HUP ⇒ 【丢弃不完整帧】+ 重置解析状态; 旧行为是
    //      把半截残留与下一条命令拼成一条 ⇒ 首字符不合法 ⇒ 新命令被【静默丢弃】⇒ 网关干等
    //      (观感 = 引擎卡死 ✗)。
    //   ② 对端关闭【绝不退出引擎】: 旧行为 `if (n <= 0) break;` = 直接跳出服务循环 ⇒ 引擎自杀;
    //      网关一被 kill/重启, 引擎与其热缓存一起没了 (事故现场: /health 不响应, 只能 stop/start 80s ✗)。
    //      新行为: 置 stdin_eof, 200ms 节流轮询; 路径被重建(旧网关会 remove+mkfifo)时重新 open 附着。
    //   ③ 预填充期间也抽空命令队列(白名单 !SDEL/!SSTAT/!SMCK/!SADD/!SPAUS/!SRESUM/!CPREF):
    //      /health 在长预填充期间照常答复, 客户端中断能立刻取消该槽; 白名单外的命令(图像塔等)
    //      推迟到预填充结束后再处理 —— 绝不在预填充中间去抢卡/动共享状态 ✗。
    // ========================================================================
    static std::vector<std::string> g_defer;
    std::string inbuf;
    std::vector<char> rbuf(65536);
    bool stdin_eof = false;
    const char *fifo_path = getenv("K200_FIFO");
    if (!fifo_path || !*fifo_path) fifo_path = "/home/caden/ornc/.ornq_cb.fifo";
    std::function<void(std::string &)> handle_cmd;   // ★ 实体在服务循环里赋值 (预填充泵也要调用)
    auto pump_cmds = [&](int to_ms, int prefill_mode) {
        struct pollfd pf; pf.fd = 0; pf.events = POLLIN; pf.revents = 0;
        int pr = poll(&pf, 1, to_ms);
        if (pr < 0 && errno != EINTR) { stdin_eof = true; inbuf.clear(); }
        if (pr > 0 && (pf.revents & (POLLIN | POLLHUP | POLLERR))) {
            ssize_t n = read(0, &rbuf[0], rbuf.size());
            if (n < 0) {
                if (errno != EINTR && errno != EAGAIN) { stdin_eof = true; inbuf.clear(); }
            } else if (n == 0) {                       // ★ 对端关闭: 丢弃不完整帧, 绝不退出
                printf("[orn] ★ 命令 FIFO 读到 EOF/HUP (客户端中断/关闭): 丢弃不完整帧 %d 字节 + 重置解析状态;"
                       " 空闲时自动重新附着 %s (无需重启引擎)\\n", (int)inbuf.size(), fifo_path);
                fflush(stdout);
                stdin_eof = true; inbuf.clear();
            } else {
                if (stdin_eof) printf("[orn] ★ 命令 FIFO 对端恢复, 继续服务 (引擎未重启)\\n");
                stdin_eof = false;
                inbuf.append(&rbuf[0], (size_t)n);
            }
        }
        size_t nl;
        while ((nl = inbuf.find('\n')) != std::string::npos) {
            std::string s = inbuf.substr(0, nl);
            inbuf.erase(0, nl + 1);
            while (!s.empty() && (s[s.size() - 1] == '\r' || s[s.size() - 1] == ' ')) s.erase(s.size() - 1);
            if (s.empty()) continue;
            if (prefill_mode && s[0] == '!') {
                bool safe = (s.compare(0, 5, "!SDEL") == 0) || (s.compare(0, 6, "!SSTAT") == 0) ||
                            (s.compare(0, 5, "!SMCK") == 0) || (s.compare(0, 5, "!SADD") == 0) ||
                            (s.compare(0, 6, "!SPAUS") == 0) || (s.compare(0, 7, "!SRESUM") == 0) ||
                            (s.compare(0, 6, "!CPREF") == 0);
                if (!safe) {
                    printf("[orn] ★ 预填充期间推迟命令(不在预填充中间抢卡): %.40s\\n", s.c_str());
                    fflush(stdout);
                    g_defer.push_back(s);
                    continue;
                }
            }
            handle_cmd(s);
        }
        if (inbuf.size() > ((size_t)8 << 20)) inbuf.clear();     // 防御: 超长无换行的垃圾
    };
'''
rep("pump-insert", "    // ---- 入槽 + 预填充 (+ 前缀复用) ----", PUMP + "    // ---- 入槽 + 预填充 (+ 前缀复用) ----")

# ------------------------------------------------- 7) 预填充循环里抽空命令队列
PUMP_LOOP_OLD = '''            for (size_t i = (size_t)S.reused; i < S.ids.size(); ) {
                if (S.pos >= MAXT) {   // ★ 第30轮: 入口已"保尾砍头", 这里正常不该发生 (兜底 + 告警)
                    printf("[orn] ★ WARN: 槽%d 预填充触到 MAXT=%d (入口保尾砍头未生效?)\\n", (int)(&S - &g_slots[0]), MAXT);
                    S.ids.resize(i); break; }
                size_t remain = S.ids.size() - i;'''
PUMP_LOOP_NEW = '''            unsigned _pf_pump = 0;
            for (size_t i = (size_t)S.reused; i < S.ids.size(); ) {
                if (S.pos >= MAXT) {   // ★ 第30轮: 入口已"保尾砍头", 这里正常不该发生 (兜底 + 告警)
                    printf("[orn] ★ WARN: 槽%d 预填充触到 MAXT=%d (入口保尾砍头未生效?)\\n", (int)(&S - &g_slots[0]), MAXT);
                    S.ids.resize(i); break; }
                // ★★ 第32轮: 预填充【期间】也抽空命令队列 —— ① /health(!SSTAT) 照常答复 (旧行为:
                //   预填充把命令循环整段占住 ⇒ 网关 /health 与后续请求全像"服务死"); ② 客户端中断
                //   (!SDEL) 能立刻取消本槽, 不再空跑到 MAXT。
                if (++_pf_pump > 0u) {   // ★ 每批之后都抽空一次: 深冷预填充一个批可达数秒, 8 批一次会让 /health 等近 1 分钟
                    pump_cmds(0, 1);
                    if (!S.used) {
                        cb_unbind();
                        printf("[orn] ★ 槽%d cid=%d 预填充被取消(客户端中断 !SDEL): 停在 %d/%d tok, 立即释放\\n",
                               (int)(&S - &g_slots[0]), S.reqid, S.pos, (int)S.ids.size());
                        fflush(stdout);
                        return;
                    }
                }
                size_t remain = S.ids.size() - i;'''
rep("prefill-pump", PUMP_LOOP_OLD, PUMP_LOOP_NEW)

# ------------------------------------------------- 8) !CPREF 命令
CPREF = r'''                } else if (s.compare(0, 6, "!CPREF") == 0) {
                    // ★★ 第32轮: 【冷预填充代价咨询】—— 网关在 !SADD 之前问, 只读、不占槽。★★
                    //   为什么必须问引擎: token 数要用【引擎自己的 tokenizer】的真实值 —— 字符估算
                    //   对中文/数字误差可达 3~4 倍 (中文 1 字可能 1~3 token; 数字 1 token/字符) ✗。
                    std::vector<std::string> tk; std::string r = s.substr(6);
                    {   size_t i = 0;
                        while (i < r.size()) {
                            while (i < r.size() && r[i] == ' ') i++;
                            size_t j = i;
                            while (j < r.size() && r[j] != ' ') j++;
                            if (j > i) tk.push_back(r.substr(i, j - i));
                            i = j;
                        }
                    }
                    std::string b64; uint64_t cimg = 0; int has_img = 0; size_t want_len = 0; bool has_len = false;
                    for (size_t k = 0; k < tk.size(); k++) {
                        const std::string &t = tk[k];
                        if (t.rfind("b64:", 0) == 0) b64 = t.substr(4);
                        else if (t.rfind("img=", 0) == 0) { cimg = (uint64_t)strtoull(t.c_str() + 4, NULL, 16); has_img = 1; }
                        else if (t.rfind("len=", 0) == 0) { want_len = (size_t)strtoul(t.c_str() + 4, NULL, 10); has_len = true; }
                    }
                    static const char *T64c = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                    int rev2[256]; for (int i = 0; i < 256; i++) rev2[i] = -1;
                    for (int i = 0; i < 64; i++) rev2[(unsigned char)T64c[i]] = i;
                    std::string pr; int val = 0, bits = 0;
                    for (size_t i = 0; i < b64.size(); i++) {
                        if (b64[i] == '=') break;
                        int c = rev2[(unsigned char)b64[i]]; if (c < 0) continue;
                        val = ((val << 6) | c) & 0xFFFFFF; bits += 6;
                        if (bits >= 8) { bits -= 8; pr += (char)((val >> bits) & 0xFF); }
                    }
                    if (has_len && pr.size() != want_len) {
                        printf("[orn] ★ WARN: CPREF 丢弃不完整帧 (声明 %zu 字节, 实收 %zu) —— 按探针失败回报\n",
                               want_len, pr.size());
                        printf("__CPREF__ tok=-1 raw=-1 reused=0 cold=-1 coldmax=%d err=truncated\n", g_maxcold);
                        fflush(stdout);
                    } else {
                        std::chrono::steady_clock::time_point ct0 = std::chrono::steady_clock::now();
                        std::vector<int> ids = g_tk.encode(pr);
                        double tk_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ct0).count();
                        int raw = (int)ids.size();
                        if ((int)ids.size() > MAXT) head_trim_keep_tail(ids, MAXT);
                        int haspad = 0;
                        if (g_img_pad >= 0) for (size_t i = 0; i < ids.size(); i++) if (ids[(size_t)i] == g_img_pad) { haspad = 1; break; }
                        int hs = -1, hl = 0, la = 0;
                        if (!haspad || has_img) cb_lcp_probe(ids, cimg, &hs, &hl, &la);   // 带图却没给指纹 ⇒ 保守: 不认复用
                        int reused = (hs >= 0) ? hl : 0;
                        int cold = (int)ids.size() - reused;
                        printf("[orn] ★ CPREF 实测(引擎 tokenizer %.1f ms): raw=%d tok, 保尾砍头后=%d, 可复用=%d,"
                               " 冷预填充=%d (上限 %d)\\n", tk_ms, raw, (int)ids.size(), reused, cold, g_maxcold);
                        printf("__CPREF__ tok=%d raw=%d reused=%d cold=%d coldmax=%d lcp=%d slot=%d haspad=%d tk_ms=%.1f\\n",
                               (int)ids.size(), raw, reused, cold, g_maxcold, la, hs, haspad, tk_ms);
                        fflush(stdout);
                    }
'''
rep("cpref-cmd", '                } else if (s.compare(0, 6, "!SSTAT") == 0) {', CPREF + '                } else if (s.compare(0, 6, "!SSTAT") == 0) {')

# ------------------------------------------------- 9) !SADD: len= 校验 + 半截帧拒收
rep("sadd-len-decl",
    "                    std::string sid, b64; uint64_t want_img = 0; int solo = 0;\n",
    "                    std::string sid, b64; uint64_t want_img = 0; int solo = 0;\n"
    "                    size_t want_len = 0; bool has_len = false;   // ★ 第32轮: 帧完整性校验\n")
rep("sadd-len-parse",
    '                        else if (t.rfind("solo=", 0) == 0) solo = atoi(t.c_str() + 5);\n',
    '                        else if (t.rfind("solo=", 0) == 0) solo = atoi(t.c_str() + 5);\n'
    '                        else if (t.rfind("len=", 0) == 0) { want_len = (size_t)strtoul(t.c_str() + 4, NULL, 10); has_len = true; }\n')
rep("sadd-len-check",
    '                    if (g_slots.empty()) { printf("__SREJ__ %d no-slots\\n", cid); fflush(stdout); return; }\n',
    '                    if (has_len && pr.size() != want_len) {   // ★ 半截帧: 绝不拿半截 prompt 去算\n'
    '                        printf("[orn] ★ WARN: 丢弃不完整命令帧 (cid=%d 声明 %zu 字节, 实收 %zu) —— 不入槽\\n",\n'
    '                               cid, want_len, pr.size());\n'
    '                        printf("__SREJ__ %d truncated-frame got=%zu want=%zu\\n", cid, pr.size(), want_len);\n'
    '                        fflush(stdout); return;\n'
    '                    }\n'
    '                    if (g_slots.empty()) { printf("__SREJ__ %d no-slots\\n", cid); fflush(stdout); return; }\n')

# ------------------------------------------------- 10) 非法帧: 不再静默丢弃 + 重同步
BAD_OLD = "            if (s.size() < 3 || s[0] != '@') return;\n"
BAD_NEW = '''            if (s.size() < 3 || s[0] != '@') {
                // ★★ 第32轮: 非法/半截帧【不再静默丢弃】★★
                //   旧行为: 直接 return ⇒ 若上一帧是半截(对端中途死掉)残留, 与这一帧拼在一起时
                //   新命令被静默吞掉 ⇒ 网关一直等 (观感 = 引擎卡死 ✗)。
                //   新行为: ① 先在行内【重同步】到最后一个命令起点再解析一次 (半截 b64 会被 len=
                //   校验拦下); ② 仍然不合法就大声告警 —— 绝不悄悄吞命令。
                size_t p = std::string::npos;
                static const char *mk[7] = {"!SADD ", "!CPREF ", "!SDEL ", "!SSTAT", "!SMCK", "!VIS ", "!VISR"};
                for (int k = 0; k < 7; k++) {
                    size_t q = s.rfind(mk[k]);
                    if (q != std::string::npos && q > 0 && (p == std::string::npos || q > p)) p = q;
                }
                if (p != std::string::npos) {
                    printf("[orn] ★ WARN: 半截残留帧 + 新命令拼在一起 (%d 字节) ⇒ 重同步到偏移 %d 的命令\\n",
                           (int)s.size(), (int)p);
                    fflush(stdout);
                    std::string s2 = s.substr(p);
                    handle_cmd(s2);
                    return;
                }
                printf("[orn] ★ WARN: 丢弃非法命令帧 (%d 字节, 首字节=%02x): %.40s\\n",
                       (int)s.size(), s.empty() ? 0 : (unsigned char)s[0], s.c_str());
                fflush(stdout);
                return;
            }
'''
rep("bad-frame-resync", BAD_OLD, BAD_NEW)

# ------------------------------------------------- 11) handle_cmd 变成 std::function 赋值
rep("handle-cmd-assign",
    "auto handle_cmd = [&](std::string &s) {",
    "handle_cmd = [&](std::string &s) {")

# ------------------------------------------------- 12) 主循环重写
LOOP_OLD = '''        std::string inbuf;
        std::vector<char> rbuf(65536);
        for (;;) {
            bool busy = g_cb_run && cb_busy();
            struct pollfd pf; pf.fd = 0; pf.events = POLLIN; pf.revents = 0;
            int pr = poll(&pf, 1, busy ? 0 : -1);
            if (pr < 0) { if (errno == EINTR) continue; break; }
            if (pr > 0 && (pf.revents & (POLLIN | POLLHUP))) {
                ssize_t n = read(0, &rbuf[0], rbuf.size());
                if (n <= 0) break;
                inbuf.append(&rbuf[0], (size_t)n);
            }
            size_t nl;
            while ((nl = inbuf.find('\\n')) != std::string::npos) {
                std::string s = inbuf.substr(0, nl);
                inbuf.erase(0, nl + 1);
                while (!s.empty() && (s[s.size() - 1] == '\\r' || s[s.size() - 1] == ' ')) s.erase(s.size() - 1);
                if (!s.empty()) handle_cmd(s);
            }
            if (inbuf.size() > ((size_t)8 << 20)) inbuf.clear();     // 防御: 超长无换行的垃圾
            if (g_cb_run) {
                if (!g_squeue.empty()) cb_drain_queue();
                if (cb_busy()) cb_step();
            }
            fflush(stdout);
        }'''
LOOP_NEW = '''        for (;;) {
            while (!g_defer.empty()) {        // ★ 预填充期间被推迟的命令 (图像塔等) 在这里补做
                std::string sd = g_defer.front(); g_defer.erase(g_defer.begin());
                printf("[orn] ★ 预填充结束, 补做被推迟的命令: %.40s\\n", sd.c_str());
                handle_cmd(sd);
            }
            bool busy = g_cb_run && cb_busy();
            if (!stdin_eof) {
                pump_cmds(busy ? 0 : -1, 0);   // 空闲时阻塞等命令; 忙时非阻塞抽空
            } else {
                pump_cmds(200, 0);             // ★ 对端不在: 200ms 节流轮询, 绝不空转/绝不退出
                struct stat s0, s1;
                bool need_reopen = (fstat(0, &s0) != 0) || (stat(fifo_path, &s1) != 0) ||
                                   (s0.st_ino != s1.st_ino) || (s0.st_dev != s1.st_dev);
                if (need_reopen && !(g_cb_run && cb_busy())) {
                    int nf = open(fifo_path, O_RDONLY);      // 路径被重建(旧网关 remove+mkfifo) ⇒ 重开
                    if (nf >= 0) {
                        dup2(nf, 0); if (nf != 0) close(nf);
                        printf("[orn] ★ 重新附着命令 FIFO %s (客户端中断/重连; 引擎与其热缓存都不需要重启)\\n", fifo_path);
                        fflush(stdout);
                    } else usleep(200000);
                }
            }
            if (g_cb_run) {
                if (!g_squeue.empty()) cb_drain_queue();
                if (cb_busy()) cb_step();
            }
            fflush(stdout);
        }'''
rep("main-loop", LOOP_OLD, LOOP_NEW)

open(DST, "w", encoding="utf-8").write(s)
print("== 已写出 %s (%d 字节, 原 %d) ==" % (DST, len(s), len(orig)))
print("== 应用的改动 %d 项: %s ==" % (len(applied), ", ".join(applied)))
