/* ============================================================================
 * crashbt.c —— LD_PRELOAD 崩溃回溯垫片 (零算式风险: 只装信号处理器 + 打印回溯)
 *
 *   为什么需要它:
 *     - /proc/sys/kernel/core_pattern 是 root 的 apport 管道, caden 无 sudo, 改不了;
 *     - apport 日志实测记录: "core limit 0" + "executable does not belong to a
 *       package, ignoring"  =>  自家编译的 orn3 崩溃时既无 core 也无堆栈;
 *     - 本机没有 gdb / eu-stack / lldb。
 *   所以把回溯能力做进进程自身: 崩溃瞬间把 PC/栈帧地址写盘, 事后用 nm 符号表还原。
 *
 *   安全边界:
 *     - 只拦截 SIGSEGV/SIGABRT/SIGBUS/SIGILL/SIGFPE (SIGKILL 不可拦, safe_run 仍能秒杀);
 *     - 打印完恢复 SIG_DFL 并 raise(), 进程依旧以原信号退出 => safe_run 的 rc 语义不变;
 *     - 不 hook malloc/free/任何数值路径, 不修改任何全局状态 => 推理结果逐字节不受影响。
 * ==========================================================================*/
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>
#include <time.h>
#include <link.h>
#include <sys/syscall.h>
#include <execinfo.h>

static int          crashfd = -1;
static unsigned long g_base = 0;
static const char  *g_log  = "/home/caden/ornc/orn3_crash.log";

/* 主可执行映像的加载基址 (PIE: 运行地址 = 基址 + 符号值) */
static int phdr_cb(struct dl_phdr_info *info, size_t sz, void *data)
{
    (void)sz; (void)data;
    if (info->dlpi_name == NULL || info->dlpi_name[0] == '\0') {
        g_base = (unsigned long)info->dlpi_addr;
        return 1;                 /* 只取主映像 */
    }
    return 0;
}

static void wboth(const char *s)
{
    size_t n = strlen(s);
    if (crashfd >= 0) { ssize_t r = write(crashfd, s, n); (void)r; }
    ssize_t r2 = write(2, s, n); (void)r2;
}

static void on_crash(int sig)
{
    char buf[256];
    int  n;

    n = snprintf(buf, sizeof buf,
                 "\n=== [crashbt] pid=%d tid=%ld signal=%d(%s) time=%ld base=0x%lx ===\n",
                 (int)getpid(), (long)syscall(SYS_gettid), sig,
                 (sig > 0 && sig < NSIG && strsignal(sig)) ? strsignal(sig) : "?",
                 (long)time(NULL), g_base);
    if (n > 0) { if (crashfd >= 0) { ssize_t r = write(crashfd, buf, n); (void)r; } ssize_t r2 = write(2, buf, n); (void)r2; }

    void *bt[64];
    int m = backtrace(bt, 64);
    for (int i = 0; i < m; i++) {
        n = snprintf(buf, sizeof buf, "[crashbt] frame %2d addr=0x%016lx sym_off=0x%lx\n",
                     i, (unsigned long)bt[i],
                     g_base ? (unsigned long)bt[i] - g_base : 0UL);
        if (crashfd >= 0) { ssize_t r = write(crashfd, buf, n); (void)r; }
        ssize_t r2 = write(2, buf, n); (void)r2;
    }
    if (crashfd >= 0) backtrace_symbols_fd(bt, m, crashfd);
    backtrace_symbols_fd(bt, m, 2);
    wboth("=== [crashbt] 解析: python3 /home/caden/ornc/bt_resolve.py "
          "/home/caden/ornc/orn3_crash.log /home/caden/orn_engine/orn3 ===\n");

    signal(sig, SIG_DFL);
    raise(sig);
    _exit(128 + sig);
}

__attribute__((constructor)) static void crashbt_init(void)
{
    const char *lg = getenv("K200_CRASHBT_LOG");
    if (lg && *lg) g_log = lg;
    crashfd = open(g_log, O_WRONLY | O_CREAT | O_APPEND, 0644);

    dl_iterate_phdr(phdr_cb, NULL);

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_crash;
    sa.sa_flags   = SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    int sigs[] = { SIGSEGV, SIGABRT, SIGBUS, SIGILL, SIGFPE };
    for (unsigned i = 0; i < sizeof(sigs) / sizeof(sigs[0]); i++)
        sigaction(sigs[i], &sa, NULL);
}
