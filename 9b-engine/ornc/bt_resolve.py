#!/usr/bin/env python3
# =============================================================================
# bt_resolve.py —— 把 crashbt.so 写下的崩渍日志还原成函数名+偏移
#   用法: python3 bt_resolve.py <crash.log> <elf>
#   原理: 日志里有 base=0x... 与每帧 addr=0x...;  PIE 下 符号值 = addr - base;
#         用 ELF 的 .symtab (orn3 "not stripped") 做最近符号匹配 => 函数名+偏移。
#         若 ELF 带调试信息, 再尝试 addr2line 拿 文件:行。
# =============================================================================
import re, subprocess, sys, bisect

def symtab(elf):
    out = subprocess.run(["nm", "-n", "--defined-only", elf],
                         capture_output=True, text=True).stdout
    syms = []
    for ln in out.splitlines():
        p = ln.split()
        if len(p) >= 3:
            try:
                syms.append((int(p[0], 16), p[2]))
            except ValueError:
                pass
    syms.sort()
    return syms

def nearest(syms, off):
    i = bisect.bisect_right(syms, (off, "\xff")) - 1
    if i < 0:
        return "?", 0
    a, nm_ = syms[i]
    return nm_, off - a

def main():
    log = sys.argv[1] if len(sys.argv) > 1 else "/home/caden/ornc/orn3_crash.log"
    elf = sys.argv[2] if len(sys.argv) > 2 else "/home/caden/orn_engine/orn3"
    txt = open(log, "rb").read().decode("utf-8", "replace")
    syms = symtab(elf)
    print("符号表条目 = %d (%s)" % (len(syms), elf))
    blocks = txt.split("[crashbt] pid=")
    for b in blocks[1:]:
        head = "[crashbt] pid=" + b.splitlines()[0]
        base = 0
        m = re.search(r"base=0x([0-9a-f]+)", head)
        if m:
            base = int(m.group(1), 16)
        print("\n--- " + head)
        for fm in re.finditer(r"frame\s+(\d+)\s+addr=0x([0-9a-f]+)\s+sym_off=0x([0-9a-f]+)", b):
            idx, addr, off = int(fm.group(1)), int(fm.group(2), 16), int(fm.group(3), 16)
            if off > 0x4000000:          # 不在主映像里(共享库) => 主映像符号表无意义
                print("  #%-2d ---- 共享库帧 addr=0x%x (查上面 backtrace_symbols_fd 行里的模块名)" % (idx, addr))
                continue
            f, d = nearest(syms, off)
            line = "  #%-2d %s+0x%x   (sym_off=0x%x addr=0x%x)" % (idx, f, d, off, addr)
            try:
                r = subprocess.run(["addr2line", "-f", "-C", "-e", elf, hex(off)],
                                   capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
                if r and not r[0].startswith("??"):
                    line += "  <= %s:%s" % (r[1] if len(r) > 1 else "?", r[0])
            except Exception:
                pass
            print(line)

if __name__ == "__main__":
    main()
