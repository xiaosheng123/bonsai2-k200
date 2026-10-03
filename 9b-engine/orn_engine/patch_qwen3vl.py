#!/usr/bin/env python3
# 给 qwen3vl.cpp 打临时探针: MTMD_DUMP_STAGE=<k> 时把图输出换成中间张量, 靠官方
# MTMD_DEBUG_EMBEDDINGS 落盘 (图输出是真正物化的张量, 数据可信 —— 回调里回读中间节点不可信)
#   1 = patch+pos_emb [1152, n_pos]
#   2..28 = 第 (k-2) 层输出
#   99 = post_ln 输出
import sys, shutil, os
P = "/home/caden/llamacpp/tools/mtmd/models/qwen3vl.cpp"
BAK = P + ".preimg"

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--revert":
        shutil.copy(BAK, P); print("reverted"); return
    if not os.path.exists(BAK):
        shutil.copy(P, BAK); print("backup ->", BAK)
    src = open(BAK).read()
    a1 = "    // loop over layers\n    for (int il = 0; il < n_layer; il++) {"
    assert src.count(a1) == 1
    src = src.replace(a1, "    std::vector<ggml_tensor *> dump_stack;   // ★TEMP\n" + a1, 1)
    a2 = "        inpL = cur;\n    }"
    assert src.count(a2) == 1
    src = src.replace(a2, "        inpL = cur;\n        dump_stack.push_back(cur);   // ★TEMP\n    }", 1)
    a3 = "    // multimodal projection\n    ggml_tensor * embeddings = inpL;"
    assert src.count(a3) == 1
    hook = """    // ★TEMP: 可选把图输出换成中间张量 (便于逐段对拍)
    if (const char * st = std::getenv("MTMD_DUMP_STAGE")) {
        int s = atoi(st);
        ggml_tensor * out = nullptr;
        if (s == 1) out = inp;
        else if (s == 99) out = inpL;
        else if (s >= 2 && s <= 1 + (int) dump_stack.size()) out = dump_stack[s - 2];
        if (out) { ggml_build_forward_expand(gf, out); return gf; }
    }

"""
    src = src.replace(a3, hook + a3, 1)
    open(P, "w").write(src)
    print("patched qwen3vl.cpp")

main()
