#!/usr/bin/env python3
# 给 clip.cpp 打一个临时探针: MTMD_DEBUG_TENSORS=<dir> 时把图中每个具名节点落盘
# 用 [int32 type][int32 ne0..ne3][raw] 格式; 结束用 --revert 还原
import sys, shutil, os, re
P = "/home/caden/llamacpp/tools/mtmd/clip.cpp"
BAK = P + ".preimg"
MARK = "// ★TEMP-TENSOR-DUMP★"

BLOCK = MARK + r"""
    if (const char * tdir = std::getenv("MTMD_DEBUG_TENSORS")) {
        FILE * idx = fopen((std::string(tdir) + "/index.txt").c_str(), "w");
        const int n_nodes = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n_nodes; i++) {
            ggml_tensor * t = ggml_graph_node(gf, i);
            if (t->name[0] == '\0') continue;
            std::string nm(t->name);
            if (nm.find("attn_out") == std::string::npos && nm.find("layer_out") == std::string::npos &&
                nm.find("norm_w")    == std::string::npos && nm.find("Qcur")      == std::string::npos &&
                nm.find("patch_bias")== std::string::npos && nm.find("inp_pos_emb") == std::string::npos &&
                nm.find("ffn_inp_normed") == std::string::npos && nm.find("embeddings") == std::string::npos) continue;
            for (auto & c : nm) if (c == '/') c = '_';
            size_t nb = ggml_nbytes(t);
            std::vector<uint8_t> buf(nb);
            ggml_backend_tensor_get(t, buf.data(), 0, nb);
            FILE * f = fopen((std::string(tdir) + "/" + nm + ".bin").c_str(), "wb");
            if (!f) continue;
            int32_t hdr[5] = { (int32_t)t->type, (int32_t)t->ne[0], (int32_t)t->ne[1], (int32_t)t->ne[2], (int32_t)t->ne[3] };
            fwrite(hdr, sizeof(hdr), 1, f);
            fwrite(buf.data(), 1, nb, f);
            fclose(f);
            if (idx) fprintf(idx, "%s %d %d %d %d %zu\n", nm.c_str(), hdr[1], hdr[2], hdr[3], hdr[4], nb);
        }
        if (idx) fclose(idx);
    }

    // Debug: dump final embeddings if MTMD_DEBUG_EMBEDDINGS is set"""

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--revert":
        shutil.copy(BAK, P)
        print("reverted", P)
        return
    src = open(P).read()
    if MARK in src:
        print("already patched")
        return
    if not os.path.exists(BAK):
        shutil.copy(P, BAK)
        print("backup ->", BAK)
    anchor = "    // Debug: dump final embeddings if MTMD_DEBUG_EMBEDDINGS is set"
    assert src.count(anchor) == 1, src.count(anchor)
    src = src.replace(anchor, BLOCK, 1)
    open(P, "w").write(src)
    print("patched", P)

main()
