#!/usr/bin/env python3
# 给 clip.cpp 打临时探针 v2: MTMD_DEBUG_TENSORS=<dir> 时注册 eval 回调,
# 在【每个节点刚算完】的时刻落盘 (旧版在图算完后回读, 缓冲区已被复用 => 数据是垃圾)
# 用法: patch_clip2.py  |  patch_clip2.py --revert
import sys, shutil, os
P = "/home/caden/llamacpp/tools/mtmd/clip.cpp"
BAK = P + ".preimg"
MARK = "clip_dump_eval_cb"

CB = r"""
// ★TEMP-TENSOR-DUMP★ 每个节点算完立刻落盘 (eval callback)
static bool clip_dump_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true;
    const char * dir = (const char *) user_data;
    if (dir == nullptr || dir[0] == '\0') return true;
    std::string nm(t->name);
    if (nm.empty()) return true;
    if (nm.find("attn_out") == std::string::npos && nm.find("layer_out") == std::string::npos &&
        nm.find("norm_w")    == std::string::npos && nm.find("Qcur")      == std::string::npos &&
        nm.find("patch_bias")== std::string::npos && nm.find("inp_pos_emb") == std::string::npos &&
        nm.find("ffn_inp_normed") == std::string::npos) return true;
    for (auto & c : nm) if (c == '/' || c == ' ') c = '_';
    static std::string cur_dir;
    cur_dir = dir;
    FILE * f = fopen((cur_dir + "/" + nm + ".bin").c_str(), "wb");
    if (!f) return true;
    size_t nb = ggml_nbytes(t);
    std::vector<uint8_t> buf(nb);
    ggml_backend_tensor_get(t, buf.data(), 0, nb);
    int32_t hdr[5] = { (int32_t)t->type, (int32_t)t->ne[0], (int32_t)t->ne[1], (int32_t)t->ne[2], (int32_t)t->ne[3] };
    fwrite(hdr, sizeof(hdr), 1, f);
    fwrite(buf.data(), 1, nb, f);
    fclose(f);
    return true;
}
"""

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--revert":
        shutil.copy(BAK, P); print("reverted"); return
    src = open(P).read()
    if not os.path.exists(BAK):
        shutil.copy(P, BAK); print("backup ->", BAK)
    src = open(BAK).read()      # 从干净版本开始
    anchor = "struct clip_ctx {"
    i = src.index(anchor)
    src = src[:i] + CB + "\n" + src[i:]
    anchor2 = """        if (ctx_params.cb_eval != nullptr) {
            ggml_backend_sched_set_eval_callback(sched.get(), ctx_params.cb_eval, ctx_params.cb_eval_user_data);
        }"""
    assert src.count(anchor2) == 1, src.count(anchor2)
    rep = anchor2 + """ else if (std::getenv("MTMD_DEBUG_TENSORS") != nullptr) {
            ggml_backend_sched_set_eval_callback(sched.get(), clip_dump_eval_cb, (void *) std::getenv("MTMD_DEBUG_TENSORS"));
        }"""
    src = src.replace(anchor2, rep, 1)
    open(P, "w").write(src)
    print("patched v2")

main()
