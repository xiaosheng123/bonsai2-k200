#!/usr/bin/env python3
# 放松 clip_encode 的「输出 token 数必须等于预期」断言 (探针用)
import sys
P = "/home/caden/llamacpp/tools/mtmd/clip.cpp"
old = """        if (n_tokens_out != expected_n_tokens_out) {
            LOG_ERR("%s: expected output %d tokens, got %d\\n", __func__, expected_n_tokens_out, n_tokens_out);
            GGML_ABORT("Invalid number of output tokens");
        }"""
new = """        if (n_tokens_out != expected_n_tokens_out) {
            LOG_ERR("%s: expected output %d tokens, got %d (探针: 只告警)\\n", __func__, expected_n_tokens_out, n_tokens_out);
        }"""
s = open(P).read()
assert s.count(old) == 1, s.count(old)
open(P, "w").write(s.replace(old, new, 1))
print("relaxed")
