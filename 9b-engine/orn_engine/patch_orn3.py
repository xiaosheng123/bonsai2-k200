import re
with open("/home/caden/orn_engine/orn3.cb.cpp", "r") as f:
    content = f.read()

# 1. Add extern declaration after line 28
lines = content.split("\n")
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    if i == 27:  # 0-indexed, line 28
        new_lines.append("extern void run_delta_net_batch_merged_vec(int clusters, int cores, int T, float *S, const float *packed_in, float *outT);")

content = "\n".join(new_lines)

# 2. Replace the call
content = content.replace(
    "run_delta_net_batch_merged(8, 16, 1,",
    "run_delta_net_batch_merged_vec(8, 16, 1,"
)

with open("/home/caden/orn_engine/orn3.cb.cpp", "w") as f:
    f.write(content)

print("Done")