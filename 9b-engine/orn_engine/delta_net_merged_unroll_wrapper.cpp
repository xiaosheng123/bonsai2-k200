#include <xpu/runtime.h>

extern "C" void xpu_kernel__Z20dn_k_T_merged_unrollPfPKfS_i(void **args, int num_args, void *stream);

void run_delta_net_batch_merged_unroll(int clusters, int cores, int T, float *S, const float *packed_in, float *outT) {
    void *args[] = {&S, &packed_in, &outT, &T};
    xpu_kernel__Z20dn_k_T_merged_unrollPfPKfS_i(args, 4, nullptr);
    xpu_wait();
}