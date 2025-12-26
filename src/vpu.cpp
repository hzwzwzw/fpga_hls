#include "vpu.hpp"

vpu_acc_t vpu(
    vpu_data_t vec_a[VPU_SIZE],
    vpu_data_t vec_b[VPU_SIZE]
) {
    // The accumulator for the dot product result
    vpu_acc_t acc = 0;

    // The DOT_PRODUCT_LOOP computes the dot product of the two input vectors.
    // The HLS PIPELINE directive is used to unroll the loop and allow for parallel
    // execution of the loop iterations, which is essential for high-performance hardware.
    DOT_PRODUCT_LOOP:
    for (int i = 0; i < VPU_SIZE; ++i) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
        acc += vec_a[i] * vec_b[i];
    }

    return acc;
}
