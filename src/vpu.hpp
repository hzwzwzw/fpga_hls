#ifndef VPU_HPP
#define VPU_HPP

// Use HLS-specific types only during high-level synthesis
#ifdef __SYNTHESIS__
#include <ap_int.h>
// Define HLS-specific types for synthesis
using vpu_data_t = ap_int<8>;
using vpu_acc_t = ap_int<32>;
#else
// For C++ simulation, use standard integer types
#include <cstdint>
using vpu_data_t = int8_t;
using vpu_acc_t = int32_t;
#endif

// Define the size of the vector processing unit
const int VPU_SIZE = 16;

/**
 * @brief Vector Processing Unit (VPU)
 * 
 * This function computes the dot product of two vectors, which is the fundamental
 * operation of the VPU as described in the FlightLLM paper.
 * 
 * @param vec_a Input vector A of size VPU_SIZE.
 * @param vec_b Input vector B of size VPU_SIZE.
 * @return vpu_acc_t The accumulated result of the dot product.
 */
vpu_acc_t vpu(
    vpu_data_t vec_a[VPU_SIZE],
    vpu_data_t vec_b[VPU_SIZE]
);

#endif // VPU_HPP
