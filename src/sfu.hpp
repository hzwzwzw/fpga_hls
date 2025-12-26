#ifndef SFU_HPP
#define SFU_HPP

#include <cmath>

// For HLS, we would use hls::half or ap_fixed. For C++ simulation, we use float.
#ifdef __SYNTHESIS__
#include "hls_half.h"
#include "hls_math.h"
using sfu_data_t = hls::half;
#else
#include <cstdint>
using sfu_data_t = float;
#endif

// Define a representative vector size for SFU operations
const int SFU_VECTOR_SIZE = 128;

/**
 * @brief Performs Softmax on a vector.
 * Softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j.
 * A numerically stable version is used.
 */
void sfu_softmax(sfu_data_t* input, sfu_data_t* output, int size);

/**
 * @brief Performs Layer Normalization on a vector.
 * LayerNorm(x_i) = (x_i - mean) / sqrt(variance + epsilon)
 */
void sfu_layer_norm(sfu_data_t* input, sfu_data_t* output, int size);

/**
 * @brief Performs element-wise addition of two vectors.
 */
void sfu_element_add(sfu_data_t* input_a, sfu_data_t* input_b, sfu_data_t* output, int size);

/**
 * @brief Performs SiLU (Sigmoid Linear Unit) activation on a vector.
 * SiLU(x) = x * sigmoid(x)
 */
void sfu_silu(sfu_data_t* input, sfu_data_t* output, int size);


#endif // SFU_HPP
