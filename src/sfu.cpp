#include "sfu.hpp"

// A small constant for numerical stability in layer normalization
const sfu_data_t EPSILON = 1e-5;

void sfu_softmax(sfu_data_t* input, sfu_data_t* output, int size) {
    // Phase 1: Find max for numerical stability
    sfu_data_t max_val = input[0];
    for (int i = 1; i < size; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // Phase 2: Calculate exp and sum
    sfu_data_t sum_exp = 0;
    for (int i = 0; i < size; ++i) {
#ifdef __SYNTHESIS__
        output[i] = hls::exp(input[i] - max_val);
#else
        output[i] = std::exp(input[i] - max_val);
#endif
        sum_exp += output[i];
    }

    // Phase 3: Normalize
    for (int i = 0; i < size; ++i) {
        output[i] /= sum_exp;
    }
}

void sfu_layer_norm(sfu_data_t* input, sfu_data_t* output, int size) {
    // Phase 1: Calculate mean
    sfu_data_t sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += input[i];
    }
    sfu_data_t mean = sum / size;

    // Phase 2: Calculate variance
    sfu_data_t variance_sum = 0;
    for (int i = 0; i < size; ++i) {
        sfu_data_t diff = input[i] - mean;
        variance_sum += diff * diff;
    }
    sfu_data_t variance = variance_sum / size;

    // Phase 3: Normalize
    sfu_data_t inv_sqrt;
#ifdef __SYNTHESIS__
    inv_sqrt = hls::rsqrt(variance + EPSILON);
#else
    inv_sqrt = 1.0f / std::sqrt(variance + EPSILON);
#endif

    for (int i = 0; i < size; ++i) {
        output[i] = (input[i] - mean) * inv_sqrt;
    }
}

void sfu_rms_norm(sfu_data_t* input, sfu_data_t* output, int size) {
    // Phase 1: Calculate sum of squares
    sfu_data_t sum_squares = 0;
    for (int i = 0; i < size; ++i) {
        sum_squares += input[i] * input[i];
    }
    sfu_data_t mean_squares = sum_squares / size;

    // Phase 2: Calculate inverse square root
    sfu_data_t inv_rms;
#ifdef __SYNTHESIS__
    inv_rms = hls::rsqrt(mean_squares + EPSILON);
#else
    inv_rms = 1.0f / std::sqrt(mean_squares + EPSILON);
#endif

    // Phase 3: Normalize
    for (int i = 0; i < size; ++i) {
        output[i] = input[i] * inv_rms;
    }
}

void sfu_element_add(sfu_data_t* input_a, sfu_data_t* input_b, sfu_data_t* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = input_a[i] + input_b[i];
    }
}

void sfu_element_mul(sfu_data_t* input_a, sfu_data_t* input_b, sfu_data_t* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = input_a[i] * input_b[i];
    }
}

void sfu_silu(sfu_data_t* input, sfu_data_t* output, int size) {
    for (int i = 0; i < size; ++i) {
        sfu_data_t val = input[i];
#ifdef __SYNTHESIS__
        sfu_data_t sigmoid = 1.0f / (1.0f + hls::exp(-val));
#else
        sfu_data_t sigmoid = 1.0f / (1.0f + std::exp(-val));
#endif
        output[i] = val * sigmoid;
    }
}
