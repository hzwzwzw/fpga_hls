#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <cassert>
#include "../src/sfu.hpp"

const sfu_data_t TEST_EPSILON = 1e-4;

// Helper to check if two float values are close
void check_close(sfu_data_t actual, sfu_data_t expected, const char* test_name, int index) {
    if (std::abs(actual - expected) > TEST_EPSILON) {
        std::cout << "❌ " << test_name << " FAILED at index " << index << "!" << std::endl;
        std::cout << "   Expected: " << expected << std::endl;
        std::cout << "   Actual:   " << actual << std::endl;
        exit(1);
    }
}

bool test_softmax() {
    const char* test_name = "Softmax Test";
    std::cout << "\n--- " << test_name << " ---" << std::endl;
    sfu_data_t input[SFU_VECTOR_SIZE];
    sfu_data_t output[SFU_VECTOR_SIZE];
    sfu_data_t expected_sum = 0;

    for(int i=0; i<SFU_VECTOR_SIZE; ++i) input[i] = (i % 10) * 0.5;

    sfu_softmax(input, output, SFU_VECTOR_SIZE);

    sfu_data_t sum = 0;
    for(int i=0; i<SFU_VECTOR_SIZE; ++i) sum += output[i];

    check_close(sum, 1.0, test_name, -1); // Sum of softmax should be 1.0

    std::cout << "✅ " << test_name << " PASSED!" << std::endl;
    return true;
}

bool test_layer_norm() {
    const char* test_name = "LayerNorm Test";
    std::cout << "\n--- " << test_name << " ---" << std::endl;
    sfu_data_t input[SFU_VECTOR_SIZE];
    sfu_data_t output[SFU_VECTOR_SIZE];

    for(int i=0; i<SFU_VECTOR_SIZE; ++i) input[i] = (i % 20) * 0.2 - 1.0;

    sfu_layer_norm(input, output, SFU_VECTOR_SIZE);

    sfu_data_t mean = 0, variance = 0;
    for(int i=0; i<SFU_VECTOR_SIZE; ++i) mean += output[i];
    mean /= SFU_VECTOR_SIZE;

    for(int i=0; i<SFU_VECTOR_SIZE; ++i) variance += (output[i] - mean) * (output[i] - mean);
    variance /= SFU_VECTOR_SIZE;

    check_close(mean, 0.0, test_name, -1); // Mean of a normalized vector should be 0
    check_close(variance, 1.0, test_name, -2); // Variance of a normalized vector should be 1

    std::cout << "✅ " << test_name << " PASSED!" << std::endl;
    return true;
}

bool test_element_add() {
    const char* test_name = "ElementAdd Test";
    std::cout << "\n--- " << test_name << " ---" << std::endl;
    sfu_data_t input_a[SFU_VECTOR_SIZE], input_b[SFU_VECTOR_SIZE], output[SFU_VECTOR_SIZE];

    for(int i=0; i<SFU_VECTOR_SIZE; ++i) {
        input_a[i] = i * 1.5;
        input_b[i] = i * -0.5;
    }

    sfu_element_add(input_a, input_b, output, SFU_VECTOR_SIZE);

    for(int i=0; i<SFU_VECTOR_SIZE; ++i) {
        check_close(output[i], i * 1.0, test_name, i);
    }

    std::cout << "✅ " << test_name << " PASSED!" << std::endl;
    return true;
}

bool test_silu() {
    const char* test_name = "SiLU Test";
    std::cout << "\n--- " << test_name << " ---" << std::endl;
    sfu_data_t input[SFU_VECTOR_SIZE], output[SFU_VECTOR_SIZE];

    for(int i=0; i<SFU_VECTOR_SIZE; ++i) input[i] = (i - SFU_VECTOR_SIZE/2) * 0.1;

    sfu_silu(input, output, SFU_VECTOR_SIZE);

    for(int i=0; i<SFU_VECTOR_SIZE; ++i) {
        float val = (i - SFU_VECTOR_SIZE/2) * 0.1;
        float expected = val / (1.0 + std::exp(-val));
        check_close(output[i], expected, test_name, i);
    }

    std::cout << "✅ " << test_name << " PASSED!" << std::endl;
    return true;
}


int main() {
    try {
        test_softmax();
        test_layer_norm();
        test_element_add();
        test_silu();
        std::cout << "\nAll SFU tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        // The exit(1) call in check_close will terminate, but this is good practice.
        return 1;
    }
}
