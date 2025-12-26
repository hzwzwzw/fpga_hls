#include <iostream>
#include "../src/vpu.hpp"

// Function to calculate the expected dot product for verification
vpu_acc_t calculate_expected(vpu_data_t vec_a[VPU_SIZE], vpu_data_t vec_b[VPU_SIZE]) {
    vpu_acc_t expected_acc = 0;
    for (int i = 0; i < VPU_SIZE; ++i) {
        expected_acc += vec_a[i] * vec_b[i];
    }
    return expected_acc;
}

int main() {
    // --- Test Data ---
    // Create two sample vectors for the test.
    vpu_data_t test_vec_a[VPU_SIZE];
    vpu_data_t test_vec_b[VPU_SIZE];

    // Initialize the vectors with some values.
    for (int i = 0; i < VPU_SIZE; ++i) {
        test_vec_a[i] = i + 1; // e.g., {1, 2, 3, ..., 16}
        test_vec_b[i] = i % 4; // e.g., {0, 1, 2, 3, 0, 1, ...}
    }

    // --- Execution ---
    // Call the VPU function to get the actual result.
    vpu_acc_t actual_result = vpu(test_vec_a, test_vec_b);

    // --- Verification ---
    // Calculate the expected result.
    vpu_acc_t expected_result = calculate_expected(test_vec_a, test_vec_b);

    // --- Report ---
    std::cout << "--- VPU Unit Test ---" << std::endl;
    std::cout << "Expected Result: " << expected_result << std::endl;
    std::cout << "Actual Result:   " << actual_result << std::endl;

    if (actual_result == expected_result) {
        std::cout << "✅ Test PASSED!" << std::endl;
        return 0; // Return 0 on success
    } else {
        std::cout << "❌ Test FAILED!" << std::endl;
        return 1; // Return 1 on failure
    }
}
