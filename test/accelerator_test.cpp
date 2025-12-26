#include <iostream>
#include <vector>
#include "../src/accelerator.hpp"

// Test dimensions - must be a multiple of TILE size for this simple test
const int TEST_M = 32;
const int TEST_K = 32;
const int TEST_N = 32;

// For C++ simulation, we need to allocate memory for our large matrices
large_matrix_t matrix_a_g;
large_matrix_t matrix_b_g;
large_result_matrix_t matrix_c_actual_g;
large_result_matrix_t matrix_c_expected_g;


void init_matrices() {
    for (int i = 0; i < TEST_M; ++i) {
        for (int j = 0; j < TEST_K; ++j) {
            matrix_a_g[i][j] = (i + j) % 7;
        }
    }
    for (int i = 0; i < TEST_K; ++i) {
        for (int j = 0; j < TEST_N; ++j) {
            matrix_b_g[i][j] = (i * 3 + j * 2) % 11;
        }
    }
}

void calculate_expected() {
    for (int i = 0; i < TEST_M; ++i) {
        for (int j = 0; j < TEST_N; ++j) {
            matrix_c_expected_g[i][j] = 0;
            for (int k = 0; k < TEST_K; ++k) {
                matrix_c_expected_g[i][j] += matrix_a_g[i][k] * matrix_b_g[k][j];
            }
        }
    }
}

int main() {
    std::cout << "--- Accelerator Top-Level Test (Tiled Matrix Multiplication) ---" << std::endl;
    std::cout << "Matrix Dimensions: " << TEST_M << "x" << TEST_K << " * " << TEST_K << "x" << TEST_N << std::endl;
    std::cout << "Tile Size: " << TILE_ROWS << "x" << TILE_COLS << std::endl;

    // 1. Initialize input matrices
    init_matrices();

    // 2. Run the accelerator function
    flight_llm_accelerator(matrix_a_g, matrix_b_g, matrix_c_actual_g, TEST_M, TEST_K, TEST_N);

    // 3. Calculate the expected result using a standard algorithm
    calculate_expected();

    // 4. Compare the results
    for (int i = 0; i < TEST_M; ++i) {
        for (int j = 0; j < TEST_N; ++j) {
            if (matrix_c_actual_g[i][j] != matrix_c_expected_g[i][j]) {
                std::cout << "❌ Test FAILED at C(" << i << "," << j << ")!" << std::endl;
                std::cout << "   Expected: " << matrix_c_expected_g[i][j] << std::endl;
                std::cout << "   Actual:   " << matrix_c_actual_g[i][j] << std::endl;
                return 1; // Failure
            }
        }
    }

    std::cout << "✅ Test PASSED!" << std::endl;
    return 0; // Success
}
