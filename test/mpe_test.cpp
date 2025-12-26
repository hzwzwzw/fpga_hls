#include <iostream>
#include "../src/mpe.hpp"

// --- Helper functions for testing ---

// Initialize a matrix with some values
void init_matrix(matrix_t m) {
    for (int i = 0; i < MPE_ROWS; ++i) {
        for (int j = 0; j < MPE_COLS; ++j) {
            m[i][j] = (i + j) % 5; // Example values
        }
    }
}

// Initialize a vector with some values
void init_vector(vector_t v) {
    for (int i = 0; i < MPE_COLS; ++i) {
        v[i] = i % 3; // Example values
    }
}

// Print a matrix for debugging
void print_matrix(result_matrix_t m) {
    for (int i = 0; i < MPE_ROWS; ++i) {
        for (int j = 0; j < MPE_COLS; ++j) {
            std::cout << m[i][j] << "\t";
        }
        std::cout << std::endl;
    }
}

// --- Test Cases ---

bool test_mv() {
    std::cout << "\n--- MPE Matrix-Vector Test ---" << std::endl;
    matrix_t A;
    vector_t B;
    result_vector_t C_actual;
    result_vector_t C_expected;

    init_matrix(A);
    init_vector(B);

    // Calculate actual result from the hardware function
    mpe_mv(A, B, C_actual);

    // Calculate expected result for verification
    for (int i = 0; i < MPE_ROWS; ++i) {
        C_expected[i] = 0;
        for (int j = 0; j < MPE_COLS; ++j) {
            C_expected[i] += A[i][j] * B[j];
        }
    }

    // Compare results
    for (int i = 0; i < MPE_ROWS; ++i) {
        if (C_actual[i] != C_expected[i]) {
            std::cout << "❌ MV Test FAILED at index " << i << ": Expected " << C_expected[i] << ", Got " << C_actual[i] << std::endl;
            return false;
        }
    }

    std::cout << "✅ MV Test PASSED!" << std::endl;
    return true;
}

bool test_mm() {
    std::cout << "\n--- MPE Matrix-Matrix Test ---" << std::endl;
    matrix_t A, B;
    result_matrix_t C_actual;
    result_matrix_t C_expected;

    init_matrix(A);
    init_matrix(B); // Using the same init for simplicity

    // Calculate actual result from the hardware function
    mpe_mm(A, B, C_actual);

    // Calculate expected result for verification
    for (int i = 0; i < MPE_ROWS; ++i) {
        for (int j = 0; j < MPE_COLS; ++j) {
            C_expected[i][j] = 0;
            for (int k = 0; k < MPE_COLS; ++k) {
                C_expected[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // Compare results
    for (int i = 0; i < MPE_ROWS; ++i) {
        for (int j = 0; j < MPE_COLS; ++j) {
            if (C_actual[i][j] != C_expected[i][j]) {
                std::cout << "❌ MM Test FAILED at (" << i << "," << j << "): Expected " << C_expected[i][j] << ", Got " << C_actual[i][j] << std::endl;
                // print_matrix(C_expected);
                // print_matrix(C_actual);
                return false;
            }
        }
    }

    std::cout << "✅ MM Test PASSED!" << std::endl;
    return true;
}

int main() {
    bool mv_passed = test_mv();
    bool mm_passed = test_mm();

    if (mv_passed && mm_passed) {
        std::cout << "\nAll MPE tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\nSome MPE tests failed." << std::endl;
        return 1;
    }
}
