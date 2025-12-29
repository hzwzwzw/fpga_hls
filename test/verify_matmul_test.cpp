#include "llama2_block.hpp" // For DIM and mat_vec_mul declaration
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

// --- Test Helper Functions ---

bool read_file_to_buffer(const std::string& path, float* buffer, size_t expected_elements) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "ERROR: Could not open file: " << path << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(buffer), expected_elements * sizeof(float));
    if (!file) {
        std::cerr << "ERROR: Could not read expected number of bytes from " << path << std::endl;
        return false;
    }
    return true;
}

// --- Main Verification Logic ---

int main() {
    std::cout << "--- Running Fundamental Matrix-Vector Multiplication Test ---" << std::endl;

    // 1. Allocate memory
    static token_t weights[DIM][DIM];
    static token_t input_vec[DIM];
    static token_t cpp_output[DIM];
    std::vector<float> golden_output(DIM);

    // 2. Load data from files
    std::cout << "Loading verification files..." << std::endl;
    if (!read_file_to_buffer("verify_out/verify_weights.bin", &weights[0][0], DIM * DIM)) return 1;
    if (!read_file_to_buffer("verify_out/verify_input.bin", input_vec, DIM)) return 1;
    if (!read_file_to_buffer("verify_out/verify_output_golden.bin", golden_output.data(), DIM)) return 1;
    std::cout << "Files loaded." << std::endl;

    // 3. Execute C++ mat_vec_mul
    std::cout << "Executing C++ mat_vec_mul..." << std::endl;
    mat_vec_mul(weights, input_vec, cpp_output);

    // 4. Compare results
    std::cout << "Verifying results..." << std::endl;
    float max_err = 0.0f;
    int first_mismatch = -1;
    float tolerance = 1e-5;

    for (int i = 0; i < DIM; ++i) {
        float err = std::abs(cpp_output[i] - golden_output[i]);
        if (err > max_err) {
            max_err = err;
        }
        if (err > tolerance && first_mismatch == -1) {
            first_mismatch = i;
        }
    }

    if (first_mismatch != -1) {
        std::cerr << "\n[FAIL]" << std::endl;
        std::cerr << "Fundamental mat_vec_mul verification FAILED." << std::endl;
        std::cerr << "This proves a subtle difference between the C++ loop and PyTorch matmul." << std::endl;
        std::cerr << "Max error: " << max_err << std::endl;
        std::cerr << "First mismatch at index " << first_mismatch << ":" << std::endl;
        std::cerr << "  - C++ value:    " << cpp_output[first_mismatch] << std::endl;
        std::cerr << "  - Golden value: " << golden_output[first_mismatch] << std::endl;
        return 1;
    }

    std::cout << "\n[PASS]" << std::endl;
    std::cout << "Fundamental mat_vec_mul verification PASSED." << std::endl;
    std::cout << "This proves the C++ loop is correct. The bug must be in another function." << std::endl;
    std::cout << "Max error: " << max_err << std::endl;

    return 0;
}
