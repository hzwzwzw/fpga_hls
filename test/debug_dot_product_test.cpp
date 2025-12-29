#include "config.hpp" // For HEAD_DIM
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

// --- Test Helper Function ---
bool read_file_to_buffer(const std::string& path, float* buffer, size_t expected_elements) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "ERROR: Could not open file: " << path << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(buffer), expected_elements * sizeof(float));
    if (!file) {
        std::cerr << "ERROR: Could not read expected bytes from " << path << std::endl;
        return false;
    }
    return true;
}

// --- Main Verification Logic ---
int main() {
    std::cout << "--- Running Fundamental Dot Product Verification Test ---" << std::endl;

    // 1. Allocate memory
    static float q_vec[HEAD_DIM];
    static float k_vec[HEAD_DIM];
    float golden_score;

    // 2. Load data from files
    std::cout << "Loading verification files..." << std::endl;
    if (!read_file_to_buffer("debug_dot_product_out/dot_input_q.bin", q_vec, HEAD_DIM)) return 1;
    if (!read_file_to_buffer("debug_dot_product_out/dot_input_k.bin", k_vec, HEAD_DIM)) return 1;
    if (!read_file_to_buffer("debug_dot_product_out/dot_output_golden.bin", &golden_score, 1)) return 1;
    std::cout << "Files loaded." << std::endl;

    // 3. Execute C++ dot product
    std::cout << "Executing C++ dot product..." << std::endl;
    double cpp_acc = 0.0; // Use double for precision
    for (int i = 0; i < HEAD_DIM; ++i) {
        cpp_acc += (double)q_vec[i] * (double)k_vec[i];
    }
    float cpp_score = (float)cpp_acc;

    // 4. Compare results
    std::cout << "Verifying results..." << std::endl;
    float final_error = std::abs(cpp_score - golden_score);

    std::cout << "\n--- Verification Info ---" << std::endl;
    std::cout << "PyTorch Golden Score: " << golden_score << std::endl;
    std::cout << "C++ Loop Score:       " << cpp_score << std::endl;
    std::cout << "Absolute Error:       " << final_error << std::endl;

    if (final_error > 1e-7) { // Use a very strict tolerance
        std::cout << "\n[FAIL]" << std::endl;
        std::cout << "The C++ dot product loop result DOES NOT MATCH the PyTorch result." << std::endl;
        std::cout << "This provides strong evidence that the issue is a fundamental" << std::endl;
        std::cout << "numerical difference in the underlying floating-point operations." << std::endl;
        return 1;
    }
    
    std::cout << "\n[PASS]" << std::endl;
    std::cout << "The C++ dot product loop result MATCHES the PyTorch result." << std::endl;
    std::cout << "This means the numerical stability theory is likely incorrect." << std::endl;

    return 0;
}
