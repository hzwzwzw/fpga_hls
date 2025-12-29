#include "flight_llama2.hpp"
#include <iostream>
#include <fstream>
#include <memory>

// This test verifies if the C++ `load_weights` function correctly populates
// the C++ struct in memory by comparing its memory layout to a golden file.

int main() {
    // 1. Load the full model into the C++ struct
    auto model = std::make_unique<Llama2Model>();
    if (!load_weights(model.get(), "model.bin")) {
        std::cerr << "Failed to load model.bin" << std::endl;
        return 1;
    }
    std::cout << "model.bin loaded into C++ struct." << std::endl;

    // 2. Get a pointer to the specific weight matrix in the C++ struct
    const void* cpp_wq_ptr = &model->layers[0].attention.wq;
    size_t wq_size = sizeof(model->layers[0].attention.wq);

    // 3. Write the raw memory content of the C++ wq matrix to a new file
    std::ofstream cpp_file("wq_cpp.bin", std::ios::binary);
    if (!cpp_file) {
        std::cerr << "Failed to create wq_cpp.bin" << std::endl;
        return 1;
    }
    cpp_file.write(static_cast<const char*>(cpp_wq_ptr), wq_size);
    cpp_file.close();
    std::cout << "Raw memory of C++ wq matrix saved to wq_cpp.bin." << std::endl;

    std::cout << "\nVerification ready. Please compare the files in your shell:" << std::endl;
    std::cout << "  cmp -l wq_golden.bin wq_cpp.bin | head" << std::endl;
    std::cout << "If there is any output, the loading process is incorrect." << std::endl;

    return 0;
}
