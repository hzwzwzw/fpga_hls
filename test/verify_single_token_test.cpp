#include "flight_llama2.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <cmath>
#include <cstring>

// --- Test Helper Functions ---
bool read_golden_file(const std::string& path, std::vector<float>& data) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "  - ERROR: Could not open golden file: " << path << std::endl;
        return false;
    }
    file.seekg(0, std::ios::end);
    size_t filesize = file.tellg();
    file.seekg(0, std::ios::beg);
    data.resize(filesize / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), filesize);
    return true;
}

int sample_argmax(const token_t* logits, int vocab_size) {
    int max_idx = 0;
    token_t max_val = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// --- Main Verification Logic ---
int main() {
    std::cout << "--- Running Single Token End-to-End Verification ---" << std::endl;

    // 1. Load Model
    auto model = std::make_unique<Llama2Model>();
    if (!load_weights(model.get(), "model.bin")) return 1;
    init_rope_tables();
    std::cout << "Model weights and RoPE tables loaded." << std::endl;

    // 2. Execute C++ inference for a single token
    int token = 1; // BOS token
    int pos = 0;
    std::cout << "Executing C++ inference for token " << token << " at pos " << pos << "..." << std::endl;
    token_t* cpp_logits = flight_llama2_inference(token, pos, model.get());
    
    // 3. Load golden logits from Python
    std::cout << "Loading golden logits from Python..." << std::endl;
    std::vector<float> golden_logits;
    if (!read_golden_file("final_logits_golden.bin", golden_logits)) return 1;

    // 4. Rigorously compare the full logits vectors
    std::cout << "Comparing C++ logits vs Golden logits..." << std::endl;
    float max_err = 0.0f;
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        float err = std::abs(cpp_logits[i] - golden_logits[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    std::cout << "  - Max absolute error between logit vectors: " << max_err << std::endl;

    // 5. Find and print the C++ argmax result
    int cpp_next_token_id = sample_argmax(cpp_logits, VOCAB_SIZE);
    std::cout << "\n--- Verification Info ---" << std::endl;
    std::cout << "C++ Argmax Token ID: " << cpp_next_token_id << std::endl;

    // The Python script will print the golden token ID. You can compare them.
    if (max_err > 1e-3) { // Use a reasonable tolerance for end-to-end float comparison
        std::cout << "\n[FAIL] Logit vectors diverge significantly." << std::endl;
        return 1;
    }
    
    std::cout << "\n[PASS] Logit vectors are closely matched." << std::endl;
    return 0;
}
