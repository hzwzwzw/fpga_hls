#include "../src/model.hpp"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    std::cout << "Initializing TinyLlama Model..." << std::endl;
    TinyLlama model;

    // Create dummy input: Batch=1, SeqLen=16, Hidden=2048
    int seq_len = 16;
    int hidden_size = MODEL_HIDDEN_SIZE; // 2048
    
    std::cout << "Creating input tensor [" << seq_len << " x " << hidden_size << "]..." << std::endl;
    Tensor<float> input(seq_len, hidden_size);
    
    // Fill input with some values
    for(size_t i=0; i<input.size(); ++i) {
        input.data[i] = (float)(i % 10) / 10.0f;
    }

    std::cout << "Running Forward Pass (Prefill)..." << std::endl;
    try {
        // Prefill: start_pos = 0
        model.forward(input, 0);
        std::cout << "Prefill Pass Completed." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during forward pass: " << e.what() << std::endl;
        return 1;
    }

    // Check output stats
    float mean = 0;
    for(float val : input.data) mean += val;
    mean /= input.size();
    std::cout << "Prefill Output Mean: " << mean << std::endl;

    // Test Decoding Step (Next Token)
    std::cout << "Running Forward Pass (Decode Step)..." << std::endl;
    Tensor<float> next_token_input(1, hidden_size);
    // Fill with some data
    for(size_t i=0; i<next_token_input.size(); ++i) next_token_input.data[i] = 0.5f;

    try {
        // Decode: start_pos = 16 (previous seq_len)
        model.forward(next_token_input, 16);
        std::cout << "Decode Pass Completed." << std::endl;
    } catch (const std::exception& e) {
         std::cerr << "Error during decode pass: " << e.what() << std::endl;
         return 1;
    }
    
    mean = 0;
    for(float val : next_token_input.data) mean += val;
    mean /= next_token_input.size();
    std::cout << "Decode Output Mean: " << mean << std::endl;

    if (mean == 0) {
        std::cerr << "Error: Output is all zeros!" << std::endl;
        return 1;
    }

    std::cout << "Test Passed!" << std::endl;
    return 0;
}
