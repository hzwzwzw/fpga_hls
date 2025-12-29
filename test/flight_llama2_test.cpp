#include "flight_llama2.hpp"
#include <iostream>
#include <memory>

// External function from rope.cpp to initialize tables for simulation
void init_rope_tables();

int main() {
    std::cout << "--- Running End-to-End LLaMA-2 Inference Test with Binary Weights ---" << std::endl;

    // 1. Allocate memory for weights
    // Use a smart pointer to manage this large allocation on the heap
    auto weights = std::make_unique<Llama2ModelWeights>();
    
    // 2. Load weights from binary file
    std::cout << "Loading weights from model.bin..." << std::endl;
    if (!load_weights(weights.get(), "model.bin")) {
        std::cerr << "Failed to load weights. Please ensure 'model.bin' exists." << std::endl;
        std::cerr << "You can generate it by running: python3 scripts/export_weights.py" << std::endl;
        return 1;
    }
    std::cout << "Weights loaded successfully." << std::endl;

    // 3. Initialize RoPE tables
    init_rope_tables();
    std::cout << "RoPE tables initialized." << std::endl;

    // 4. Define input token and position
    int token_id = 1; // Example: "start of sentence" token
    int position = 0;
    std::cout << "Input token ID: " << token_id << " at position " << position << std::endl;

    // 5. Execute the full inference pipeline
    std::cout << "Executing flight_llama2_inference..." << std::endl;
    token_t* logits = flight_llama2_inference(token_id, position, weights.get());
    std::cout << "Inference finished." << std::endl;

    // 6. Print a sample of the output logits
    std::cout << "Sample output logit[0]: " << logits[0] << std::endl;

    std::cout << "\n--- End-to-End Test Finished ---" << std::endl;

    return 0;
}