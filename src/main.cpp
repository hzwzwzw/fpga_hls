#include "model.hpp"
#include <iostream>
#include <string>
#include <chrono>

int main(int argc, char** argv) {
    std::string weights_dir = "weights_bin";
    std::string tokenizer_path = weights_dir + "/tokenizer.bin";

    if (argc > 1) {
        weights_dir = argv[1];
        tokenizer_path = weights_dir + "/tokenizer.bin";
    }

    TinyLlama model;

    std::cout << "Loading tokenizer from " << tokenizer_path << "..." << std::endl;
    if (!model.load_tokenizer(tokenizer_path)) {
        std::cerr << "Failed to load tokenizer!" << std::endl;
        return 1;
    }

    std::cout << "Loading weights from " << weights_dir << "..." << std::endl;
    if (!model.load_weights(weights_dir)) {
        std::cerr << "Failed to load weights!" << std::endl;
        return 1;
    }

    std::string prompt = "The capital of France is";
    if (argc > 2) {
        prompt = argv[2];
    }

    std::cout << "\nPrompt: " << prompt << std::endl;
    std::cout << "Generating..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    
    std::string output = model.generate(prompt, 20); // Generate 20 tokens
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "\n--- Result ---" << std::endl;
    std::cout << output << std::endl;
    std::cout << "--------------" << std::endl;
    std::cout << "Time: " << elapsed.count() << "s" << std::endl;

    return 0;
}
