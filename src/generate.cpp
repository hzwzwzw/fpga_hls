#include "flight_llama2.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <algorithm>

// External function from rope.cpp to initialize tables for simulation
void init_rope_tables();

// --- Vocabulary Loading ---
bool load_vocab(const char* path, std::vector<std::string>& vocab) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Error: Could not open vocabulary file: " << path << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(file, line)) {
        vocab.push_back(line);
    }
    return true;
}

// --- Sampling ---
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

int main(int argc, char *argv[]) {
    // --- Model and Vocab Loading ---
    auto model = std::make_unique<Llama2Model>();
    std::vector<std::string> vocab;

    if (!load_weights(model.get(), "model.bin")) return 1;
    if (!load_vocab("vocab.txt", vocab)) return 1;
    
    init_rope_tables();
    std::cout << "Model and vocabulary loaded." << std::endl;

    // --- Generation Loop ---
    int start_token = 1; // BOS (Beginning of Sentence) token
    int next_token = start_token;
    int pos = 0;

    std::cout << "--- Generated Text ---" << std::endl;
    while (pos < 10) { // Run for 10 steps
        token_t* logits = flight_llama2_inference(next_token, pos, model.get());
        
        // In a real implementation, we would sample from the full logits vector.
        // Here we just use our dummy sampler.
        next_token = sample_argmax(logits, vocab.size());

        // Stop at EOS (End of Sentence) token, which is typically ID 2
        if (next_token == 2) {
            break;
        }

        // Print the generated token
        if (next_token < vocab.size()) {
            std::cout << vocab[next_token] << std::flush;
        } else {
            std::cout << "[UNK:" << next_token << "]" << std::flush;
        }
        
        pos++;
    }
    std::cout << std::endl;

    return 0;
}
