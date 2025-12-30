#include "../src/model.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>

void create_dummy_tokenizer(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    int max_len = 10;
    file.write((char*)&max_len, sizeof(int));
    
    int vocab_size = 32000;
    for(int i=0; i<vocab_size; ++i) {
        float score = 1.0f;
        file.write((char*)&score, sizeof(float));
        
        std::string str;
        if (i==1) str = "<s>";
        else if (i==2) str = "</s>";
        else if (i==32) str = " "; // Space
        else if (i==33) str = "H";
        else if (i==34) str = "e";
        else if (i==35) str = "l";
        else if (i==36) str = "o";
        else str = "t" + std::to_string(i); // dummy
        
        int len = str.length();
        file.write((char*)&len, sizeof(int));
        file.write(str.c_str(), len);
    }
}

void create_dummy_weights(const std::string& dir) {
    std::filesystem::create_directories(dir);
    // Just ensure the dir exists, the model will fail to load individual files but continue if we don't return false on partial failure?
    // My load_weights returns true if dir exists. Inside it checks files. 
    // Wait, it loops and tries to read. If file not found, it skips.
    // So empty dir is fine for testing logic flow (weights remain initialized to defaults).
}

int main() {
    std::string tok_path = "dummy_tokenizer.bin";
    create_dummy_tokenizer(tok_path);
    
    std::string weight_path = "dummy_weights";
    create_dummy_weights(weight_path);

    TinyLlama model;
    if (!model.load_tokenizer(tok_path)) {
        std::cerr << "Failed to load tokenizer" << std::endl;
        return 1;
    }
    
    // We don't strictly need to call load_weights for the test to run (it uses random init),
    // but we check if it handles the call.
    model.load_weights(weight_path);

    std::cout << "Generating..." << std::endl;
    std::string output = model.generate("Hello", 5);
    
    std::cout << "Generated Output: [" << output << "]" << std::endl;
    
    // Check if output is non-empty
    if (output.empty()) return 1;

    std::cout << "Test Passed!" << std::endl;
    return 0;
}
