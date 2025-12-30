#include "tokenizer.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>

Tokenizer::Tokenizer() {}

Tokenizer::~Tokenizer() {}

bool Tokenizer::load(const std::string& filename, int v_size) {
    this->vocab_size = v_size;
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open tokenizer file " << filename << std::endl;
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read max_token_length
    int max_token_length;
    file.read(reinterpret_cast<char*>(&max_token_length), sizeof(int));

    vocab.resize(vocab_size);
    vocab_scores.resize(vocab_size);
    token_map.clear();

    for (int i = 0; i < vocab_size; ++i) {
        float score;
        file.read(reinterpret_cast<char*>(&score), sizeof(float));
        vocab_scores[i] = score;

        int len;
        file.read(reinterpret_cast<char*>(&len), sizeof(int));

        std::string str(len, '\0');
        file.read(&str[0], len);
        vocab[i] = str;
        
        // Insert into map (prefer longer tokens or first one? Llama usually has unique strings)
        token_map[str] = i;
    }

    return true;
}

std::string Tokenizer::decode(int token_id) {
    if (token_id < 0 || token_id >= vocab_size) return "";
    return vocab[token_id];
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
    std::string result;
    for (int id : tokens) {
        result += decode(id);
    }
    return result;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool bos, bool eos) {
    std::vector<int> tokens;
    if (bos) tokens.push_back(token_bos_id);
    
    if (text.empty()) return tokens;

    // Greedy matching approach (Simple replacement for proper BPE)
    // We try to match the longest possible token starting at 'pos'.
    // Note: This is NOT BPE. BPE is "merge frequent pairs".
    // But for "The capital...", "The" exists. " capital" exists.
    // So greedy match often works for whole words.
    
    // However, SentencePiece usually treats space as a prefix `_` (U+2581).
    // So "The capital" -> "The", "_capital".
    // We need to handle spaces.
    
    // Pre-processing: Replace space with U+2581
    std::string processed_text = "";
    for(char c : text) {
        if(c == ' ') processed_text += "\xe2\x96\x81";
        else processed_text += c;
    }
    
    // Wait, usually the first token doesn't have space? "The".
    // "The capital" -> "The" (No space), " capital" (Space).
    // My pre-proc adds space to all?
    // "The capital" -> "The_capital".
    // Let's try to match on the raw text but checking both "word" and " word".
    
    // Simpler: Just greedy match on the string.
    // But we need to handle the space correctly.
    // Let's just use the `processed_text` where space is `_`.
    // But leading space? "The" might be "The".
    // " capital" is "_capital".
    // So: Replace " " with "_".
    
    // Exception: If text starts with space?
    // Let's stick to the prompt provided "The capital...".
    // It has spaces.
    
    size_t pos = 0;
    while(pos < processed_text.length()) {
        int best_id = -1;
        int best_len = 0;
        
        // Search for longest match
        // Optimization: limit search length
        int max_len = 32; // arbitrary
        for(int len=1; len<=max_len && pos+len <= processed_text.length(); ++len) {
            std::string sub = processed_text.substr(pos, len);
            if(token_map.count(sub)) {
                best_id = token_map[sub];
                best_len = len;
            }
        }
        
        if (best_id != -1) {
            tokens.push_back(best_id);
            pos += best_len;
        } else {
            // Unknown char? Skip or unk.
            // Llama fallback is usually byte tokens.
            // Try to map byte?
            // <0xXX> tokens.
            // For now, just skip to avoid infinite loop or infinite 0s.
             pos++;
        }
    }

    if (eos) tokens.push_back(token_eos_id);
    return tokens;
}