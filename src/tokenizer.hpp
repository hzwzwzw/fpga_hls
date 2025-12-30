#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <vector>
#include <map>
#include <unordered_map>

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    // Load tokenizer from a binary format (compatible with llama2.c tokenizer.bin)
    bool load(const std::string& filename, int vocab_size);

    // Encode text into token IDs
    // bos: whether to prepend Beginning Of Sentence token
    // eos: whether to append End Of Sentence token
    std::vector<int> encode(const std::string& text, bool bos, bool eos);

    // Decode token ID into piece
    std::string decode(int token_id);

    // Decode a sequence of IDs
    std::string decode(const std::vector<int>& tokens);

    int vocab_size;
    
    // Special tokens for Llama 2 / TinyLlama
    int token_bos_id{1};
    int token_eos_id{2};

private:
    struct TokenIndex {
        std::string str;
        int id;
    };

    std::vector<std::string> vocab;
    std::vector<float> vocab_scores;
    
    // Map for encoding
    std::map<std::string, int> token_map;

    unsigned char *vocab_bin_buffer = nullptr;
};

#endif // TOKENIZER_HPP
