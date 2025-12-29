#include "flight_llama2.hpp"
#include <fstream>
#include <iostream>

// Forward declaration for the new mat_vec_mul overload
void mat_vec_mul_wcls(const token_t matrix[VOCAB_SIZE][DIM], const token_t vector[DIM], token_t* output);

// Helper macro to read a specific field from the file
#define READ_WEIGHT(file, field) \
    file.read(reinterpret_cast<char*>(&field), sizeof(field)); \
    if (!file) { \
        std::cerr << "Error reading weights for " #field << std::endl; \
        return false; \
    }

bool load_weights(Llama2Model* weights, const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open weights file: " << path << std::endl;
        return false;
    }

    // Read each weight tensor individually to avoid struct padding issues.
    READ_WEIGHT(file, weights->token_embedding_table);

    for (int l = 0; l < N_LAYERS; ++l) {
        auto& layer = weights->layers[l];
        READ_WEIGHT(file, layer.rms_att_weight);
        READ_WEIGHT(file, layer.rms_ffn_weight);
        
        READ_WEIGHT(file, layer.attention.wq);
        READ_WEIGHT(file, layer.attention.wk);
        READ_WEIGHT(file, layer.attention.wv);
        READ_WEIGHT(file, layer.attention.wo);
        
        READ_WEIGHT(file, layer.ffn.w1);
        READ_WEIGHT(file, layer.ffn.w2);
        READ_WEIGHT(file, layer.ffn.w3);
    }

    READ_WEIGHT(file, weights->rms_final_weight);
    READ_WEIGHT(file, weights->wcls);

    file.peek();
    if (file.eof()) {
         std::cout << "Successfully read the entire weights file." << std::endl;
    } else {
        std::cerr << "Warning: Did not read the entire weights file. More data remains." << std::endl;
    }

    file.close();
    return true;
}

// Global buffer for the activations (the "x" vector)
static token_t activations[DIM];

// Global buffer for the final logits
static token_t logits[VOCAB_SIZE];

#include <cstring> // For memcpy

// ... (rest of the file up to the inference function)

token_t* flight_llama2_inference(int token, int pos, Llama2Model* model) {

    // --- 1. Token Embedding ---
    // Correctly look up the embedding for the current token
    memcpy(activations, &model->token_embedding_table[token][0], DIM * sizeof(token_t));

    // --- 2. Transformer Blocks ---
    for (int l = 0; l < N_LAYERS; ++l) {
        llama2_block(activations, &model->layers[l], pos, model->k_cache[l], model->v_cache[l]);
    }

    // --- 3. Final RMSNorm ---
    sfu_rms_norm(activations, (token_t*)model->rms_final_weight, activations, DIM);

    // --- 4. Classifier ---
    mat_vec_mul_wcls(model->wcls, activations, logits);

    return logits;
}

// --- Implementation for the new mat_vec_mul overload ---
void mat_vec_mul_wcls(const token_t matrix[VOCAB_SIZE][DIM], const token_t vector[DIM], token_t* output) {
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        double acc = 0.0;
        for (int j = 0; j < DIM; ++j) {
            acc += (double)vector[j] * (double)matrix[j][i]; // Corrected: matrix[j][i]
        }
        output[i] = (token_t)acc;
    }
}