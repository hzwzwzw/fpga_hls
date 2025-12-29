#ifndef FLIGHT_LLAMA2_HPP
#define FLIGHT_LLAMA2_HPP

#include "llama2_block.hpp"

// --- LLaMA-2 Model Configuration ---
// This configuration is for a tiny version of LLaMA-2 for prototyping.
// A real model would have dozens of layers.
const int N_LAYERS = 2; // For prototyping, we'll stack 2 layers.

struct Llama2Model {
    // Token embedding table
    token_t token_embedding_table[VOCAB_SIZE][DIM];

    // Weights for all transformer blocks
    TransformerBlockWeights layers[N_LAYERS];

    // Final RMS norm
    token_t rms_final_weight[DIM];

    // Classifier weights
    token_t wcls[VOCAB_SIZE][DIM];

    // Key-Value cache
    token_t k_cache[N_LAYERS][MAX_SEQ_LEN][KV_DIM];
    token_t v_cache[N_LAYERS][MAX_SEQ_LEN][KV_DIM];
};

/**
 * @brief Performs one step of end-to-end inference for a LLaMA-2 model.
 *
 * @param token     The input token ID.
 * @param pos       The position of the token in the sequence.
 * @param model     A pointer to the complete model (weights and KV cache).
 * @return token_t* Pointer to the final logits vector.
 */
token_t* flight_llama2_inference(int token, int pos, Llama2Model* model);


/**
 * @brief Loads model weights from a binary file into the Llama2ModelWeights struct.
 *
 * @param weights   Pointer to the weights struct to be filled.
 * @param path      Path to the binary weights file.
 * @return true     If loading was successful.
 * @return false    If an error occurred.
 */
bool load_weights(Llama2Model* weights, const char* path);



#endif // FLIGHT_LLAMA2_HPP
