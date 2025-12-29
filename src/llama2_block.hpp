#ifndef LLAMA2_BLOCK_HPP
#define LLAMA2_BLOCK_HPP

#include "mpe.hpp"
#include "sfu.hpp"
#include "rope.hpp"
#include "config.hpp"

// For simplicity in this prototype, we assume MHA (Multi-Head Attention)
// where N_HEADS == N_KV_HEADS.
// We also assume the data types for now.
using token_t = sfu_data_t;

// --- Weight structures for one Transformer Block ---
struct AttentionWeights {
    token_t wq[DIM][DIM];
    token_t wk[KV_DIM][DIM]; // Corrected for GQA
    token_t wv[KV_DIM][DIM]; // Corrected for GQA
    token_t wo[DIM][DIM];
};

struct FFNWeights {
    token_t w1[HIDDEN_DIM][DIM]; // Corresponds to W_gate in SwiGLU
    token_t w2[DIM][HIDDEN_DIM]; // Corresponds to W_down in SwiGLU
    token_t w3[HIDDEN_DIM][DIM]; // Corresponds to W_up in SwiGLU
};

struct TransformerBlockWeights {
    // Normalization weights
    token_t rms_att_weight[DIM];
    token_t rms_ffn_weight[DIM];

    // Attention weights
    AttentionWeights attention;

    // FFN weights
    FFNWeights ffn;
};


/**
 * @brief Executes one block of the LLaMA-2 transformer architecture with KV cache.
 *
 * @param x         Input token vector (size DIM). Modified in-place for output.
 * @param weights   A pointer to the weights for this block.
 * @param pos       The sequence position of the input token.
 * @param k_cache   The Key cache for this layer [MAX_SEQ_LEN, KV_DIM].
 * @param v_cache   The Value cache for this layer [MAX_SEQ_LEN, KV_DIM].
 */
void llama2_block(token_t* x, const TransformerBlockWeights* weights, int pos, 
                  token_t k_cache[MAX_SEQ_LEN][KV_DIM], 
                  token_t v_cache[MAX_SEQ_LEN][KV_DIM]);

// Forward declarations for matrix-vector multiplication helpers
void mat_vec_mul(const token_t matrix[DIM][DIM], const token_t vector[DIM], token_t* output);
void mat_vec_mul_ffn_w13(const token_t matrix[HIDDEN_DIM][DIM], const token_t vector[DIM], token_t* output);
void mat_vec_mul_ffn_w2(const token_t matrix[DIM][HIDDEN_DIM], const token_t vector[HIDDEN_DIM], token_t* output);
void mat_vec_mul_kv(const token_t matrix[KV_DIM][DIM], const token_t vector[DIM], token_t* output);




#endif // LLAMA2_BLOCK_HPP
