#include "llama2_block.hpp"
#include <cstring>
#include <cstdio> // For printf

// --- Implementations of the mat_vec_mul functions ---
// Using double for accumulator precision and CORRECT transpose implementation

void mat_vec_mul(const token_t matrix[DIM][DIM], const token_t vector[DIM], token_t* output) {
    for (int i = 0; i < DIM; ++i) {
        double acc = 0.0;
        for (int j = 0; j < DIM; ++j) {
            acc += (double)vector[j] * (double)matrix[j][i]; // Corrected: input @ W.T
        }
        output[i] = (token_t)acc;
    }
}

void mat_vec_mul_ffn_w13(const token_t matrix[HIDDEN_DIM][DIM], const token_t vector[DIM], token_t* output) {
    for (int i = 0; i < HIDDEN_DIM; ++i) {
        double acc = 0.0;
        for (int j = 0; j < DIM; ++j) {
            acc += (double)vector[j] * (double)matrix[j][i]; // Corrected: input @ W.T
        }
        output[i] = (token_t)acc;
    }
}

void mat_vec_mul_ffn_w2(const token_t matrix[DIM][HIDDEN_DIM], const token_t vector[HIDDEN_DIM], token_t* output) {
    for (int i = 0; i < DIM; ++i) {
        double acc = 0.0;
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            acc += (double)vector[j] * (double)matrix[j][i]; // Corrected: input @ W.T
        }
        output[i] = (token_t)acc;
    }
}

void mat_vec_mul_kv(const token_t matrix[KV_DIM][DIM], const token_t vector[DIM], token_t* output) {
    for (int i = 0; i < KV_DIM; ++i) {
        double acc = 0.0;
        for (int j = 0; j < DIM; ++j) {
            acc += (double)vector[j] * (double)matrix[j][i]; // Corrected: input @ W.T
        }
        output[i] = (token_t)acc;
    }
}


// --- Main llama2_block function implementation ---
void llama2_block(token_t* x, const TransformerBlockWeights* weights, int pos,
                  token_t k_cache[MAX_SEQ_LEN][KV_DIM],
                  token_t v_cache[MAX_SEQ_LEN][KV_DIM]) {
    
    token_t buffer[DIM];
    token_t buffer2[DIM];
    token_t q[DIM], k[KV_DIM], v[KV_DIM];
    token_t attention_output[DIM];
    token_t residual[DIM];
    memcpy(residual, x, DIM * sizeof(token_t));

    // --- 1. Pre-Attention RMSNorm ---
    sfu_rms_norm(x, (token_t*)weights->rms_att_weight, buffer, DIM);

    // --- 2. QKV Calculation & Caching ---
    mat_vec_mul(weights->attention.wq, buffer, q);
    mat_vec_mul_kv(weights->attention.wk, buffer, k);
    mat_vec_mul_kv(weights->attention.wv, buffer, v);

    // Store UN-ROTATED K and V into the cache
    memcpy(k_cache[pos], k, KV_DIM * sizeof(token_t));
    memcpy(v_cache[pos], v, KV_DIM * sizeof(token_t));
    
    // --- 3. Attention Calculation ---
    token_t concatenated_heads[DIM] = {0};
    for (int h = 0; h < N_HEADS; ++h) {
        token_t q_head[HEAD_DIM];
        memcpy(q_head, &q[h * HEAD_DIM], HEAD_DIM * sizeof(token_t));
        apply_rope(q_head, nullptr, pos);

        int kv_head_idx = h % N_KV_HEADS;
        token_t att_scores[MAX_SEQ_LEN];

        for (int t = 0; t <= pos; ++t) {
            token_t k_head_temp[HEAD_DIM];
            memcpy(k_head_temp, &k_cache[t][kv_head_idx * HEAD_DIM], HEAD_DIM * sizeof(token_t));
            apply_rope(nullptr, k_head_temp, t);

            token_t score = 0;
            for (int i = 0; i < HEAD_DIM; ++i) score += q_head[i] * k_head_temp[i];
            att_scores[t] = score / std::sqrt((float)HEAD_DIM);
        }

        sfu_softmax(att_scores, att_scores, pos + 1);

        token_t head_att_out[HEAD_DIM] = {0};
        for (int t = 0; t <= pos; ++t) {
            token_t* v_head_past = &v_cache[t][kv_head_idx * HEAD_DIM];
            for (int i = 0; i < HEAD_DIM; ++i) head_att_out[i] += att_scores[t] * v_head_past[i];
        }
        memcpy(&concatenated_heads[h * HEAD_DIM], head_att_out, HEAD_DIM * sizeof(token_t));
    }

    mat_vec_mul(weights->attention.wo, concatenated_heads, attention_output);

    // --- Residuals & FFN ---
    sfu_element_add(residual, attention_output, x, DIM);
    memcpy(residual, x, DIM * sizeof(token_t));

    sfu_rms_norm(x, (token_t*)weights->rms_ffn_weight, buffer, DIM);

    token_t ffn_hidden_gate[HIDDEN_DIM], ffn_hidden_up[HIDDEN_DIM];
    mat_vec_mul_ffn_w13(weights->ffn.w1, buffer, ffn_hidden_gate);
    mat_vec_mul_ffn_w13(weights->ffn.w3, buffer, ffn_hidden_up);
    sfu_silu(ffn_hidden_gate, ffn_hidden_gate, HIDDEN_DIM);
    sfu_element_mult(ffn_hidden_gate, ffn_hidden_up, ffn_hidden_gate, HIDDEN_DIM);
    mat_vec_mul_ffn_w2(weights->ffn.w2, ffn_hidden_gate, buffer2);

    sfu_element_add(residual, buffer2, x, DIM);
}