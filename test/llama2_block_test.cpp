#include "../src/llama2_block.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// Golden reference implementations use `float` for high precision.
using golden_token_t = float;

// Forward declarations for golden model
void llama2_block_golden(golden_token_t* x, const TransformerBlockWeights* weights, int pos);
void init_rope_tables();

// --- Helper function to initialize weights and data for testing ---
void initialize_data(TransformerBlockWeights& weights, token_t* x, golden_token_t* x_golden) {
    // Initialize weights with small, non-trivial values
    for (int i = 0; i < DIM; ++i) {
        weights.rms_att_weight[i] = (i % 10) * 0.1f + 1.0f;
        weights.rms_ffn_weight[i] = (i % 20) * 0.05f + 1.0f;
        x[i] = (token_t)((i % 5) * 0.05f);
        x_golden[i] = (golden_token_t)((i % 5) * 0.05f);
    }

    // Initialize matrices with some values
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            weights.attention.wq[i][j] = ((i + j) % 100) * 0.001f;
            weights.attention.wk[i][j] = ((i + j + 1) % 100) * 0.001f;
            weights.attention.wv[i][j] = ((i + j + 2) % 100) * 0.001f;
            weights.attention.wo[i][j] = ((i + j + 3) % 100) * 0.001f;
        }
    }
     for (int i = 0; i < HIDDEN_DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            weights.ffn.w1[i][j] = (i+j % 90) * 0.0012f;
            weights.ffn.w3[i][j] = (i+j % 80) * 0.0013f;
        }
    }
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            weights.ffn.w2[i][j] = (i+j % 70) * 0.0014f;
        }
    }
}

// Golden reference implementation of the LLaMA-2 block
void llama2_block_golden(golden_token_t* x, const TransformerBlockWeights* weights, int pos) {
    // Use vectors for easier management
    std::vector<golden_token_t> buffer(DIM);
    std::vector<golden_token_t> buffer2(HIDDEN_DIM);
    std::vector<golden_token_t> q(DIM), k(DIM), v(DIM);
    std::vector<golden_token_t> attention_output(DIM);
    std::vector<golden_token_t> x_input(x, x + DIM);

    // 1. Pre-Attention RMSNorm
    float ss = 0.0f;
    for(int i=0; i<DIM; ++i) ss += x_input[i] * x_input[i];
    ss /= DIM;
    ss = 1.0f / std::sqrt(ss + 1e-5f);
    for(int i=0; i<DIM; ++i) buffer[i] = (x_input[i] * ss) * weights->rms_att_weight[i];

    // 2. Self-Attention
    // 2.1. QKV
    for(int i=0; i<DIM; ++i) {
        q[i] = k[i] = v[i] = 0.0f;
        for(int j=0; j<DIM; ++j) {
            q[i] += weights->attention.wq[i][j] * buffer[j];
            k[i] += weights->attention.wk[i][j] * buffer[j];
            v[i] += weights->attention.wv[i][j] * buffer[j];
        }
    }
    
    // 2.2. RoPE
    // (The golden RoPE logic must be identical to the one in rope.cpp)
    // This is a simplified placeholder for the test.
    // A real test would share the exact same RoPE implementation.
    for (int h = 0; h < N_HEADS; ++h) {
        for (int i = 0; i < HEAD_DIM; i += 2) {
            float theta = pos / std::pow(10000.0f, (float)(h*HEAD_DIM + i) / DIM);
            float cos_theta = std::cos(theta);
            float sin_theta = std::sin(theta);
            int q_idx = h * HEAD_DIM + i;
            float q0 = q[q_idx], q1 = q[q_idx+1];
            q[q_idx]   = q0 * cos_theta - q1 * sin_theta;
            q[q_idx+1] = q0 * sin_theta + q1 * cos_theta;
            int k_idx = h * HEAD_DIM + i;
            float k0 = k[k_idx], k1 = k[k_idx+1];
            k[k_idx]   = k0 * cos_theta - k1 * sin_theta;
            k[k_idx+1] = k0 * sin_theta + k1 * cos_theta;
        }
    }

    // 2.3. MHA
    std::vector<golden_token_t> concatenated_heads(DIM, 0.0f);
    for (int h = 0; h < N_HEADS; ++h) {
        float score = 0.0f;
        for(int i=0; i<HEAD_DIM; ++i) score += q[h*HEAD_DIM+i] * k[h*HEAD_DIM+i];
        score /= std::sqrt((float)HEAD_DIM);
        float softmax_score = 1.0f; // Simplified for seq_len=1
        for(int i=0; i<HEAD_DIM; ++i) concatenated_heads[h*HEAD_DIM+i] = v[h*HEAD_DIM+i] * softmax_score;
    }

    // 2.4. Final projection
    for(int i=0; i<DIM; ++i) {
        attention_output[i] = 0.0f;
        for(int j=0; j<DIM; ++j) attention_output[i] += weights->attention.wo[i][j] * concatenated_heads[j];
    }

    // 3. First Residual
    for(int i=0; i<DIM; ++i) x_input[i] += attention_output[i];

    // 4. Pre-FFN RMSNorm
    ss = 0.0f;
    for(int i=0; i<DIM; ++i) ss += x_input[i] * x_input[i];
    ss /= DIM;
    ss = 1.0f / std::sqrt(ss + 1e-5f);
    for(int i=0; i<DIM; ++i) buffer[i] = (x_input[i] * ss) * weights->rms_ffn_weight[i];

    // 5. FFN
    std::vector<golden_token_t> ffn_hidden_gate(HIDDEN_DIM), ffn_hidden_up(HIDDEN_DIM);
    for(int i=0; i<HIDDEN_DIM; ++i) {
        ffn_hidden_gate[i] = ffn_hidden_up[i] = 0.0f;
        for(int j=0; j<DIM; ++j) {
            ffn_hidden_gate[i] += weights->ffn.w1[i][j] * buffer[j];
            ffn_hidden_up[i]   += weights->ffn.w3[i][j] * buffer[j];
        }
    }
    for(int i=0; i<HIDDEN_DIM; ++i) {
        golden_token_t val = ffn_hidden_gate[i];
        golden_token_t sigmoid = 1.0f / (1.0f + std::exp(-val));
        ffn_hidden_gate[i] = (val * sigmoid) * ffn_hidden_up[i];
    }
    for(int i=0; i<DIM; ++i) {
        buffer[i] = 0.0f;
        for(int j=0; j<HIDDEN_DIM; ++j) buffer[i] += weights->ffn.w2[i][j] * ffn_hidden_gate[j];
    }

    // 6. Second Residual
    for(int i=0; i<DIM; ++i) x[i] = x_input[i] + buffer[i];
}


int main() {
    std::cout << "--- Running LLaMA-2 Block Verification Test ---" << std::endl;

    init_rope_tables();

    static TransformerBlockWeights weights;
    static token_t x[DIM];
    static golden_token_t x_golden[DIM];
    static golden_token_t x_golden_out[DIM];

    initialize_data(weights, x, x_golden);
    // Copy golden input to the output buffer to be worked on by the golden function
    for(int i=0; i<DIM; ++i) x_golden_out[i] = x_golden[i];

    int position = 10;

    std::cout << "Executing llama2_block (DUT)..." << std::endl;
    llama2_block(x, &weights, position);

    std::cout << "Executing llama2_block_golden (Reference)..." << std::endl;
    llama2_block_golden(x_golden_out, &weights, position);

    // Verification
    std::cout << "Verifying results..." << std::endl;
    float max_error = 0.0f;
    int errors = 0;
    const float tolerance = 1e-3;

    for (int i = 0; i < DIM; ++i) {
        float error = std::abs(x[i] - x_golden_out[i]);
        if (error > max_error) {
            max_error = error;
        }
        if (error > tolerance) {
            errors++;
        }
    }

    if (errors > 0) {
        std::cout << "[FAIL]" << std::endl;
        std::cout << "Verification failed with " << errors << " mismatches." << std::endl;
        std::cout << "Max error: " << max_error << std::endl;
    } else {
        std::cout << "[PASS]" << std::endl;
        std::cout << "Verification successful. Max error: " << max_error << std::endl;
    }

    return errors > 0 ? 1 : 0;
}

