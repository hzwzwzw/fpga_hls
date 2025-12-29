#include "flight_llama2.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <cmath>
#include <cstring>

// --- Test Helper Functions (Implementations moved before main) ---

bool read_golden_file(const std::string& path, std::vector<float>& data) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "  - ERROR: Could not open golden file: " << path << std::endl;
        return false;
    }
    file.seekg(0, std::ios::end);
    size_t filesize = file.tellg();
    file.seekg(0, std::ios::beg);
    data.resize(filesize / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), filesize);
    return true;
}

bool compare_buffers(const std::string& step_name, const float* cpp_buf, const std::vector<float>& golden_buf, float tolerance = 1e-4) {
    std::cout << "--- Verifying Step: " << step_name << " ---" << std::endl;
    if (cpp_buf == nullptr || golden_buf.empty()) {
        std::cerr << "  - FAIL: Buffers are not valid." << std::endl;
        return false;
    }

    float max_err = 0.0f;
    int first_mismatch = -1;
    for (size_t i = 0; i < golden_buf.size(); ++i) {
        float err = std::abs(cpp_buf[i] - golden_buf[i]);
        if (err > max_err) max_err = err;
        if (err > tolerance && first_mismatch == -1) first_mismatch = i;
    }

    if (first_mismatch != -1) {
        std::cerr << "  - FAIL!" << std::endl;
        std::cerr << "  - Max error: " << max_err << std::endl;
        std::cerr << "  - First mismatch at index " << first_mismatch << ":" << std::endl;
        std::cerr << "    - C++ value:    " << cpp_buf[first_mismatch] << std::endl;
        std::cerr << "    - Golden value: " << golden_buf[first_mismatch] << std::endl;
        return false;
    }

    std::cout << "  - PASS. Max error: " << max_err << std::endl;
    return true;
}


// --- Main Verification Logic: Error Accumulation Trace ---
int main() {
    // --- Load Model & Init ---
    auto model = std::make_unique<Llama2Model>();
    if (!load_weights(model.get(), "model.bin")) return 1;
    init_rope_tables();
    std::cout << "Model weights and RoPE tables loaded." << std::endl;

    // --- Buffers and Golden Data ---
    std::vector<float> golden_input, golden_output;
    static token_t activations[DIM];
    static token_t buffer[DIM];
    static token_t q[DIM], k[KV_DIM], v[KV_DIM];
    static token_t attention_output[DIM];
    static token_t ffn_hidden_gate[HIDDEN_DIM], ffn_hidden_up[HIDDEN_DIM];

    int token = 1;
    int pos = 0;
    const auto& layer0 = model->layers[0];

    // --- Trace Step 1: Embedding ---
    memcpy(activations, &model->token_embedding_table[token][0], DIM * sizeof(token_t));
    if (!read_golden_file("debug_out/1_embedding_out.bin", golden_output)) return 1;
    if (!compare_buffers("1. Embedding", activations, golden_output)) return 1;

    // --- Trace Step 2: Pre-Attention RMSNorm ---
    // INPUT: C++ activations from Step 1
    sfu_rms_norm(activations, (token_t*)layer0.rms_att_weight, buffer, DIM);
    if (!read_golden_file("debug_out/2_rms_att_out.bin", golden_output)) return 1;
    if (!compare_buffers("2. Pre-Att RMSNorm", buffer, golden_output)) return 1;

    // --- Trace Step 3: Q, K, V Projections ---
    // INPUT: C++ buffer from Step 2
    mat_vec_mul(layer0.attention.wq, buffer, q);
    if (!read_golden_file("debug_out/3_q_proj_out.bin", golden_output)) return 1;
    if (!compare_buffers("3. Q Projection", q, golden_output)) return 1;
    
    mat_vec_mul_kv(layer0.attention.wk, buffer, k);
    if (!read_golden_file("debug_out/3_k_proj_out.bin", golden_output)) return 1;
    if (!compare_buffers("3. K Projection", k, golden_output)) return 1;

    mat_vec_mul_kv(layer0.attention.wv, buffer, v);
    if (!read_golden_file("debug_out/3_v_proj_out.bin", golden_output)) return 1;
    if (!compare_buffers("3. V Projection", v, golden_output)) return 1;

    // --- Trace Step 4: RoPE ---
    // INPUT: C++ q and k from Step 3
    apply_rope(q, k, pos);
    if (!read_golden_file("debug_out/4_q_rope_out.bin", golden_output)) return 1;
    if (!compare_buffers("4. RoPE (Q)", q, golden_output)) return 1;
    if (!read_golden_file("debug_out/4_k_rope_out.bin", golden_output)) return 1;
    if (!compare_buffers("4. RoPE (K)", k, golden_output)) return 1;

    // --- Trace Step 5: Attention Output Projection ---
    // This step combines score, softmax, and value weighting.
    // We will perform it and compare with the final projection output.
    // INPUT: C++ q, k, v from previous steps.
    static token_t k_cache[MAX_SEQ_LEN][KV_DIM] = {0};
    static token_t v_cache[MAX_SEQ_LEN][KV_DIM] = {0};
    memcpy(k_cache[pos], k, KV_DIM * sizeof(token_t));
    memcpy(v_cache[pos], v, KV_DIM * sizeof(token_t));
    
    // Manual attention block to trace errors
    static token_t concatenated_heads[DIM] = {0};
    for (int h = 0; h < N_HEADS; ++h) {
        token_t q_head[HEAD_DIM];
        memcpy(q_head, &q[h * HEAD_DIM], HEAD_DIM * sizeof(token_t));
        int kv_head_idx = h % N_KV_HEADS;
        token_t att_scores[MAX_SEQ_LEN];

        // --- This is the critical part: Attention Score Calculation ---
        // For pos=0, the loop runs once (t=0)
        for (int t = 0; t <= pos; ++t) {
            token_t k_head_temp[HEAD_DIM];
            memcpy(k_head_temp, &k_cache[t][kv_head_idx * HEAD_DIM], HEAD_DIM * sizeof(token_t));
            
            // **THE MISSING PIECE**: RoPE must be applied to the cached K vector just-in-time.
            apply_rope(nullptr, k_head_temp, t);

            token_t score = 0;
            for (int i = 0; i < HEAD_DIM; ++i) score += q_head[i] * k_head_temp[i];
            att_scores[t] = score / std::sqrt((float)HEAD_DIM);
        }
        
        sfu_softmax(att_scores, att_scores, pos + 1);
        
        token_t head_att_out[HEAD_DIM] = {0};
        token_t* v_head_past = &v_cache[pos][kv_head_idx * HEAD_DIM];
        for (int i = 0; i < HEAD_DIM; ++i) head_att_out[i] += att_scores[0] * v_head_past[i];
        memcpy(&concatenated_heads[h * HEAD_DIM], head_att_out, HEAD_DIM * sizeof(token_t));
    }
    mat_vec_mul(layer0.attention.wo, concatenated_heads, attention_output);
    if (!read_golden_file("debug_out/6_attn_proj_out.bin", golden_output)) return 1;
    if (!compare_buffers("6. Attention Projection", attention_output, golden_output)) return 1;

    // --- Trace Step 7: First Residual ---
    sfu_element_add(activations, attention_output, activations, DIM);
    if (!read_golden_file("debug_out/7_residual1_out.bin", golden_output)) return 1;
    if (!compare_buffers("7. First Residual", activations, golden_output)) return 1;

    // --- Trace Step 8 & 9 & 10 ---
    sfu_rms_norm(activations, (token_t*)layer0.rms_ffn_weight, buffer, DIM);
    if (!read_golden_file("debug_out/8_ffn_norm_out.bin", golden_output)) return 1;
    if (!compare_buffers("8. Pre-FFN RMSNorm", buffer, golden_output)) return 1;

    mat_vec_mul_ffn_w13(layer0.ffn.w1, buffer, ffn_hidden_gate);
    mat_vec_mul_ffn_w13(layer0.ffn.w3, buffer, ffn_hidden_up);
    sfu_silu(ffn_hidden_gate, ffn_hidden_gate, HIDDEN_DIM);
    sfu_element_mult(ffn_hidden_gate, ffn_hidden_up, ffn_hidden_gate, HIDDEN_DIM);
    mat_vec_mul_ffn_w2(layer0.ffn.w2, ffn_hidden_gate, buffer);
    if (!read_golden_file("debug_out/9_ffn_out.bin", golden_output)) return 1;
    if (!compare_buffers("9. FFN Output", buffer, golden_output)) return 1;

    sfu_element_add(activations, buffer, activations, DIM);
    if (!read_golden_file("debug_out/10_block0_output.bin", golden_output)) return 1;
    if (!compare_buffers("10. Final Block Output", activations, golden_output)) return 1;

    std::cout << "\n\n--- ALL TRACE STEPS PASSED! ---" << std::endl;
    return 0;
}