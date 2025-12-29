#include "flight_llama2.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <cmath>
#include <cstring>

// --- Test Helper Functions ---
bool read_golden_file(const std::string& path, std::vector<float>& data) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "ERROR: Cannot open " << path << std::endl; return false; }
    file.seekg(0, std::ios::end);
    size_t filesize = file.tellg();
    file.seekg(0, std::ios::beg);
    data.resize(filesize / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), filesize);
    return true;
}

bool compare_buffers(const std::string& name, const float* cpp, const std::vector<float>& golden, float tol = 1e-5) {
    std::cout << "--- Verifying: " << name << " ---" << std::endl;
    float max_err = 0.0f;
    for (size_t i = 0; i < golden.size(); ++i) {
        max_err = std::max(max_err, std::abs(cpp[i] - golden[i]));
    }
    if (max_err > tol) {
        std::cerr << "  - FAIL! Max error: " << max_err << std::endl;
        return false;
    }
    std::cout << "  - PASS. Max error: " << max_err << std::endl;
    return true;
}

// --- Main Verification Logic ---
int main() {
    auto model = std::make_unique<Llama2Model>();
    if (!load_weights(model.get(), "model.bin")) return 1;
    init_rope_tables();
    std::cout << "Model and RoPE tables loaded." << std::endl;

    // --- Buffers ---
    std::vector<float> golden;
    static token_t h_norm[DIM], q[DIM], k[KV_DIM], v[KV_DIM];
    static token_t q_rotated[DIM], k_rotated[KV_DIM];
    static token_t scores[N_HEADS * MAX_SEQ_LEN]; // Simplified for pos=0
    static token_t heads_out[DIM], attn_proj_out[DIM];
    const auto& layer0 = model->layers[0];
    int pos = 0;

    // --- Get Input ---
    if (!read_golden_file("debug_attn_out/0_attn_input.bin", golden)) return 1;
    memcpy(h_norm, golden.data(), DIM * sizeof(token_t));

    // --- 1. QKV Projections ---
    mat_vec_mul(layer0.attention.wq, h_norm, q);
    mat_vec_mul_kv(layer0.attention.wk, h_norm, k);
    mat_vec_mul_kv(layer0.attention.wv, h_norm, v);

    // --- 2. RoPE ---
    memcpy(q_rotated, q, DIM * sizeof(token_t));
    memcpy(k_rotated, k, KV_DIM * sizeof(token_t));
    for (int h = 0; h < N_HEADS; ++h) {
        apply_rope(&q_rotated[h * HEAD_DIM], &k_rotated[(h % N_KV_HEADS) * HEAD_DIM], pos);
    }
    if (!read_golden_file("debug_attn_out/1_q_rope.bin", golden)) return 1;
    if (!compare_buffers("RoPE Q", q_rotated, golden)) return 1;
    if (!read_golden_file("debug_attn_out/2_k_rope.bin", golden)) return 1;
    if (!compare_buffers("RoPE K", k_rotated, golden)) return 1;

    // --- 3. Pre-Softmax Scores ---
    for (int h = 0; h < N_HEADS; ++h) {
        token_t* q_head = &q_rotated[h * HEAD_DIM];
        token_t* k_head = &k_rotated[(h % N_KV_HEADS) * HEAD_DIM];
        token_t score = 0.0f;
        for (int i = 0; i < HEAD_DIM; ++i) score += q_head[i] * k_head[i];
        scores[h] = score / std::sqrt((float)HEAD_DIM);
    }
    if (!read_golden_file("debug_attn_out/3_presoftmax_scores.bin", golden)) return 1;
    if (!compare_buffers("Pre-Softmax Scores", scores, golden)) return 1;

    // --- 4. Softmax ---
    // We test each head's softmax individually (for pos=0, it's a softmax over 1 element)
    for (int h = 0; h < N_HEADS; ++h) {
        sfu_softmax(&scores[h], &scores[h], 1);
    }
    if (!read_golden_file("debug_attn_out/4_postsoftmax_scores.bin", golden)) return 1;
    if (!compare_buffers("Post-Softmax Scores", scores, golden)) return 1;

    // --- 5. Value Weighting ---
    for (int h = 0; h < N_HEADS; ++h) {
        token_t* v_head = &v[(h % N_KV_HEADS) * HEAD_DIM];
        token_t score = scores[h];
        for (int i = 0; i < HEAD_DIM; ++i) {
            heads_out[h * HEAD_DIM + i] = v_head[i] * score;
        }
    }
    if (!read_golden_file("debug_attn_out/5_attn_heads_out.bin", golden)) return 1;
    if (!compare_buffers("Value Weighting (Heads Out)", heads_out, golden)) return 1;

    // --- 6. Output Projection ---
    mat_vec_mul(layer0.attention.wo, heads_out, attn_proj_out);
    if (!read_golden_file("debug_attn_out/6_attn_proj_out.bin", golden)) return 1;
    if (!compare_buffers("Attention Projection", attn_proj_out, golden)) return 1;

    std::cout << "\n\n--- ALL ATTENTION MICRO-STEPS PASSED! ---" << std::endl;
    return 0;
}
