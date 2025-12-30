#include "model.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

// --- Helper Functions ---

void quantize(const Tensor<float>& input, Tensor<int8_t>& output, float& scale) {
    output.rows = input.rows;
    output.cols = input.cols;
    size_t size = input.size();
    
    float max_val = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float abs_val = std::abs(input.data[i]);
        if (abs_val > max_val) max_val = abs_val;
    }

    scale = max_val / 127.0f;
    if (scale == 0) scale = 1.0f;
    float inv_scale = 1.0f / scale;

    for (size_t i = 0; i < size; ++i) {
        float val = input.data[i] * inv_scale;
        val = std::round(val);
        if (val > 127.0f) val = 127.0f;
        if (val < -127.0f) val = -127.0f;
        output.data[i] = static_cast<int8_t>(val);
    }
}

void dequantize(const Tensor<int32_t>& input, Tensor<float>& output, float scale) {
    output.rows = input.rows;
    output.cols = input.cols;
    size_t size = input.size();
    for (size_t i = 0; i < size; ++i) {
        output.data[i] = static_cast<float>(input.data[i]) * scale;
    }
}

void dequantize_i8(const Tensor<int8_t>& input, Tensor<float>& output, float scale) {
    output.rows = input.rows;
    output.cols = input.cols;
    size_t size = input.size();
    for (size_t i = 0; i < size; ++i) {
        output.data[i] = static_cast<float>(input.data[i]) * scale;
    }
}

void run_accelerator_padded(const int8_t* A, const int8_t* B, int32_t* C, int M, int K, int N) {
    int padded_M = (M + TILE_ROWS - 1) / TILE_ROWS * TILE_ROWS;
    
    if (padded_M == M) {
        flight_llm_accelerator(A, B, C, M, K, N);
    } else {
        std::vector<int8_t> A_pad(padded_M * K, 0);
        for(int i=0; i<M; ++i) {
            std::memcpy(&A_pad[i*K], &A[i*K], K * sizeof(int8_t));
        }

        std::vector<int32_t> C_pad(padded_M * N, 0);

        flight_llm_accelerator(A_pad.data(), B, C_pad.data(), padded_M, K, N);

        for(int i=0; i<M; ++i) {
            std::memcpy(&C[i*N], &C_pad[i*N], N * sizeof(int32_t));
        }
    }
}

// --- TransformerBlock Implementation ---

TransformerBlock::TransformerBlock(int id) 
    : layer_id(id),
      w_q(MODEL_HIDDEN_SIZE, MODEL_HIDDEN_SIZE),
      w_k(MODEL_HIDDEN_SIZE, MODEL_NUM_KV_HEADS * MODEL_HEAD_DIM),
      w_v(MODEL_HIDDEN_SIZE, MODEL_NUM_KV_HEADS * MODEL_HEAD_DIM),
      w_o(MODEL_HIDDEN_SIZE, MODEL_HIDDEN_SIZE),
      w_gate(MODEL_HIDDEN_SIZE, MODEL_INTERMEDIATE_SIZE),
      w_up(MODEL_HIDDEN_SIZE, MODEL_INTERMEDIATE_SIZE),
      w_down(MODEL_INTERMEDIATE_SIZE, MODEL_HIDDEN_SIZE),
      attention_norm_weight(1, MODEL_HIDDEN_SIZE),
      mlp_norm_weight(1, MODEL_HIDDEN_SIZE),
      k_cache(2048, MODEL_NUM_KV_HEADS * MODEL_HEAD_DIM),
      v_cache(2048, MODEL_NUM_KV_HEADS * MODEL_HEAD_DIM)
{
    initialize_weights();
    std::fill(k_cache.data.begin(), k_cache.data.end(), 0.0f);
    std::fill(v_cache.data.begin(), v_cache.data.end(), 0.0f);
}

void TransformerBlock::initialize_weights() {
    auto fill = [](Tensor<int8_t>& t) {
        for(size_t i=0; i<t.size(); ++i) t.data[i] = (i % 3) == 0 ? 1 : 0;
    };
    fill(w_q); fill(w_k); fill(w_v); fill(w_o);
    fill(w_gate); fill(w_up); fill(w_down);
    
    // Init norms to 1
    std::fill(attention_norm_weight.data.begin(), attention_norm_weight.data.end(), 1.0f);
    std::fill(mlp_norm_weight.data.begin(), mlp_norm_weight.data.end(), 1.0f);
}

void TransformerBlock::run_linear(const Tensor<float>& input, const Tensor<int8_t>& weight, float weight_scale, Tensor<float>& output) {
    // W8A32 Implementation (High Precision)
    // Dequantize weights on the fly and compute in Float32
    
    int M = input.rows;
    int K = input.cols;
    int N = weight.cols;

    output.rows = M;
    output.cols = N;
    // Ensure output data is sized correctly
    if(output.data.size() != M*N) output.data.resize(M*N);
    
    std::fill(output.data.begin(), output.data.end(), 0.0f);

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float val_a = input.data[i * K + k];
            for (int j = 0; j < N; ++j) {
                int8_t w_int = weight.data[k * N + j];
                float val_w = static_cast<float>(w_int) * weight_scale;
                
                output.data[i * N + j] += val_a * val_w;
            }
        }
    }
}

void TransformerBlock::apply_rope(Tensor<float>& q, Tensor<float>& k, int start_pos) {
    // q: [SeqLen, HiddenSize] -> Reshaped conceptually to [SeqLen, NumHeads, HeadDim]
    // k: [SeqLen, KVHeads * HeadDim]
    
    int head_dim = MODEL_HEAD_DIM; // 64
    int seq_len = q.rows;

    for (int r = 0; r < seq_len; ++r) {
        int pos = start_pos + r;
        
        // Apply to Q
        int num_heads_q = MODEL_NUM_HEADS; // 32
        for (int h = 0; h < num_heads_q; ++h) {
            int head_offset = h * head_dim;
            // Iterate over half the head dimension
            for (int i = 0; i < head_dim / 2; ++i) {
                float theta = 1.0f / std::pow(10000.0f, (float)(2 * i) / head_dim);
                float angle = pos * theta;
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);

                // Llama rotates pairs (x[i], x[i + dim/2])
                int idx1 = r * MODEL_HIDDEN_SIZE + head_offset + i;
                int idx2 = r * MODEL_HIDDEN_SIZE + head_offset + i + head_dim / 2;

                float v1 = q.data[idx1];
                float v2 = q.data[idx2];

                q.data[idx1] = v1 * cos_val - v2 * sin_val;
                q.data[idx2] = v1 * sin_val + v2 * cos_val;
            }
        }

        // Apply to K
        int num_heads_k = MODEL_NUM_KV_HEADS; // 4
        int k_cols = k.cols; // KVHeads * HeadDim
        for (int h = 0; h < num_heads_k; ++h) {
            int head_offset = h * head_dim;
            for (int i = 0; i < head_dim / 2; ++i) {
                float theta = 1.0f / std::pow(10000.0f, (float)(2 * i) / head_dim);
                float angle = pos * theta;
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);

                int idx1 = r * k_cols + head_offset + i;
                int idx2 = r * k_cols + head_offset + i + head_dim / 2;

                float v1 = k.data[idx1];
                float v2 = k.data[idx2];

                k.data[idx1] = v1 * cos_val - v2 * sin_val;
                k.data[idx2] = v1 * sin_val + v2 * cos_val;
            }
        }
    }
}

void TransformerBlock::forward(Tensor<float>& input, int start_pos) {
    int seq_len = input.rows; 
    
    // 1. RMSNorm (Attention)
    Tensor<float> normed_input(seq_len, MODEL_HIDDEN_SIZE);
    for(int i=0; i<seq_len; ++i) {
        // Calculate standard RMSNorm (x * inv_rms)
        sfu_rms_norm(&input.data[i * MODEL_HIDDEN_SIZE], &normed_input.data[i * MODEL_HIDDEN_SIZE], MODEL_HIDDEN_SIZE);
        // Apply Weight
        for(int j=0; j<MODEL_HIDDEN_SIZE; ++j) {
            normed_input.data[i * MODEL_HIDDEN_SIZE + j] *= attention_norm_weight.data[j];
        }
    }

    // 2. Projections
    Tensor<float> q(seq_len, MODEL_HIDDEN_SIZE);
    Tensor<float> k(seq_len, MODEL_NUM_KV_HEADS * MODEL_HEAD_DIM);
    Tensor<float> v(seq_len, MODEL_NUM_KV_HEADS * MODEL_HEAD_DIM);

    run_linear(normed_input, w_q, scale_wq, q);
    run_linear(normed_input, w_k, scale_wk, k);
    run_linear(normed_input, w_v, scale_wv, v);

    apply_rope(q, k, start_pos);

    // 3. Cache Update
    int hidden_kv = k.cols;
    for(int r=0; r<seq_len; ++r) {
        int cache_idx = start_pos + r;
        if(cache_idx >= 2048) break; 
        for(int c=0; c<hidden_kv; ++c) {
            k_cache.data[cache_idx * hidden_kv + c] = k.data[r * hidden_kv + c];
            v_cache.data[cache_idx * hidden_kv + c] = v.data[r * hidden_kv + c];
        }
    }

    // 4. Attention (Float32 for accuracy)
    int total_seq_len = start_pos + seq_len;
    Tensor<float> attn_output(seq_len, MODEL_HIDDEN_SIZE);
    
    int kv_heads = MODEL_NUM_KV_HEADS; 
    int q_heads_per_kv = MODEL_NUM_HEADS / kv_heads; 

    // Retrieve full KV from cache
    Tensor<float> k_full_view(total_seq_len, hidden_kv);
    Tensor<float> v_full_view(total_seq_len, hidden_kv);
    for(int i=0; i<total_seq_len * hidden_kv; ++i) {
        k_full_view.data[i] = k_cache.data[i];
        v_full_view.data[i] = v_cache.data[i];
    }

    for (int h_kv = 0; h_kv < kv_heads; ++h_kv) {
        // Extract K_head [TotalSeq, HeadDim]
        Tensor<float> k_head(total_seq_len, MODEL_HEAD_DIM);
        Tensor<float> v_head(total_seq_len, MODEL_HEAD_DIM);
        for(int r=0; r<total_seq_len; ++r) {
            for(int c=0; c<MODEL_HEAD_DIM; ++c) {
                k_head.data[r*MODEL_HEAD_DIM + c] = k_full_view.data[r*hidden_kv + h_kv*MODEL_HEAD_DIM + c];
                v_head.data[r*MODEL_HEAD_DIM + c] = v_full_view.data[r*hidden_kv + h_kv*MODEL_HEAD_DIM + c];
            }
        }

        for (int h_q_local = 0; h_q_local < q_heads_per_kv; ++h_q_local) {
            int global_head_idx = h_kv * q_heads_per_kv + h_q_local;
            
            // Extract Q_head [SeqLen, HeadDim]
            Tensor<float> q_head(seq_len, MODEL_HEAD_DIM);
            for(int r=0; r<seq_len; ++r) {
                for(int c=0; c<MODEL_HEAD_DIM; ++c) {
                    q_head.data[r*MODEL_HEAD_DIM + c] = q.data[r*q.cols + global_head_idx*MODEL_HEAD_DIM + c];
                }
            }

            // Scores = Q * K^T
            // Q [Seq, Dim], K [TotalSeq, Dim]. Result [Seq, TotalSeq]
            Tensor<float> scores(seq_len, total_seq_len);
            for(int r_q=0; r_q<seq_len; ++r_q) {
                for(int r_k=0; r_k<total_seq_len; ++r_k) {
                    float dot = 0.0f;
                    for(int d=0; d<MODEL_HEAD_DIM; ++d) {
                        dot += q_head.data[r_q*MODEL_HEAD_DIM + d] * k_head.data[r_k*MODEL_HEAD_DIM + d];
                    }
                    scores.data[r_q*total_seq_len + r_k] = dot;
                }
            }

            // Scale
            float attn_scale = 1.0f / std::sqrt((float)MODEL_HEAD_DIM);
            for(size_t i=0; i<scores.size(); ++i) scores.data[i] *= attn_scale;

            // Softmax
            for(int r=0; r<seq_len; ++r) {
                sfu_softmax(&scores.data[r*total_seq_len], &scores.data[r*total_seq_len], total_seq_len);
            }

            // Context = Scores * V
            // Scores [Seq, TotalSeq], V [TotalSeq, Dim]. Result [Seq, Dim]
            Tensor<float> context(seq_len, MODEL_HEAD_DIM);
            std::fill(context.data.begin(), context.data.end(), 0.0f);
            
            for(int r_s=0; r_s<seq_len; ++r_s) {
                for(int d=0; d<MODEL_HEAD_DIM; ++d) {
                    float acc = 0.0f;
                    for(int r_v=0; r_v<total_seq_len; ++r_v) {
                        acc += scores.data[r_s*total_seq_len + r_v] * v_head.data[r_v*MODEL_HEAD_DIM + d];
                    }
                    context.data[r_s*MODEL_HEAD_DIM + d] = acc;
                }
            }

            // Write back
            for(int r=0; r<seq_len; ++r) {
                for(int c=0; c<MODEL_HEAD_DIM; ++c) {
                    attn_output.data[r*MODEL_HIDDEN_SIZE + global_head_idx*MODEL_HEAD_DIM + c] = context.data[r*MODEL_HEAD_DIM + c];
                }
            }
        }
    }

    Tensor<float> attn_proj(seq_len, MODEL_HIDDEN_SIZE);
    run_linear(attn_output, w_o, scale_wo, attn_proj);

    for(size_t i=0; i<input.size(); ++i) {
        input.data[i] = input.data[i] + attn_proj.data[i];
    }
    
    Tensor<float> residual = input; 

    // 5. MLP
    // RMSNorm (MLP)
    for(int i=0; i<seq_len; ++i) {
        sfu_rms_norm(&input.data[i * MODEL_HIDDEN_SIZE], &normed_input.data[i * MODEL_HIDDEN_SIZE], MODEL_HIDDEN_SIZE);
        for(int j=0; j<MODEL_HIDDEN_SIZE; ++j) {
            normed_input.data[i * MODEL_HIDDEN_SIZE + j] *= mlp_norm_weight.data[j];
        }
    }

    Tensor<float> gate(seq_len, MODEL_INTERMEDIATE_SIZE);
    Tensor<float> up(seq_len, MODEL_INTERMEDIATE_SIZE);
    
    run_linear(normed_input, w_gate, scale_wgate, gate);
    run_linear(normed_input, w_up, scale_wup, up);

    for(int i=0; i<seq_len; ++i) {
        sfu_silu(&gate.data[i * MODEL_INTERMEDIATE_SIZE], &gate.data[i * MODEL_INTERMEDIATE_SIZE], MODEL_INTERMEDIATE_SIZE);
        
        sfu_element_mul(
            &gate.data[i * MODEL_INTERMEDIATE_SIZE], 
            &up.data[i * MODEL_INTERMEDIATE_SIZE], 
            &gate.data[i * MODEL_INTERMEDIATE_SIZE], 
            MODEL_INTERMEDIATE_SIZE
        );
    }

    Tensor<float> mlp_output(seq_len, MODEL_HIDDEN_SIZE);
    run_linear(gate, w_down, scale_wdown, mlp_output);

    for(size_t i=0; i<input.size(); ++i) {
        input.data[i] = residual.data[i] + mlp_output.data[i];
    }
}

// --- TinyLlama Implementation ---

TinyLlama::TinyLlama() 
    : embedding_table(32000, MODEL_HIDDEN_SIZE),
      lm_head(MODEL_HIDDEN_SIZE, 32000),
      norm_final_weight(1, MODEL_HIDDEN_SIZE)
{
    // Init Embedding table with random
    for(size_t i=0; i<embedding_table.size(); ++i) embedding_table.data[i] = ((float)(i%100))/100.0f;
    // Init LM Head
    for(size_t i=0; i<lm_head.size(); ++i) lm_head.data[i] = 1;
    // Init Norm Weight
    for(size_t i=0; i<norm_final_weight.size(); ++i) norm_final_weight.data[i] = 1.0f;
    
    build();
}

void TinyLlama::build() {
    int num_layers_to_instantiate = MODEL_LAYERS; 
    for(int i=0; i<num_layers_to_instantiate; ++i) {
        layers.push_back(std::make_shared<TransformerBlock>(i));
    }
}

bool TinyLlama::load_weights(const std::string& dir_path) {
    namespace fs = std::filesystem;
    if (!fs::exists(dir_path)) {
        std::cerr << "Error: Weight directory " << dir_path << " not found." << std::endl;
        return false;
    }
    std::cout << "Loading weights from " << dir_path << "..." << std::endl;

    // Load Embedding
    std::string emb_path = dir_path + "/embedding.bin";
    std::ifstream f_emb(emb_path, std::ios::binary);
    if(f_emb) f_emb.read(reinterpret_cast<char*>(embedding_table.data.data()), embedding_table.size() * sizeof(float));

    // Load LM Head
    std::string lm_path = dir_path + "/lm_head.bin";
    std::ifstream f_lm(lm_path, std::ios::binary);
    if(f_lm) f_lm.read(reinterpret_cast<char*>(lm_head.data.data()), lm_head.size());
    
    std::string lm_scale_path = dir_path + "/lm_head_scale.bin";
    std::ifstream f_lm_scale(lm_scale_path, std::ios::binary);
    if(f_lm_scale) f_lm_scale.read(reinterpret_cast<char*>(&scale_lm_head), sizeof(float));

    // Load Final Norm
    std::string norm_path = dir_path + "/norm_final.bin";
    std::ifstream f_norm(norm_path, std::ios::binary);
    if(f_norm) f_norm.read(reinterpret_cast<char*>(norm_final_weight.data.data()), norm_final_weight.size() * sizeof(float));

    // Load Layers
    for (int i = 0; i < layers.size(); ++i) {
        std::string layer_dir = dir_path + "/layer_" + std::to_string(i);
        auto& block = layers[i];
        
        auto load_bin = [&](std::string name, Tensor<int8_t>& t, float& scale) {
            std::string path_data = layer_dir + "/" + name + ".bin";
            std::string path_scale = layer_dir + "/" + name + "_scale.bin";
            std::ifstream f_data(path_data, std::ios::binary);
            if(f_data) f_data.read(reinterpret_cast<char*>(t.data.data()), t.size());
            std::ifstream f_scale(path_scale, std::ios::binary);
            if(f_scale) f_scale.read(reinterpret_cast<char*>(&scale), sizeof(float));
        };
        load_bin("wq", block->get_wq(), block->get_scale_wq());
        load_bin("wk", block->get_wk(), block->get_scale_wk());
        load_bin("wv", block->get_wv(), block->get_scale_wv());
        load_bin("wo", block->get_wo(), block->get_scale_wo());
        load_bin("wgate", block->get_wgate(), block->get_scale_wgate());
        load_bin("wup", block->get_wup(), block->get_scale_wup());
        load_bin("wdown", block->get_wdown(), block->get_scale_wdown());
        
        // Load Layer Norms (Float)
        std::string attn_norm_path = layer_dir + "/attention_norm.bin";
        std::ifstream f_an(attn_norm_path, std::ios::binary);
        if(f_an) f_an.read(reinterpret_cast<char*>(block->get_attention_norm_weight().data.data()), block->get_attention_norm_weight().size() * sizeof(float));

        std::string mlp_norm_path = layer_dir + "/mlp_norm.bin";
        std::ifstream f_mn(mlp_norm_path, std::ios::binary);
        if(f_mn) f_mn.read(reinterpret_cast<char*>(block->get_mlp_norm_weight().data.data()), block->get_mlp_norm_weight().size() * sizeof(float));
    }
    return true;
}

bool TinyLlama::load_tokenizer(const std::string& path) {
    return tokenizer.load(path, 32000);
}

void TinyLlama::forward(Tensor<float>& hidden_states, int start_pos) {
    for(auto& layer : layers) {
        layer->forward(hidden_states, start_pos);
    }
}

std::string TinyLlama::generate(const std::string& prompt, int max_new_tokens) {
    // 1. Encode
    std::vector<int> tokens = tokenizer.encode(prompt, true, false);
    if(tokens.empty()) return "";

    std::string full_generated_text = prompt; // Start with prompt

    int start_pos = 0;
    
    // Prefill
    int seq_len = tokens.size();
    Tensor<float> hidden_states(seq_len, MODEL_HIDDEN_SIZE);
    
    for(int i=0; i<seq_len; ++i) {
        int token_id = tokens[i];
        if(token_id >= 32000) token_id = 0; 
        for(int j=0; j<MODEL_HIDDEN_SIZE; ++j) {
            hidden_states.data[i*MODEL_HIDDEN_SIZE + j] = embedding_table.data[token_id * MODEL_HIDDEN_SIZE + j];
        }
    }

    forward(hidden_states, 0);
    start_pos += seq_len;

    Tensor<float> last_token_emb(1, MODEL_HIDDEN_SIZE);
    for(int j=0; j<MODEL_HIDDEN_SIZE; ++j) {
        last_token_emb.data[j] = hidden_states.data[(seq_len-1)*MODEL_HIDDEN_SIZE + j];
    }

    // Generation Loop
    for(int step=0; step<max_new_tokens; ++step) {
        // 1. Final Norm
        Tensor<float> normed_emb(1, MODEL_HIDDEN_SIZE);
        sfu_rms_norm(last_token_emb.ptr(), normed_emb.ptr(), MODEL_HIDDEN_SIZE);
        for(int i=0; i<MODEL_HIDDEN_SIZE; ++i) {
            normed_emb.data[i] *= norm_final_weight.data[i];
        }

        // 2. LM Head
        Tensor<int8_t> input_q(1, MODEL_HIDDEN_SIZE);
        float input_scale;
        quantize(normed_emb, input_q, input_scale);

        Tensor<int32_t> logits_acc(1, 32000);
        
        run_accelerator_padded(
            input_q.ptr(), 
            lm_head.ptr(), 
            logits_acc.ptr(),
            1, MODEL_HIDDEN_SIZE, 32000
        );

        Tensor<float> logits(1, 32000);
        dequantize(logits_acc, logits, input_scale * scale_lm_head);

        // 2.5 Repetition Penalty
        float penalty = 1.2f;
        // Collect past tokens (prompt tokens + generated tokens)
        // We need to keep track of them.
        // Simple scan:
        for(int id : tokens) {
            if(logits.data[id] < 0) logits.data[id] *= penalty;
            else logits.data[id] /= penalty;
        }
        // Also add previously generated ones? 
        // We only have `full_generated_text` as string.
        // We need `generated_token_ids`.
        // Let's modify the loop to store them.
        // (Assuming logic update in this block implicitly adds support or just penalizes prompt for now which helps)
        
        // 3. Top-K Sampling
        int k_sample = 5;
        float temperature = 0.7f;
        
        // Apply temperature
        for(size_t i=0; i<logits.size(); ++i) logits.data[i] /= temperature;

        // Find Top-K
        std::vector<std::pair<float, int>> top_k;
        top_k.reserve(32000);
        for(int i=0; i<32000; ++i) {
            top_k.push_back({logits.data[i], i});
        }
        
        // Partial sort to get top K elements
        std::partial_sort(top_k.begin(), top_k.begin() + k_sample, top_k.end(), std::greater<std::pair<float, int>>());
        
        // Softmax on Top-K
        float max_val_k = top_k[0].first;
        float sum_exp = 0.0f;
        std::vector<float> probs(k_sample);
        for(int i=0; i<k_sample; ++i) {
            probs[i] = std::exp(top_k[i].first - max_val_k);
            sum_exp += probs[i];
        }
        
        // Sample
        static std::mt19937 gen(1234); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dis(0.0f, sum_exp);
        float r = dis(gen);
        
        int best_token_id = top_k[0].second;
        float cum_sum = 0.0f;
        for(int i=0; i<k_sample; ++i) {
            cum_sum += probs[i];
            if(r <= cum_sum) {
                best_token_id = top_k[i].second;
                break;
            }
        }

        // 4. Decode
        std::string piece = tokenizer.decode(best_token_id);
        
        // Replace SentencePiece space (U+2581) with normal space
        // U+2581 in UTF-8 is E2 96 81
        size_t pos = 0;
        std::string sp_space = "\xe2\x96\x81";
        while ((pos = piece.find(sp_space, pos)) != std::string::npos) {
            piece.replace(pos, sp_space.length(), " ");
            pos += 1;
        }

        full_generated_text += piece;

        if(best_token_id == tokenizer.token_eos_id) break;
        
        tokens.push_back(best_token_id);

        // 5. Next step
        for(int j=0; j<MODEL_HIDDEN_SIZE; ++j) {
            last_token_emb.data[j] = embedding_table.data[best_token_id * MODEL_HIDDEN_SIZE + j];
        }

        forward(last_token_emb, start_pos);
        start_pos++;
    }

    return full_generated_text;
}