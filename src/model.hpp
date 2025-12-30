#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include <string>
#include <memory>
#include "config.hpp"
#include "accelerator.hpp"
#include "sfu.hpp"

// Simple Tensor structure for Host memory management
template <typename T>
struct Tensor {
    std::vector<T> data;
    int rows;
    int cols;

    Tensor(int r, int c) : rows(r), cols(c) {
        data.resize(r * c, 0);
    }
    
    T* ptr() { return data.data(); }
    const T* ptr() const { return data.data(); }
    size_t size() const { return data.size(); }
};

#include "tokenizer.hpp"

// Host-side helper functions
struct QuantParams {
    float scale;
};

void quantize(const Tensor<float>& input, Tensor<int8_t>& output, float& scale);
void dequantize(const Tensor<int32_t>& input, Tensor<float>& output, float scale);
// For residual addition inputs or re-quantization
void dequantize_i8(const Tensor<int8_t>& input, Tensor<float>& output, float scale); 

class TransformerBlock {
// ... (keep existing content) ...
public:
    TransformerBlock(int layer_id);

    // Initialize weights (dummy/random for now)
    void initialize_weights();

    // Forward pass with KV Cache support
    // input: [SeqLen, HiddenSize]
    // start_pos: The starting position in the sequence (0 for prompt, >0 for decode)
    void forward(Tensor<float>& input, int start_pos);

    // Accessors for weights to load data
    Tensor<int8_t>& get_wq() { return w_q; }
    Tensor<int8_t>& get_wk() { return w_k; }
    Tensor<int8_t>& get_wv() { return w_v; }
    Tensor<int8_t>& get_wo() { return w_o; }
    Tensor<int8_t>& get_wgate() { return w_gate; }
    Tensor<int8_t>& get_wup() { return w_up; }
    Tensor<int8_t>& get_wdown() { return w_down; }

    // Accessors for scales
    float& get_scale_wq() { return scale_wq; }
    float& get_scale_wk() { return scale_wk; }
    float& get_scale_wv() { return scale_wv; }
    float& get_scale_wo() { return scale_wo; }
    float& get_scale_wgate() { return scale_wgate; }
    float& get_scale_wup() { return scale_wup; }
    float& get_scale_wdown() { return scale_wdown; }

    // Accessors for Norms
    Tensor<float>& get_attention_norm_weight() { return attention_norm_weight; }
    Tensor<float>& get_mlp_norm_weight() { return mlp_norm_weight; }

private:
    int layer_id;
    
    // Weights (Storage as int8 for Accelerator)
    Tensor<int8_t> w_q; float scale_wq = 1.0f;
    Tensor<int8_t> w_k; float scale_wk = 1.0f;
    Tensor<int8_t> w_v; float scale_wv = 1.0f;
    Tensor<int8_t> w_o; float scale_wo = 1.0f;
    
    Tensor<int8_t> w_gate; float scale_wgate = 1.0f;
    Tensor<int8_t> w_up;   float scale_wup = 1.0f;
    Tensor<int8_t> w_down; float scale_wdown = 1.0f;

    // RMSNorm weights
    Tensor<float> attention_norm_weight;
    Tensor<float> mlp_norm_weight;

    // KV Cache
    // Shape: [MaxSeqLen, NumKVHeads, HeadDim]
    // We flatten it to 1D for Tensor but treat it logically.
    // However, to keep it simple, we can use [MaxSeqLen, HiddenKV] where HiddenKV = NumKVHeads * HeadDim
    Tensor<float> k_cache; 
    Tensor<float> v_cache;

    // Helper to run GEMM: C = A * B
    // Wraps flight_llm_accelerator and handles type conversions and scaling
    void run_linear(const Tensor<float>& input, const Tensor<int8_t>& weight, float weight_scale, Tensor<float>& output);

    // RoPE rotation (Host implementation for now)
    void apply_rope(Tensor<float>& q, Tensor<float>& k, int start_pos);
};

class TinyLlama {
public:
    TinyLlama();
    void build();
    
    // Load weights from a binary directory
    bool load_weights(const std::string& dir_path);
    
    // Load tokenizer
    bool load_tokenizer(const std::string& path);

    // Full Generation Pipeline
    // Returns the generated text
    std::string generate(const std::string& prompt, int max_new_tokens);

    // Forward pass for the whole model
    // hidden_states: [SeqLen, HiddenSize]
    // start_pos: 0 for prefill, >0 for decode
    void forward(Tensor<float>& hidden_states, int start_pos);
    
    // Accessor for final norm
    Tensor<float>& get_norm_final_weight() { return norm_final_weight; }

private:
    std::vector<std::shared_ptr<TransformerBlock>> layers;
    
    Tokenizer tokenizer;
    
    // Embedding Table
    // Size: [VocabSize, HiddenSize]
    // For simplicity, store as Float. In optimized version, could be int8 or quantized.
    Tensor<float> embedding_table;
    
    // RMSNorm Final
    // We usually have a final norm before the LM Head
    // We can reuse the SFU function, but we need the weight vector.
    // Wait, TinyLlama has a weight for the final RMSNorm.
    Tensor<float> norm_final_weight; // [HiddenSize]

    // LM Head (Output Projection)
    // [HiddenSize, VocabSize]
    // Usually shared with embedding table in some models, but Llama usually has separate or same?
    // TinyLlama: Separate usually, or tied. We'll implement as separate for generality.
    Tensor<int8_t> lm_head; float scale_lm_head = 1.0f;
};

#endif // MODEL_HPP
