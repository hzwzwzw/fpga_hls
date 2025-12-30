import os
import struct
import torch
import numpy as np
from safetensors.torch import load_file
import json

# Configuration
INPUT_DIR = "weights"
OUTPUT_DIR = "weights_bin"
TOKENIZER_MODEL = os.path.join(INPUT_DIR, "tokenizer.model")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading model from {INPUT_DIR}...")
    
    # Try loading safetensors first, then bin
    state_dict = {}
    safetensors_path = os.path.join(INPUT_DIR, "model.safetensors")
    bin_path = os.path.join(INPUT_DIR, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        # Maybe sharded? TinyLlama is usually 1 file. 
        # Check for index
        index_path = os.path.join(INPUT_DIR, "model.safetensors.index.json")
        if os.path.exists(index_path):
            print("Detected sharded safetensors. Loading all shards...")
            with open(index_path, 'r') as f:
                index = json.load(f)
            files = set(index['weight_map'].values())
            for filename in files:
                path = os.path.join(INPUT_DIR, filename)
                print(f"Loading {filename}...")
                shard = load_file(path)
                state_dict.update(shard)
        else:
            print("Error: Could not find model.safetensors or pytorch_model.bin")
            return

    print(f"Model loaded. Keys: {len(state_dict)}")

    # Helper to quantize and save
    def save_quantized(tensor, path_prefix):
        # tensor: torch.Tensor
        # If 2D (Linear layer), Transpose it to [In, Out] for C++ GEMM (A * B)
        if len(tensor.shape) == 2:
            tensor = tensor.t()
            
        data = tensor.float().numpy()
        
        # Calculate scale
        max_val = np.max(np.abs(data))
        scale = max_val / 127.0
        if scale == 0: scale = 1.0
        
        # Quantize
        data_int8 = np.round(data / scale).astype(np.int8)
        
        # Save Data
        with open(path_prefix + ".bin", "wb") as f:
            f.write(data_int8.tobytes())
            
        # Save Scale
        with open(path_prefix + "_scale.bin", "wb") as f:
            f.write(struct.pack('f', scale))
            
    def save_float(tensor, path):
        # Norms are 1D, no transpose needed.
        # Embedding is 2D but [Vocab, Dim] which matches memory layout we want.
        # So generally only transpose "Weights" of Linear layers.
        # But save_float is used for Embedding and Norms.
        # Embedding in PyTorch: [Vocab, Dim]. 
        # C++ lookup: row-major. So we want [Vocab, Dim].
        # So NO Transpose for save_float.
        data = tensor.float().numpy()
        with open(path, "wb") as f:
            f.write(data.tobytes())

    # 1. Embedding
    print("Exporting Embedding...")
    embed_weight = state_dict.get("model.embed_tokens.weight")
    if embed_weight is None: embed_weight = state_dict.get("embed_tokens.weight") # Try fallback
    save_float(embed_weight, os.path.join(OUTPUT_DIR, "embedding.bin"))

    # 2. Layers
    n_layers = 22 # TinyLlama default
    # Detect layers
    layer_indices = set()
    for key in state_dict.keys():
        if "layers." in key:
            try:
                # model.layers.0.xxx
                parts = key.split('.')
                idx = int(parts[2])
                layer_indices.add(idx)
            except:
                pass
    if layer_indices:
        n_layers = max(layer_indices) + 1
    
    print(f"Exporting {n_layers} layers...")
    
    for i in range(n_layers):
        layer_dir = os.path.join(OUTPUT_DIR, f"layer_{i}")
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir)
            
        prefix = f"model.layers.{i}."
        
        # Attention
        save_quantized(state_dict[prefix + "self_attn.q_proj.weight"], os.path.join(layer_dir, "wq"))
        save_quantized(state_dict[prefix + "self_attn.k_proj.weight"], os.path.join(layer_dir, "wk"))
        save_quantized(state_dict[prefix + "self_attn.v_proj.weight"], os.path.join(layer_dir, "wv"))
        save_quantized(state_dict[prefix + "self_attn.o_proj.weight"], os.path.join(layer_dir, "wo"))
        
        # MLP
        save_quantized(state_dict[prefix + "mlp.gate_proj.weight"], os.path.join(layer_dir, "wgate"))
        save_quantized(state_dict[prefix + "mlp.up_proj.weight"], os.path.join(layer_dir, "wup"))
        save_quantized(state_dict[prefix + "mlp.down_proj.weight"], os.path.join(layer_dir, "wdown"))
        
        # Norms (RMSNorm weights are just scales, 1D float)
        # Note: My C++ code implements RMSNorm purely on input? 
        # Wait, C++ `TransformerBlock` does `sfu_rms_norm` but WHERE IS THE WEIGHT?
        # AHH, I missed the RMSNorm weights in `TransformerBlock` C++ implementation!
        # In `TransformerBlock::forward`, I do `sfu_rms_norm` which standardizes.
        # But Llama RMSNorm is `x * rsqrt(mean(x^2)) * weight`.
        # I only implemented the first part.
        # I NEED TO ADD `attention_norm` and `mlp_norm` weights to `TransformerBlock`.
        
        # I will save them now, and I SHOULD UPDATE C++ code to load and use them.
        save_float(state_dict[prefix + "input_layernorm.weight"], os.path.join(layer_dir, "attention_norm.bin"))
        save_float(state_dict[prefix + "post_attention_layernorm.weight"], os.path.join(layer_dir, "mlp_norm.bin"))

    # 3. Final Norm & LM Head
    print("Exporting Final Norm & LM Head...")
    save_float(state_dict["model.norm.weight"], os.path.join(OUTPUT_DIR, "norm_final.bin")) # Wait, C++ uses `norm_final_weight` and checks hardcoded path?
    # In C++ `load_weights`: reads `norm_final_weight`? No, I checked `model.cpp`:
    # `load_weights` does NOT load `norm_final_weight` in the loop I wrote earlier?
    # Let me check `model.cpp` content again. 
    # I wrote: `load_bin` loop inside `layers` loop.
    # And BEFORE that: `load Embedding`, `load LM Head`.
    # I DID NOT load `norm_final_weight` from file in C++ `load_weights`! I initialized it to 1s.
    
    # I need to fix C++ code to load `norm_final.bin` and layer norms.
    
    save_quantized(state_dict["lm_head.weight"], os.path.join(OUTPUT_DIR, "lm_head"))

    # 4. Tokenizer
    print("Exporting Tokenizer...")
    export_tokenizer()
    
    print("Done.")

def export_tokenizer():
    # Attempt to load tokenizer.model using sentencepiece
    try:
        from sentencepiece import SentencePieceProcessor
    except ImportError:
        print("Error: sentencepiece module not found. Please install it: pip install sentencepiece")
        return

    if not os.path.exists(TOKENIZER_MODEL):
        print(f"Error: {TOKENIZER_MODEL} not found.")
        return

    sp = SentencePieceProcessor(model_file=TOKENIZER_MODEL)
    vocab_size = sp.vocab_size()
    
    with open(os.path.join(OUTPUT_DIR, "tokenizer.bin"), "wb") as f:
        # Write max_token_length (dummy/estimate or max len)
        f.write(struct.pack('i', 32)) # Estimate max token len?
        
        for i in range(vocab_size):
            piece = sp.id_to_piece(i)
            score = sp.get_score(i)
            piece_bytes = piece.encode('utf-8')
            
            f.write(struct.pack('f', score))
            f.write(struct.pack('i', len(piece_bytes)))
            f.write(piece_bytes)

if __name__ == "__main__":
    main()
