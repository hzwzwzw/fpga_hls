import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import numpy as np
import struct

# --- Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_FILE = "model.bin" # Output to a binary file
VOCAB_OUTPUT_FILE = "vocab.txt"
N_LAYERS = 2 # Must match N_LAYERS in flight_llama2.hpp
DIM = 2048     # Must match DIM in llama2_block.hpp
HIDDEN_DIM = 5632 # Must match HIDDEN_DIM for TinyLlama

# Global variable to hold the current tensor name for debugging
current_name = ""

def write_tensor_binary(f, tensor):
    """Writes a tensor's raw bytes to the binary file."""
    print(f"  - Tensor Name: {current_name}, Shape: {tensor.shape}")
    d = tensor.detach().cpu().contiguous().view(-1).numpy().astype(np.float32)
    b = d.tobytes()
    f.write(b)
    print(f"    - Wrote {tensor.numel()} params, {len(b)} bytes")

def export_vocab(tokenizer, filepath):
    print(f"Writing vocabulary to {filepath}...")
    with open(filepath, "w", encoding="utf-8") as f:
        for i in range(tokenizer.vocab_size):
            token = tokenizer.decode([i])
            f.write(f"{token}\n")

def main():
    global current_name
    print(f"Loading model: {MODEL_NAME}...")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    state_dict = model.state_dict()

    export_vocab(tokenizer, VOCAB_OUTPUT_FILE)
    
    print(f"Writing weights to binary file: {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "wb") as f:
        # The order of writing MUST EXACTLY MATCH the C++ struct layout.
        
        print("Writing token embedding table...")
        current_name = "token_embedding_table"
        emb = state_dict['model.embed_tokens.weight']
        write_tensor_binary(f, emb)

        # --- Transformer Layers ---
        for l in range(N_LAYERS):
            print(f"Writing layer {l}...")
            
            # RMS Norm weights
            current_name = f"layers.{l}.rms_att_weight"
            write_tensor_binary(f, state_dict[f'model.layers.{l}.input_layernorm.weight'])
            current_name = f"layers.{l}.rms_ffn_weight"
            write_tensor_binary(f, state_dict[f'model.layers.{l}.post_attention_layernorm.weight'])
            
            # Attention weights
            current_name = f"layers.{l}.attention.wq"
            write_tensor_binary(f, state_dict[f'model.layers.{l}.self_attn.q_proj.weight'])
            current_name = f"layers.{l}.attention.wk"
            write_tensor_binary(f, state_dict[f'model.layers.{l}.self_attn.k_proj.weight'])
            current_name = f"layers.{l}.attention.wv"
            write_tensor_binary(f, state_dict[f'model.layers.{l}.self_attn.v_proj.weight'])
            current_name = f"layers.{l}.attention.wo"
            write_tensor_binary(f, state_dict[f'model.layers.{l}.self_attn.o_proj.weight'])

            # FFN weights
            current_name = f"layers.{l}.ffn.w1"
            write_tensor_binary(f, state_dict[f'model.layers.{l}.mlp.gate_proj.weight'])
            current_name = f"layers.{l}.ffn.w2"
            write_tensor_binary(f, state_dict[f'model.layers.{l}.mlp.down_proj.weight'])
            current_name = f"layers.{l}.ffn.w3"
            write_tensor_binary(f, state_dict[f'model.layers.{l}.mlp.up_proj.weight'])

        # --- Final RMS Norm ---
        print("Writing final RMS norm...")
        current_name = "rms_final_weight"
        write_tensor_binary(f, state_dict['model.norm.weight'])

        # --- Classifier Weights ---
        print("Writing classifier weights...")
        current_name = "wcls"
        wcls = state_dict['lm_head.weight']
        write_tensor_binary(f, wcls)
        
    print(f"\nDone. Binary weights saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()