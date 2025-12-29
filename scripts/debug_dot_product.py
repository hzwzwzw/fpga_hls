import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import numpy as np
import os

# This script isolates a single dot product operation to provide definitive
# proof of numerical differences between C++ and PyTorch.

# --- Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEBUG_DIR = "debug_dot_product_out"

# --- Model Dimensions ---
DIM = 2048
N_HEADS = 32
N_KV_HEADS = 4
HEAD_DIM = DIM // N_HEADS

def save_tensor(tensor, name):
    """Saves a tensor to the debug directory."""
    filepath = os.path.join(DEBUG_DIR, name)
    print(f"  - Saving {name}, shape: {tensor.shape}")
    d = tensor.detach().cpu().view(-1).numpy().astype(np.float32)
    with open(filepath, "wb") as f:
        f.write(d.tobytes())

def main():
    if not os.path.exists(DEBUG_DIR):
        os.makedirs(DEBUG_DIR)

    print(f"Loading model: {MODEL_NAME}...")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    
    input_token_id = torch.tensor([[1]]) # BOS token
    pos = 0
    
    # --- Get Inputs for the Attention Block ---
    h = model.model.embed_tokens(input_token_id)
    layer = model.model.layers[0]
    h_norm = layer.input_layernorm(h)
    
    # --- Get Q and K vectors ---
    bsz, q_len, _ = h_norm.shape
    q_states = layer.self_attn.q_proj(h_norm)
    k_states = layer.self_attn.k_proj(h_norm)
    v_states = layer.self_attn.v_proj(h_norm) # Needed for rotary_emb call
    
    q_for_rope = q_states.view(bsz, q_len, N_HEADS, HEAD_DIM).transpose(1, 2)
    k_for_rope = k_states.view(bsz, q_len, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
    
    position_ids = torch.arange(pos, pos + q_len, dtype=torch.long).unsqueeze(0)
    cos, sin = model.model.rotary_emb(v_states, position_ids=position_ids)
    q_rotated, k_rotated = apply_rotary_pos_emb(q_for_rope, k_for_rope, cos, sin, position_ids)

    # --- Isolate the first head ---
    q_head_0 = q_rotated[0, 0, 0, :] # First batch, first head, first token
    k_head_0 = k_rotated[0, 0, 0, :] # First batch, first head, first token
    
    save_tensor(q_head_0, "dot_input_q.bin")
    save_tensor(k_head_0, "dot_input_k.bin")

    # --- Perform dot product in PyTorch ---
    score = torch.dot(q_head_0, k_head_0)
    
    save_tensor(score, "dot_output_golden.bin")
        
    print("\nDot product verification files generated in 'debug_dot_product_out/' directory.")

if __name__ == "__main__":
    main()
