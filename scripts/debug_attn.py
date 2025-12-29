import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import numpy as np
import os

# This script focuses exclusively on the attention block of the first layer.
# It saves the output of every micro-step within the attention calculation
# to definitively identify the point of divergence.

# --- Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEBUG_DIR = "debug_attn_out"

# --- Model Dimensions (must match C++ config) ---
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
    save_tensor(h_norm, "0_attn_input.bin") # The input to the whole process

    # --- Start of Attention Block ---
    print("\nExecuting Attention Block Step-by-Step...")
    bsz, q_len, _ = h_norm.shape
    
    # 1. QKV Projections
    q_states = layer.self_attn.q_proj(h_norm)
    k_states = layer.self_attn.k_proj(h_norm)
    v_states = layer.self_attn.v_proj(h_norm)
    
    # 2. Reshape and apply RoPE
    q_for_rope = q_states.view(bsz, q_len, N_HEADS, HEAD_DIM).transpose(1, 2)
    k_for_rope = k_states.view(bsz, q_len, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
    
    position_ids = torch.arange(pos, pos + q_len, dtype=torch.long).unsqueeze(0)
    cos, sin = model.model.rotary_emb(v_states, position_ids=position_ids)
    q_rotated, k_rotated = apply_rotary_pos_emb(q_for_rope, k_for_rope, cos, sin, position_ids)
    save_tensor(q_rotated, "1_q_rope.bin")
    save_tensor(k_rotated, "2_k_rope.bin")

    # 3. Attention score calculation
    k_repeated = k_rotated.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)
    attn_weights = torch.matmul(q_rotated, k_repeated.transpose(2, 3)) / np.sqrt(HEAD_DIM)
    save_tensor(attn_weights, "3_presoftmax_scores.bin")
    
    # 4. Softmax
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_states.dtype)
    save_tensor(attn_weights, "4_postsoftmax_scores.bin")
    
    # 5. Value weighting
    v_for_attn = v_states.view(bsz, q_len, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v_repeated = v_for_attn.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)
    attn_output_heads = torch.matmul(attn_weights, v_repeated)
    save_tensor(attn_output_heads, "5_attn_heads_out.bin")

    # 6. Output projection
    attn_output = attn_output_heads.transpose(1, 2).contiguous().view(bsz, q_len, -1)
    attn_output = layer.self_attn.o_proj(attn_output)
    save_tensor(attn_output, "6_attn_proj_out.bin")

    print("\nAttention block debug export finished.")

if __name__ == "__main__":
    main()
