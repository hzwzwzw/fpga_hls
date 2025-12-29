import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import numpy as np
import os

# This script performs a single forward pass and PRINTS the state of
# key tensors at each micro-step to create a golden trace log.

# --- Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# --- Model Dimensions (must match C++ config) ---
DIM = 2048
HIDDEN_DIM = 5632
N_HEADS = 32
N_KV_HEADS = 4
HEAD_DIM = DIM // N_HEADS

def trace_tensor(name, tensor):
    """Prints the trace information for a tensor."""
    tensor = tensor.detach().cpu().view(-1).to(torch.float32)
    # Truncate for printing
    head = ", ".join([f"{x:.6f}" for x in tensor[:3].tolist()])
    tensor_sum = tensor.sum().item()
    print(f"--- {name} ---")
    print(f"  - Head: [{head}, ...]")
    print(f"  - Sum:  {tensor_sum:.6f}")
    print()

def main():
    print(f"Loading model: {MODEL_NAME}...")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    
    input_token_id = torch.tensor([[1]]) # BOS token
    pos = 0
    
    # --- 1. Embedding ---
    h = model.model.embed_tokens(input_token_id).squeeze(0) # Shape: [1, 2048]
    trace_tensor("1. Embedding Output", h)

    layer = model.model.layers[0]
    residual = h

    # --- 2. Pre-Attention RMSNorm ---
    h_norm = layer.input_layernorm(h)
    trace_tensor("2. Pre-Att RMSNorm Output", h_norm)

    # --- 3. Q, K, V Projections ---
    q_states = layer.self_attn.q_proj(h_norm)
    k_states = layer.self_attn.k_proj(h_norm)
    v_states = layer.self_attn.v_proj(h_norm)
    trace_tensor("3. Q Proj Output", q_states)
    trace_tensor("3. K Proj Output", k_states)
    trace_tensor("3. V Proj Output", v_states)
    
    # --- 4. RoPE ---
    bsz, q_len, _ = h_norm.unsqueeze(0).shape
    q_for_rope = q_states.view(bsz, q_len, N_HEADS, HEAD_DIM).transpose(1, 2)
    k_for_rope = k_states.view(bsz, q_len, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
    
    position_ids = torch.arange(pos, pos + q_len, dtype=torch.long).unsqueeze(0)
    cos, sin = model.model.rotary_emb(v_states, position_ids=position_ids)
    q_rotated, k_rotated = apply_rotary_pos_emb(q_for_rope, k_for_rope, cos, sin, position_ids)
    trace_tensor("4. Q RoPE Output", q_rotated)
    trace_tensor("4. K RoPE Output", k_rotated)

    # --- 5. Attention Calculation ---
    k_repeated = k_rotated.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)
    attn_weights = torch.matmul(q_rotated, k_repeated.transpose(2, 3)) / np.sqrt(HEAD_DIM)
    trace_tensor("5. Pre-Softmax Scores", attn_weights)
    
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_states.dtype)
    trace_tensor("5. Post-Softmax Scores", attn_weights)
    
    v_for_attn = v_states.view(bsz, q_len, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v_repeated = v_for_attn.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)
    attn_output_heads = torch.matmul(attn_weights, v_repeated)
    trace_tensor("5. Attn Heads Output", attn_output_heads)

    # --- 6. Attention Output Projection ---
    attn_output = attn_output_heads.transpose(1, 2).contiguous().view(bsz, q_len, -1)
    attn_output = layer.self_attn.o_proj(attn_output).squeeze(0)
    trace_tensor("6. Attn Proj Output", attn_output)

    # --- 7. First Residual ---
    h = residual + attn_output
    trace_tensor("7. First Residual Output", h)
    residual = h

    # --- 8. Pre-FFN RMSNorm ---
    h_norm = layer.post_attention_layernorm(h)
    trace_tensor("8. Pre-FFN RMSNorm Output", h_norm)

    # --- 9. FFN ---
    ffn_out = layer.mlp(h_norm)
    trace_tensor("9. FFN Output", ffn_out)

    # --- 10. Second Residual ---
    h = residual + ffn_out
    trace_tensor("10. Final Block Output", h)

    print("\nPython trace finished.")

if __name__ == "__main__":
    main()
