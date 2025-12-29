import torch
from transformers import LlamaForCausalLM
import numpy as np
import os

# This script isolates a single matrix-vector multiplication to definitively
# verify the correctness of the C++ mat_vec_mul loop against PyTorch.

# --- Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
VERIFY_DIR = "verify_out"
DIM = 2048

def save_tensor(tensor, name):
    """Saves a tensor to the verification directory."""
    filepath = os.path.join(VERIFY_DIR, name)
    d = tensor.detach().cpu().view(-1).numpy().astype(np.float32)
    with open(filepath, "wb") as f:
        f.write(d.tobytes())
    print(f"Saved {name}, shape: {tensor.shape}")

def main():
    if not os.path.exists(VERIFY_DIR):
        os.makedirs(VERIFY_DIR)

    print(f"Loading model: {MODEL_NAME}...")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    
    # 1. Get a single weight matrix (e.g., layer 0, q_proj)
    weights = model.model.layers[0].self_attn.q_proj.weight
    save_tensor(weights, "verify_weights.bin")

    # 2. Create a simple input vector (all ones)
    input_vec = torch.ones(DIM, dtype=torch.float32)
    save_tensor(input_vec, "verify_input.bin")

    # 3. Perform matmul in PyTorch (W @ x)
    # Note: PyTorch nn.Linear does y = x @ W.T, so we do W @ x to match C++
    output_vec = torch.matmul(weights, input_vec)
    save_tensor(output_vec, "verify_output_golden.bin")
        
    print("\nVerification files generated in 'verify_out/' directory.")

if __name__ == "__main__":
    main()
