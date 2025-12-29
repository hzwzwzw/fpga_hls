import torch
from transformers import LlamaForCausalLM
import numpy as np
import os

# This script performs a SINGLE forward pass and saves the final logits vector.
# It also prints the token ID with the highest logit (the argmax result).

# --- Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_FILE = "final_logits_golden.bin"
DIM = 2048

def save_tensor(tensor, name):
    """Saves a tensor to a binary file."""
    print(f"  - Saving {name}, shape: {tensor.shape}")
    d = tensor.detach().cpu().view(-1).numpy().astype(np.float32)
    with open(name, "wb") as f:
        f.write(d.tobytes())

def main():
    print(f"Loading model: {MODEL_NAME}...")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    
    # --- Prepare Input: BOS token at pos 0 ---
    input_token_id = torch.tensor([[1]]) # BOS token
    
    # --- Perform a full forward pass ---
    print("Performing forward pass...")
    with torch.no_grad():
        outputs = model(input_token_id)
        logits = outputs.logits

    # The logits tensor for a single token has shape [1, 1, VOCAB_SIZE]
    logits = logits.squeeze() # Shape: [VOCAB_SIZE]
    
    # --- Save golden logits vector ---
    save_tensor(logits, OUTPUT_FILE)

    # --- Save a sample weight tensor for loading verification ---
    wq_weights = model.model.layers[0].self_attn.q_proj.weight
    save_tensor(wq_weights, "wq_golden.bin")

    # --- Find and print the argmax result ---
    next_token_id = torch.argmax(logits).item()
    print(f"\n--- Verification Info ---")
    print(f"Golden Logits saved to: {OUTPUT_FILE}")
    print(f"Golden WQ weights saved to: wq_golden.bin")
    print(f"PyTorch Argmax Token ID: {next_token_id}")
    
    print("\nVerification script finished.")

if __name__ == "__main__":
    main()
