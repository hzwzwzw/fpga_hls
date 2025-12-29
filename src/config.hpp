#ifndef CONFIG_HPP
#define CONFIG_HPP

#include "mpe.hpp"

// Define the tile dimensions. For our architecture, the tile size
// must match the MPE size, as the MPE is our core processing block.
#define TILE_ROWS MPE_ROWS
#define TILE_COLS MPE_COLS

// For now, we assume a single MPE. We will explore NUM_MPES later.
#define NUM_MPES 1

// --- LLaMA-2 Model Configuration ---
// These are configured for TinyLlama-1.1B
const int DIM = 2048;       // Model dimension
const int HIDDEN_DIM = 5632; // FFN hidden dimension
const int N_HEADS = 32;     // Number of heads
const int N_KV_HEADS = 4;   // Number of KV heads for GQA
const int HEAD_DIM = DIM / N_HEADS; // Dimension of each head
const int KV_DIM = 256; // Dimension for K and V weights in GQA
const int VOCAB_SIZE = 32000; // Vocabulary size

// Maximum sequence length for our KV cache
const int MAX_SEQ_LEN = 512;



#endif // CONFIG_HPP
