#ifndef CONFIG_HPP
#define CONFIG_HPP

#include "mpe.hpp"

// Define the tile dimensions. For our architecture, the tile size
// must match the MPE size, as the MPE is our core processing block.
#define TILE_ROWS MPE_ROWS
#define TILE_COLS MPE_COLS

// For now, we assume a single MPE. We will explore NUM_MPES later.
#define NUM_MPES 1

// TinyLlama-1.1B Configuration
// Hidden size: 2048
// Intermediate size: 5632
// Number of heads: 32
// Number of KV heads: 4 (Grouped Query Attention)
// Head dimension: 2048 / 32 = 64
// Layers: 22
// Vocab size: 32000

#define MODEL_HIDDEN_SIZE 2048
#define MODEL_INTERMEDIATE_SIZE 5632
#define MODEL_NUM_HEADS 32
#define MODEL_NUM_KV_HEADS 4
#define MODEL_HEAD_DIM (MODEL_HIDDEN_SIZE / MODEL_NUM_HEADS)
#define MODEL_LAYERS 22

#endif // CONFIG_HPP
