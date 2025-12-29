#ifndef ROPE_HPP
#define ROPE_HPP

#include "config.hpp"
#include "sfu.hpp"
#include "vpu.hpp"

// Define the dimension of the rotary embeddings.
// This must match the head dimension of the model.
const int ROPE_DIM = HEAD_DIM;

// The maximum sequence length for which we precompute embeddings.


/**
 * @brief Applies Rotary Positional Embeddings (RoPE) to query and key vectors.
 *
 * This function modifies the input vectors in-place.
 *
 * @param q     Pointer to the query vector (or a row in a query matrix).
 * @param k     Pointer to the key vector (or a row in a key matrix).
 * @param pos   The position of the token in the sequence.
 */
void apply_rope(sfu_data_t* q, sfu_data_t* k, int pos);

/**
 * @brief Initializes the sin/cos tables for RoPE. For simulation only.
 */
void init_rope_tables();


#endif // ROPE_HPP
