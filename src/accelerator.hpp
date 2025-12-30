#ifndef ACCELERATOR_HPP
#define ACCELERATOR_HPP

#include "config.hpp"

// Define a maximum matrix size for simulation purposes.
// In a real HLS project, these would be pointers to global memory (e.g., DDR).
#define MAX_MATRIX_ROWS 64
#define MAX_MATRIX_COLS 64

// Redefine matrix types for clarity at the accelerator level
// In the pointer-based interface, these represent the data type in memory.
using global_data_t = vpu_data_t;
using global_acc_t = vpu_acc_t;

/**
 * @brief The top-level accelerator function for large matrix multiplication.
 * 
 * This function implements a tiled matrix multiplication algorithm, allowing
 * a fixed-size MPE to process large matrices.
 * 
 * @param A         Input Matrix A (pointer to global memory).
 * @param B         Input Matrix B (pointer to global memory).
 * @param C         Output Matrix C (pointer to global memory).
 * @param M         Number of rows in matrix A and C.
 * @param K         Number of columns in A / rows in B.
 * @param N         Number of columns in matrix B and C.
 */
void flight_llm_accelerator(
    const global_data_t* A,
    const global_data_t* B,
    global_acc_t* C,
    int M, int K, int N
);

#endif // ACCELERATOR_HPP
