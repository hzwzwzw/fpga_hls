#ifndef MPE_HPP
#define MPE_HPP

#include "vpu.hpp"

// Define the dimensions for the Matrix Processing Engine (MPE)
// For simplicity, we'll define a square matrix.
// The number of columns in the matrix must match VPU_SIZE
// to allow direct use of the vpu function.
const int MPE_ROWS = 16;
const int MPE_COLS = VPU_SIZE; // Should be 16

// Type definitions for matrix and vector
using matrix_t = vpu_data_t[MPE_ROWS][MPE_COLS];
using vector_t = vpu_data_t[MPE_COLS];
using result_vector_t = vpu_acc_t[MPE_ROWS];
using result_matrix_t = vpu_acc_t[MPE_ROWS][MPE_COLS];

/**
 * @brief Matrix-Vector multiplication using MPE.
 * 
 * @param matrix_a Input matrix A.
 * @param vector_b Input vector B.
 * @param result   Output vector where result = matrix_a * vector_b.
 */
void mpe_mv(
    matrix_t matrix_a,
    vector_t vector_b,
    result_vector_t result
);

/**
 * @brief Matrix-Matrix multiplication using MPE.
 * 
 * @param matrix_a Input matrix A.
 * @param matrix_b Input matrix B.
 * @param result   Output matrix where result = matrix_a * matrix_b.
 */
void mpe_mm(
    matrix_t matrix_a,
    matrix_t matrix_b,
    result_matrix_t result
);

#endif // MPE_HPP
