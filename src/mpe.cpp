#include "mpe.hpp"

void mpe_mv(
    matrix_t matrix_a,
    vector_t vector_b,
    result_vector_t result
) {
    // Iterate over each row of matrix_a
    // Each iteration computes one element of the output vector.
    MV_ROW_LOOP:
    for (int i = 0; i < MPE_ROWS; ++i) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL factor=MPE_ROWS
#endif
        // For each row, compute the dot product with vector_b.
        // This is exactly what our VPU does.
        result[i] = vpu(matrix_a[i], vector_b);
    }
}

void mpe_mm(
    matrix_t matrix_a,
    matrix_t matrix_b,
    result_matrix_t result
) {
    // To compute C = A * B, we can compute each column of C as C_j = A * B_j,
    // where B_j is the j-th column of B. This is a series of matrix-vector multiplications.

    MM_COL_LOOP:
    for (int j = 0; j < MPE_COLS; ++j) {
        // Temporary vector to hold the j-th column of matrix_b
        vector_t col_b;

        // Extract the j-th column from matrix_b
        EXTRACT_COL_LOOP:
        for (int i = 0; i < MPE_ROWS; ++i) {
            col_b[i] = matrix_b[i][j];
        }

        // Temporary result vector for the matrix-vector product
        result_vector_t res_col;

        // Calculate the j-th column of the result matrix
        mpe_mv(matrix_a, col_b, res_col);

        // Write the result to the final output matrix
        WRITE_COL_LOOP:
        for (int i = 0; i < MPE_ROWS; ++i) {
            result[i][j] = res_col[i];
        }
    }
}
