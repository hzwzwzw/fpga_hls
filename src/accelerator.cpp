#include "accelerator.hpp"

// Helper functions for loading/storing tiles.
// In HLS, these would be optimized to burst-read/write from global memory.
void load_tile_a(const vpu_data_t A[MAX_MATRIX_ROWS][MAX_MATRIX_COLS], matrix_t tile_a, int tile_row, int k_tile) {
    for (int i = 0; i < TILE_ROWS; ++i) {
        for (int j = 0; j < TILE_COLS; ++j) {
            tile_a[i][j] = A[tile_row * TILE_ROWS + i][k_tile * TILE_COLS + j];
        }
    }
}

void load_tile_b(const vpu_data_t B[MAX_MATRIX_ROWS][MAX_MATRIX_COLS], matrix_t tile_b, int k_tile, int tile_col) {
    for (int i = 0; i < TILE_ROWS; ++i) {
        for (int j = 0; j < TILE_COLS; ++j) {
            tile_b[i][j] = B[k_tile * TILE_ROWS + i][tile_col * TILE_COLS + j];
        }
    }
}

void store_tile_c(vpu_acc_t C[MAX_MATRIX_ROWS][MAX_MATRIX_COLS], result_matrix_t tile_c, int tile_row, int tile_col) {
    for (int i = 0; i < TILE_ROWS; ++i) {
        for (int j = 0; j < TILE_COLS; ++j) {
            C[tile_row * TILE_ROWS + i][tile_col * TILE_COLS + j] = tile_c[i][j];
        }
    }
}


void flight_llm_accelerator(
    const vpu_data_t A[MAX_MATRIX_ROWS][MAX_MATRIX_COLS],
    const vpu_data_t B[MAX_MATRIX_ROWS][MAX_MATRIX_COLS],
    vpu_acc_t C[MAX_MATRIX_ROWS][MAX_MATRIX_COLS],
    int M, int K, int N
) {
    // Tiling loops
    // These loops iterate over the large matrices tile by tile.
    for (int tile_row = 0; tile_row < M / TILE_ROWS; ++tile_row) {
        for (int tile_col = 0; tile_col < N / TILE_COLS; ++tile_col) {

            // --- On-chip buffers for one output tile and its inputs ---
            // These will be synthesized into BRAMs on the FPGA.
            matrix_t tile_a;
            matrix_t tile_b;
            result_matrix_t tile_c = {0}; // Accumulator tile for C, must be initialized to 0.
            result_matrix_t partial_sum;  // Temporary storage for MPE result

            // K-loop: iterates through the tiles of A's columns and B's rows
            // to compute one full output tile of C.
            for (int k_tile = 0; k_tile < K / TILE_COLS; ++k_tile) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE
#endif
                // 1. Load tiles from global memory into on-chip BRAMs
                load_tile_a(A, tile_a, tile_row, k_tile);
                load_tile_b(B, tile_b, k_tile, tile_col);

                // 2. Compute: Perform matrix multiplication on the tiles using the MPE
                mpe_mm(tile_a, tile_b, partial_sum);

                // 3. Accumulate the partial result into the output tile buffer
                for (int i = 0; i < TILE_ROWS; ++i) {
                    for (int j = 0; j < TILE_COLS; ++j) {
                        tile_c[i][j] += partial_sum[i][j];
                    }
                }
            }

            // 4. Store the final computed tile from BRAM back to global memory
            store_tile_c(C, tile_c, tile_row, tile_col);
        }
    }
}
