#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// CUDA Kernel for SGEMM: C = alpha * (A * B^T) + beta * C
// A: m x k (Column Major)
// B: n x k (Column Major) - effectively B^T in the math formula
// C: m x n (Column Major)
__global__ void sgemm_kernel(int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global row and column indices for C
    int row = blockIdx.y * TILE_SIZE + ty; 
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Loop over tiles covering the K dimension
    int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile from A into shared memory
        // We use a transposed loading scheme to ensure coalesced global memory access
        // As[tx][ty] stores A[row_idx, col_idx] where row_idx varies with tx (contiguous in memory)
        
        int a_row = blockIdx.y * TILE_SIZE + tx; // Varies with tx (coalesced)
        int a_col = t * TILE_SIZE + ty;          // Constant across warp (for same ty)

        if (a_row < m && a_col < k) {
            As[tx][ty] = A[a_row + a_col * lda];
        } else {
            As[tx][ty] = 0.0f;
        }

        // Load tile from B into shared memory
        // Similar coalesced loading
        // Bs[tx][ty] stores B[row_idx, col_idx]
        
        int b_row = blockIdx.x * TILE_SIZE + tx; // Varies with tx (coalesced) - this is 'col' in C
        int b_col = t * TILE_SIZE + ty;          // Constant across warp - this is 'k' dim

        if (b_row < n && b_col < k) {
            Bs[tx][ty] = B[b_row + b_col * ldb];
        } else {
            Bs[tx][ty] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        // We need A[row, k] * B[col, k]
        // 'row' corresponds to threadIdx.y in our thread mapping for C
        // 'col' corresponds to threadIdx.x in our thread mapping for C
        
        // In As, we stored A[r, c] at As[r%TILE][c%TILE] (relative to block base)
        // With our loading:
        // As[tx][ty] = A[block_row + tx, block_k + ty]
        // We need A[block_row + ty, block_k + i]
        // So we access As[ty][i]
        
        // In Bs, we stored B[r, c] at Bs[r%TILE][c%TILE]
        // With our loading:
        // Bs[tx][ty] = B[block_col + tx, block_k + ty]
        // We need B[block_col + tx, block_k + i]
        // So we access Bs[tx][i]
        
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[ty][i] * Bs[tx][i];
        }
        __syncthreads();
    }

    // Write result
    if (row < m && col < n) {
        int idx = row + col * ldc;
        if (beta == 0.0f) {
             C[idx] = alpha * sum;
        } else {
             C[idx] = alpha * sum + beta * C[idx];
        }
    }
}

// Wrapper function to match the signature used in main.cc
void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc) {
    if ((transa != 'N') && (transa != 'n')) {
        std::cerr << "unsupported value of 'transa'" << std::endl;
        return;
    }
    
    if ((transb != 'T') && (transb != 't')) {
        std::cerr << "unsupported value of 'transb'" << std::endl;
        return;
    }

    float *d_A, *d_B, *d_C;
    
    // Allocate device memory
    cudaMalloc((void**)&d_A, lda * k * sizeof(float));
    cudaMalloc((void**)&d_B, ldb * k * sizeof(float));
    cudaMalloc((void**)&d_C, ldc * n * sizeof(float));

    // Copy inputs
    cudaMemcpy(d_A, A, lda * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, ldb * k * sizeof(float), cudaMemcpyHostToDevice);
    if (beta != 0.0f) {
        cudaMemcpy(d_C, C, ldc * n * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Launch kernel
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

    sgemm_kernel<<<gridDim, blockDim>>>(m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
    
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(C, d_C, ldc * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
