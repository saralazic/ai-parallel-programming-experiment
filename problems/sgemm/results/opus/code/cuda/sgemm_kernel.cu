/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*
 * CUDA implementation of SGEMM: C = beta*C + alpha*A*B^T
 *
 * A is m×k, B^T is n×k, C is m×n — all column-major.
 *
 * Uses shared-memory tiling with coalesced global loads and
 * bank-conflict-free shared memory access.  Each thread block
 * computes a BLOCK_SIZE×BLOCK_SIZE tile of C.
 */

#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 16

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                  << " — " << cudaGetErrorString(err) << std::endl;    \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

/* ------------------------------------------------------------------ */
/*  Device kernel                                                      */
/* ------------------------------------------------------------------ */
/*
 * Thread mapping:
 *   row = blockIdx.y * BLOCK_SIZE + threadIdx.y   (row of C)
 *   col = blockIdx.x * BLOCK_SIZE + threadIdx.x   (col of C)
 *
 * Tile loads use threadIdx.x for the stride-1 (row) index of the
 * column-major matrices so that consecutive threads in a warp touch
 * consecutive addresses → coalesced 128-byte transactions.
 *
 * Shared memory layout  As[p][q]  stores  A(blockRow+q, tileStart+p)
 *                       Bs[p][q]  stores  B^T(blockCol+q, tileStart+p)
 *
 * Compute step reads As[kk][ty] and Bs[kk][tx]:
 *   • As[kk][ty] — all threads with the same ty read the same address → broadcast
 *   • Bs[kk][tx] — consecutive tx → consecutive bank addresses → no conflict
 */
__global__ void sgemmKernel(int m, int n, int k,
                            float alpha,
                            const float *A, int lda,
                            const float *B, int ldb,
                            float beta,
                            float *C, int ldc)
{
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCK_SIZE + ty;
    const int col = blockIdx.x * BLOCK_SIZE + tx;

    float acc = 0.0f;
    const int numTiles = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        const int tileStart = t * BLOCK_SIZE;

        /*
         * Load A tile — tx indexes the row dimension (coalesced).
         * Stored with the k-offset in the first index so that the
         * compute loop walks down columns of As.
         */
        int aRow = blockIdx.y * BLOCK_SIZE + tx;
        int aCol = tileStart + ty;
        As[ty][tx] = (aRow < m && aCol < k) ? A[aRow + (size_t)aCol * lda]
                                             : 0.0f;

        /*
         * Load B^T tile — same coalescing strategy.
         */
        int bRow = blockIdx.x * BLOCK_SIZE + tx;
        int bCol = tileStart + ty;
        Bs[ty][tx] = (bRow < n && bCol < k) ? B[bRow + (size_t)bCol * ldb]
                                             : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BLOCK_SIZE; ++kk)
            acc += As[kk][ty] * Bs[kk][tx];

        __syncthreads();
    }

    if (row < m && col < n)
        C[row + (size_t)col * ldc] = beta * C[row + (size_t)col * ldc]
                                   + alpha * acc;
}

/* ------------------------------------------------------------------ */
/*  Host wrapper — keeps the same interface as the CPU version         */
/* ------------------------------------------------------------------ */
void basicSgemm(char transa, char transb,
                int m, int n, int k,
                float alpha, const float *A, int lda,
                const float *B, int ldb,
                float beta,  float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
        std::cerr << "unsupported value of 'transa'" << std::endl;
        return;
    }
    if ((transb != 'T') && (transb != 't')) {
        std::cerr << "unsupported value of 'transb'" << std::endl;
        return;
    }

    /* ----- Allocate device memory --------------------------------- */
    float *d_A, *d_B, *d_C;
    size_t sizeA = (size_t)lda * k * sizeof(float);
    size_t sizeB = (size_t)ldb * k * sizeof(float);
    size_t sizeC = (size_t)ldc * n * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    /* ----- Host → Device transfers -------------------------------- */
    CUDA_CHECK(cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice));
    if (beta != 0.0f)
        CUDA_CHECK(cudaMemcpy(d_C, C, sizeC, cudaMemcpyHostToDevice));

    /* ----- Kernel launch ------------------------------------------ */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);               // 16×16 = 256 threads
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,      // covers columns of C
              (m + BLOCK_SIZE - 1) / BLOCK_SIZE);      // covers rows of C

    sgemmKernel<<<grid, block>>>(m, n, k,
                                 alpha, d_A, lda,
                                        d_B, ldb,
                                 beta,  d_C, ldc);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ----- Device → Host transfer --------------------------------- */
    CUDA_CHECK(cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost));

    /* ----- Free device memory ------------------------------------- */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}
