#include <cuda_runtime.h>

#include <iostream>

#include "sgemm_kernel.h"

namespace {

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;

inline bool checkCuda(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error during " << context << ": "
              << cudaGetErrorString(status) << std::endl;
    return false;
  }
  return true;
}

__global__ void sgemmNTKernel(int m, int n, int k, float alpha, const float* A,
                              int lda, const float* BT, int ldb, float beta,
                              float* C, int ldc) {
  __shared__ float aTile[TILE_M][TILE_K];
  __shared__ float bTile[TILE_N][TILE_K];

  const int localCol = threadIdx.x;
  const int localRow = threadIdx.y;

  const int globalRow = blockIdx.y * TILE_M + localRow;
  const int globalCol = blockIdx.x * TILE_N + localCol;

  float accum = 0.0f;

  for (int tileStart = 0; tileStart < k; tileStart += TILE_K) {
    const int aRow = blockIdx.y * TILE_M + localCol;
    const int aCol = tileStart + localRow;
    if (aRow < m && aCol < k) {
      aTile[localCol][localRow] = A[aRow + aCol * lda];
    } else {
      aTile[localCol][localRow] = 0.0f;
    }

    const int bRow = blockIdx.x * TILE_N + localCol;
    const int bCol = tileStart + localRow;
    if (bRow < n && bCol < k) {
      bTile[localCol][localRow] = BT[bRow + bCol * ldb];
    } else {
      bTile[localCol][localRow] = 0.0f;
    }

    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < TILE_K; ++kk) {
      accum += aTile[localRow][kk] * bTile[localCol][kk];
    }

    __syncthreads();
  }

  if (globalRow < m && globalCol < n) {
    const int cIdx = globalRow + globalCol * ldc;
    const float oldC = (beta == 0.0f) ? 0.0f : C[cIdx];
    C[cIdx] = beta * oldC + alpha * accum;
  }
}

}  // namespace

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha,
                const float* A, int lda, const float* B, int ldb, float beta,
                float* C, int ldc) {
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in basicSgemm()"
              << std::endl;
    return;
  }

  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in basicSgemm()"
              << std::endl;
    return;
  }

  const size_t bytesA = static_cast<size_t>(m) * k * sizeof(float);
  const size_t bytesB = static_cast<size_t>(n) * k * sizeof(float);
  const size_t bytesC = static_cast<size_t>(m) * n * sizeof(float);

  float* dA = nullptr;
  float* dB = nullptr;
  float* dC = nullptr;

  if (!checkCuda(cudaMalloc(&dA, bytesA), "cudaMalloc(A)") ||
      !checkCuda(cudaMalloc(&dB, bytesB), "cudaMalloc(B)") ||
      !checkCuda(cudaMalloc(&dC, bytesC), "cudaMalloc(C)")) {
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return;
  }

  if (!checkCuda(cudaMemcpy(dA, A, bytesA, cudaMemcpyHostToDevice),
                 "cudaMemcpy(A H2D)") ||
      !checkCuda(cudaMemcpy(dB, B, bytesB, cudaMemcpyHostToDevice),
                 "cudaMemcpy(B H2D)")) {
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return;
  }

  if (beta == 0.0f) {
    if (!checkCuda(cudaMemset(dC, 0, bytesC), "cudaMemset(C)")) {
      cudaFree(dA);
      cudaFree(dB);
      cudaFree(dC);
      return;
    }
  } else if (!checkCuda(cudaMemcpy(dC, C, bytesC, cudaMemcpyHostToDevice),
                        "cudaMemcpy(C H2D)")) {
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return;
  }

  dim3 block(TILE_N, TILE_M);
  dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);

  sgemmNTKernel<<<grid, block>>>(m, n, k, alpha, dA, lda, dB, ldb, beta, dC,
                                 ldc);

  if (!checkCuda(cudaGetLastError(), "kernel launch") ||
      !checkCuda(cudaDeviceSynchronize(), "kernel execution") ||
      !checkCuda(cudaMemcpy(C, dC, bytesC, cudaMemcpyDeviceToHost),
                 "cudaMemcpy(C D2H)")) {
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return;
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}
