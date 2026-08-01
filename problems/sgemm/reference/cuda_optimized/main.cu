/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*
 * Standalone entry point for Parboil SGEMM CUDA optimized reference.
 * Uses register-tiled kernel (regtileSgemm) with shared memory.
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <vector>
#include <iostream>

extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v);
extern bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float>&);

extern void regtileSgemm(char transa, char transb, int m, int n, int k,
    float alpha, const float *A, int lda, const float *B, int ldb,
    float beta, float *C, int ldc);

int
main (int argc, char *argv[]) {

  if (argc < 4) {
    fprintf(stderr, "Usage: %s <matA> <matB> <matBT> [outFile]\n", argv[0]);
    exit(-1);
  }

  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;

  readColMajorMatrixFile(argv[1], matArow, matAcol, matA);
  readColMajorMatrixFile(argv[3], matBcol, matBrow, matBT);

  size_t A_sz = matArow * matAcol * sizeof(float);
  size_t B_sz = matBrow * matBcol * sizeof(float);
  size_t C_sz = matArow * matBcol * sizeof(float);

  std::vector<float> matC(matArow * matBcol);

  float *dA, *dB, *dC;
  cudaMalloc((void**)&dA, A_sz);
  cudaMalloc((void**)&dB, B_sz);
  cudaMalloc((void**)&dC, C_sz);

  cudaMemcpy(dA, &matA.front(), A_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, &matBT.front(), B_sz, cudaMemcpyHostToDevice);
  cudaMemset(dC, 0, C_sz);

  cudaDeviceSynchronize();

  regtileSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f,
      dA, matArow, dB, matBcol, 0.0f, dC, matArow);

  cudaDeviceSynchronize();

  cudaMemcpy(&matC.front(), dC, C_sz, cudaMemcpyDeviceToHost);

  if (argc > 4) {
    writeColMajorMatrixFile(argv[4], matArow, matBcol, matC);
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return 0;
}
