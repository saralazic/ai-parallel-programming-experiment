/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*
 * Main entry of dense matrix-matrix multiplication kernel (CUDA version)
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <vector>
#include <iostream>

// SGEMM kernel (defined in sgemm_kernel.cu)
extern void basicSgemm(char transa, char transb,
                       int m, int n, int k,
                       float alpha, const float *A, int lda,
                       const float *B, int ldb,
                       float beta,  float *C, int ldc);

// I/O routines (defined in io.cc)
extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col,
                                   std::vector<float> &v);
extern bool writeColMajorMatrixFile(const char *fn, int, int,
                                    std::vector<float> &);

static double getTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[]) {

  if (argc < 4) {
    fprintf(stderr, "Usage: %s <matA> <matB> <matBT> [outFile]\n", argv[0]);
    exit(-1);
  }

  double wallStart = getTime();

  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;

  readColMajorMatrixFile(argv[1], matArow, matAcol, matA);
  readColMajorMatrixFile(argv[3], matBcol, matBrow, matBT);

  std::vector<float> matC(matArow * matBcol);

  double computeStart = getTime();

  basicSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f,
             &matA.front(), matArow, &matBT.front(), matBcol, 0.0f,
             &matC.front(), matArow);

  double computeTime = getTime() - computeStart;

  if (argc > 4) {
    writeColMajorMatrixFile(argv[4], matArow, matBcol, matC);
  }

  double wallTime = getTime() - wallStart;

  std::cout << "Timer Wall Time: " << wallTime << "s" << std::endl;
  std::cout << "Compute Time: "    << computeTime << "s" << std::endl;
  std::cout << "GFLOPs = "
            << 2.0 * matArow * matBcol * matAcol / computeTime / 1e9
            << std::endl;
  return 0;
}
