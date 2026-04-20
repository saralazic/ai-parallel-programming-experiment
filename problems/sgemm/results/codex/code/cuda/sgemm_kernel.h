#ifndef SGEMM_KERNEL_H
#define SGEMM_KERNEL_H

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha,
                const float *A, int lda, const float *B, int ldb, float beta,
                float *C, int ldc);

#endif  // SGEMM_KERNEL_H
