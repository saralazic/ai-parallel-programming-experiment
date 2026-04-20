/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * OpenMP-parallelized implementation of MM
 */
#include <iostream>
#include <omp.h>

void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }

  /*
   * PARALLELIZATION STRATEGY:
   *
   * We parallelize the two outer loops (mm, nn) using a collapsed 2D
   * parallel region. The inner loop (i) is left sequential because it
   * carries a RAW dependency on the accumulator variable 'c'.
   *
   * Directive breakdown:
   *
   *   #pragma omp parallel for
   *     - Creates a team of threads and distributes loop iterations
   *       among them. Combines the 'parallel' (fork threads) and
   *       'for' (work-sharing) constructs into one directive.
   *
   *   collapse(2)
   *     - Fuses the mm and nn loops into a single iteration space
   *       of size m*n. This is valid because neither loop carries
   *       cross-iteration dependencies. Collapsing increases the
   *       total number of schedulable work units from m to m*n,
   *       which improves load balance when m alone is small or
   *       not evenly divisible by the thread count.
   *
   *   schedule(static)
   *     - Divides the m*n iterations into contiguous chunks of
   *       approximately equal size, one per thread, assigned at
   *       compile time. Chosen because every (mm,nn) work unit
   *       performs exactly k multiply-accumulate operations — the
   *       workload is perfectly uniform, so static scheduling
   *       avoids the overhead of dynamic work-stealing while
   *       providing optimal balance.
   *
   *   shared(A, B, C, m, n, k, alpha, beta, lda, ldb, ldc)
   *     - All array pointers and scalar parameters are shared among
   *       threads. A and B are read-only — concurrent reads are safe
   *       with no synchronization. C is written, but each (mm,nn)
   *       pair maps to a unique element C[mm + nn*ldc], so no two
   *       threads ever write to the same location.
   *
   * Variables declared INSIDE the loop body (c, a, b, i) are
   * automatically private to each thread because they are local
   * to the scope entered by each iteration. No explicit clause
   * is needed for them.
   *
   * RACE CONDITION ANALYSIS:
   *
   *   - C[mm + nn*ldc]: Each (mm,nn) pair produces a unique address.
   *     No two threads write to the same element. NO RACE.
   *
   *   - c (accumulator): Declared at the top of the nn loop body,
   *     so each thread iteration has its own stack-local copy.
   *     NO RACE.
   *
   *   - a, b: Declared inside the i loop body, stack-local. NO RACE.
   *
   *   - A[], B[]: Read-only arrays. Concurrent reads are always safe.
   *     NO RACE.
   *
   * MEMORY ACCESS CONSIDERATIONS:
   *
   *   With collapse(2) and static scheduling, the m*n iteration space
   *   is divided into contiguous chunks. Threads with adjacent mm
   *   ranges access adjacent A and C memory (stride-1 in column-major),
   *   promoting L2/L3 cache sharing. B reads are independent of mm,
   *   so all threads share the same B data — constructive sharing in
   *   shared caches.
   */

  #pragma omp parallel for collapse(2) schedule(static) \
    shared(A, B, C, m, n, k, alpha, beta, lda, ldb, ldc)
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      /*
       * Everything below is private to the thread executing this
       * (mm, nn) iteration — c, a, b, and i are block-scoped locals.
       */
      float c = 0.0f;

      /*
       * The inner loop is NOT parallelized.
       * Reason: c += a * b is a loop-carried RAW dependency.
       * Iteration i reads the value of c written by iteration i-1.
       * Since we already have m*n independent work units from the
       * outer loops, there is abundant parallelism without touching
       * this loop. Keeping it sequential also preserves numerical
       * determinism (accumulation order is fixed).
       */
      for (int i = 0; i < k; ++i) {
        float a = A[mm + i * lda]; 
        float b = B[nn + i * ldb];
        c += a * b;
      }
      C[mm+nn*ldc] = C[mm+nn*ldc] * beta + alpha * c;
    }
  }
}
