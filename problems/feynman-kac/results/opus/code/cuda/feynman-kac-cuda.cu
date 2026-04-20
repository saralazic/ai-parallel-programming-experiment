#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define THREADS_PER_BLOCK 256

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

/*
 * Device-side potential function V(x).
 * Inlined by the compiler into every thread — pure register arithmetic.
 */
__device__ __forceinline__ double potential_d(double a, double x)
{
    double xa2 = x / (a * a);
    return 2.0 * xa2 * xa2 + 1.0 / (a * a);
}

/*
 * Monte Carlo kernel — one thread per trajectory.
 *
 * Grid layout (2D):
 *   blockIdx.y                              → grid point index  [0, num_points)
 *   blockIdx.x * blockDim.x + threadIdx.x   → trajectory index  [0, n)
 *
 * Memory usage:
 *   Registers / local : per-thread walk state (x1, w, rng state)
 *   Shared             : partial sums for intra-block reduction
 *   Global              : d_wt_sums[], d_step_sums[] (one element per grid point)
 */
__global__ void feynman_kac_kernel(
    double              a,
    double              h,
    double              rth,
    int                 n,
    int                 ni,
    unsigned long long  base_seed,
    double             *d_wt_sums,
    int                *d_step_sums)
{
    int point_idx = blockIdx.y;
    int traj_idx  = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double s_wt[THREADS_PER_BLOCK];
    __shared__ int    s_steps[THREADS_PER_BLOCK];

    double my_w     = 0.0;
    int    my_steps = 0;

    double x = ((double)(ni - point_idx) * (-a)
              + (double)(point_idx - 1)  *   a)
             /  (double)(ni - 1);

    double test = a * a - x * x;

    if (traj_idx < n && test >= 0.0)
    {
        /*
         * Philox4_32_10: counter-based PRNG — fast init, excellent
         * parallelism.  Each thread gets a unique (seed, sequence)
         * pair so streams are statistically independent.
         */
        curandStatePhilox4_32_10_t rng;
        curand_init(base_seed,
                    (unsigned long long)point_idx * n + traj_idx,
                    0, &rng);

        double x1  = x;
        double w   = 1.0;
        double chk = 0.0;

        while (chk < 1.0)
        {
            double us = curand_uniform_double(&rng) - 0.5;
            double dx = (us < 0.0) ? -rth : rth;

            double vs = potential_d(a, x1);

            x1 += dx;
            my_steps++;

            double vh = potential_d(a, x1);

            double we = (1.0 - h * vs) * w;
            w = w - 0.5 * h * (vh * we + vs * w);

            chk = (x1 / a) * (x1 / a);
        }

        my_w = w;
    }

    /*
     * Shared-memory tree reduction.
     * Collapses THREADS_PER_BLOCK partial results into a single
     * value in log2(THREADS_PER_BLOCK) steps, avoiding expensive
     * global atomics from every thread.
     */
    s_wt[threadIdx.x]    = my_w;
    s_steps[threadIdx.x] = my_steps;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            s_wt[threadIdx.x]    += s_wt[threadIdx.x + stride];
            s_steps[threadIdx.x] += s_steps[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&d_wt_sums[point_idx],   s_wt[0]);
        atomicAdd(&d_step_sums[point_idx], s_steps[0]);
    }
}

/******************************************************************************/

void timestamp(void)
{
#define TIME_SIZE 40
    static char time_buffer[TIME_SIZE];
    const struct tm *tm_info;
    time_t now;

    now = time(NULL);
    tm_info = localtime(&now);
    strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_info);
    printf("%s\n", time_buffer);
#undef TIME_SIZE
}

/******************************************************************************/

int main(int argc, char **argv)
{
    double a  = 2.0;
    double h  = 0.0001;
    int    n  = 10000;
    int    ni = 21;
    int    num_points = ni + 2;

    double             rth       = sqrt(h);
    unsigned long long base_seed = 123456789ULL;

    timestamp();

    printf("\n");
    printf("FEYNMAN_KAC_1D:\n");
    printf("  CUDA version.\n");
    printf("\n");
    printf("  Program parameters:\n");
    printf("\n");
    printf("  The calculation takes place inside an interval.\n");
    printf("  The solution will be estimated at points\n");
    printf("  on a regular spaced grid within the interval.\n");
    printf("  Each solution will be estimated by computing %d trajectories\n", n);
    printf("  from the point to the boundary.\n");
    printf("\n");
    printf("    (X/A)^2 = 1\n");
    printf("\n");
    printf("  The interval parameter A is:\n");
    printf("\n");
    printf("    A = %g\n", a);
    printf("\n");
    printf("  Path stepsize H = %g\n", h);
    printf("\n");
    printf("  X coordinate discretized by %d points\n", num_points);

    /* ------------------------------------------------------------------
     * DEVICE MEMORY ALLOCATION
     *
     * Only two small arrays live in global device memory:
     *   d_wt_sums   [num_points doubles] — accumulated w across trajectories
     *   d_step_sums [num_points ints]    — accumulated step counts
     *
     * Everything else (walk state, PRNG state) is in per-thread registers
     * and local memory, managed automatically by the compiler.
     * ------------------------------------------------------------------ */
    double *d_wt_sums   = NULL;
    int    *d_step_sums = NULL;

    CUDA_CHECK(cudaMalloc(&d_wt_sums,   num_points * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_step_sums, num_points * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_wt_sums,   0, num_points * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_step_sums, 0, num_points * sizeof(int)));

    /* ------------------------------------------------------------------
     * KERNEL LAUNCH CONFIGURATION
     *
     * 2D grid:
     *   Y dimension (num_points = 23 blocks) — one row per grid point
     *   X dimension (blocks_x ≈ 40 blocks)   — tiles trajectories
     *
     * Block: 256 threads (1D), each running one trajectory.
     *
     * Shared memory per block:
     *   256 * sizeof(double) + 256 * sizeof(int) = 3072 bytes
     *   (well within the 48 KB default shared memory limit)
     * ------------------------------------------------------------------ */
    int blocks_x = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    dim3 grid(blocks_x, num_points);
    dim3 block(THREADS_PER_BLOCK);

    printf("\n");
    printf("  CUDA launch configuration:\n");
    printf("    Grid:  (%d, %d) blocks\n", blocks_x, num_points);
    printf("    Block: %d threads\n", THREADS_PER_BLOCK);
    printf("    Total threads: %d\n", blocks_x * num_points * THREADS_PER_BLOCK);
    printf("    Shared memory per block: %lu bytes\n",
           THREADS_PER_BLOCK * (sizeof(double) + sizeof(int)));

    /* ------------------------------------------------------------------
     * KERNEL EXECUTION with timing
     * ------------------------------------------------------------------ */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));

    feynman_kac_kernel<<<grid, block>>>(
        a, h, rth, n, ni, base_seed,
        d_wt_sums, d_step_sums);

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float kernel_ms;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop));
    printf("\n");
    printf("  Kernel execution time: %.3f ms\n", kernel_ms);

    /* ------------------------------------------------------------------
     * DEVICE → HOST TRANSFER
     * ------------------------------------------------------------------ */
    double *h_wt_sums   = (double *)malloc(num_points * sizeof(double));
    int    *h_step_sums = (int    *)malloc(num_points * sizeof(int));

    CUDA_CHECK(cudaMemcpy(h_wt_sums,   d_wt_sums,
                           num_points * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_step_sums, d_step_sums,
                           num_points * sizeof(int),    cudaMemcpyDeviceToHost));

    /* ------------------------------------------------------------------
     * HOST-SIDE POST-PROCESSING
     * Exact solutions, averaging, and error computation stay on the host
     * because the work is trivial (23 points) and requires formatted I/O.
     * ------------------------------------------------------------------ */
    printf("\n");
    printf("     I     K       X           W exact");
    printf("      W Approx        Error      Ave Steps  Test\n");
    printf("\n");

    double err   = 0.0;
    int    n_int = 0;
    int    k     = 0;

    for (int i = 0; i <= ni + 1; i++)
    {
        double x = ((double)(ni - i) * (-a)
                  + (double)(i - 1)  *   a)
                 /  (double)(ni - 1);
        k++;

        double test = a * a - x * x;

        if (test < 0.0)
        {
            printf("  %4d  %4d  %12g  %12g  %12g  %12g  %8d  %8g\n",
                   i, k, x, 1.0, 1.0, 0.0, 0, test);
            continue;
        }

        n_int++;

        double w_exact   = exp((x / a) * (x / a) - 1.0);
        double wt        = h_wt_sums[i] / (double)n;
        int    steps_ave = h_step_sums[i] / n;

        err += (w_exact - wt) * (w_exact - wt);

        printf("  %4d  %4d  %12g  %12g  %12g  %12g  %8d  %8g\n",
               i, k, x, w_exact, wt, fabs(w_exact - wt), steps_ave, test);
    }

    err = sqrt(err / (double)n_int);

    printf("\n");
    printf("  RMS absolute error in solution = %g\n", err);

    /* ------------------------------------------------------------------
     * CLEANUP
     * ------------------------------------------------------------------ */
    free(h_wt_sums);
    free(h_step_sums);
    CUDA_CHECK(cudaFree(d_wt_sums));
    CUDA_CHECK(cudaFree(d_step_sums));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    printf("\n");
    printf("FEYNMAN_KAC_1D:\n");
    printf("  Normal end of execution.\n");
    printf("\n");
    timestamp();

    return 0;
}
