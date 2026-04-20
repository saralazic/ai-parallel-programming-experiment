#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>

// Polyfill for atomicAdd on double for older architectures (< Compute Capability 6.0)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__device__ double potential(double a, double x) {
    return 2.0 * pow(x / a / a, 2) + 1.0 / (a * a);
}

// Kernel to initialize the parallel random number generator state per thread
__global__ void init_curand(curandState *state, unsigned long long seed, int total_threads) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < total_threads) {
        // Give each thread a unique sequence (id)
        curand_init(seed, id, 0, &state[id]);
    }
}

// Main computational kernel for the Feynman-Kac 1D random walk
__global__ void feynman_kac_kernel(
    int ni, int n, double a, double h, double rth, 
    curandState *state, double *d_wt, int *d_steps) 
{
    // Grid Setup:
    // Y-dimension is the point index 'i' (0 to ni+1)
    // X-dimension is the trajectory index for that specific point
    int i = blockIdx.y;
    int traj_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate the spatial coordinate 'x' for this grid point
    double x = ( (double)(ni - i) * (-a) + (double)(i - 1) * a ) / (double)(ni - 1);
    double test = a * a - x * x;
    
    double wt_local = 0.0;
    int steps_local = 0;
    
    // Only process trajectories that start strictly inside the interval
    if (test >= 0.0 && traj_idx < n) {
        // Global flat index for the RNG state
        int id = i * n + traj_idx; 
        
        // Copy state to local memory for faster access during the loop
        curandState local_state = state[id];
        
        double x1 = x;
        double w = 1.0;
        double chk = 0.0;
        
        while (chk < 1.0) {
            // Generate uniform random double in (0, 1]
            double us = curand_uniform_double(&local_state) - 0.5;
            double dx = (us < 0.0) ? -rth : rth;
            
            double vs = potential(a, x1);
            x1 = x1 + dx;
            steps_local++;
            
            double vh = potential(a, x1);
            double we = (1.0 - h * vs) * w;
            w = w - 0.5 * h * (vh * we + vs * w);
            
            chk = pow(x1 / a, 2);
        }
        
        // Save RNG state back to global memory (optional here since we only run once)
        state[id] = local_state;
        wt_local = w;
    }
    
    // ========================================================
    // Block-level reduction using Shared Memory
    // ========================================================
    __shared__ double s_wt[256];
    __shared__ int s_steps[256];
    
    int tid = threadIdx.x;
    s_wt[tid] = wt_local;
    s_steps[tid] = steps_local;
    
    __syncthreads();
    
    // Parallel tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_wt[tid] += s_wt[tid + s];
            s_steps[tid] += s_steps[tid + s];
        }
        __syncthreads();
    }
    
    // The first thread in the block adds the block's sum to Global Memory
    if (tid == 0) {
        atomicAdd(&d_wt[i], s_wt[0]);
        atomicAdd(&d_steps[i], s_steps[0]);
    }
}

void timestamp(void) {
#define TIME_SIZE 40
    static char time_buffer[TIME_SIZE];
    const struct tm *tm;
    size_t len;
    time_t now;

    now = time(NULL);
    tm = localtime(&now);

    len = strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);
    printf("%s\n", time_buffer);
#undef TIME_SIZE
}

int main(int argc, char **argv) {
    double a = 2.0;
    double h = 0.0001;
    int n = 10000;
    int ni = 21;
    int num_points = ni + 2;
    double rth = sqrt(h);
    int seed = 123456789;

    timestamp();

    printf("\n");
    printf("FEYNMAN_KAC_1D:\n");
    printf("  CUDA parallel version.\n");
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

    // ========================================================
    // 1. Allocate host memory
    // ========================================================
    double *h_wt = (double*)malloc(num_points * sizeof(double));
    int *h_steps = (int*)malloc(num_points * sizeof(int));

    // ========================================================
    // 2. Allocate and initialize device memory
    // ========================================================
    double *d_wt;
    int *d_steps;
    cudaMalloc((void**)&d_wt, num_points * sizeof(double));
    cudaMalloc((void**)&d_steps, num_points * sizeof(int));
    cudaMemset(d_wt, 0, num_points * sizeof(double));
    cudaMemset(d_steps, 0, num_points * sizeof(int));

    // Allocate memory for curand states
    int total_threads = num_points * n;
    curandState *d_state;
    cudaMalloc((void**)&d_state, total_threads * sizeof(curandState));

    // ========================================================
    // 3. Initialize parallel RNG states
    // ========================================================
    int blockSize = 256;
    int gridSizeInit = (total_threads + blockSize - 1) / blockSize;
    init_curand<<<gridSizeInit, blockSize>>>(d_state, seed, total_threads);
    cudaDeviceSynchronize();

    // ========================================================
    // 4. Organize threads into blocks and grids
    // ========================================================
    dim3 block(blockSize, 1);
    // Grid: X-dim covers the `n` trajectories, Y-dim covers the `num_points` grid points
    dim3 grid((n + blockSize - 1) / blockSize, num_points);
    
    // Launch main computation kernel
    feynman_kac_kernel<<<grid, block>>>(ni, n, a, h, rth, d_state, d_wt, d_steps);
    cudaDeviceSynchronize();

    // ========================================================
    // 5. Transfer results from device back to host
    // ========================================================
    cudaMemcpy(h_wt, d_wt, num_points * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_steps, d_steps, num_points * sizeof(int), cudaMemcpyDeviceToHost);

    // ========================================================
    // 6. Final Host Computation and Output
    // ========================================================
    double err = 0.0;
    int n_int = 0;
    int k = 0;

    printf("\n");
    printf("     I     K       X           W exact");
    printf("      W Approx        Error      Ave Steps  Test\n");
    printf("\n");

    for (int i = 0; i <= ni + 1; i++) {
        double x = ( (double)(ni - i) * (-a) + (double)(i - 1) * a ) / (double)(ni - 1);
        k = k + 1;
        double test = a * a - x * x;

        double w_exact, wt;
        int steps_ave;

        // If outside the interval, handle it directly on CPU (Kernel also skipped it)
        if (test < 0.0) {
            w_exact = 1.0;
            wt = 1.0;
            steps_ave = 0;
            printf("  %4d  %4d  %12g  %12g  %12g  %12g  %8d  %8g\n",
                   i, k, x, w_exact, wt, fabs(w_exact - wt), steps_ave, test);
            continue;
        }

        n_int = n_int + 1;
        w_exact = exp(pow(x / a, 2) - 1.0);
        
        // Extract the accumulated sum for this grid point
        wt = h_wt[i] / (double)(n);
        steps_ave = h_steps[i] / n;

        err = err + pow(w_exact - wt, 2);

        printf("  %4d  %4d  %12g  %12g  %12g  %12g  %8d  %8g\n",
               i, k, x, w_exact, wt, fabs(w_exact - wt), steps_ave, test);
    }

    err = sqrt(err / (double)(n_int));

    printf("\n");
    printf("  RMS absolute error in solution = %g\n", err);
    printf("\n");
    printf("FEYNMAN_KAC_1D:\n");
    printf("  Normal end of execution.\n");
    printf("\n");
    timestamp();

    // ========================================================
    // 7. Cleanup memory (Host & Device)
    // ========================================================
    cudaFree(d_wt);
    cudaFree(d_steps);
    cudaFree(d_state);
    free(h_wt);
    free(h_steps);

    return 0;
}
