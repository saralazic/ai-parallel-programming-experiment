#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define M 500
#define N 500
#define COUNT_MAX 2000
#define BLOCK_SIZE 16

/*
  Constant memory for read-only parameters shared by all threads.
  Stored in GPU constant cache — broadcast to all threads in a warp
  in a single transaction, avoiding redundant global memory reads.
*/
__constant__ double d_x_min, d_x_max, d_y_min, d_y_max;
__constant__ int    d_m, d_n, d_count_max;

/*
  CUDA kernel: one thread per pixel.
  Each thread independently computes the escape-time iteration for its
  assigned (row, col) and writes the resulting RGB values to global memory.
  All intermediate iteration variables (x1, y1, x2, y2, k) live in
  registers (local memory) — fastest storage on the GPU.
*/
__global__ void mandelbrot_kernel(int *r, int *g, int *b)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= d_m || col >= d_n)
        return;

    double y = ((double)(row - 1) * d_y_max
              + (double)(d_m - row) * d_y_min)
              / (double)(d_m - 1);

    double x = ((double)(col - 1) * d_x_max
              + (double)(d_n - col) * d_x_min)
              / (double)(d_n - 1);

    double x1 = x;
    double y1 = y;
    int count = 0;

    for (int k = 1; k <= d_count_max; k++)
    {
        double x2 = x1 * x1 - y1 * y1 + x;
        double y2 = 2.0 * x1 * y1 + y;

        if (x2 < -2.0 || 2.0 < x2 || y2 < -2.0 || 2.0 < y2)
        {
            count = k;
            break;
        }
        x1 = x2;
        y1 = y2;
    }

    int idx = row * d_n + col;

    if ((count % 2) == 1)
    {
        r[idx] = 255;
        g[idx] = 255;
        b[idx] = 255;
    }
    else
    {
        int c = (int)(255.0 * sqrt(sqrt(sqrt(
                    (double)count / (double)d_count_max))));
        r[idx] = 3 * c / 5;
        g[idx] = 3 * c / 5;
        b[idx] = c;
    }
}

void timestamp(void)
{
#define TIME_SIZE 40
    static char time_buffer[TIME_SIZE];
    const struct tm *tm_ptr;
    time_t now;

    now = time(NULL);
    tm_ptr = localtime(&now);
    strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr);
    printf("%s\n", time_buffer);
#undef TIME_SIZE
}

static inline int i4_min(int i1, int i2) { return i1 < i2 ? i1 : i2; }

int main(void)
{
    int m = M;
    int n = N;
    int count_max = COUNT_MAX;
    double x_max =  1.25;
    double x_min = -2.25;
    double y_max =  1.75;
    double y_min = -1.75;

    size_t img_bytes = (size_t)m * n * sizeof(int);

    timestamp();
    printf("\nMANDELBROT_CUDA\n");

    /* ------------------------------------------------------------------ */
    /*  Host memory allocation (pinned for faster DMA transfers)          */
    /* ------------------------------------------------------------------ */
    int *h_r, *h_g, *h_b;
    cudaHostAlloc((void **)&h_r, img_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_g, img_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_b, img_bytes, cudaHostAllocDefault);

    /* ------------------------------------------------------------------ */
    /*  Device memory allocation (global memory for output arrays)        */
    /* ------------------------------------------------------------------ */
    int *d_r, *d_g, *d_b;
    cudaMalloc((void **)&d_r, img_bytes);
    cudaMalloc((void **)&d_g, img_bytes);
    cudaMalloc((void **)&d_b, img_bytes);

    /* ------------------------------------------------------------------ */
    /*  Copy read-only parameters to constant memory                      */
    /* ------------------------------------------------------------------ */
    cudaMemcpyToSymbol(d_x_min,     &x_min,     sizeof(double));
    cudaMemcpyToSymbol(d_x_max,     &x_max,     sizeof(double));
    cudaMemcpyToSymbol(d_y_min,     &y_min,     sizeof(double));
    cudaMemcpyToSymbol(d_y_max,     &y_max,     sizeof(double));
    cudaMemcpyToSymbol(d_m,         &m,         sizeof(int));
    cudaMemcpyToSymbol(d_n,         &n,         sizeof(int));
    cudaMemcpyToSymbol(d_count_max, &count_max, sizeof(int));

    /* ------------------------------------------------------------------ */
    /*  Configure 2D grid and block dimensions                            */
    /*                                                                    */
    /*  Block: 16x16 = 256 threads (multiple of warp size 32).            */
    /*  Grid:  ceil(N/16) x ceil(M/16) blocks to cover the full image.    */
    /*  threadIdx.x maps to columns so that adjacent threads in a warp    */
    /*  access adjacent memory addresses (coalesced writes).              */
    /* ------------------------------------------------------------------ */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* ------------------------------------------------------------------ */
    /*  Launch kernel with GPU timing via CUDA events                     */
    /* ------------------------------------------------------------------ */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mandelbrot_kernel<<<grid, block>>>(d_r, d_g, d_b);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    printf("\n  GPU kernel time = %g ms  (%g seconds)\n",
           gpu_ms, gpu_ms / 1000.0);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* Check for kernel launch errors */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /*  Copy results from device global memory back to host               */
    /* ------------------------------------------------------------------ */
    cudaMemcpy(h_r, d_r, img_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g, d_g, img_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, img_bytes, cudaMemcpyDeviceToHost);

    /* ------------------------------------------------------------------ */
    /*  Write PPM image on host (sequential I/O — cannot be parallelized) */
    /* ------------------------------------------------------------------ */
    const char *output_filename = "mandelbrot_cuda.ppm";
    FILE *output_unit = fopen(output_filename, "wt");

    fprintf(output_unit, "P3\n");
    fprintf(output_unit, "%d  %d\n", n, m);
    fprintf(output_unit, "%d\n", 255);

    for (int i = 0; i < m; i++)
    {
        for (int jlo = 0; jlo < n; jlo += 4)
        {
            int jhi = i4_min(jlo + 4, n);
            for (int j = jlo; j < jhi; j++)
            {
                int idx = i * n + j;
                fprintf(output_unit, "  %d  %d  %d",
                        h_r[idx], h_g[idx], h_b[idx]);
            }
            fprintf(output_unit, "\n");
        }
    }

    fclose(output_unit);
    printf("\n  Graphics data written to \"%s\".\n", output_filename);

    /* ------------------------------------------------------------------ */
    /*  Free device and host memory                                       */
    /* ------------------------------------------------------------------ */
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFreeHost(h_r);
    cudaFreeHost(h_g);
    cudaFreeHost(h_b);

    printf("\nMANDELBROT_CUDA\n");
    printf("  Normal end of execution.\n\n");
    timestamp();

    return 0;
}
