/** @file mandelbrot.cu - Mandelbrot CUDA reference (matches sequential algorithm) */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define M 500
#define N 500
#define COUNT_MAX 2000
#define BLOCK_SIZE 16

#define CUCHECK(call)                                                          \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/*
 * Same iteration / coloring as the sequential program:
 *   - coordinate mapping with (i-1)/(j-1) formula
 *   - escape when |x|>2 or |y|>2 (box test, not |z|^2 > 4)
 *   - count stays 0 if point never escapes
 */
__global__ void mandelbrot_kernel(int *r, int *g, int *b,
                                  int m, int n, int count_max,
                                  double x_min, double x_max,
                                  double y_min, double y_max)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= m || col >= n)
        return;

    double y = ((double)(row - 1) * y_max + (double)(m - row) * y_min) /
               (double)(m - 1);
    double x = ((double)(col - 1) * x_max + (double)(n - col) * x_min) /
               (double)(n - 1);

    double x1 = x;
    double y1 = y;
    int count = 0;

    for (int k = 1; k <= count_max; k++) {
        double x2 = x1 * x1 - y1 * y1 + x;
        double y2 = 2.0 * x1 * y1 + y;

        if (x2 < -2.0 || 2.0 < x2 || y2 < -2.0 || 2.0 < y2) {
            count = k;
            break;
        }
        x1 = x2;
        y1 = y2;
    }

    int idx = row * n + col;
    if ((count % 2) == 1) {
        r[idx] = 255;
        g[idx] = 255;
        b[idx] = 255;
    } else {
        int c = (int)(255.0 * sqrt(sqrt(sqrt(
            (double)count / (double)count_max))));
        r[idx] = 3 * c / 5;
        g[idx] = 3 * c / 5;
        b[idx] = c;
    }
}

int main(void)
{
    const int m = M;
    const int n = N;
    const int count_max = COUNT_MAX;
    const double x_min = -2.25, x_max = 1.25;
    const double y_min = -1.75, y_max = 1.75;
    const char *output_filename = "mandelbrot_cuda.ppm";

    size_t nbytes = (size_t)m * n * sizeof(int);
    int *h_r = (int *)malloc(nbytes);
    int *h_g = (int *)malloc(nbytes);
    int *h_b = (int *)malloc(nbytes);
    int *d_r, *d_g, *d_b;

    CUCHECK(cudaMalloc(&d_r, nbytes));
    CUCHECK(cudaMalloc(&d_g, nbytes));
    CUCHECK(cudaMalloc(&d_b, nbytes));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    mandelbrot_kernel<<<grid, block>>>(d_r, d_g, d_b, m, n, count_max,
                                       x_min, x_max, y_min, y_max);
    CUCHECK(cudaDeviceSynchronize());

    CUCHECK(cudaMemcpy(h_r, d_r, nbytes, cudaMemcpyDeviceToHost));
    CUCHECK(cudaMemcpy(h_g, d_g, nbytes, cudaMemcpyDeviceToHost));
    CUCHECK(cudaMemcpy(h_b, d_b, nbytes, cudaMemcpyDeviceToHost));

    FILE *fp = fopen(output_filename, "wt");
    if (!fp) {
        perror("fopen");
        return EXIT_FAILURE;
    }
    fprintf(fp, "P3\n");
    fprintf(fp, "%d  %d\n", n, m);
    fprintf(fp, "%d\n", 255);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(fp, "  %d  %d  %d", h_r[i * n + j], h_g[i * n + j],
                    h_b[i * n + j]);
            if ((j + 1) % 4 == 0 || j == n - 1)
                fprintf(fp, "\n");
        }
    }
    fclose(fp);

    printf("Mandelbrot CUDA reference: %dx%d, max %d iterations\n", n, m,
           count_max);
    printf("  Graphics data written to \"%s\".\n", output_filename);

    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    free(h_r);
    free(h_g);
    free(h_b);
    return 0;
}
