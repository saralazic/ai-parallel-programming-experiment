#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// CUDA Kernel for Mandelbrot set computation
// -----------------------------------------------------------------------------
__global__ void mandelbrot_kernel(
    unsigned char *r, unsigned char *g, unsigned char *b,
    int m, int n, int count_max,
    double x_max, double x_min, double y_max, double y_min) 
{
    // Calculate global thread coordinates (pixel coordinates)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row

    // Ensure we don't process outside the image boundaries
    if (i < m && j < n) {
        // Map pixel coordinates to the complex plane
        double y = ((double)(i - 1) * y_max + (double)(m - i) * y_min) / (double)(m - 1);
        double x = ((double)(j - 1) * x_max + (double)(n - j) * x_min) / (double)(n - 1);

        int count = 0;
        double x1 = x;
        double y1 = y;

        // Mandelbrot iteration
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

        // Calculate 1D index from 2D coordinates
        int index = i * n + j;

        // Assign colors based on iterations
        if (count % 2 == 1) {
            r[index] = 255;
            g[index] = 255;
            b[index] = 255;
        } else {
            // Using sqrt repeatedly to smooth the color gradient
            int c = (int)(255.0 * sqrt(sqrt(sqrt(((double)count / (double)count_max)))));
            r[index] = 3 * c / 5;
            g[index] = 3 * c / 5;
            b[index] = c;
        }
    }
}

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------
int i4_min(int i1, int i2) {
    return (i1 < i2) ? i1 : i2;
}

void timestamp() {
#define TIME_SIZE 40
    static char time_buffer[TIME_SIZE];
    const struct tm *tm;
    time_t now;

    now = time(NULL);
    tm = localtime(&now);

    strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);
    printf("%s\n", time_buffer);
#undef TIME_SIZE
}

// -----------------------------------------------------------------------------
// Main Function
// -----------------------------------------------------------------------------
int main() {
    int m = 500;
    int n = 500;
    int count_max = 2000;
    
    double x_max = 1.25;
    double x_min = -2.25;
    double y_max = 1.75;
    double y_min = -1.75;

    char *output_filename = "mandelbrot_cuda.ppm";
    FILE *output_unit;

    timestamp();
    printf("\n");
    printf("MANDELBROT CUDA\n");
    printf("  Generating %d x %d image.\n", n, m);

    // 1. Allocate host (CPU) memory
    // Using unsigned char instead of int reduces memory transfer by 75%
    size_t image_size = m * n * sizeof(unsigned char);
    unsigned char *h_r = (unsigned char *)malloc(image_size);
    unsigned char *h_g = (unsigned char *)malloc(image_size);
    unsigned char *h_b = (unsigned char *)malloc(image_size);

    // 2. Allocate device (GPU) memory (Global Memory)
    unsigned char *d_r, *d_g, *d_b;
    cudaMalloc((void **)&d_r, image_size);
    cudaMalloc((void **)&d_g, image_size);
    cudaMalloc((void **)&d_b, image_size);

    // 3. Set up execution configuration
    // Organizing threads into 2D blocks and a 2D grid
    dim3 threadsPerBlock(16, 16); // 256 threads per block
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Timing with CUDA events for accurate device timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 4. Launch kernel
    mandelbrot_kernel<<<numBlocks, threadsPerBlock>>>(
        d_r, d_g, d_b, m, n, count_max, x_max, x_min, y_max, y_min
    );

    // 5. Manage memory transfers from device (GPU) to host (CPU)
    cudaMemcpy(h_r, d_r, image_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g, d_g, image_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, image_size, cudaMemcpyDeviceToHost);

    // Record stop time and compute elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\n");
    printf("  Time = %g seconds.\n", milliseconds / 1000.0);

    // Write data to an ASCII PPM file.
    output_unit = fopen(output_filename, "wt");
    if (!output_unit) {
        printf("  Error: Could not open output file.\n");
        return 1;
    }

    fprintf(output_unit, "P3\n");
    fprintf(output_unit, "%d  %d\n", n, m);
    fprintf(output_unit, "%d\n", 255);

    for (int i = 0; i < m; i++) {
        for (int jlo = 0; jlo < n; jlo = jlo + 4) {
            int jhi = i4_min(jlo + 4, n);
            for (int j = jlo; j < jhi; j++) {
                int index = i * n + j;
                fprintf(output_unit, "  %d  %d  %d", h_r[index], h_g[index], h_b[index]);
            }
            fprintf(output_unit, "\n");
        }
    }

    fclose(output_unit);
    printf("\n");
    printf("  Graphics data written to \"%s\".\n", output_filename);

    // Terminate
    printf("\n");
    printf("MANDELBROT_CUDA\n");
    printf("  Normal end of execution.\n");
    printf("\n");
    timestamp();

    // 6. Free memory
    free(h_r);
    free(h_g);
    free(h_b);
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
